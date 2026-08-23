"""
The EigenBible

Kernel PCA over a bible's chapter embeddings, with an LLM-written label per component.

Pulls every vector out of a Milvus collection built by embed_biblia.py, fits a Kernel PCA
on them, and for each of the requested components picks a neighbourhood of chapters that
characterizes that axis - via one of two NeighbourhoodStrategy implementations (see below) -
reads their text back off disk, and hands it to an Ollama chat model, which is asked to name
the shared theme in a few words. Each labelled component is written to a dedicated results
collection as soon as it's ready, since the whole pipeline (kernel eigendecomposition + one
Ollama call per component) can take a while and a crash or Ctrl-C shouldn't lose everything
computed so far.
"""
import argparse
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import requests
from pymilvus import DataType, MilvusClient
from sklearn.decomposition import KernelPCA

from embed_biblia import BIBLES, COMBINED_COLLECTION, MILVUS_URI, OLLAMA_URL

OLLAMA_LABEL_MODEL = "qwen3.5:4b"
N_COMPONENTS = 50
N_NEIGHBOURS = 8
KERNEL = "cosine"  # matches the COSINE index the source collections already use
SNIPPET_CHARS = 600  # per-neighbour text budget kept in the labelling prompt

logger = logging.getLogger(__name__)

# collection name -> bible key, for collections holding a single bible's chapters
COLLECTION_TO_BIBLE = {collection: key for key, (_dirname, collection) in BIBLES.items()}


class VectorCollectionReader:
    """Pages through a Milvus collection built by embed_biblia.py and returns every row."""

    BATCH_SIZE = 1000

    def __init__(self, client: MilvusClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name
        self.has_bible_field = collection_name == COMBINED_COLLECTION

    def fetch_all(self) -> list[dict]:
        output_fields = ["id", "vector", "file_name"] + (["bible"] if self.has_bible_field else [])
        rows = []
        offset = 0
        while True:
            batch = self.client.query(
                collection_name=self.collection_name,
                filter="id >= 0",
                output_fields=output_fields,
                limit=self.BATCH_SIZE,
                offset=offset,
            )
            if not batch:
                break
            rows.extend(batch)
            offset += len(batch)
            logger.info("Fetched %d rows from '%s' so far", offset, self.collection_name)
        return rows

    def bible_of(self, row: dict) -> str:
        return row["bible"] if self.has_bible_field else COLLECTION_TO_BIBLE[self.collection_name]


class ChapterTextResolver:
    """Maps a (bible, file_name) pair back to the chapter markdown on disk - Milvus only
    stores the vector and file_name, not the text itself."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def read(self, bible: str, file_name: str) -> str:
        source_dirname, _collection = BIBLES[bible]
        return (self.base_dir / source_dirname / file_name).read_text(encoding="utf-8")


class KernelPCAReducer:
    def __init__(self, n_components: int, kernel: str):
        self.model = KernelPCA(n_components=n_components, kernel=kernel)

    def fit_transform(self, vectors: np.ndarray) -> np.ndarray:
        logger.info(
            "Fitting kernel PCA (kernel=%s, n_components=%d) on %d vectors of dimension %d - "
            "this is an O(n^2..n^3) eigendecomposition, so it may take a while for a large "
            "collection",
            self.model.kernel, self.model.n_components, *vectors.shape,
        )
        transformed = self.model.fit_transform(vectors)
        logger.info("Kernel PCA fit complete")
        return transformed

    @property
    def eigenvalues(self) -> np.ndarray:
        return self.model.eigenvalues_


class MilvusNearestNeighbourSearch:
    def __init__(self, client: MilvusClient, collection_name: str, has_bible_field: bool):
        self.client = client
        self.collection_name = collection_name
        self.has_bible_field = has_bible_field
        self.client.load_collection(collection_name)

    def search(self, query_vector: list, k: int) -> list[dict]:
        output_fields = ["file_name"] + (["bible"] if self.has_bible_field else [])
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=k,
            output_fields=output_fields,
        )
        return [hit["entity"] for hit in results[0]]


class NeighbourhoodStrategy(ABC):
    """Picks the group(s) of chapters that get summarized into a label for one kPCA component.
    A group is (sign, representative_row_index, neighbour_rows): the sign and representative
    row are what get stored alongside the label, so downstream readers can tell which pole of
    the axis (or single anchor) a given label describes."""

    @abstractmethod
    def neighbourhoods_for_component(
        self, component_index: int, transformed: np.ndarray, rows: list[dict],
    ) -> list[tuple[str, int, list[dict]]]:
        raise NotImplementedError


class AnchorKNNNeighbourhoodStrategy(NeighbourhoodStrategy):
    """Finds the single chapter that sits furthest out on this component's axis (in either
    direction), then asks Milvus for its nearest neighbours in the *original* embedding space.
    One Milvus round-trip per component, one label per component. The neighbours are close to
    the anchor generally, but aren't necessarily the chapters that score highest on this
    specific axis - a weaker signal for the less dominant components."""

    def __init__(self, search: MilvusNearestNeighbourSearch, k: int):
        self.search = search
        self.k = k

    def neighbourhoods_for_component(self, component_index, transformed, rows):
        projections = transformed[:, component_index]
        anchor_idx = int(np.argmax(np.abs(projections)))
        sign = "+" if projections[anchor_idx] >= 0 else "-"
        neighbours = self.search.search(rows[anchor_idx]["vector"], self.k)
        return [(sign, anchor_idx, neighbours)]


class TopProjectionNeighbourhoodStrategy(NeighbourhoodStrategy):
    """For each component, takes the K chapters that score highest on that axis, and
    separately the K that score lowest - a kPCA component is a contrast between two poles,
    not a single point, so this labels both ends. Ranking comes straight from the already
    computed kPCA projections, no Milvus search involved."""

    def __init__(self, k: int):
        self.k = k

    def neighbourhoods_for_component(self, component_index, transformed, rows):
        order = np.argsort(transformed[:, component_index])
        top_negative = order[:self.k]
        top_positive = order[::-1][:self.k]
        return [
            ("+", int(top_positive[0]), [rows[i] for i in top_positive]),
            ("-", int(top_negative[0]), [rows[i] for i in top_negative]),
        ]


class ChapterSummarizer:
    """Asks an Ollama chat model to name the shared theme of a handful of bible passages."""

    def __init__(self, base_url: str, model: str):
        self.url = f"{base_url.rstrip('/')}/api/generate"
        self.model = model
        logger.info("Labelling with Ollama model '%s' at %s", model, self.url)

    def label(self, texts: list[str]) -> str:
        passages = "\n\n---\n\n".join(text[:SNIPPET_CHARS] for text in texts)
        prompt = (
            "The following are excerpts from several bible chapters that were found to be "
            "closely related. In a few words, or at most one short sentence, name the theme "
            "they share. Answer with only the label itself - no preamble, no quotes.\n\n"
            f"{passages}\n\nShared theme:"
        )
        response = requests.post(
            self.url,
            # think=False: qwen3.5:4b is a hybrid reasoning model that otherwise burns its
            # whole generation budget on a hidden "thinking" field for prompts this long,
            # frequently leaving the actual "response" empty (and taking ~80s to do it).
            json={"model": self.model, "prompt": prompt, "stream": False, "think": False},
            timeout=300,
        )
        response.raise_for_status()
        label = response.json()["response"].strip()
        if not label:
            logger.warning("Ollama returned an empty label for a component - keeping it empty rather than failing the run")
        return label


class LabelResultsStore:
    """Where each component's label lands, one row at a time, so a long run checkpoints its
    progress instead of losing everything if it's interrupted before the last component."""

    LABEL_MAX_LENGTH = 4096  # generous headroom - the prompt asks for a sentence but isn't always obeyed

    def __init__(self, client: MilvusClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name
        self.ready = False

    def ensure_collection(self, n_components: int) -> None:
        if self.ready:
            return
        if not self.client.has_collection(self.collection_name):
            logger.info(
                "Creating results collection '%s' (kpca_vector dim=%d)",
                self.collection_name, n_components,
            )
            schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="component_index", datatype=DataType.INT64)
            schema.add_field(field_name="eigenvalue", datatype=DataType.DOUBLE)
            schema.add_field(field_name="sign", datatype=DataType.VARCHAR, max_length=1)
            schema.add_field(field_name="anchor_bible", datatype=DataType.VARCHAR, max_length=64)
            schema.add_field(field_name="anchor_file_name", datatype=DataType.VARCHAR, max_length=1024)
            schema.add_field(field_name="label", datatype=DataType.VARCHAR, max_length=self.LABEL_MAX_LENGTH)
            schema.add_field(field_name="kpca_vector", datatype=DataType.FLOAT_VECTOR, dim=n_components)

            index_params = self.client.prepare_index_params()
            index_params.add_index(field_name="kpca_vector", index_type="AUTOINDEX", metric_type="L2")
            self.client.create_collection(
                collection_name=self.collection_name, schema=schema, index_params=index_params,
            )
            logger.info("Results collection '%s' created", self.collection_name)
        self.ready = True

    def insert(self, record: dict) -> None:
        self.ensure_collection(len(record["kpca_vector"]))
        # Belt and braces: the prompt asks for a short label but the model doesn't always
        # comply, and a VARCHAR overflow would otherwise crash the whole run on insert.
        if len(record["label"]) > self.LABEL_MAX_LENGTH:
            record = {**record, "label": record["label"][:self.LABEL_MAX_LENGTH - 1] + "…"}
        self.client.insert(collection_name=self.collection_name, data=[record])
        logger.debug("Stored label for component %d in '%s'", record["component_index"], self.collection_name)


class KPCALabeler:
    def __init__(
        self,
        reader: VectorCollectionReader,
        text_resolver: ChapterTextResolver,
        reducer: KernelPCAReducer,
        strategy: NeighbourhoodStrategy,
        summarizer: ChapterSummarizer,
        results_store: LabelResultsStore,
    ):
        self.reader = reader
        self.text_resolver = text_resolver
        self.reducer = reducer
        self.strategy = strategy
        self.summarizer = summarizer
        self.results_store = results_store

    def run(self) -> int:
        rows = self.reader.fetch_all()
        if not rows:
            raise ValueError(f"Collection '{self.reader.collection_name}' is empty")
        vectors = np.array([row["vector"] for row in rows])

        transformed = self.reducer.fit_transform(vectors)
        eigenvalues = self.reducer.eigenvalues
        n_components = transformed.shape[1]

        logger.info("Labelling %d components by summarizing each one's characteristic chapters", n_components)
        labelled = 0
        for component_index in range(n_components):
            neighbourhoods = self.strategy.neighbourhoods_for_component(component_index, transformed, rows)
            for sign, representative_idx, neighbour_rows in neighbourhoods:
                representative_row = rows[representative_idx]
                texts = [
                    self.text_resolver.read(self._bible_of(row), row["file_name"])
                    for row in neighbour_rows
                ]
                label = self.summarizer.label(texts)

                self.results_store.insert({
                    "component_index": component_index,
                    "eigenvalue": float(eigenvalues[component_index]),
                    "sign": sign,
                    "anchor_bible": self._bible_of(representative_row),
                    "anchor_file_name": representative_row["file_name"],
                    "label": label,
                    "kpca_vector": transformed[representative_idx].tolist(),
                })
                labelled += 1
                logger.info(
                    "[component %d/%d, %s] eigenvalue=%.4f anchor=%s/%s -> %s",
                    component_index + 1, n_components, sign, eigenvalues[component_index],
                    self._bible_of(representative_row), representative_row["file_name"], label,
                )
        return labelled

    def _bible_of(self, row: dict) -> str:
        return row["bible"] if "bible" in row else COLLECTION_TO_BIBLE[self.reader.collection_name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c", "--collection", required=True,
        choices=[*COLLECTION_TO_BIBLE, COMBINED_COLLECTION],
        help="Milvus collection to analyze",
    )
    parser.add_argument("--results-collection", default=None, help="defaults to '<collection>_kpca_labels_<strategy>'")
    parser.add_argument("-n", "--components", type=int, default=N_COMPONENTS)
    parser.add_argument("-k", "--neighbours", type=int, default=N_NEIGHBOURS)
    parser.add_argument("--kernel", default=KERNEL, help="sklearn KernelPCA kernel (default: %(default)s)")
    parser.add_argument(
        "--strategy", choices=["top-projection", "anchor-knn"], default="top-projection",
        help=(
            "top-projection (default): label the K highest- and K lowest-scoring chapters on "
            "each axis directly, no Milvus search. anchor-knn: label the Milvus neighbours of "
            "the single most extreme chapter on each axis."
        ),
    )
    parser.add_argument("--ollama-model", default=OLLAMA_LABEL_MODEL)
    parser.add_argument("--milvus-uri", default=MILVUS_URI)
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    client = MilvusClient(uri=args.milvus_uri)
    has_bible_field = args.collection == COMBINED_COLLECTION
    results_collection = args.results_collection or f"{args.collection}_kpca_labels_{args.strategy.replace('-', '_')}"

    if args.strategy == "anchor-knn":
        strategy = AnchorKNNNeighbourhoodStrategy(
            MilvusNearestNeighbourSearch(client, args.collection, has_bible_field), k=args.neighbours,
        )
    else:
        strategy = TopProjectionNeighbourhoodStrategy(k=args.neighbours)

    labeler = KPCALabeler(
        reader=VectorCollectionReader(client, args.collection),
        text_resolver=ChapterTextResolver(Path(__file__).parent),
        reducer=KernelPCAReducer(n_components=args.components, kernel=args.kernel),
        strategy=strategy,
        summarizer=ChapterSummarizer(args.ollama_url, args.ollama_model),
        results_store=LabelResultsStore(client, results_collection),
    )
    labelled = labeler.run()
    logger.info("Done: labelled %d components (%s) into '%s'", labelled, args.strategy, results_collection)


if __name__ == "__main__":
    main()
