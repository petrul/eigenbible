"""The EigenBible / EigenBooks — PCA and Kernel PCA axis discovery + LLM labelling.

Runs linear PCA or Kernel PCA on any local FAISS dataset (e.g. `data/gesta_hungarorum`,
`data/ureche_letopisetul`, `data/miron_letopisetul`), local memmap collection, or Qdrant
collection, picks characteristic passages per component, and asks an Ollama chat model to
name the shared theme of each axis/pole. Results are checkpointed to JSONL and rendered
to a readable Markdown report.
"""
import argparse
import logging
from pathlib import Path

from qdrant_client import QdrantClient

from eigenbible.bibles import COLLECTION_TO_BIBLE, COMBINED_COLLECTION
from eigenbible.chapter_text_resolver import ChapterTextResolver, InlineTextResolver
from eigenbible.kpca_labeler import AxisLabeler
from eigenbible.label_results_store import JsonlResultsStore
from eigenbible.nearest_neighbour_search import FaissNearestNeighbourSearch, QdrantNearestNeighbourSearch
from eigenbible.neighbourhood_strategy import (
    AnchorKNNNeighbourhoodStrategy,
    TopProjectionNeighbourhoodStrategy,
)
from eigenbible.reducers import build_reducer
from eigenbible.settings import OLLAMA_URL, QDRANT_URL
from eigenbible.summarizer import ChapterSummarizer, ShuffledLinesSummarizer
from eigenbible.vector_collection_reader import (
    FaissCollectionReader,
    LocalDiskCollectionReader,
    QdrantCollectionReader,
)

OLLAMA_LABEL_MODEL = "qwen3.5:9b"
N_COMPONENTS = 30
N_NEIGHBOURS = 30
KERNEL = "cosine"

PROJECT_ROOT = Path(__file__).parent.parent

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "-d", "--dataset", type=Path,
        help="path to a local FAISS dataset directory (e.g. data/gesta_hungarorum)",
    )
    source.add_argument(
        "-c", "--collection",
        help="collection name under --data-dir (faiss/local) or Qdrant",
    )
    parser.add_argument(
        "--backend", choices=["faiss", "qdrant", "local"], default="faiss",
        help="backend to read --collection from (ignored when --dataset is given; default: faiss)",
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--method", choices=["pca", "kpca"], default="kpca", help="dimensionality reduction method")
    parser.add_argument("--kernel", default=KERNEL, help="KernelPCA kernel when --method kpca (default: %(default)s)")
    parser.add_argument("--gamma", type=float, default=None, help="KernelPCA gamma (e.g. 1.0-3.0 for non-linear RBF on unit vectors)")
    parser.add_argument("-n", "--components", type=int, default=N_COMPONENTS)
    parser.add_argument("-k", "--neighbours", type=int, default=N_NEIGHBOURS)
    parser.add_argument(
        "--strategy", choices=["top-projection", "anchor-knn"], default="top-projection",
        help="top-projection (default): label both poles of each axis. anchor-knn: kNN around the extreme point.",
    )
    parser.add_argument(
        "--summarizer", choices=["excerpt", "shuffled-lines"], default="excerpt",
        help="excerpt (default): feed passage excerpts intact. shuffled-lines: pool and shuffle lines.",
    )
    parser.add_argument("--source-kind", default="passages", help="noun describing the text units in the LLM prompt")
    parser.add_argument("--output", type=Path, default=None, help="JSONL output path")
    parser.add_argument("--report", type=Path, default=None, help="Markdown summary output path")
    parser.add_argument("--ollama-model", default=OLLAMA_LABEL_MODEL)
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    parser.add_argument("--qdrant-url", default=QDRANT_URL)
    parser.add_argument("--vector-name", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.dataset is not None:
        dataset_dir = args.dataset
        reader = FaissCollectionReader(dataset_dir)
        dataset = reader.fetch_dataset()
        has_inline_text = bool(dataset.records and "text" in dataset.records[0])
        text_resolver = (
            InlineTextResolver()
            if has_inline_text
            else ChapterTextResolver(PROJECT_ROOT, default_bible=COLLECTION_TO_BIBLE.get(dataset.name))
        )
        default_results_dir = dataset_dir / "results"
        metric = dataset.meta.get("metric", "cosine")
        search_backend = FaissNearestNeighbourSearch(reader.index, dataset.records, metric=metric)
    else:
        if args.backend == "faiss":
            dataset_dir = args.data_dir / args.collection
            reader = FaissCollectionReader(dataset_dir)
            dataset = reader.fetch_dataset()
            search_backend = FaissNearestNeighbourSearch(
                reader.index, dataset.records, metric=dataset.meta.get("metric", "cosine")
            )
            default_results_dir = dataset_dir / "results"
        elif args.backend == "local":
            reader = LocalDiskCollectionReader(args.data_dir, args.collection)
            dataset = reader.fetch_dataset()
            search_backend = None
            default_results_dir = args.data_dir / args.collection / "results"
        else:
            reader = QdrantCollectionReader(args.qdrant_url, args.collection, vector_name=args.vector_name)
            dataset = reader.fetch_dataset()
            search_backend = QdrantNearestNeighbourSearch(
                QdrantClient(url=args.qdrant_url, check_compatibility=False), args.collection, vector_name=args.vector_name
            )
            default_results_dir = PROJECT_ROOT / "results" / args.collection

        has_inline_text = bool(dataset.records and "text" in dataset.records[0])
        default_bible = None if args.collection == COMBINED_COLLECTION else COLLECTION_TO_BIBLE.get(args.collection)
        text_resolver = (
            InlineTextResolver()
            if has_inline_text
            else ChapterTextResolver(PROJECT_ROOT, default_bible=default_bible)
        )

    if args.strategy == "anchor-knn":
        if search_backend is None:
            raise ValueError("anchor-knn requires a FAISS or Qdrant backend")
        strategy = AnchorKNNNeighbourhoodStrategy(search_backend, k=args.neighbours)
    else:
        strategy = TopProjectionNeighbourhoodStrategy(k=args.neighbours)

    reducer = build_reducer(method=args.method, n_components=args.components, kernel=args.kernel, gamma=args.gamma)
    if args.summarizer == "shuffled-lines":
        summarizer = ShuffledLinesSummarizer(args.ollama_url, args.ollama_model)
    else:
        summarizer = ChapterSummarizer(args.ollama_url, args.ollama_model, source_kind=args.source_kind)

    stem = f"{reducer.name}_{args.strategy.replace('-', '_')}"
    output_jsonl = args.output or (default_results_dir / f"{stem}.jsonl")
    report_md = args.report or (default_results_dir / f"{stem}.md")
    results_store = JsonlResultsStore(output_jsonl)

    labeler = AxisLabeler(
        reader=reader,
        text_resolver=text_resolver,
        reducer=reducer,
        strategy=strategy,
        summarizer=summarizer,
        results_store=results_store,
    )
    labelled = labeler.run()
    results_store.write_axes_markdown(
        report_md,
        title=f"{dataset.name} — {reducer.name.upper()} ({args.strategy})",
    )
    logger.info("Done: labelled %d poles -> %s and %s", labelled, output_jsonl, report_md)


if __name__ == "__main__":
    main()
