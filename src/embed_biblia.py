import argparse
import logging
from pathlib import Path

import requests
from pymilvus import DataType, MilvusClient


OLLAMA_URL = "http://zmeu.local:11434"
OLLAMA_MODEL = "qwen3-embedding:4b"
MILVUS_URI = "http://zmeu.local:19530"

# key -> (source directory name under this file, Milvus collection to store its vectors in)
BIBLES = {
    "ortodox": ("biblia-ortdx-capitole", "biblia_ortodoxa_subcapitole"),
    "darby": ("biblia-darby-fr-capitole", "biblia_darby_fr_subcapitole"),
}
COMBINED_COLLECTION = "biblia_all_subcapitole"

logger = logging.getLogger(__name__)


def create_vector_collection(client: MilvusClient, name: str, dimension: int, extra_varchar_fields=()) -> None:
    """Create a collection with a 'file_name' field, an optional set of extra VARCHAR fields
    (e.g. a 'bible' discriminator), and a COSINE index on the vector field - built as part of
    collection creation so it's searchable/loadable right away rather than needing a separate
    create_index step before it can be loaded."""
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
    schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=1024)
    for field_name in extra_varchar_fields:
        schema.add_field(field_name=field_name, datatype=DataType.VARCHAR, max_length=64)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")

    client.create_collection(collection_name=name, schema=schema, index_params=index_params)


class MarkdownFileReader:
    def __init__(self, source_directory: Path):
        self.source_directory = source_directory

    def files(self) -> list[Path]:
        files = sorted(
            path for path in self.source_directory.rglob("*") if path.is_file()
        )
        logger.info("Found %d chapter files under %s", len(files), self.source_directory)
        return files

    def read(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def relative_name(self, path: Path) -> str:
        return path.relative_to(self.source_directory).as_posix()


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str):
        self.url = f"{base_url.rstrip('/')}/api/embed"
        self.model = model
        logger.info("Embedding with Ollama model '%s' at %s", model, self.url)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        logger.debug("Requesting embeddings for a batch of %d texts from Ollama", len(texts))
        response = requests.post(
            self.url,
            json={"model": self.model, "input": texts},
            timeout=600,
        )
        response.raise_for_status()
        embeddings = response.json().get("embeddings")
        if not embeddings or len(embeddings) != len(texts):
            raise ValueError("Ollama returned an unexpected embedding response")
        logger.debug("Got %d embeddings back, %d-dimensional each", len(embeddings), len(embeddings[0]))
        return embeddings


class MilvusVectorStore:
    def __init__(self, uri: str, collection_name: str):
        logger.info("Connecting to Milvus at %s (collection '%s')", uri, collection_name)
        self.client = MilvusClient(uri=uri)
        self.collection_name = collection_name
        self.dimension: int | None = None

    def ensure_collection(self, dimension: int) -> None:
        if self.client.has_collection(self.collection_name):
            self.dimension = dimension
            return

        logger.info(
            "Collection '%s' doesn't exist yet, creating it (dim=%d) with a COSINE index so the "
            "chapter vectors have somewhere to land and are loadable/searchable right away",
            self.collection_name,
            dimension,
        )
        create_vector_collection(self.client, self.collection_name, dimension)
        self.dimension = dimension
        logger.info("Collection '%s' created", self.collection_name)

    def insert_batch(self, file_names: list[str], vectors: list[list[float]]) -> None:
        self.ensure_collection(len(vectors[0]))
        for vector in vectors:
            if self.dimension != len(vector):
                raise ValueError(
                    f"Vector dimension changed from {self.dimension} to {len(vector)}"
                )
        logger.debug("Storing a batch of %d vectors in Milvus so they become searchable", len(vectors))
        self.client.insert(
            collection_name=self.collection_name,
            data=[
                {"vector": vector, "file_name": file_name}
                for file_name, vector in zip(file_names, vectors)
            ],
        )


class CollectionMerger:
    """Builds a combined collection out of per-bible collections that are already populated,
    by copying their vectors instead of asking Ollama to embed every chapter a second time.
    Each copied row is tagged with a 'bible' field, so the combined collection can still be
    scoped back down to a single bible, or searched across all of them at once."""

    BATCH_SIZE = 1000

    def __init__(self, uri: str):
        self.client = MilvusClient(uri=uri)

    def merge(self, sources: dict, target_collection: str) -> int:
        total = 0
        target_ready = False
        for bible_key, source_collection in sources.items():
            logger.info("Copying vectors from '%s' (%s) into '%s'", source_collection, bible_key, target_collection)
            offset = 0
            while True:
                rows = self.client.query(
                    collection_name=source_collection,
                    filter="id >= 0",
                    output_fields=["vector", "file_name"],
                    limit=self.BATCH_SIZE,
                    offset=offset,
                )
                if not rows:
                    break
                if not target_ready:
                    self._ensure_collection(target_collection, len(rows[0]["vector"]))
                    target_ready = True
                self.client.insert(
                    collection_name=target_collection,
                    data=[
                        {"vector": row["vector"], "file_name": row["file_name"], "bible": bible_key}
                        for row in rows
                    ],
                )
                offset += len(rows)
                total += len(rows)
                logger.info("  ... copied %d rows from '%s' so far", offset, source_collection)
        return total

    def _ensure_collection(self, name: str, dimension: int) -> None:
        if self.client.has_collection(name):
            return
        logger.info(
            "Collection '%s' doesn't exist yet, creating it (dim=%d) with a 'bible' field so "
            "rows from every bible can share one collection and still be told apart, plus a "
            "COSINE index so it's loadable/searchable right away",
            name,
            dimension,
        )
        create_vector_collection(self.client, name, dimension, extra_varchar_fields=["bible"])
        logger.info("Collection '%s' created", name)


class BibliaEmbeddingImporter:
    def __init__(
        self,
        reader: MarkdownFileReader,
        embedder: OllamaEmbedder,
        vector_store: MilvusVectorStore,
        batch_size: int = 200,
    ):
        self.reader = reader
        self.embedder = embedder
        self.vector_store = vector_store
        self.batch_size = batch_size

    def run(self) -> int:
        imported = 0
        files = self.reader.files()
        batches = [files[i:i + self.batch_size] for i in range(0, len(files), self.batch_size)]
        logger.info(
            "Embedding chapters in batches of %d and storing each batch's vectors in Milvus in "
            "one go, so later a query can be embedded the same way and matched by similarity "
            "against these chapters",
            self.batch_size,
        )
        for batch_index, batch in enumerate(batches, start=1):
            file_names = [self.reader.relative_name(path) for path in batch]
            texts = [self.reader.read(path) for path in batch]
            vectors = self.embedder.embed_batch(texts)
            self.vector_store.insert_batch(file_names, vectors)
            imported += len(batch)
            logger.info(
                "[batch %d/%d] embedded and stored %d chapters (%d/%d total)",
                batch_index, len(batches), len(batch), imported, len(files),
            )
        return imported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-b", "--bible", choices=[*sorted(BIBLES), "all", "none"], default="all",
        help="which bible to embed, or 'none' to skip straight to --combine (default: all)",
    )
    parser.add_argument(
        "--combine", action="store_true",
        help=f"also (re)build '{COMBINED_COLLECTION}' by copying vectors out of the per-bible "
             f"collections, tagged with a 'bible' field - no re-embedding involved",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.bible != "none":
        keys = list(BIBLES) if args.bible == "all" else [args.bible]
        for key in keys:
            source_dirname, collection = BIBLES[key]
            logger.info("=== %s: embedding into Milvus collection '%s' ===", key, collection)
            importer = BibliaEmbeddingImporter(
                reader=MarkdownFileReader(Path(__file__).parent / source_dirname),
                embedder=OllamaEmbedder(OLLAMA_URL, OLLAMA_MODEL),
                vector_store=MilvusVectorStore(MILVUS_URI, collection),
            )
            imported = importer.run()
            logger.info("Done: imported %d files into Milvus collection '%s'", imported, collection)

    if args.combine:
        sources = {key: collection for key, (_source_dirname, collection) in BIBLES.items()}
        merger = CollectionMerger(MILVUS_URI)
        total = merger.merge(sources, COMBINED_COLLECTION)
        logger.info("Done: combined %d rows from %d bibles into '%s'", total, len(sources), COMBINED_COLLECTION)


if __name__ == "__main__":
    main()
