"""Embeds every bible chapter with Ollama (default: bge-m3:567m-fp16) and stores
the vectors in FAISS (default), Qdrant, or raw local-disk memmap.
"""
import argparse
import logging
from pathlib import Path

from eigenbible.bibles import BIBLES, COMBINED_COLLECTION
from eigenbible.collection_merger import (
    FaissCollectionMerger,
    LocalDiskCollectionMerger,
    QdrantCollectionMerger,
)
from eigenbible.embedder import OllamaEmbedder
from eigenbible.importer import BibliaEmbeddingImporter
from eigenbible.markdown_reader import MarkdownFileReader
from eigenbible.settings import OLLAMA_EMBED_MODEL, OLLAMA_URL, QDRANT_URL
from eigenbible.vector_store import FaissVectorStore, LocalDiskVectorStore, QdrantVectorStore

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-b", "--bible", choices=[*sorted(BIBLES), "all", "none"], default="all",
        help="which bible to embed, or 'none' to skip straight to --combine (default: all)",
    )
    parser.add_argument(
        "--combine", action="store_true",
        help=f"also (re)build '{COMBINED_COLLECTION}' by copying vectors out of the per-bible "
             "collections, tagged with a 'bible' field - no re-embedding involved",
    )
    parser.add_argument(
        "--backend", choices=["faiss", "qdrant", "local"], default="faiss",
        help="where to store vectors: local FAISS (default), remote Qdrant, or raw memmap (local)",
    )
    parser.add_argument(
        "--data-dir", "--local-dir", dest="data_dir", type=Path, default=DEFAULT_DATA_DIR,
        help=f"directory for faiss/local vector data (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument("--qdrant-url", default=QDRANT_URL)
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    parser.add_argument("--ollama-model", default=OLLAMA_EMBED_MODEL)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.bible != "none":
        keys = list(BIBLES) if args.bible == "all" else [args.bible]
        for key in keys:
            source_dirname, collection = BIBLES[key]
            logger.info("=== %s: embedding into %s collection '%s' ===", key, args.backend, collection)
            reader = MarkdownFileReader(PROJECT_ROOT / source_dirname)

            if args.backend == "faiss":
                vector_store = FaissVectorStore(
                    args.data_dir,
                    collection,
                    metric="cosine",
                    meta={"embedding_model": args.ollama_model, "source_dir": source_dirname},
                )
            elif args.backend == "qdrant":
                vector_store = QdrantVectorStore(args.qdrant_url, collection)
            else:
                vector_store = LocalDiskVectorStore(args.data_dir, collection, total_rows=len(reader.files()))

            importer = BibliaEmbeddingImporter(
                reader=reader,
                embedder=OllamaEmbedder(args.ollama_url, args.ollama_model),
                vector_store=vector_store,
            )
            imported = importer.run()
            vector_store.close()
            logger.info("Done: imported %d files into %s collection '%s'", imported, args.backend, collection)

    if args.combine:
        sources = {key: collection for key, (_source_dirname, collection) in BIBLES.items()}
        if args.backend == "faiss":
            merger = FaissCollectionMerger(args.data_dir)
        elif args.backend == "qdrant":
            merger = QdrantCollectionMerger(args.qdrant_url)
        else:
            merger = LocalDiskCollectionMerger(args.data_dir)
        total = merger.merge(sources, COMBINED_COLLECTION)
        logger.info("Done: combined %d rows from %d bibles into '%s'", total, len(sources), COMBINED_COLLECTION)


if __name__ == "__main__":
    main()
