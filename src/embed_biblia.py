"""Embeds every bible chapter with Ollama and stores the vectors, either in

Milvus (default) or in a local disk memory-mapped format that needs no
server (--backend local) - see eigenbible.vector_store for both.
"""
import argparse
import logging
from pathlib import Path

from eigenbible.bibles import BIBLES, COMBINED_COLLECTION
from eigenbible.collection_merger import LocalDiskCollectionMerger, MilvusCollectionMerger
from eigenbible.embedder import OllamaEmbedder
from eigenbible.importer import BibliaEmbeddingImporter
from eigenbible.markdown_reader import MarkdownFileReader
from eigenbible.settings import MILVUS_URI, OLLAMA_URL
from eigenbible.vector_store import LocalDiskVectorStore, MilvusVectorStore

OLLAMA_MODEL = "qwen3-embedding:4b"

# The bible source directories and .env both live at the project root, one
# level up from this script under src/.
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_LOCAL_DIR = PROJECT_ROOT / "vector_data"

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
             f"collections, tagged with a 'bible' field - no re-embedding involved",
    )
    parser.add_argument(
        "--backend", choices=["milvus", "local"], default="milvus",
        help="where to store vectors: Milvus (default, needs MILVUS_URI/a running server), or "
             "a local disk memory-mapped format (--local-dir) that needs neither",
    )
    parser.add_argument(
        "--local-dir", type=Path, default=DEFAULT_LOCAL_DIR,
        help=f"directory for --backend local's vector data (default: {DEFAULT_LOCAL_DIR})",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.bible != "none":
        keys = list(BIBLES) if args.bible == "all" else [args.bible]
        for key in keys:
            source_dirname, collection = BIBLES[key]
            logger.info("=== %s: embedding into %s collection '%s' ===", key, args.backend, collection)
            reader = MarkdownFileReader(PROJECT_ROOT / source_dirname)

            if args.backend == "milvus":
                vector_store = MilvusVectorStore(MILVUS_URI, collection)
            else:
                vector_store = LocalDiskVectorStore(args.local_dir, collection, total_rows=len(reader.files()))

            importer = BibliaEmbeddingImporter(
                reader=reader,
                embedder=OllamaEmbedder(OLLAMA_URL, OLLAMA_MODEL),
                vector_store=vector_store,
            )
            imported = importer.run()
            if isinstance(vector_store, LocalDiskVectorStore):
                vector_store.close()
            logger.info("Done: imported %d files into %s collection '%s'", imported, args.backend, collection)

    if args.combine:
        sources = {key: collection for key, (_source_dirname, collection) in BIBLES.items()}
        merger = MilvusCollectionMerger(MILVUS_URI) if args.backend == "milvus" else LocalDiskCollectionMerger(args.local_dir)
        total = merger.merge(sources, COMBINED_COLLECTION)
        logger.info("Done: combined %d rows from %d bibles into '%s'", total, len(sources), COMBINED_COLLECTION)


if __name__ == "__main__":
    main()
