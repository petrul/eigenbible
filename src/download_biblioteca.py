"""Downloads paragraph vectors from Qdrant (zmeu) and verified UTF-8 text from
biblioteca.scriptorium.ro into self-contained local FAISS datasets under data/.
"""
import argparse
import logging
from pathlib import Path

from eigenbible.qdrant_downloader import (
    BIBLIOTECA_WORKS,
    BibliotecaDatasetExporter,
    BibliotecaTextFetcher,
)
from eigenbible.settings import BIBLIOTECA_BASE_URL, QDRANT_COLLECTION, QDRANT_URL, QDRANT_VECTOR_NAME

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "data"

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-w", "--work",
        choices=[*sorted(BIBLIOTECA_WORKS), "all"],
        default="all",
        help="preset Biblioteca work to download (default: all three)",
    )
    parser.add_argument(
        "--opus-path",
        default=None,
        help="download an arbitrary Qdrant opus_path instead of a preset (requires --name)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="local dataset directory name when using --opus-path",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"root directory for downloaded FAISS datasets (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument("--qdrant-url", default=QDRANT_URL)
    parser.add_argument("--qdrant-collection", default=QDRANT_COLLECTION)
    parser.add_argument("--vector-name", default=QDRANT_VECTOR_NAME)
    parser.add_argument("--biblioteca-url", default=BIBLIOTECA_BASE_URL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--include-headings",
        action="store_true",
        help="keep paragraph_num == 1 section headings (excluded by default)",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=100,
        help="drop paragraphs shorter than this many characters (default: 100; 0 disables)",
    )
    parser.add_argument(
        "--exclude-sections",
        default="about",
        help="comma-separated section prefixes to drop (default: about)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    exclude_sections = tuple(s.strip() for s in args.exclude_sections.split(",") if s.strip())

    exporter = BibliotecaDatasetExporter(
        out_root=args.output_dir,
        qdrant_url=args.qdrant_url,
        collection_name=args.qdrant_collection,
        vector_name=args.vector_name,
        fetcher=BibliotecaTextFetcher(base_url=args.biblioteca_url, workers=args.workers),
    )

    if args.opus_path:
        work_name = args.name or args.opus_path.replace("/", "_")
        targets = [(work_name, args.opus_path)]
    else:
        keys = list(BIBLIOTECA_WORKS) if args.work == "all" else [args.work]
        targets = [(key, BIBLIOTECA_WORKS[key]) for key in keys]

    for work_name, opus_path in targets:
        summary = exporter.export_work(
            work_name,
            opus_path,
            exclude_headings=not args.include_headings,
            exclude_sections=exclude_sections,
            min_chars=args.min_chars,
        )
        logger.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
