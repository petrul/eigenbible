"""
The EigenBible

Kernel PCA over a bible's chapter embeddings, with an LLM-written label per component.

Pulls every vector out of a Milvus collection built by embed_biblia.py, fits a Kernel PCA
on them, and for each of the requested components picks a neighbourhood of chapters that
characterizes that axis - via one of two NeighbourhoodStrategy implementations (see
eigenbible.neighbourhood_strategy) - reads their text back off disk, and hands it to an
Ollama chat model, which is asked to name the shared theme in a few words. Each labelled
component is written to a dedicated results collection as soon as it's ready, since the
whole pipeline (kernel eigendecomposition + one Ollama call per component) can take a
while and a crash or Ctrl-C shouldn't lose everything computed so far.

This script is just the CLI; see eigenbible.kpca_labeler.KPCALabeler for the pipeline itself.
"""
import argparse
import logging
from pathlib import Path

from pymilvus import MilvusClient

from eigenbible.bibles import COLLECTION_TO_BIBLE, COMBINED_COLLECTION
from eigenbible.chapter_text_resolver import ChapterTextResolver
from eigenbible.kpca_labeler import KPCALabeler
from eigenbible.label_results_store import LabelResultsStore
from eigenbible.nearest_neighbour_search import MilvusNearestNeighbourSearch
from eigenbible.neighbourhood_strategy import AnchorKNNNeighbourhoodStrategy, TopProjectionNeighbourhoodStrategy
from eigenbible.settings import MILVUS_URI, OLLAMA_URL
from eigenbible.summarizer import ChapterSummarizer, ShuffledLinesSummarizer
from eigenbible.vector_collection_reader import VectorCollectionReader

OLLAMA_LABEL_MODEL = "qwen3.5:4b"
N_COMPONENTS = 50
N_NEIGHBOURS = 8
KERNEL = "cosine"  # matches the COSINE index the source collections already use

# The bible source directories live at the project root, one level up from
# this script under src/ (same layout embed_biblia.py uses).
PROJECT_ROOT = Path(__file__).parent.parent

logger = logging.getLogger(__name__)


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
    parser.add_argument(
        "--summarizer", choices=["excerpt", "shuffled-lines"], default="excerpt",
        help=(
            "excerpt (default): label each neighbour chapter's own text kept intact and "
            "separate. shuffled-lines: pool every neighbour's lines together and shuffle "
            "them first, so the label can't lean on chapter boundaries/narrative order."
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

    summarizer_cls = ShuffledLinesSummarizer if args.summarizer == "shuffled-lines" else ChapterSummarizer

    labeler = KPCALabeler(
        reader=VectorCollectionReader(client, args.collection),
        text_resolver=ChapterTextResolver(PROJECT_ROOT),
        strategy=strategy,
        summarizer=summarizer_cls(args.ollama_url, args.ollama_model),
        results_store=LabelResultsStore(client, results_collection),
        n_components=args.components,
        kernel=args.kernel,
    )
    labelled = labeler.run()
    logger.info("Done: labelled %d components (%s) into '%s'", labelled, args.strategy, results_collection)


if __name__ == "__main__":
    main()
