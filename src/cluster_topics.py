"""Runs HDBSCAN and/or BERTopic clustering over any local FAISS dataset using its
precomputed embeddings, optionally labelling discovered clusters with Ollama.
"""
import argparse
import logging
from pathlib import Path

from eigenbible.bibles import COLLECTION_TO_BIBLE, COMBINED_COLLECTION
from eigenbible.chapter_text_resolver import ChapterTextResolver, InlineTextResolver
from eigenbible.clusterer import BERTopicClusterer, HDBSCANClusterer
from eigenbible.label_results_store import JsonlResultsStore
from eigenbible.settings import OLLAMA_URL
from eigenbible.summarizer import ChapterSummarizer, ShuffledLinesSummarizer
from eigenbible.vector_collection_reader import FaissCollectionReader

OLLAMA_LABEL_MODEL = "qwen3.5:9b"
PROJECT_ROOT = Path(__file__).parent.parent

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d", "--dataset", type=Path, required=True,
        help="path to a local FAISS dataset directory (e.g. data/gesta_hungarorum)",
    )
    parser.add_argument(
        "-a", "--algorithm", choices=["hdbscan", "bertopic", "both"], default="both",
        help="clustering algorithm to execute (default: both)",
    )
    parser.add_argument("--min-cluster-size", type=int, default=None, help="HDBSCAN min_cluster_size (auto-scaled if omitted)")
    parser.add_argument("--min-samples", type=int, default=None, help="HDBSCAN min_samples")
    parser.add_argument(
        "--cluster-selection-method", choices=["eom", "leaf"], default="eom",
        help="HDBSCAN cluster selection method: eom (default) or leaf (finer-grained clusters)",
    )
    parser.add_argument("--reduce-method", choices=["umap", "pca", "none"], default="umap", help="pre-reduction for HDBSCAN")
    parser.add_argument("--reduce-dim", type=int, default=5, help="target dimension before HDBSCAN/BERTopic")
    parser.add_argument("--n-neighbors", type=int, default=10, help="UMAP n_neighbors")
    parser.add_argument("-k", "--representatives", type=int, default=30, help="centroid-nearest passages per cluster")
    parser.add_argument("--top-n-words", type=int, default=8, help="BERTopic c-TF-IDF keywords per topic")
    parser.add_argument("--summarizer", choices=["excerpt", "shuffled-lines"], default="excerpt")
    parser.add_argument("--no-llm", action="store_true", help="skip Ollama theme labelling")
    parser.add_argument("--ollama-model", default=OLLAMA_LABEL_MODEL)
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    parser.add_argument("--output-dir", type=Path, default=None, help="where to save JSONL/Markdown results")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    reader = FaissCollectionReader(args.dataset)
    dataset = reader.fetch_dataset()
    has_inline_text = bool(dataset.records and "text" in dataset.records[0])
    default_bible = None if dataset.name == COMBINED_COLLECTION else COLLECTION_TO_BIBLE.get(dataset.name)
    text_resolver = (
        InlineTextResolver()
        if has_inline_text
        else ChapterTextResolver(PROJECT_ROOT, default_bible=default_bible)
    )
    # Sensible default cluster size scaled to corpus size (e.g. 5 for Gesta's 122 rows, ~10 for 600-700 rows)
    min_cluster_size = args.min_cluster_size or max(5, min(15, len(dataset) // 40))

    summarizer = None
    if not args.no_llm:
        if args.summarizer == "shuffled-lines":
            summarizer = ShuffledLinesSummarizer(args.ollama_url, args.ollama_model)
        else:
            source_kind = "paragraphs" if has_inline_text else "bible chapters"
            summarizer = ChapterSummarizer(args.ollama_url, args.ollama_model, source_kind=source_kind)

    out_dir = args.output_dir or (args.dataset / "results")
    algos = ["hdbscan", "bertopic"] if args.algorithm == "both" else [args.algorithm]

    for algo in algos:
        jsonl_path = out_dir / f"{algo}.jsonl"
        md_path = out_dir / f"{algo}.md"
        results_store = JsonlResultsStore(jsonl_path)

        if algo == "hdbscan":
            runner = HDBSCANClusterer(
                reader=reader,
                results_store=results_store,
                text_resolver=text_resolver,
                summarizer=summarizer,
                min_cluster_size=min_cluster_size,
                min_samples=args.min_samples,
                cluster_selection_method=args.cluster_selection_method,
                reduce_method=args.reduce_method,
                reduce_dim=args.reduce_dim,
                n_neighbors=args.n_neighbors,
                n_representatives=args.representatives,
            )
            title = (
                f"{dataset.name} — HDBSCAN (reduce={args.reduce_method}:{args.reduce_dim}, "
                f"min_cluster_size={min_cluster_size}, selection={args.cluster_selection_method})"
            )
        else:
            runner = BERTopicClusterer(
                reader=reader,
                results_store=results_store,
                text_resolver=text_resolver,
                summarizer=summarizer,
                min_cluster_size=min_cluster_size,
                min_samples=args.min_samples,
                cluster_selection_method=args.cluster_selection_method,
                reduce_dim=args.reduce_dim,
                n_neighbors=args.n_neighbors,
                n_representatives=args.representatives,
                top_n_words=args.top_n_words,
            )
            title = (
                f"{dataset.name} — BERTopic (min_cluster_size={min_cluster_size}, "
                f"selection={args.cluster_selection_method})"
            )

        summary = runner.run()
        results_store.write_clusters_markdown(md_path, title=title)
        logger.info("Done (%s): %s -> %s and %s", algo, summary, jsonl_path, md_path)


if __name__ == "__main__":
    main()
