"""End-to-end automation script:
1. Optionally filters headings (paragraph_num == 1), short paragraphs (< min_chars),
   and 'about/' boilerplate out of existing local Biblioteca FAISS datasets in place.
2. Runs PCA, RBF Kernel PCA (gamma=2.0), HDBSCAN, and BERTopic on the target corpora
   (default: the 7 Biblioteca paragraph corpora, leaving the 2 unchanged Bibles untouched).
3. Regenerates standalone eigenbooks_synthesis.xhtml and exports the CMS XHTML, meta JSON,
   public/data/eigenbooks-viz.json, and public/data/eigenbooks-results.json into
   scriptorium-blog-cms.

Usage:
  uv run python scripts/rebuild_and_publish.py --clean
  uv run python scripts/rebuild_and_publish.py --skip-analyze
  uv run python scripts/rebuild_and_publish.py --corpora ureche_letopisetul gesta_hungarorum
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CMS_DIR = Path("/home/petru/work/scriptorium.ro/scriptorium-blog-cms")

BIBLIOTECA_CORPORA = [
    "gesta_hungarorum",
    "ureche_letopisetul",
    "miron_letopisetul",
    "creanga_amintiri",
    "london_white_fang",
    "hugo_notre_dame",
    "hugo_notre_dame_en",
]
BIBLE_CORPORA = [
    "biblia_ortodoxa_subcapitole",
    "biblia_darby_fr_subcapitole",
]

# Per-corpus HDBSCAN/BERTopic min_cluster_size tuned with leaf selection to yield ~15-32 crisp topics
MIN_CLUSTER_SIZE: dict[str, int] = {
    "gesta_hungarorum": 3,
    "creanga_amintiri": 5,
    "ureche_letopisetul": 6,
    "miron_letopisetul": 6,
    "london_white_fang": 7,
    "biblia_darby_fr_subcapitole": 10,
    "biblia_ortodoxa_subcapitole": 12,
    "hugo_notre_dame": 12,
    "hugo_notre_dame_en": 12,
}

# Leaf selection unpacks fine-grained narrative & thematic episodes (~20-30 topics) across all corpora
DEFAULT_CLUSTER_SELECTION = "leaf"
CLUSTER_SELECTION: dict[str, str] = {}

logger = logging.getLogger(__name__)


def _run(cmd: list[str]) -> None:
    logger.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def clean_corpora(corpora: list[str], *, exclude_headings: bool, min_chars: int, exclude_sections: tuple[str, ...]) -> None:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from eigenbible.dataset_cleaner import filter_faiss_dataset

    for name in corpora:
        ds_dir = DATA_DIR / name
        if not (ds_dir / "index.faiss").exists():
            logger.warning("Skipping missing dataset: %s", ds_dir)
            continue
        stats = filter_faiss_dataset(
            ds_dir,
            exclude_headings=exclude_headings,
            exclude_sections=exclude_sections,
            min_chars=min_chars,
        )
        print(f"Cleaned {name:24s}: {stats['before']:4d} -> {stats['after']:4d} (dropped {stats['dropped']})")


def analyze_corpora(
    corpora: list[str],
    *,
    n_components: int,
    n_neighbours: int,
    n_representatives: int,
    gamma: float,
    ollama_model: str | None,
) -> None:
    py = sys.executable
    for name in corpora:
        ds_dir = DATA_DIR / name
        if not (ds_dir / "index.faiss").exists():
            logger.warning("Skipping missing dataset: %s", ds_dir)
            continue
        results_dir = ds_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        model_args = ["--ollama-model", ollama_model] if ollama_model else []

        # 1. Linear PCA
        _run([
            py, "src/kpca_label.py",
            "--dataset", str(ds_dir),
            "--method", "pca",
            "--components", str(n_components),
            "--neighbours", str(n_neighbours),
            "--output", str(results_dir / "pca_top_projection.jsonl"),
            "--report", str(results_dir / "pca_top_projection.md"),
            *model_args,
        ])

        # 2. Non-linear RBF Kernel PCA (gamma=2.0)
        _run([
            py, "src/kpca_label.py",
            "--dataset", str(ds_dir),
            "--method", "kpca",
            "--kernel", "rbf",
            "--gamma", str(gamma),
            "--components", str(n_components),
            "--neighbours", str(n_neighbours),
            "--output", str(results_dir / "kpca_rbf_g2.jsonl"),
            "--report", str(results_dir / "kpca_rbf_g2.md"),
            *model_args,
        ])

        # 3. Cosine kPCA identity check on Gesta Hungarorum (top 10 axes)
        if name == "gesta_hungarorum":
            _run([
                py, "src/kpca_label.py",
                "--dataset", str(ds_dir),
                "--method", "kpca",
                "--kernel", "cosine",
                "--components", str(min(10, n_components)),
                "--neighbours", str(n_neighbours),
                "--output", str(results_dir / "kpca_cosine_top_projection.jsonl"),
                "--report", str(results_dir / "kpca_cosine_top_projection.md"),
                *model_args,
            ])

        # 4. HDBSCAN + BERTopic
        min_cs = MIN_CLUSTER_SIZE.get(name, 10)
        sel = CLUSTER_SELECTION.get(name, DEFAULT_CLUSTER_SELECTION)
        _run([
            py, "src/cluster_topics.py",
            "--dataset", str(ds_dir),
            "--algorithm", "both",
            "--min-cluster-size", str(min_cs),
            "--cluster-selection-method", sel,
            "--representatives", str(n_representatives),
            "--output-dir", str(results_dir),
            *model_args,
        ])


def publish_reports(cms_dir: Path | None, *, skip_standalone: bool) -> None:
    cmd = [sys.executable, "src/build_blog_report.py"]
    if skip_standalone:
        cmd.append("--skip-standalone")
    if cms_dir is not None:
        cmd.extend(["--cms-dir", str(cms_dir)])
    _run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpora",
        nargs="+",
        default=None,
        help="specific dataset names to clean/analyze (default: all 7 Biblioteca works)",
    )
    parser.add_argument("--all", action="store_true", help="include the 2 Bible corpora when analyzing")
    parser.add_argument("--clean", action="store_true", help="filter headings & short paragraphs in place before analyzing")
    parser.add_argument("--include-headings", action="store_true", help="do not drop paragraph_num == 1 headings")
    parser.add_argument("--min-chars", type=int, default=100, help="minimum paragraph length in characters (default: 100)")
    parser.add_argument("--exclude-sections", default="about", help="comma-separated section prefixes to drop (default: about)")
    parser.add_argument("--skip-analyze", action="store_true", help="skip PCA/kPCA/HDBSCAN/BERTopic runs")
    parser.add_argument("--skip-publish", action="store_true", help="skip XHTML report build & blog-cms export")
    parser.add_argument("--skip-standalone", action="store_true", help="skip standalone eigenbooks_synthesis.xhtml")
    parser.add_argument("--components", type=int, default=30, help="PCA/kPCA components per corpus (default: 30)")
    parser.add_argument("--neighbours", type=int, default=30, help="top-projection passages per PCA/kPCA pole (default: 30)")
    parser.add_argument("--representatives", type=int, default=30, help="centroid-nearest passages per cluster (default: 30)")
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--ollama-model", default="qwen3.5:9b")
    parser.add_argument("--cms-dir", type=Path, default=DEFAULT_CMS_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.corpora:
        targets = args.corpora
    elif args.all:
        targets = [*BIBLIOTECA_CORPORA, *BIBLE_CORPORA]
    else:
        targets = list(BIBLIOTECA_CORPORA)

    exclude_sections = tuple(s.strip() for s in args.exclude_sections.split(",") if s.strip())

    if args.clean:
        # Never strip headings from chapter-level Bibles; only clean Biblioteca paragraph corpora
        clean_targets = [c for c in targets if c not in BIBLE_CORPORA]
        clean_corpora(
            clean_targets,
            exclude_headings=not args.include_headings,
            min_chars=args.min_chars,
            exclude_sections=exclude_sections,
        )

    if not args.skip_analyze:
        analyze_corpora(
            targets,
            n_components=args.components,
            n_neighbours=args.neighbours,
            n_representatives=args.representatives,
            gamma=args.gamma,
            ollama_model=args.ollama_model,
        )

    if not args.skip_publish:
        publish_reports(args.cms_dir, skip_standalone=args.skip_standalone)


if __name__ == "__main__":
    main()
