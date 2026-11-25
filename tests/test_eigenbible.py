"""Tests for the eigenbible package.

Covers local FAISS and LocalDisk vector stores, mergers, readers, kNN search,
text resolvers, JSONL results storage, PCA/kPCA axis labelling, URL parsing,
HDBSCAN clustering, summarizers, and (when Ollama is reachable) real embedding
and label stability integration checks.

Run with: rake test
      or: uv run python -m unittest discover -s tests -v
"""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import requests

from eigenbible.chapter_text_resolver import InlineTextResolver
from eigenbible.clusterer import HDBSCANClusterer
from eigenbible.collection_merger import FaissCollectionMerger, LocalDiskCollectionMerger
from eigenbible.dataset import VectorDataset
from eigenbible.embedder import OllamaEmbedder
from eigenbible.importer import BibliaEmbeddingImporter
from eigenbible.kpca_labeler import AxisLabeler
from eigenbible.label_results_store import JsonlResultsStore
from eigenbible.markdown_reader import MarkdownFileReader
from eigenbible.nearest_neighbour_search import FaissNearestNeighbourSearch
from eigenbible.neighbourhood_strategy import (
    AnchorKNNNeighbourhoodStrategy,
    TopProjectionNeighbourhoodStrategy,
)
from eigenbible.qdrant_downloader import parse_biblioteca_url
from eigenbible.reducers import KPCAReducer, PCAReducer
from eigenbible.settings import OLLAMA_EMBED_MODEL, OLLAMA_URL, QDRANT_URL
from eigenbible.summarizer import ChapterSummarizer, ShuffledLinesSummarizer, Summarizer
from eigenbible.vector_collection_reader import FaissCollectionReader
from eigenbible.vector_store import FaissVectorStore, LocalDiskVectorStore

RES_DIR = Path(__file__).parent / "res"
TEST_EMBED_MODEL = "nomic-embed-text:v1.5"
TEST_LABEL_MODEL = "qwen2.5vl:3b"


def has_ollama() -> bool:
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).raise_for_status()
        return True
    except Exception:
        return False


class FakeSummarizer(Summarizer):
    def __init__(self):
        self.calls: list[list[str]] = []

    def _build_prompt(self, texts: list[str]) -> str:
        return " | ".join(texts)

    def label(self, texts: list[str]) -> str:
        self.calls.append(texts)
        return f"theme({len(texts)}): {texts[0][:20]}"


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)


class SettingsTests(unittest.TestCase):
    def test_qdrant_and_ollama_settings_are_configured(self):
        self.assertTrue(OLLAMA_URL)
        self.assertEqual(QDRANT_URL, "http://zmeu.local:6333")
        self.assertEqual(OLLAMA_EMBED_MODEL, "bge-m3:567m-fp16")


class MarkdownFileReaderTests(TempDirTestCase):
    def test_lists_reads_and_names_files_relative_to_the_source_dir(self):
        (self.tmp / "sub").mkdir()
        (self.tmp / "sub" / "a.md").write_text("hello")
        (self.tmp / "b.md").write_text("world")

        reader = MarkdownFileReader(self.tmp)
        files = reader.files()

        self.assertEqual(len(files), 2)
        names = {reader.relative_name(f) for f in files}
        self.assertEqual(names, {"sub/a.md", "b.md"})
        contents = {reader.read(f) for f in files}
        self.assertEqual(contents, {"hello", "world"})


class FaissVectorStoreTests(TempDirTestCase):
    def test_writes_reconstructs_and_searches_vectors_and_records(self):
        store = FaissVectorStore(self.tmp, "work", metric="cosine", meta={"embedding_model": "bge-m3"})
        store.insert_batch(
            ["p1", "p2"],
            [[1.0, 0.0], [0.0, 1.0]],
            extra_fields=[{"text": "first"}, {"text": "second"}],
        )
        store.insert_batch(
            ["p3"],
            [[0.6, 0.8]],
            extra_fields=[{"text": "third"}],
        )
        store.close()

        dataset, index = FaissVectorStore.load_dataset_dir(self.tmp / "work")
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.dimension, 2)
        self.assertEqual(dataset.meta["embedding_model"], "bge-m3")
        self.assertEqual([r["text"] for r in dataset.records], ["first", "second", "third"])
        np.testing.assert_array_almost_equal(dataset.vectors, [[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])

        search = FaissNearestNeighbourSearch(index, dataset.records, metric="cosine")
        hits = search.search([0.99, 0.05], k=2)
        self.assertEqual(hits[0]["file_name"], "p1")
        self.assertEqual(hits[1]["file_name"], "p3")

    def test_rejects_dimension_change_mid_collection(self):
        store = FaissVectorStore(self.tmp, "work")
        store.insert_batch(["a"], [[1.0, 2.0]])
        with self.assertRaises(ValueError):
            store.insert_batch(["b"], [[1.0, 2.0, 3.0]])


class FaissCollectionMergerTests(TempDirTestCase):
    def test_merges_two_faiss_collections_with_source_tags(self):
        s1 = FaissVectorStore(self.tmp, "src_a")
        s1.insert_batch(["a1", "a2"], [[1.0, 0.0], [0.0, 1.0]], extra_fields=[{"text": "t1"}, {"text": "t2"}])
        s1.close()

        s2 = FaissVectorStore(self.tmp, "src_b")
        s2.insert_batch(["b1"], [[0.6, 0.8]], extra_fields=[{"text": "t3"}])
        s2.close()

        merger = FaissCollectionMerger(self.tmp, source_field="corpus")
        total = merger.merge({"work_a": "src_a", "work_b": "src_b"}, "combined")

        self.assertEqual(total, 3)
        dataset, _ = FaissVectorStore.load_dataset_dir(self.tmp / "combined")
        self.assertEqual(
            [(r["file_name"], r["corpus"], r["text"]) for r in dataset.records],
            [("a1", "work_a", "t1"), ("a2", "work_a", "t2"), ("b1", "work_b", "t3")],
        )


class LocalDiskVectorStoreTests(TempDirTestCase):
    def test_writes_and_reads_back_vectors_and_file_names(self):
        store = LocalDiskVectorStore(self.tmp, "coll", total_rows=3)
        store.insert_batch(["a", "b"], [[1.0, 2.0], [3.0, 4.0]])
        store.insert_batch(["c"], [[5.0, 6.0]])
        store.close()

        records, vectors = LocalDiskVectorStore.read_all(self.tmp, "coll")
        self.assertEqual([r["file_name"] for r in records], ["a", "b", "c"])
        np.testing.assert_array_almost_equal(
            np.asarray(vectors), [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        )

    def test_rejects_a_batch_that_would_exceed_total_rows(self):
        store = LocalDiskVectorStore(self.tmp, "coll", total_rows=1)
        self.addCleanup(store.close)
        with self.assertRaises(ValueError):
            store.insert_batch(["a", "b"], [[1.0], [2.0]])

    def test_rejects_a_dimension_change_mid_collection(self):
        store = LocalDiskVectorStore(self.tmp, "coll", total_rows=2)
        self.addCleanup(store.close)
        store.insert_batch(["a"], [[1.0, 2.0]])
        with self.assertRaises(ValueError):
            store.insert_batch(["b"], [[1.0, 2.0, 3.0]])

    def test_extra_fields_get_merged_into_each_record(self):
        store = LocalDiskVectorStore(self.tmp, "coll", total_rows=2)
        store.insert_batch(
            ["a", "b"], [[1.0], [2.0]], extra_fields=[{"bible": "x"}, {"bible": "y"}]
        )
        store.close()
        records, _ = LocalDiskVectorStore.read_all(self.tmp, "coll")
        self.assertEqual([r["bible"] for r in records], ["x", "y"])


class LocalDiskCollectionMergerTests(TempDirTestCase):
    def _seed(self, name: str, file_names: list[str], vectors: list[list[float]]):
        store = LocalDiskVectorStore(self.tmp, name, total_rows=len(file_names))
        store.insert_batch(file_names, vectors)
        store.close()

    def test_merges_two_source_collections_tagging_each_row_with_its_bible(self):
        self._seed("src_a", ["a1", "a2"], [[1.0, 0.0], [0.0, 1.0]])
        self._seed("src_b", ["b1"], [[1.0, 1.0]])

        merger = LocalDiskCollectionMerger(self.tmp)
        total = merger.merge({"bible_a": "src_a", "bible_b": "src_b"}, "combined")

        self.assertEqual(total, 3)
        records, vectors = LocalDiskVectorStore.read_all(self.tmp, "combined")
        self.assertEqual(
            [(r["file_name"], r["bible"]) for r in records],
            [("a1", "bible_a"), ("a2", "bible_a"), ("b1", "bible_b")],
        )
        self.assertEqual(vectors.shape, (3, 2))


class BibliotecaUrlParserTests(unittest.TestCase):
    def test_parses_section_and_paragraph_number(self):
        path, section, pnum = parse_biblioteca_url(
            "http://server:8080/trad/gesta_hungarorum_la/de_duce_bycoriensy/_2",
            "trad/gesta_hungarorum_la",
        )
        self.assertEqual(path, "/trad/gesta_hungarorum_la/de_duce_bycoriensy/_2")
        self.assertEqual(section, "de_duce_bycoriensy")
        self.assertEqual(pnum, 2)

    def test_parses_nested_sections(self):
        path, section, pnum = parse_biblioteca_url(
            "http://server:8080/ureche/letopisetul_tarii_moldovei/domniia_lui_alexandru/misail_calugarul/_19",
            "ureche/letopisetul_tarii_moldovei",
        )
        self.assertEqual(section, "domniia_lui_alexandru/misail_calugarul")
        self.assertEqual(pnum, 19)


class AxisLabelerAndClusteringTests(TempDirTestCase):
    def _build_synthetic_faiss(self) -> Path:
        rng = np.random.default_rng(7)
        c1 = rng.normal(loc=[5.0, 0.0, 0.0, 0.0], scale=0.15, size=(12, 4))
        c2 = rng.normal(loc=[-5.0, 0.0, 0.0, 0.0], scale=0.15, size=(12, 4))
        c3 = rng.normal(loc=[0.0, 5.0, 0.0, 0.0], scale=0.15, size=(12, 4))
        vectors = np.vstack([c1, c2, c3]).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        names = [f"sec_{i // 12}/_{i % 12}.txt" for i in range(len(vectors))]
        extras = [
            {"section": f"sec_{i // 12}", "paragraph_num": i % 12, "text": f"Passage from cluster {i // 12} item {i}"}
            for i in range(len(vectors))
        ]
        store = FaissVectorStore(self.tmp, "synth", metric="cosine")
        store.insert_batch(names, vectors, extra_fields=extras)
        store.close()
        return self.tmp / "synth"

    def test_pca_and_kpca_axis_labeler_with_faiss_backend(self):
        dataset_dir = self._build_synthetic_faiss()
        reader = FaissCollectionReader(dataset_dir)
        out_jsonl = self.tmp / "pca.jsonl"
        out_md = self.tmp / "pca.md"
        store = JsonlResultsStore(out_jsonl)
        summarizer = FakeSummarizer()

        labeler = AxisLabeler(
            reader=reader,
            text_resolver=InlineTextResolver(),
            reducer=PCAReducer(n_components=2),
            strategy=TopProjectionNeighbourhoodStrategy(k=3),
            summarizer=summarizer,
            results_store=store,
        )
        labelled = labeler.run()
        self.assertEqual(labelled, 4)  # 2 components * 2 poles
        self.assertEqual(len(JsonlResultsStore.read_all(out_jsonl)), 4)

        store.write_axes_markdown(out_md, title="Synthetic PCA")
        self.assertIn("Component 1", out_md.read_text(encoding="utf-8"))

        # Also verify AnchorKNN with FaissNearestNeighbourSearch and KPCAReducer
        dataset = reader.fetch_dataset()
        knn = FaissNearestNeighbourSearch(reader.index, dataset.records, metric="cosine")
        kpca_store = JsonlResultsStore(self.tmp / "kpca.jsonl")
        kpca_labeler = AxisLabeler(
            reader=reader,
            text_resolver=InlineTextResolver(),
            reducer=KPCAReducer(n_components=2, kernel="cosine"),
            strategy=AnchorKNNNeighbourhoodStrategy(knn, k=3),
            summarizer=FakeSummarizer(),
            results_store=kpca_store,
        )
        self.assertEqual(kpca_labeler.run(), 2)

    def test_hdbscan_clusterer_recovers_separated_clusters(self):
        dataset_dir = self._build_synthetic_faiss()
        reader = FaissCollectionReader(dataset_dir)
        out_jsonl = self.tmp / "hdbscan.jsonl"
        out_md = self.tmp / "hdbscan.md"
        store = JsonlResultsStore(out_jsonl)

        clusterer = HDBSCANClusterer(
            reader=reader,
            results_store=store,
            text_resolver=InlineTextResolver(),
            summarizer=FakeSummarizer(),
            min_cluster_size=5,
            reduce_method="pca",
            reduce_dim=3,
            n_representatives=3,
        )
        summary = clusterer.run()
        self.assertEqual(summary["n_clusters"], 3)
        store.write_clusters_markdown(out_md, title="Synthetic HDBSCAN")
        self.assertIn("Cluster 0", out_md.read_text(encoding="utf-8"))

    def test_vector_dataset_filter_by(self):
        ds = VectorDataset(
            name="books",
            records=[{"book": "a", "text": "x"}, {"book": "b", "text": "y"}],
            vectors=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
        sub = ds.filter_by(book="b")
        self.assertEqual(len(sub), 1)
        self.assertEqual(sub.texts(), ["y"])


class ChapterSummarizerTests(unittest.TestCase):
    def test_prompt_keeps_each_passage_intact_and_separate_up_to_the_snippet_budget(self):
        summarizer = ChapterSummarizer("http://example.invalid", "fake-model")
        prompt = summarizer._build_prompt(["short text", "x" * (ChapterSummarizer.SNIPPET_CHARS + 50)])

        self.assertIn("short text", prompt)
        self.assertIn("x" * ChapterSummarizer.SNIPPET_CHARS, prompt)
        self.assertNotIn("x" * (ChapterSummarizer.SNIPPET_CHARS + 1), prompt)
        self.assertIn("---", prompt)


class ShuffledLinesSummarizerTests(unittest.TestCase):
    def test_prompt_contains_every_non_blank_line_from_every_text(self):
        summarizer = ShuffledLinesSummarizer("http://example.invalid", "fake-model")
        texts = ["line a\nline b\n\n", "line c\n   \nline d"]
        prompt = summarizer._build_prompt(texts)

        for line in ["line a", "line b", "line c", "line d"]:
            self.assertIn(line, prompt)
        self.assertIn("random sequences", prompt)

    def test_caps_the_number_of_lines_at_max_lines(self):
        summarizer = ShuffledLinesSummarizer("http://example.invalid", "fake-model")
        many_lines = "\n".join(f"line {i}" for i in range(ShuffledLinesSummarizer.MAX_LINES + 20))
        prompt = summarizer._build_prompt([many_lines])

        _instructions, shuffled_block, _label_trailer = prompt.split("\n\n")
        self.assertEqual(len(shuffled_block.splitlines()), ShuffledLinesSummarizer.MAX_LINES)

    def test_blank_lines_are_dropped(self):
        summarizer = ShuffledLinesSummarizer("http://example.invalid", "fake-model")
        prompt = summarizer._build_prompt(["a\n\n\n   \nb"])
        _instructions, shuffled_block, _label_trailer = prompt.split("\n\n")
        self.assertEqual(sorted(shuffled_block.splitlines()), ["a", "b"])


@unittest.skipUnless(has_ollama(), "Ollama not reachable")
class LabelStabilityTests(unittest.TestCase):
    N_RUNS = 3
    MIN_PAIRWISE_COSINE_SIMILARITY = 0.5

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        a, b = np.asarray(a), np.asarray(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def test_repeated_labelling_of_the_same_neighbourhood_is_semantically_stable(self):
        reader = MarkdownFileReader(RES_DIR)
        texts = [reader.read(f) for f in reader.files()]

        summarizer = ChapterSummarizer(OLLAMA_URL, TEST_LABEL_MODEL)
        labels = [summarizer.label(texts) for _ in range(self.N_RUNS)]
        for label in labels:
            self.assertTrue(label, "expected a non-empty label")

        embedder = OllamaEmbedder(OLLAMA_URL, TEST_EMBED_MODEL)
        embeddings = embedder.embed_batch(labels)

        similarities = [
            self._cosine_similarity(embeddings[i], embeddings[j])
            for i in range(len(embeddings))
            for j in range(i + 1, len(embeddings))
        ]
        self.assertTrue(
            all(s >= self.MIN_PAIRWISE_COSINE_SIMILARITY for s in similarities),
            f"labels weren't semantically stable across {self.N_RUNS} repeats: "
            f"{labels} (pairwise cosine similarities={similarities})",
        )


@unittest.skipUnless(has_ollama(), "Ollama not reachable")
class BibliaEmbeddingImporterIntegrationTests(TempDirTestCase):
    """Real, non-mocked embedding call against Ollama."""

    def test_embeds_real_chapters_into_the_local_disk_store(self):
        reader = MarkdownFileReader(RES_DIR)
        files = reader.files()
        store = LocalDiskVectorStore(self.tmp, "test_collection", total_rows=len(files))
        importer = BibliaEmbeddingImporter(
            reader=reader,
            embedder=OllamaEmbedder(OLLAMA_URL, TEST_EMBED_MODEL),
            vector_store=store,
            batch_size=2,
        )
        imported = importer.run()
        store.close()

        self.assertEqual(imported, len(files))
        records, vectors = LocalDiskVectorStore.read_all(self.tmp, "test_collection")
        self.assertEqual(len(records), len(files))
        self.assertEqual(vectors.shape[0], len(files))
        self.assertGreater(vectors.shape[1], 0)
        self.assertFalse(np.allclose(vectors, 0))


if __name__ == "__main__":
    unittest.main()
