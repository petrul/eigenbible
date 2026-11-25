"""Unit tests for Qdrant/Biblioteca downloader, dimensionality reducers,
AxisLabeler edge cases, HDBSCANClusterer, and BERTopicClusterer.
"""
import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from eigenbible.chapter_text_resolver import InlineTextResolver
from eigenbible.clusterer import (
    BERTopicClusterer,
    HDBSCANClusterer,
    _reduce_for_clustering,
    _top_centroid_members,
)
from eigenbible.dataset import VectorDataset
from eigenbible.kpca_labeler import AxisLabeler, _row_ref
from eigenbible.label_results_store import JsonlResultsStore
from eigenbible.neighbourhood_strategy import TopProjectionNeighbourhoodStrategy
from eigenbible.qdrant_downloader import BibliotecaDatasetExporter, BibliotecaTextFetcher
from eigenbible.reducers import KPCAReducer, PCAReducer, build_reducer
from eigenbible.summarizer import Summarizer
from eigenbible.vector_collection_reader import FaissCollectionReader
from eigenbible.vector_store import FaissVectorStore


class FakeSummarizer(Summarizer):
    def __init__(self):
        self.calls: list[list[str]] = []

    def _build_prompt(self, texts: list[str]) -> str:
        return " | ".join(texts)

    def label(self, texts: list[str]) -> str:
        self.calls.append(texts)
        return f"label({len(texts)}): {texts[0][:24]}"


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)


class BibliotecaDownloaderTests(TempDirTestCase):
    @patch("eigenbible.qdrant_downloader.requests.get")
    def test_text_fetcher_verifies_sha256_and_retries_transient_errors(self, mock_get):
        payload = b"Dux Arpad venit in Pannoniam."
        good_sha = hashlib.sha256(payload).hexdigest()

        ok_resp = MagicMock()
        ok_resp.content = payload
        ok_resp.raise_for_status = MagicMock()
        mock_get.side_effect = [RuntimeError("transient"), ok_resp]

        fetcher = BibliotecaTextFetcher(base_url="https://example.test", retries=2, timeout=5)
        text, sha_ok = fetcher.fetch_one("/trad/gesta/_1", expected_sha256=good_sha)
        self.assertEqual(text, "Dux Arpad venit in Pannoniam.")
        self.assertTrue(sha_ok)
        self.assertEqual(mock_get.call_count, 2)

        # Mismatched SHA-256 is flagged rather than silently ignored
        mock_get.side_effect = [ok_resp]
        _text, sha_ok = fetcher.fetch_one("/trad/gesta/_1", expected_sha256="0" * 64)
        self.assertFalse(sha_ok)

    @patch("eigenbible.qdrant_downloader.QdrantCollectionReader")
    def test_exporter_sorts_paragraphs_aligns_vectors_and_writes_texts(self, mock_reader_cls):
        # Deliberately out of order as Qdrant scroll returns UUID order
        t_ch2_p1 = "Chapter two paragraph one."
        t_ch1_p2 = "Chapter one paragraph two."
        t_ch1_p1 = "Chapter one paragraph one."
        records = [
            {
                "id": "u3",
                "opus_path": "trad/sample",
                "url": "http://server:8080/trad/sample/ch2/_1",
                "sha256": hashlib.sha256(t_ch2_p1.encode()).hexdigest(),
            },
            {
                "id": "u2",
                "opus_path": "trad/sample",
                "url": "http://server:8080/trad/sample/ch1/_2",
                "sha256": hashlib.sha256(t_ch1_p2.encode()).hexdigest(),
            },
            {
                "id": "u1",
                "opus_path": "trad/sample",
                "url": "http://server:8080/trad/sample/ch1/_1",
                "sha256": "wrong-sha",
            },
        ]
        # Distinctive vectors so we can prove FAISS vectors stayed aligned after sorting!
        vectors = np.array(
            [
                [0.0, 0.0, 1.0],  # ch2/_1
                [0.0, 1.0, 0.0],  # ch1/_2
                [1.0, 0.0, 0.0],  # ch1/_1
            ],
            dtype=np.float32,
        )
        mock_reader_cls.return_value.fetch_dataset.return_value = VectorDataset(
            name="trad/sample", records=records, vectors=vectors
        )

        fake_fetcher = MagicMock()
        fake_fetcher.base_url = "https://biblioteca.scriptorium.ro"
        # fetch_many receives jobs in sorted order: ch1/_1, ch1/_2, ch2/_1
        fake_fetcher.fetch_many.return_value = [
            (t_ch1_p1, False),
            (t_ch1_p2, True),
            (t_ch2_p1, True),
        ]

        exporter = BibliotecaDatasetExporter(out_root=self.tmp, fetcher=fake_fetcher)
        summary = exporter.export_work("sample_work", "trad/sample", exclude_headings=False, min_chars=0)

        self.assertEqual(summary["paragraphs"], 3)
        self.assertEqual(summary["sections"], 2)
        self.assertEqual(summary["sha256_mismatches"], 1)

        dataset, _index = FaissVectorStore.load_dataset_dir(self.tmp / "sample_work")
        self.assertEqual(
            [(r["section"], r["paragraph_num"], r["sha256_verified"]) for r in dataset.records],
            [("ch1", 1, False), ("ch1", 2, True), ("ch2", 1, True)],
        )
        # Vectors must be reordered identically to records!
        np.testing.assert_array_almost_equal(
            dataset.vectors,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )
        ch1_text = (self.tmp / "sample_work" / "texts" / "ch1.txt").read_text(encoding="utf-8")
        self.assertEqual(ch1_text, f"{t_ch1_p1}\n\n{t_ch1_p2}\n")

    @patch("eigenbible.qdrant_downloader.QdrantCollectionReader")
    def test_exporter_raises_on_empty_qdrant_result(self, mock_reader_cls):
        mock_reader_cls.return_value.fetch_dataset.return_value = VectorDataset(
            name="empty", records=[], vectors=np.empty((0, 0), dtype=np.float32)
        )
        exporter = BibliotecaDatasetExporter(out_root=self.tmp, fetcher=MagicMock())
        with self.assertRaises(ValueError):
            exporter.export_work("empty_work", "trad/missing")


class ReducersAndAxisLabelerEdgeCaseTests(TempDirTestCase):
    def test_build_reducer_and_component_clamping(self):
        pca = build_reducer("pca", n_components=50)
        kpca = build_reducer("kpca", n_components=50, kernel="cosine")
        self.assertIsInstance(pca, PCAReducer)
        self.assertIsInstance(kpca, KPCAReducer)
        with self.assertRaises(ValueError):
            build_reducer("unknown", n_components=2)

        rng = np.random.default_rng(0)
        small = rng.normal(size=(5, 4)).astype(np.float32)
        proj_pca, var_ratio = pca.fit_transform(small)
        self.assertEqual(proj_pca.shape, (5, 4))  # clamped to min(n-1, d) = 4
        self.assertEqual(len(var_ratio), 4)

        proj_kpca, eigvals = kpca.fit_transform(small)
        self.assertEqual(proj_kpca.shape, (5, 4))  # clamped to n-1 = 4
        self.assertEqual(len(eigvals), 4)

    def test_row_ref_formats_biblioteca_bible_and_fallback_rows(self):
        self.assertEqual(_row_ref({"section": "de_scithia", "paragraph_num": 3}), "de_scithia/_3")
        self.assertEqual(_row_ref({"bible": "darby", "file_name": "a/b.md"}), "darby/a/b.md")
        self.assertEqual(_row_ref({"file_name": "plain.md"}), "plain.md")
        self.assertEqual(_row_ref({"id": "uuid-1"}), "uuid-1")

    def test_axis_labeler_rejects_empty_dataset(self):
        empty_reader = MagicMock()
        empty_reader.collection_name = "empty"
        empty_reader.fetch_dataset.return_value = VectorDataset(
            "empty", [], np.empty((0, 4), dtype=np.float32)
        )
        labeler = AxisLabeler(
            reader=empty_reader,
            text_resolver=InlineTextResolver(),
            reducer=PCAReducer(2),
            strategy=TopProjectionNeighbourhoodStrategy(k=2),
            summarizer=FakeSummarizer(),
            results_store=JsonlResultsStore(self.tmp / "empty.jsonl"),
        )
        with self.assertRaises(ValueError):
            labeler.run()


class ClusteringHelpersAndBERTopicTests(TempDirTestCase):
    def _build_three_topic_faiss(self) -> Path:
        rng = np.random.default_rng(42)
        topics = [
            ([6.0, 0.0, 0.0, 0.0, 0.0, 0.0], "arpad military conquest army battle fortress siege"),
            ([0.0, 6.0, 0.0, 0.0, 0.0, 0.0], "church monastery bishop orthodox liturgy consecration prayer"),
            ([0.0, 0.0, 6.0, 0.0, 0.0, 0.0], "treaty envoy tribute diplomacy alliance peace negotiation"),
        ]
        vecs = []
        names = []
        extras = []
        for t_idx, (center, vocab) in enumerate(topics):
            cluster_vecs = rng.normal(loc=center, scale=0.12, size=(12, 6)).astype(np.float32)
            cluster_vecs /= np.linalg.norm(cluster_vecs, axis=1, keepdims=True)
            vecs.append(cluster_vecs)
            for j in range(12):
                names.append(f"topic_{t_idx}/_{j + 1}.txt")
                extras.append({
                    "section": f"topic_{t_idx}",
                    "paragraph_num": j + 1,
                    "text": f"Passage {j} about {vocab} in medieval chronicle",
                })
        matrix = np.vstack(vecs)
        store = FaissVectorStore(self.tmp, "topics", metric="cosine")
        store.insert_batch(names, matrix, extra_fields=extras)
        store.close()
        return self.tmp / "topics"

    def test_top_centroid_members_and_reduce_for_clustering(self):
        vectors = np.array(
            [[1.0, 0.0], [0.98, 0.2], [0.0, 1.0], [0.0, 0.0]],
            dtype=np.float32,
        )
        top = _top_centroid_members(vectors, np.array([0, 1, 2, 3]), k=2)
        self.assertEqual(top[0], 1)  # closest to mean direction of the 3 non-zero vectors
        self.assertEqual(len(top), 2)

        unchanged = _reduce_for_clustering(vectors, method="none", n_components=2, n_neighbors=2)
        np.testing.assert_array_equal(unchanged, vectors)
        with self.assertRaises(ValueError):
            _reduce_for_clustering(vectors, method="tsne", n_components=1, n_neighbors=2)

    def test_hdbscan_without_summarizer_runs_cleanly_offline(self):
        dataset_dir = self._build_three_topic_faiss()
        store = JsonlResultsStore(self.tmp / "hdbscan_nollm.jsonl")
        runner = HDBSCANClusterer(
            reader=FaissCollectionReader(dataset_dir),
            results_store=store,
            text_resolver=InlineTextResolver(),
            summarizer=None,
            min_cluster_size=5,
            reduce_method="none",
        )
        summary = runner.run()
        self.assertEqual(summary["n_clusters"], 3)
        self.assertTrue(all(rec["label"] == "" for rec in store.records))

    def test_bertopic_uses_precomputed_embeddings_and_extracts_ctfidf_keywords(self):
        dataset_dir = self._build_three_topic_faiss()
        store = JsonlResultsStore(self.tmp / "bertopic.jsonl")
        summarizer = FakeSummarizer()

        runner = BERTopicClusterer(
            reader=FaissCollectionReader(dataset_dir),
            results_store=store,
            text_resolver=InlineTextResolver(),
            summarizer=summarizer,
            min_cluster_size=5,
            cluster_selection_method="eom",
            reduce_dim=3,
            n_neighbors=8,
            n_representatives=3,
            top_n_words=5,
        )
        summary = runner.run()
        self.assertGreaterEqual(summary["n_clusters"], 2)
        non_noise = [rec for rec in store.records if rec["cluster_id"] != -1]
        self.assertEqual(len(non_noise), summary["n_clusters"])
        for rec in non_noise:
            self.assertTrue(rec["keywords"], "expected non-empty c-TF-IDF keywords")
            self.assertTrue(rec["label"].startswith("label("))
            self.assertGreater(len(rec["representatives"]), 0)


if __name__ == "__main__":
    unittest.main()
