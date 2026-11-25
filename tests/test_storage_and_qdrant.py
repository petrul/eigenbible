"""Unit tests for VectorDataset, FAISS/Qdrant stores, readers, mergers, kNN search,
text resolvers, and JSONL results storage (all fast and network-free).
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from eigenbible.chapter_text_resolver import ChapterTextResolver, InlineTextResolver
from eigenbible.collection_merger import FaissCollectionMerger, QdrantCollectionMerger
from eigenbible.dataset import VectorDataset
from eigenbible.label_results_store import JsonlResultsStore
from eigenbible.nearest_neighbour_search import (
    FaissNearestNeighbourSearch,
    QdrantNearestNeighbourSearch,
)
from eigenbible.vector_collection_reader import (
    LocalDiskCollectionReader,
    QdrantCollectionReader,
)
from eigenbible.vector_store import (
    FaissVectorStore,
    LocalDiskVectorStore,
    QdrantVectorStore,
)


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)


class VectorDatasetTests(unittest.TestCase):
    def test_rejects_mismatched_record_and_vector_counts(self):
        with self.assertRaises(ValueError):
            VectorDataset(
                name="bad",
                records=[{"text": "only one"}],
                vectors=np.zeros((2, 3), dtype=np.float32),
            )

    def test_normalized_vectors_handles_regular_and_zero_vectors(self):
        ds = VectorDataset(
            name="norms",
            records=[{"text": "a"}, {"text": "b"}],
            vectors=np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32),
        )
        normed = ds.normalized_vectors()
        np.testing.assert_array_almost_equal(normed[0], [0.6, 0.8])
        np.testing.assert_array_almost_equal(normed[1], [0.0, 0.0])

    def test_texts_and_empty_filter_preserve_dimension(self):
        ds = VectorDataset(
            name="books",
            records=[{"book": "a", "body": "hello"}, {"book": "b"}],
            vectors=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        )
        self.assertEqual(ds.texts("body"), ["hello", ""])
        empty = ds.filter_by(book="missing")
        self.assertEqual(len(empty), 0)
        self.assertEqual(empty.vectors.shape, (0, 3))
        self.assertEqual(empty.dimension, 0)


class FaissStoreAndSearchEdgeCaseTests(TempDirTestCase):
    def test_l2_metric_round_trip_and_nearest_neighbour_ordering(self):
        store = FaissVectorStore(self.tmp, "l2_work", metric="l2")
        store.insert_batch([], [])  # empty batch is a safe no-op
        store.insert_batch(
            ["origin", "far", "near"],
            [[0.0, 0.0], [10.0, 10.0], [0.2, 0.1]],
        )
        store.close()

        dataset, index = FaissVectorStore.load_dataset_dir(self.tmp / "l2_work")
        self.assertEqual(dataset.meta["metric"], "l2")
        search = FaissNearestNeighbourSearch(index, dataset.records, metric="l2")
        hits = search.search([0.05, 0.05], k=10)  # k > ntotal clamps cleanly
        self.assertEqual([h["file_name"] for h in hits], ["origin", "near", "far"])
        self.assertIn("_score", hits[0])

    def test_rejects_unsupported_metric(self):
        store = FaissVectorStore(self.tmp, "bad_metric", metric="manhattan")
        with self.assertRaises(ValueError):
            store.insert_batch(["a"], [[1.0, 2.0]])

    def test_load_rejects_record_and_index_count_mismatch(self):
        store = FaissVectorStore(self.tmp, "corrupt")
        store.insert_batch(["a", "b"], [[1.0, 0.0], [0.0, 1.0]])
        store.close()
        (self.tmp / "corrupt" / FaissVectorStore.RECORDS_FILENAME).write_text(
            json.dumps({"file_name": "a"}) + "\n", encoding="utf-8"
        )
        with self.assertRaises(ValueError):
            FaissVectorStore.load_dataset_dir(self.tmp / "corrupt")

    def test_faiss_merger_rejects_dimension_mismatch_and_handles_empty(self):
        s1 = FaissVectorStore(self.tmp, "dim2")
        s1.insert_batch(["a"], [[1.0, 0.0]])
        s1.close()

        s2 = FaissVectorStore(self.tmp, "dim3")
        s2.insert_batch(["b"], [[1.0, 0.0, 0.0]])
        s2.close()

        merger = FaissCollectionMerger(self.tmp)
        with self.assertRaises(ValueError):
            merger.merge({"x": "dim2", "y": "dim3"}, "combined")

        self.assertEqual(merger.merge({}, "empty_combined"), 0)


class LocalDiskReaderTests(TempDirTestCase):
    def test_loads_local_disk_store_as_vector_dataset(self):
        store = LocalDiskVectorStore(self.tmp, "mem_coll", total_rows=2)
        store.insert_batch(["f1", "f2"], [[1.0, 2.0], [3.0, 4.0]])
        store.close()

        reader = LocalDiskCollectionReader(self.tmp, "mem_coll")
        self.assertEqual(reader.collection_name, "mem_coll")
        ds = reader.fetch_dataset()
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds.dimension, 2)
        self.assertEqual([r["file_name"] for r in ds.records], ["f1", "f2"])


class QdrantComponentsTests(unittest.TestCase):
    @patch("eigenbible.vector_store.QdrantClient")
    def test_qdrant_vector_store_creates_named_collection_and_uses_idempotent_ids(self, mock_cls):
        client = MagicMock()
        client.collection_exists.return_value = False
        mock_cls.return_value = client

        store = QdrantVectorStore("http://zmeu.local:6333", "test_coll", vector_name="embedding")
        store.insert_batch(["ch1.md"], [[0.5, 0.5]], extra_fields=[{"bible": "darby"}])
        client.create_collection.assert_called_once()
        first_points = client.upsert.call_args.kwargs["points"]
        self.assertEqual(list(first_points[0].vector.keys()), ["embedding"])

        # Second insert with same key produces identical deterministic UUID5 point ID
        client.collection_exists.return_value = True
        store.insert_batch(["ch1.md"], [[0.5, 0.5]], extra_fields=[{"bible": "darby"}])
        second_points = client.upsert.call_args.kwargs["points"]
        self.assertEqual(first_points[0].id, second_points[0].id)

        with self.assertRaises(ValueError):
            store.insert_batch(["ch2.md"], [[1.0, 2.0, 3.0]])

    @patch("eigenbible.vector_collection_reader.QdrantClient")
    def test_qdrant_reader_paginates_scroll_and_extracts_named_vectors(self, mock_cls):
        client = MagicMock()
        mock_cls.return_value = client
        page1 = (
            [SimpleNamespace(id="id-1", vector={"embedding": [1.0, 0.0]}, payload={"opus_path": "x", "url": "/a/_1"})],
            "cursor-2",
        )
        page2 = (
            [SimpleNamespace(id="id-2", vector={"embedding": [0.0, 1.0]}, payload={"opus_path": "x", "url": "/a/_2"})],
            None,
        )
        client.scroll.side_effect = [page1, page2]

        reader = QdrantCollectionReader(
            url="http://zmeu.local:6333",
            collection_name="biblioteca_paragraphs_bge_m3",
            vector_name="embedding",
            opus_path="x",
            batch_size=1,
        )
        ds = reader.fetch_dataset()
        self.assertEqual(len(ds), 2)
        self.assertEqual([r["id"] for r in ds.records], ["id-1", "id-2"])
        np.testing.assert_array_almost_equal(ds.vectors, [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(client.scroll.call_count, 2)

        # Ambiguous multi-named vector dict without vector_name raises ValueError
        unnamed_reader = QdrantCollectionReader("http://zmeu.local:6333", "coll", vector_name=None)
        with self.assertRaises(ValueError):
            unnamed_reader._extract_vector({"v1": [1.0], "v2": [2.0]})

    def test_qdrant_nearest_neighbour_search_unpacks_query_points(self):
        client = MagicMock()
        client.query_points.return_value = SimpleNamespace(
            points=[
                SimpleNamespace(id="p1", score=0.95, payload={"file_name": "a.md"}),
                SimpleNamespace(id="p2", score=0.80, payload={"file_name": "b.md"}),
            ]
        )
        search = QdrantNearestNeighbourSearch(client, "my_coll", vector_name="embedding")
        hits = search.search(np.array([1.0, 0.0], dtype=np.float32), k=2)
        self.assertEqual(hits[0]["file_name"], "a.md")
        self.assertEqual(hits[0]["id"], "p1")
        self.assertAlmostEqual(hits[0]["_score"], 0.95)
        client.query_points.assert_called_once()

    @patch("eigenbible.collection_merger.QdrantVectorStore")
    @patch("eigenbible.collection_merger.QdrantCollectionReader")
    def test_qdrant_collection_merger_tags_each_source_and_writes_target(self, mock_reader_cls, mock_store_cls):
        ds_a = VectorDataset("src_a", [{"file_name": "a1", "id": "1"}], np.array([[1.0, 0.0]], dtype=np.float32))
        ds_b = VectorDataset("src_b", [{"file_name": "b1", "id": "2"}], np.array([[0.0, 1.0]], dtype=np.float32))
        mock_reader_cls.side_effect = [
            MagicMock(fetch_dataset=MagicMock(return_value=ds_a)),
            MagicMock(fetch_dataset=MagicMock(return_value=ds_b)),
        ]
        target_store = MagicMock()
        mock_store_cls.return_value = target_store

        merger = QdrantCollectionMerger("http://zmeu.local:6333", source_field="corpus")
        total = merger.merge({"first": "src_a", "second": "src_b"}, "merged")

        self.assertEqual(total, 2)
        self.assertEqual(target_store.insert_batch.call_count, 2)
        first_extra = target_store.insert_batch.call_args_list[0].kwargs["extra_fields"]
        second_extra = target_store.insert_batch.call_args_list[1].kwargs["extra_fields"]
        self.assertEqual(first_extra, [{"corpus": "first"}])
        self.assertEqual(second_extra, [{"corpus": "second"}])


class TextResolverAndResultsStoreTests(TempDirTestCase):
    def test_inline_text_resolver_reads_custom_field_and_errors_when_missing(self):
        resolver = InlineTextResolver(field_name="body")
        self.assertEqual(resolver.read_row({"body": "hello"}), "hello")
        with self.assertRaises(KeyError):
            resolver.read_row({"other": "nope"})

    def test_chapter_text_resolver_resolves_explicit_and_default_bible(self):
        chapter = self.tmp / "biblia-ortdx-capitole" / "vechiul" / "facerea-1.md"
        chapter.parent.mkdir(parents=True)
        chapter.write_text("La inceput", encoding="utf-8")

        resolver = ChapterTextResolver(self.tmp, default_bible="ortodox")
        self.assertEqual(resolver.read_row({"file_name": "vechiul/facerea-1.md"}), "La inceput")
        self.assertEqual(
            resolver.read_row({"bible": "ortodox", "file_name": "vechiul/facerea-1.md"}),
            "La inceput",
        )

        strict_resolver = ChapterTextResolver(self.tmp)
        with self.assertRaises(KeyError):
            strict_resolver.read_row({"file_name": "vechiul/facerea-1.md"})

    def test_jsonl_results_store_truncates_long_labels_and_supports_append_mode(self):
        path = self.tmp / "out" / "results.jsonl"
        store = JsonlResultsStore(path, overwrite=True)
        store.insert({"component_index": 0, "label": "x" * (JsonlResultsStore.LABEL_MAX_LENGTH + 50)})

        appended = JsonlResultsStore(path, overwrite=False)
        appended.insert({"component_index": 1, "label": "short"})

        rows = JsonlResultsStore.read_all(path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rows[0]["label"]), JsonlResultsStore.LABEL_MAX_LENGTH)
        self.assertTrue(rows[0]["label"].endswith("…"))
        self.assertEqual(rows[1]["label"], "short")


if __name__ == "__main__":
    unittest.main()
