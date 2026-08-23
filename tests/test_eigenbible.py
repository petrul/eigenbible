"""Tests for the eigenbible package.

Covers the new local-disk vector store backend and its merge path with
fast, network-free fixtures/fakes, plus a real (non-mocked) Ollama
embedding call end-to-end using a small, fast model (nomic-embed-text)
rather than the production qwen3-embedding:4b - a wrong
embed_batch()/insert_batch() wiring is exactly the kind of bug mocking the
network call out entirely would hide, and nomic is fast enough (a couple
of seconds once warm) to actually run this routinely.

Run with: rake test
      or: uv run python -m unittest discover -s tests -v
"""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import requests

from eigenbible.collection_merger import LocalDiskCollectionMerger
from eigenbible.embedder import OllamaEmbedder
from eigenbible.importer import BibliaEmbeddingImporter
from eigenbible.markdown_reader import MarkdownFileReader
from eigenbible.settings import MILVUS_URI, OLLAMA_URL
from eigenbible.vector_store import LocalDiskVectorStore

RES_DIR = Path(__file__).parent / "res"
TEST_EMBED_MODEL = "nomic-embed-text:v1.5"  # small/fast - keeps the real network test quick


def has_ollama() -> bool:
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).raise_for_status()
        return True
    except Exception:
        return False


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)


class SettingsTests(unittest.TestCase):
    def test_ollama_and_milvus_urls_load_from_env(self):
        self.assertTrue(OLLAMA_URL)
        self.assertTrue(MILVUS_URI)


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


@unittest.skipUnless(has_ollama(), "Ollama not reachable")
class BibliaEmbeddingImporterIntegrationTests(TempDirTestCase):
    """Real, non-mocked embedding call against Ollama (nomic-embed-text)."""

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
        self.assertFalse(np.allclose(vectors, 0))  # real embeddings, not placeholder zeros


if __name__ == "__main__":
    unittest.main()
