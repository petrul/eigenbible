import tempfile
import unittest
from pathlib import Path

import numpy as np

from eigenbible.dataset_cleaner import filter_faiss_dataset, should_keep_record
from eigenbible.vector_collection_reader import FaissCollectionReader
from eigenbible.vector_store import FaissVectorStore


class TestDatasetCleaner(unittest.TestCase):
    def test_should_keep_record_filters_headings_about_and_short_rows(self) -> None:
        long_text = "A" * 120
        self.assertFalse(should_keep_record({"paragraph_num": 1, "section": "ch1", "text": long_text}))
        self.assertFalse(should_keep_record({"paragraph_num": 2, "section": "about", "text": long_text}))
        self.assertFalse(should_keep_record({"paragraph_num": 2, "section": "ch1", "text": "Short line."}))
        self.assertTrue(should_keep_record({"paragraph_num": 2, "section": "ch1", "text": long_text}))
        self.assertTrue(
            should_keep_record({"paragraph_num": 1, "section": "ch1", "text": "Short"}, exclude_headings=False, min_chars=0)
        )

    def test_filter_faiss_dataset_in_place_preserves_vector_alignment_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vecs = np.eye(4, dtype=np.float32)
            store = FaissVectorStore(directory=root, collection_name="sample", metric="cosine", meta={"source": "test"})
            store.insert_batch(
                file_names=["about/_2.txt", "ch1/_1.txt", "ch1/_2.txt", "ch1/_3.txt"],
                vectors=vecs,
                extra_fields=[
                    {"section": "about", "paragraph_num": 2, "url": "https://biblioteca.scriptorium.ro/x/about/_2", "text": "B" * 140},
                    {"section": "ch1", "paragraph_num": 1, "url": "https://biblioteca.scriptorium.ro/x/ch1/_1", "text": "Chapter Title"},
                    {"section": "ch1", "paragraph_num": 2, "url": "https://biblioteca.scriptorium.ro/x/ch1/_2", "text": "Too short"},
                    {"section": "ch1", "paragraph_num": 3, "url": "https://biblioteca.scriptorium.ro/x/ch1/_3", "text": "Substantive body paragraph " * 6},
                ],
            )
            store.close()

            ds_dir = root / "sample"
            stats1 = filter_faiss_dataset(ds_dir, exclude_headings=True, exclude_sections=("about",), min_chars=100)
            self.assertEqual(stats1, {"before": 4, "after": 1, "dropped": 3})

            ds = FaissCollectionReader(ds_dir).fetch_dataset()
            self.assertEqual(len(ds), 1)
            self.assertEqual(ds.records[0]["paragraph_num"], 3)
            np.testing.assert_allclose(ds.vectors[0], vecs[3], atol=1e-6)

            # Second call is a no-op
            stats2 = filter_faiss_dataset(ds_dir, exclude_headings=True, exclude_sections=("about",), min_chars=100)
            self.assertEqual(stats2, {"before": 1, "after": 1, "dropped": 0})


if __name__ == "__main__":
    unittest.main()
