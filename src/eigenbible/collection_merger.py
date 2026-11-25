"""Rebuilds a combined collection out of already-populated per-source collections
by copying their vectors instead of re-embedding every document a second time.

Implementations for FAISS, Qdrant, and LocalDisk backends all share the
CollectionMerger interface.
"""
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .vector_collection_reader import QdrantCollectionReader
from .vector_store import FaissVectorStore, LocalDiskVectorStore, QdrantVectorStore

logger = logging.getLogger(__name__)


class CollectionMerger(ABC):
    """Merges multiple named source collections into one combined target collection."""

    @abstractmethod
    def merge(self, sources: dict[str, str], target_collection: str) -> int: ...


class FaissCollectionMerger(CollectionMerger):
    """Combines multiple local FAISS collections into one, tagging each row with its source."""

    BATCH_SIZE = 1000

    def __init__(self, directory: Path, source_field: str = "bible"):
        self.directory = Path(directory)
        self.source_field = source_field

    def merge(self, sources: dict[str, str], target_collection: str) -> int:
        datasets = {}
        dimension = None
        metric = "cosine"
        total = 0
        for source_key, source_collection in sources.items():
            logger.info("Reading FAISS collection '%s' (%s)", source_collection, source_key)
            dataset, _index = FaissVectorStore.load_dataset_dir(self.directory / source_collection)
            if len(dataset) == 0:
                continue
            if dimension is None:
                dimension = dataset.dimension
                metric = dataset.meta.get("metric", "cosine")
            elif dimension != dataset.dimension:
                raise ValueError(
                    f"Dimension mismatch: '{source_collection}' has {dataset.dimension}, expected {dimension}"
                )
            datasets[source_key] = dataset
            total += len(dataset)

        if total == 0:
            return 0

        target = FaissVectorStore(
            self.directory,
            target_collection,
            metric=metric,
            meta={"sources": list(sources.keys())},
        )
        for source_key, dataset in datasets.items():
            for start in range(0, len(dataset), self.BATCH_SIZE):
                end = min(start + self.BATCH_SIZE, len(dataset))
                batch_records = dataset.records[start:end]
                extra_fields = [
                    {**{k: v for k, v in rec.items() if k not in ("file_name", "row_idx")}, self.source_field: source_key}
                    for rec in batch_records
                ]
                target.insert_batch(
                    file_names=[rec.get("file_name", rec.get("id", str(start + i))) for i, rec in enumerate(batch_records)],
                    vectors=dataset.vectors[start:end],
                    extra_fields=extra_fields,
                )
        target.close()
        logger.info("Wrote %d combined rows to FAISS collection '%s'", total, target_collection)
        return total


class QdrantCollectionMerger(CollectionMerger):
    """Combines multiple Qdrant collections into one, tagging each row with its source."""

    BATCH_SIZE = 256

    def __init__(
        self,
        url: str,
        vector_name: str | None = None,
        source_field: str = "bible",
    ):
        self.url = url
        self.vector_name = vector_name
        self.source_field = source_field

    def merge(self, sources: dict[str, str], target_collection: str) -> int:
        target = QdrantVectorStore(self.url, target_collection, vector_name=self.vector_name)
        total = 0
        for source_key, source_collection in sources.items():
            logger.info("Copying Qdrant vectors from '%s' (%s) into '%s'", source_collection, source_key, target_collection)
            reader = QdrantCollectionReader(
                url=self.url,
                collection_name=source_collection,
                vector_name=self.vector_name,
                batch_size=self.BATCH_SIZE,
            )
            dataset = reader.fetch_dataset()
            for start in range(0, len(dataset), self.BATCH_SIZE):
                end = min(start + self.BATCH_SIZE, len(dataset))
                batch_records = dataset.records[start:end]
                extra_fields = [
                    {**{k: v for k, v in rec.items() if k not in ("file_name", "id")}, self.source_field: source_key}
                    for rec in batch_records
                ]
                target.insert_batch(
                    file_names=[rec.get("file_name", rec.get("id", str(start + i))) for i, rec in enumerate(batch_records)],
                    vectors=dataset.vectors[start:end],
                    extra_fields=extra_fields,
                )
            total += len(dataset)
        logger.info("Combined %d rows into Qdrant collection '%s'", total, target_collection)
        return total


class LocalDiskCollectionMerger(CollectionMerger):
    """Combines multiple local-disk memmap collections into one target collection."""

    BATCH_SIZE = 1000

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def merge(self, sources: dict[str, str], target_collection: str) -> int:
        records_by_source = {}
        vectors_by_source = {}
        dimension = None
        total = 0
        for bible_key, source_collection in sources.items():
            logger.info("Reading vectors from local disk collection '%s' (%s)", source_collection, bible_key)
            records, vectors = LocalDiskVectorStore.read_all(self.directory, source_collection)
            if dimension is None:
                dimension = vectors.shape[1]
            elif dimension != vectors.shape[1]:
                raise ValueError(
                    f"Dimension mismatch: '{source_collection}' has {vectors.shape[1]}, expected {dimension}"
                )
            records_by_source[bible_key] = records
            vectors_by_source[bible_key] = vectors
            total += len(records)
            logger.info("  ... read %d rows from '%s'", len(records), source_collection)

        if total == 0:
            return 0

        target = LocalDiskVectorStore(self.directory, target_collection, total_rows=total)
        for bible_key, records in records_by_source.items():
            vectors = vectors_by_source[bible_key]
            for start in range(0, len(records), self.BATCH_SIZE):
                end = min(start + self.BATCH_SIZE, len(records))
                batch_records = records[start:end]
                target.insert_batch(
                    file_names=[r["file_name"] for r in batch_records],
                    vectors=np.asarray(vectors[start:end]).tolist(),
                    extra_fields=[{"bible": bible_key} for _ in batch_records],
                )
        target.close()
        logger.info("Wrote %d combined rows to local disk collection '%s'", total, target_collection)
        return total
