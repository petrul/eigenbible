"""Rebuilds the combined collection out of already-populated per-bible

collections, by copying their vectors instead of re-embedding every chapter
a second time. One implementation per backend (Milvus / local disk),
matching whichever --backend the embed run used.
"""
import logging

import numpy as np
from pymilvus import MilvusClient

from .vector_store import LocalDiskVectorStore, create_milvus_collection

logger = logging.getLogger(__name__)


class MilvusCollectionMerger:
    """Each copied row is tagged with a 'bible' field, so the combined

    collection can still be scoped back down to a single bible, or searched
    across all of them at once."""

    BATCH_SIZE = 1000

    def __init__(self, uri: str):
        self.client = MilvusClient(uri=uri)

    def merge(self, sources: dict, target_collection: str) -> int:
        total = 0
        target_ready = False
        for bible_key, source_collection in sources.items():
            logger.info("Copying vectors from '%s' (%s) into '%s'", source_collection, bible_key, target_collection)
            offset = 0
            while True:
                rows = self.client.query(
                    collection_name=source_collection,
                    filter="id >= 0",
                    output_fields=["vector", "file_name"],
                    limit=self.BATCH_SIZE,
                    offset=offset,
                )
                if not rows:
                    break
                if not target_ready:
                    self._ensure_collection(target_collection, len(rows[0]["vector"]))
                    target_ready = True
                self.client.insert(
                    collection_name=target_collection,
                    data=[
                        {"vector": row["vector"], "file_name": row["file_name"], "bible": bible_key}
                        for row in rows
                    ],
                )
                offset += len(rows)
                total += len(rows)
                logger.info("  ... copied %d rows from '%s' so far", offset, source_collection)
        return total

    def _ensure_collection(self, name: str, dimension: int) -> None:
        if self.client.has_collection(name):
            return
        logger.info(
            "Collection '%s' doesn't exist yet, creating it (dim=%d) with a 'bible' field so "
            "rows from every bible can share one collection and still be told apart, plus a "
            "COSINE index so it's loadable/searchable right away",
            name,
            dimension,
        )
        create_milvus_collection(self.client, name, dimension, extra_varchar_fields=["bible"])
        logger.info("Collection '%s' created", name)


class LocalDiskCollectionMerger:
    """Combined-collection equivalent for the local disk backend: reads every

    source collection's already-written vectors+records back off disk (via
    LocalDiskVectorStore.read_all) and writes them into one target
    collection, each row tagged with which bible it came from."""

    BATCH_SIZE = 1000

    def __init__(self, directory):
        self.directory = directory

    def merge(self, sources: dict, target_collection: str) -> int:
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
