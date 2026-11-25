"""Backend-agnostic readers that load a VectorDataset from FAISS, LocalDisk, or Qdrant."""
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from .dataset import VectorDataset
from .vector_store import FaissVectorStore, LocalDiskVectorStore

logger = logging.getLogger(__name__)


class VectorDatasetReader(ABC):
    """Loads vectors and parallel metadata records from any supported backend."""

    @property
    @abstractmethod
    def collection_name(self) -> str: ...

    @abstractmethod
    def fetch_dataset(self) -> VectorDataset: ...


class FaissCollectionReader(VectorDatasetReader):
    """Reads a local FAISS dataset directory (`index.faiss` + `records.jsonl` + `meta.json`)."""

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self._index: faiss.Index | None = None
        self._dataset: VectorDataset | None = None

    @property
    def collection_name(self) -> str:
        return self.dataset_dir.name

    @property
    def index(self) -> faiss.Index:
        if self._index is None:
            self.fetch_dataset()
        assert self._index is not None
        return self._index

    def fetch_dataset(self) -> VectorDataset:
        if self._dataset is None:
            self._dataset, self._index = FaissVectorStore.load_dataset_dir(self.dataset_dir)
            logger.info(
                "Loaded FAISS dataset '%s' (%d vectors, dim=%d) from %s",
                self._dataset.name, len(self._dataset), self._dataset.dimension, self.dataset_dir,
            )
        return self._dataset


class LocalDiskCollectionReader(VectorDatasetReader):
    """Reads a local numpy memmap collection written by LocalDiskVectorStore."""

    def __init__(self, directory: Path, collection_name: str):
        self.directory = Path(directory)
        self._collection_name = collection_name

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def fetch_dataset(self) -> VectorDataset:
        dataset = LocalDiskVectorStore.load_dataset(self.directory, self._collection_name)
        logger.info(
            "Loaded local-disk dataset '%s' (%d vectors, dim=%d)",
            dataset.name, len(dataset), dataset.dimension,
        )
        return dataset


class QdrantCollectionReader(VectorDatasetReader):
    """Streams points (with vectors and payloads) from a Qdrant collection via scroll()."""

    BATCH_SIZE = 256

    def __init__(
        self,
        url: str,
        collection_name: str,
        vector_name: str | None = None,
        opus_path: str | None = None,
        batch_size: int = BATCH_SIZE,
    ):
        self.client = QdrantClient(url=url, check_compatibility=False)
        self._collection_name = collection_name
        self.vector_name = vector_name
        self.opus_path = opus_path
        self.batch_size = batch_size

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def fetch_dataset(self) -> VectorDataset:
        scroll_filter = None
        if self.opus_path is not None:
            scroll_filter = Filter(
                must=[FieldCondition(key="opus_path", match=MatchValue(value=self.opus_path))]
            )

        records: list[dict[str, Any]] = []
        vectors: list[list[float]] = []
        offset = None

        while True:
            points, offset = self.client.scroll(
                collection_name=self._collection_name,
                scroll_filter=scroll_filter,
                limit=self.batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            if not points:
                break
            for point in points:
                vec = self._extract_vector(point.vector)
                payload = dict(point.payload or {})
                payload["id"] = str(point.id)
                records.append(payload)
                vectors.append(vec)
            logger.info(
                "Scrolled %d points from Qdrant '%s'%s",
                len(records),
                self._collection_name,
                f" (opus_path={self.opus_path})" if self.opus_path else "",
            )
            if offset is None:
                break

        matrix = (
            np.asarray(vectors, dtype=np.float32)
            if vectors
            else np.empty((0, 0), dtype=np.float32)
        )
        return VectorDataset(
            name=self.opus_path or self._collection_name,
            records=records,
            vectors=matrix,
            meta={
                "source": "qdrant",
                "qdrant_collection": self._collection_name,
                "vector_name": self.vector_name,
                "opus_path": self.opus_path,
            },
        )

    def _extract_vector(self, raw_vector: Any) -> list[float]:
        if isinstance(raw_vector, dict):
            if self.vector_name:
                return list(raw_vector[self.vector_name])
            if len(raw_vector) == 1:
                return list(next(iter(raw_vector.values())))
            raise ValueError(
                f"Qdrant returned multiple named vectors {list(raw_vector)}; specify vector_name"
            )
        return list(raw_vector)
