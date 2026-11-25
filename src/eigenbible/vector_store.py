"""Vector storage backends: FAISS (local index + JSONL records), Qdrant (remote
vector DB), and LocalDisk (raw numpy memmap).

All three implement VectorStore's insert_batch(...) interface so importers,
downloaders, and collection mergers can target any backend cleanly.
"""
import json
import logging
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .dataset import VectorDataset

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """Common write interface implemented by all vector storage backends."""

    @abstractmethod
    def insert_batch(
        self,
        file_names: list[str],
        vectors: list[list[float]] | np.ndarray,
        extra_fields: list[dict[str, Any]] | None = None,
    ) -> None: ...

    def close(self) -> None:
        """Flush and finalize any open files/indexes."""


class FaissVectorStore(VectorStore):
    """Stores a collection as a flat FAISS index plus parallel JSONL metadata.

    Layout under <directory>/<collection_name>/:
        index.faiss   - faiss.IndexFlatIP (cosine) or IndexFlatL2 (l2)
        records.jsonl - one JSON object per row, aligned 1-to-1 with FAISS row index
        meta.json     - collection metadata (dimension, total_rows, metric, model, ...)

    Using a flat FAISS index keeps exact uncompressed float32 vectors, giving
    both fast FAISS kNN search and zero-duplication `(N, D)` NumPy matrix
    reconstruction via `index.reconstruct_n(0, N)` for linear algebra (PCA,
    kPCA, NCA, LMNN, HDBSCAN, BERTopic).
    """

    INDEX_FILENAME = "index.faiss"
    RECORDS_FILENAME = "records.jsonl"
    META_FILENAME = "meta.json"

    def __init__(
        self,
        directory: Path,
        collection_name: str,
        metric: str = "cosine",
        meta: dict[str, Any] | None = None,
    ):
        self.collection_name = collection_name
        self.directory = Path(directory) / collection_name
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metric = metric
        self.extra_meta = dict(meta or {})
        self.dimension: int | None = None
        self._index: faiss.Index | None = None
        self._records: list[dict[str, Any]] = []
        logger.info(
            "Using FAISS vector store at %s (collection '%s', metric=%s)",
            self.directory, collection_name, metric,
        )

    def _ensure_index(self, dimension: int) -> None:
        if self._index is not None:
            return
        self.dimension = dimension
        if self.metric == "cosine":
            self._index = faiss.IndexFlatIP(dimension)
        elif self.metric == "l2":
            self._index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unsupported FAISS metric: {self.metric!r} (expected 'cosine' or 'l2')")

    def insert_batch(
        self,
        file_names: list[str],
        vectors: list[list[float]] | np.ndarray,
        extra_fields: list[dict[str, Any]] | None = None,
    ) -> None:
        if len(file_names) == 0:
            return
        matrix = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
        if matrix.ndim != 2 or matrix.shape[0] != len(file_names):
            raise ValueError(
                f"Expected 2D vector matrix matching {len(file_names)} records, got shape {matrix.shape}"
            )
        self._ensure_index(matrix.shape[1])
        if self.dimension != matrix.shape[1]:
            raise ValueError(
                f"Vector dimension changed from {self.dimension} to {matrix.shape[1]}"
            )
        start_row = self._index.ntotal
        self._index.add(matrix)
        for offset, file_name in enumerate(file_names):
            record: dict[str, Any] = {"row_idx": start_row + offset, "file_name": file_name}
            if extra_fields:
                record.update(extra_fields[offset])
            self._records.append(record)
        logger.debug("Added %d vectors to FAISS store '%s' (%d total)", len(file_names), self.collection_name, self._index.ntotal)

    def close(self) -> None:
        if self._index is None:
            return
        faiss.write_index(self._index, str(self.directory / self.INDEX_FILENAME))
        with (self.directory / self.RECORDS_FILENAME).open("w", encoding="utf-8") as fh:
            for record in self._records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        meta = {
            "name": self.collection_name,
            "dimension": self.dimension,
            "total_rows": int(self._index.ntotal),
            "metric": self.metric,
            **self.extra_meta,
        }
        (self.directory / self.META_FILENAME).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "Saved FAISS collection '%s' (%d rows, dim=%d) to %s",
            self.collection_name, self._index.ntotal, self.dimension, self.directory,
        )

    @classmethod
    def load_dataset_dir(cls, dataset_dir: Path) -> tuple[VectorDataset, faiss.Index]:
        store_dir = Path(dataset_dir)
        meta = json.loads((store_dir / cls.META_FILENAME).read_text(encoding="utf-8"))
        index = faiss.read_index(str(store_dir / cls.INDEX_FILENAME))
        records = [
            json.loads(line)
            for line in (store_dir / cls.RECORDS_FILENAME).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        n = int(index.ntotal)
        if len(records) != n:
            raise ValueError(
                f"FAISS store '{store_dir}' mismatch: {len(records)} records vs {n} indexed vectors"
            )
        vectors = (
            np.asarray(index.reconstruct_n(0, n), dtype=np.float32)
            if n > 0
            else np.empty((0, int(meta["dimension"])), dtype=np.float32)
        )
        dataset = VectorDataset(
            name=meta.get("name", store_dir.name),
            records=records,
            vectors=vectors,
            meta=meta,
        )
        return dataset, index

    @classmethod
    def read_all(cls, directory: Path, collection_name: str) -> tuple[list[dict[str, Any]], np.ndarray]:
        dataset, _index = cls.load_dataset_dir(Path(directory) / collection_name)
        return dataset.records, dataset.vectors


class QdrantVectorStore(VectorStore):
    """Writes embedded vectors into a Qdrant collection using idempotent UUID5 IDs."""

    _DISTANCE_MAP = {
        "cosine": Distance.COSINE,
        "euclid": Distance.EUCLID,
        "l2": Distance.EUCLID,
        "dot": Distance.DOT,
    }

    def __init__(
        self,
        url: str,
        collection_name: str,
        vector_name: str | None = None,
        distance: str = "cosine",
    ):
        logger.info("Connecting to Qdrant at %s (collection '%s')", url, collection_name)
        self.client = QdrantClient(url=url, check_compatibility=False)
        self.collection_name = collection_name
        self.vector_name = vector_name
        self.distance = self._DISTANCE_MAP[distance.lower()]
        self.dimension: int | None = None

    def ensure_collection(self, dimension: int) -> None:
        if self.dimension is not None:
            return
        if self.client.collection_exists(self.collection_name):
            self.dimension = dimension
            return
        vector_params = VectorParams(size=dimension, distance=self.distance)
        vectors_config = {self.vector_name: vector_params} if self.vector_name else vector_params
        logger.info(
            "Qdrant collection '%s' doesn't exist yet, creating it (dim=%d, distance=%s)",
            self.collection_name, dimension, self.distance,
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=vectors_config,
        )
        self.dimension = dimension

    def insert_batch(
        self,
        file_names: list[str],
        vectors: list[list[float]] | np.ndarray,
        extra_fields: list[dict[str, Any]] | None = None,
    ) -> None:
        if len(file_names) == 0:
            return
        rows_vec = vectors.tolist() if isinstance(vectors, np.ndarray) else vectors
        self.ensure_collection(len(rows_vec[0]))
        for vector in rows_vec:
            if self.dimension != len(vector):
                raise ValueError(
                    f"Vector dimension changed from {self.dimension} to {len(vector)}"
                )
        points = []
        for idx, (file_name, vector) in enumerate(zip(file_names, rows_vec)):
            payload: dict[str, Any] = {"file_name": file_name}
            if extra_fields:
                payload.update(extra_fields[idx])
            stable_key = json.dumps(
                {"collection": self.collection_name, "file_name": file_name, "bible": payload.get("bible")},
                sort_keys=True,
            )
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))
            vec_payload = {self.vector_name: vector} if self.vector_name else vector
            points.append(PointStruct(id=point_id, vector=vec_payload, payload=payload))
        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.debug("Upserted %d vectors into Qdrant collection '%s'", len(points), self.collection_name)


class LocalDiskVectorStore(VectorStore):
    """Stores vectors as a single numpy memmap file per collection - a
    dependency-free, no-server alternative for local experiments.

    Layout, under <directory>/<collection_name>/:
        vectors.dat   - raw float32 memmap, shape (total_rows, dimension)
        records.jsonl - one JSON object per row, same order as vectors.dat
        meta.json     - {"dimension": ..., "total_rows": ...}
    """

    VECTORS_FILENAME = "vectors.dat"
    RECORDS_FILENAME = "records.jsonl"
    META_FILENAME = "meta.json"

    def __init__(self, directory: Path, collection_name: str, total_rows: int):
        self.collection_name = collection_name
        self.directory = Path(directory) / collection_name
        self.directory.mkdir(parents=True, exist_ok=True)
        self.total_rows = total_rows
        self.dimension: int | None = None
        self._memmap: np.memmap | None = None
        self._records_file = None
        self._next_row = 0
        logger.info(
            "Using local disk vector store at %s (collection '%s', %d rows expected)",
            self.directory, collection_name, total_rows,
        )

    def _ensure_open(self, dimension: int) -> None:
        if self._memmap is not None:
            return
        self.dimension = dimension
        self._memmap = np.memmap(
            self.directory / self.VECTORS_FILENAME,
            dtype=np.float32,
            mode="w+",
            shape=(self.total_rows, dimension),
        )
        (self.directory / self.META_FILENAME).write_text(
            json.dumps({"dimension": dimension, "total_rows": self.total_rows})
        )
        self._records_file = (self.directory / self.RECORDS_FILENAME).open("w", encoding="utf-8")
        logger.info(
            "Collection '%s' doesn't exist on disk yet, allocating it (dim=%d, %d rows) at %s",
            self.collection_name, dimension, self.total_rows, self.directory,
        )

    def insert_batch(
        self,
        file_names: list[str],
        vectors: list[list[float]] | np.ndarray,
        extra_fields: list[dict[str, Any]] | None = None,
    ) -> None:
        if len(file_names) == 0:
            return
        self._ensure_open(len(vectors[0]))
        n = len(vectors)
        if self._next_row + n > self.total_rows:
            raise ValueError(
                f"Inserting {n} more rows would exceed the {self.total_rows} rows this store "
                "was sized for - total_rows must match the actual number of vectors written"
            )
        for vector in vectors:
            if self.dimension != len(vector):
                raise ValueError(f"Vector dimension changed from {self.dimension} to {len(vector)}")

        self._memmap[self._next_row:self._next_row + n] = np.asarray(vectors, dtype=np.float32)
        self._memmap.flush()
        for i, file_name in enumerate(file_names):
            record: dict[str, Any] = {"file_name": file_name}
            if extra_fields:
                record.update(extra_fields[i])
            self._records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._records_file.flush()
        self._next_row += n
        logger.debug("Wrote a batch of %d vectors to local disk store '%s'", n, self.collection_name)

    def close(self) -> None:
        if self._records_file is not None:
            self._records_file.close()
        if self._memmap is not None:
            self._memmap.flush()

    @classmethod
    def read_all(cls, directory: Path, collection_name: str) -> tuple[list[dict[str, Any]], np.memmap]:
        store_dir = Path(directory) / collection_name
        meta = json.loads((store_dir / cls.META_FILENAME).read_text(encoding="utf-8"))
        vectors = np.memmap(
            store_dir / cls.VECTORS_FILENAME,
            dtype=np.float32,
            mode="r",
            shape=(meta["total_rows"], meta["dimension"]),
        )
        records = [
            json.loads(line)
            for line in (store_dir / cls.RECORDS_FILENAME).read_text(encoding="utf-8").splitlines()
            if line
        ]
        return records, vectors

    @classmethod
    def load_dataset(cls, directory: Path, collection_name: str) -> VectorDataset:
        records, vectors = cls.read_all(directory, collection_name)
        meta = json.loads((Path(directory) / collection_name / cls.META_FILENAME).read_text(encoding="utf-8"))
        return VectorDataset(
            name=collection_name,
            records=records,
            vectors=np.asarray(vectors, dtype=np.float32),
            meta=meta,
        )
