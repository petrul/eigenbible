"""Vector storage backends for embedded bible chapters: Milvus (remote vector

DB) or a local disk memory-mapped format, selected via --backend on the
embed_biblia CLI. Both implement VectorStore's insert_batch(file_names,
vectors), so BibliaEmbeddingImporter doesn't need to know which one it's
writing to.
"""
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from pymilvus import DataType, MilvusClient

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """Common interface both backends implement, so BibliaEmbeddingImporter

    (and CollectionMerger) can write to either without knowing which."""

    @abstractmethod
    def insert_batch(self, file_names: list[str], vectors: list[list[float]]) -> None: ...


def create_milvus_collection(client: MilvusClient, name: str, dimension: int, extra_varchar_fields=()) -> None:
    """Create a collection with a 'file_name' field, an optional set of extra VARCHAR fields
    (e.g. a 'bible' discriminator), and a COSINE index on the vector field - built as part of
    collection creation so it's searchable/loadable right away rather than needing a separate
    create_index step before it can be loaded."""
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
    schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=1024)
    for field_name in extra_varchar_fields:
        schema.add_field(field_name=field_name, datatype=DataType.VARCHAR, max_length=64)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")

    client.create_collection(collection_name=name, schema=schema, index_params=index_params)


class MilvusVectorStore(VectorStore):
    def __init__(self, uri: str, collection_name: str):
        logger.info("Connecting to Milvus at %s (collection '%s')", uri, collection_name)
        self.client = MilvusClient(uri=uri)
        self.collection_name = collection_name
        self.dimension: int | None = None

    def ensure_collection(self, dimension: int) -> None:
        if self.client.has_collection(self.collection_name):
            self.dimension = dimension
            return

        logger.info(
            "Collection '%s' doesn't exist yet, creating it (dim=%d) with a COSINE index so the "
            "chapter vectors have somewhere to land and are loadable/searchable right away",
            self.collection_name,
            dimension,
        )
        create_milvus_collection(self.client, self.collection_name, dimension)
        self.dimension = dimension
        logger.info("Collection '%s' created", self.collection_name)

    def insert_batch(self, file_names: list[str], vectors: list[list[float]]) -> None:
        self.ensure_collection(len(vectors[0]))
        for vector in vectors:
            if self.dimension != len(vector):
                raise ValueError(
                    f"Vector dimension changed from {self.dimension} to {len(vector)}"
                )
        logger.debug("Storing a batch of %d vectors in Milvus so they become searchable", len(vectors))
        self.client.insert(
            collection_name=self.collection_name,
            data=[
                {"vector": vector, "file_name": file_name}
                for file_name, vector in zip(file_names, vectors)
            ],
        )


class LocalDiskVectorStore(VectorStore):
    """Stores vectors as a single numpy memmap file per collection - a

    dependency-free, no-server alternative to Milvus for local experiments.

    Layout, under <directory>/<collection_name>/:
        vectors.dat   - raw float32 memmap, shape (total_rows, dimension)
        records.jsonl - one JSON object per row, same order as vectors.dat
                         (at minimum {"file_name": ...}; LocalDiskCollectionMerger
                         adds a "bible" field per row for the combined collection)
        meta.json      - {"dimension": ..., "total_rows": ...}, so the memmap's
                          shape can be reopened later without re-deriving it

    total_rows must be known upfront (the memmap is a fixed-size file - the
    caller already lists every source file before embedding starts, so this
    is cheap to provide), but dimension is discovered lazily from the first
    batch, mirroring MilvusVectorStore.ensure_collection's lazy creation.
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
        vectors: list[list[float]],
        extra_fields: list[dict] | None = None,
    ) -> None:
        """extra_fields, when given, is one dict of additional record fields per

        row (e.g. {"bible": "darby"}) - used by LocalDiskCollectionMerger to tag
        the combined collection; plain per-bible embedding runs don't pass it."""
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
            record = {"file_name": file_name}
            if extra_fields:
                record.update(extra_fields[i])
            self._records_file.write(json.dumps(record) + "\n")
        self._records_file.flush()
        self._next_row += n
        logger.debug("Wrote a batch of %d vectors to local disk store '%s'", n, self.collection_name)

    def close(self) -> None:
        if self._records_file is not None:
            self._records_file.close()
        if self._memmap is not None:
            self._memmap.flush()

    @classmethod
    def read_all(cls, directory: Path, collection_name: str) -> tuple[list[dict], np.memmap]:
        """Reopens a completed collection for reading: a memory-mapped array of

        vectors plus the parallel list of row records (file_name, and 'bible'
        for a combined collection). Used by LocalDiskCollectionMerger, and
        reusable for downstream analysis (e.g. a local-disk mode for
        kpca_label.py, not built yet)."""
        store_dir = Path(directory) / collection_name
        meta = json.loads((store_dir / cls.META_FILENAME).read_text())
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
