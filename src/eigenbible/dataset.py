"""In-memory vector dataset container shared across readers, reducers, and clusterers.

Decouples numerical linear algebra / clustering algorithms (PCA, kPCA, NCA,
LMNN, HDBSCAN, BERTopic, etc.) from how vectors are stored on disk or in Qdrant.
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class VectorDataset:
    """Dense (N, D) float32 matrix paired 1-to-1 with row metadata records."""

    name: str
    records: list[dict[str, Any]]
    vectors: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.vectors = np.asarray(self.vectors, dtype=np.float32)
        if len(self.records) != len(self.vectors):
            raise ValueError(
                f"Record count ({len(self.records)}) does not match vector count ({len(self.vectors)})"
            )

    def __len__(self) -> int:
        return len(self.records)

    @property
    def dimension(self) -> int:
        if self.vectors.ndim != 2 or len(self.vectors) == 0:
            return 0
        return int(self.vectors.shape[1])

    def texts(self, field_name: str = "text") -> list[str]:
        return [str(record.get(field_name, "")) for record in self.records]

    def normalized_vectors(self) -> np.ndarray:
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return self.vectors / norms

    def filter_by(self, **predicates: Any) -> "VectorDataset":
        indices = [
            idx
            for idx, record in enumerate(self.records)
            if all(record.get(key) == value for key, value in predicates.items())
        ]
        filtered_records = [self.records[idx] for idx in indices]
        filtered_vectors = (
            self.vectors[indices]
            if indices
            else np.empty((0, self.dimension), dtype=np.float32)
        )
        return VectorDataset(
            name=self.name,
            records=filtered_records,
            vectors=filtered_vectors,
            meta={**self.meta, "filtered_by": predicates},
        )
