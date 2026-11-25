"""Nearest-neighbour search backends for anchor-knn and ad-hoc vector lookup."""
from abc import ABC, abstractmethod
from typing import Any

import faiss
import numpy as np
from qdrant_client import QdrantClient


class NearestNeighbourSearch(ABC):
    """Common kNN lookup interface shared by FAISS and Qdrant backends."""

    @abstractmethod
    def search(self, query_vector: list[float] | np.ndarray, k: int) -> list[dict[str, Any]]: ...


class FaissNearestNeighbourSearch(NearestNeighbourSearch):
    """Exact kNN search against an in-memory/loaded FAISS index."""

    def __init__(self, index: faiss.Index, records: list[dict[str, Any]], metric: str = "cosine"):
        self.index = index
        self.records = records
        self.metric = metric

    def search(self, query_vector: list[float] | np.ndarray, k: int) -> list[dict[str, Any]]:
        query = np.ascontiguousarray(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))
        if self.metric == "cosine":
            norm = float(np.linalg.norm(query))
            if norm > 0:
                query = query / norm
        limit = min(k, int(self.index.ntotal))
        if limit <= 0:
            return []
        distances, indices = self.index.search(query, limit)
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            results.append({**self.records[int(idx)], "_score": float(score)})
        return results


class QdrantNearestNeighbourSearch(NearestNeighbourSearch):
    """kNN search against a remote Qdrant collection."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        vector_name: str | None = None,
    ):
        self.client = client
        self.collection_name = collection_name
        self.vector_name = vector_name

    def search(self, query_vector: list[float] | np.ndarray, k: int) -> list[dict[str, Any]]:
        query = query_vector.tolist() if isinstance(query_vector, np.ndarray) else list(query_vector)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query,
            using=self.vector_name,
            limit=k,
            with_payload=True,
        )
        return [
            {**(point.payload or {}), "id": str(point.id), "_score": float(point.score)}
            for point in response.points
        ]
