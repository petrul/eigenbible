"""Strategies for picking representative passages along each reduced axis."""
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .nearest_neighbour_search import NearestNeighbourSearch


class NeighbourhoodStrategy(ABC):
    """Picks the group(s) of passages that get summarized into a label for one axis.

    A group is `(sign, representative_row_index, neighbour_rows)`: the sign and
    representative row are stored alongside the label so downstream readers can
    tell which pole of the axis (or single anchor) a given label describes.
    """

    @abstractmethod
    def neighbourhoods_for_component(
        self,
        component_index: int,
        transformed: np.ndarray,
        rows: list[dict[str, Any]],
        vectors: np.ndarray | None = None,
    ) -> list[tuple[str, int, list[dict[str, Any]]]]:
        raise NotImplementedError


class AnchorKNNNeighbourhoodStrategy(NeighbourhoodStrategy):
    """Finds the single passage furthest out on this component's axis (in either
    direction), then queries the NearestNeighbourSearch backend (FAISS or Qdrant)
    for its k nearest neighbours in the original embedding space.
    """

    def __init__(self, search: NearestNeighbourSearch, k: int):
        self.search = search
        self.k = k

    def neighbourhoods_for_component(self, component_index, transformed, rows, vectors=None):
        projections = transformed[:, component_index]
        anchor_idx = int(np.argmax(np.abs(projections)))
        sign = "+" if projections[anchor_idx] >= 0 else "-"
        query_vec = vectors[anchor_idx] if vectors is not None else rows[anchor_idx]["vector"]
        neighbours = self.search.search(query_vec, self.k)
        return [(sign, anchor_idx, neighbours)]


class TopProjectionNeighbourhoodStrategy(NeighbourhoodStrategy):
    """For each component, takes the K passages scoring highest on that axis, and
    separately the K scoring lowest — labelling both poles of the axis directly
    from the reduced projections with no extra search call.
    """

    def __init__(self, k: int):
        self.k = k

    def neighbourhoods_for_component(self, component_index, transformed, rows, vectors=None):
        order = np.argsort(transformed[:, component_index])
        limit = min(self.k, len(order))
        top_negative = order[:limit]
        top_positive = order[::-1][:limit]
        return [
            ("+", int(top_positive[0]), [rows[int(i)] for i in top_positive]),
            ("-", int(top_negative[0]), [rows[int(i)] for i in top_negative]),
        ]
