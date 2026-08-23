from abc import ABC, abstractmethod

import numpy as np

from .nearest_neighbour_search import MilvusNearestNeighbourSearch


class NeighbourhoodStrategy(ABC):
    """Picks the group(s) of chapters that get summarized into a label for one kPCA component.
    A group is (sign, representative_row_index, neighbour_rows): the sign and representative
    row are what get stored alongside the label, so downstream readers can tell which pole of
    the axis (or single anchor) a given label describes."""

    @abstractmethod
    def neighbourhoods_for_component(
        self, component_index: int, transformed: np.ndarray, rows: list[dict],
    ) -> list[tuple[str, int, list[dict]]]:
        raise NotImplementedError


class AnchorKNNNeighbourhoodStrategy(NeighbourhoodStrategy):
    """Finds the single chapter that sits furthest out on this component's axis (in either
    direction), then asks Milvus for its nearest neighbours in the *original* embedding space.
    One Milvus round-trip per component, one label per component. The neighbours are close to
    the anchor generally, but aren't necessarily the chapters that score highest on this
    specific axis - a weaker signal for the less dominant components."""

    def __init__(self, search: MilvusNearestNeighbourSearch, k: int):
        self.search = search
        self.k = k

    def neighbourhoods_for_component(self, component_index, transformed, rows):
        projections = transformed[:, component_index]
        anchor_idx = int(np.argmax(np.abs(projections)))
        sign = "+" if projections[anchor_idx] >= 0 else "-"
        neighbours = self.search.search(rows[anchor_idx]["vector"], self.k)
        return [(sign, anchor_idx, neighbours)]


class TopProjectionNeighbourhoodStrategy(NeighbourhoodStrategy):
    """For each component, takes the K chapters that score highest on that axis, and
    separately the K that score lowest - a kPCA component is a contrast between two poles,
    not a single point, so this labels both ends. Ranking comes straight from the already
    computed kPCA projections, no Milvus search involved."""

    def __init__(self, k: int):
        self.k = k

    def neighbourhoods_for_component(self, component_index, transformed, rows):
        order = np.argsort(transformed[:, component_index])
        top_negative = order[:self.k]
        top_positive = order[::-1][:self.k]
        return [
            ("+", int(top_positive[0]), [rows[i] for i in top_positive]),
            ("-", int(top_negative[0]), [rows[i] for i in top_negative]),
        ]
