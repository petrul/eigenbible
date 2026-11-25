"""Axis-labelling pipeline: reduce vectors (PCA / kPCA), pick representative
passages per component via a NeighbourhoodStrategy, resolve their text, ask an
LLM Summarizer to name each axis/pole, and checkpoint results.
"""
import logging
from typing import Any

from .chapter_text_resolver import TextResolver
from .label_results_store import ResultsStore
from .neighbourhood_strategy import NeighbourhoodStrategy
from .reducers import DimensionalityReducer, KPCAReducer
from .summarizer import Summarizer
from .vector_collection_reader import VectorDatasetReader

logger = logging.getLogger(__name__)


def _row_ref(row: dict[str, Any]) -> str:
    if "section" in row and "paragraph_num" in row:
        return f"{row['section']}/_{row['paragraph_num']}"
    if "bible" in row and "file_name" in row:
        return f"{row['bible']}/{row['file_name']}"
    return str(row.get("file_name") or row.get("url") or row.get("id") or row.get("row_idx", ""))


class AxisLabeler:
    """Backend-agnostic pipeline for discovering and naming latent axes of any text vector dataset."""

    def __init__(
        self,
        reader: VectorDatasetReader,
        text_resolver: TextResolver,
        strategy: NeighbourhoodStrategy,
        summarizer: Summarizer,
        results_store: ResultsStore,
        reducer: DimensionalityReducer | None = None,
        n_components: int = 10,
        kernel: str = "cosine",
    ):
        self.reader = reader
        self.text_resolver = text_resolver
        self.strategy = strategy
        self.summarizer = summarizer
        self.results_store = results_store
        self.reducer = reducer or KPCAReducer(n_components=n_components, kernel=kernel)

    def run(self) -> int:
        dataset = self.reader.fetch_dataset()
        if len(dataset) == 0:
            raise ValueError(f"Dataset '{self.reader.collection_name}' is empty")

        transformed, scores = self.reducer.fit_transform(dataset.vectors)
        n_components = transformed.shape[1]

        logger.info(
            "Labelling %d %s components of '%s' (%d passages)",
            n_components, self.reducer.name, dataset.name, len(dataset),
        )
        labelled = 0
        for component_index in range(n_components):
            neighbourhoods = self.strategy.neighbourhoods_for_component(
                component_index, transformed, dataset.records, dataset.vectors,
            )
            for sign, representative_idx, neighbour_rows in neighbourhoods:
                representative_row = dataset.records[representative_idx]
                # Keep all K neighbour_ids for auditability, but on very small corpora cap
                # how many passages enter the prompt so + and - poles stay distinct.
                label_cap = max(8, len(dataset) // 6)
                texts = [self.text_resolver.read_row(row) for row in neighbour_rows[:label_cap]]
                label = self.summarizer.label(texts)

                anchor_id = _row_ref(representative_row)
                neighbour_ids = [_row_ref(row) for row in neighbour_rows]
                score_val = float(scores[component_index])

                self.results_store.insert({
                    "dataset": dataset.name,
                    "method": self.reducer.name,
                    "component_index": component_index,
                    "eigenvalue": score_val,
                    "score_name": self.reducer.score_name,
                    "sign": sign,
                    "anchor_id": anchor_id,
                    "anchor_file_name": representative_row.get("file_name", anchor_id),
                    "neighbour_ids": neighbour_ids,
                    "label": label,
                    "projected_vector": transformed[representative_idx].tolist(),
                })
                labelled += 1
                logger.info(
                    "[component %d/%d, %s] %s=%.4f anchor=%s -> %s",
                    component_index + 1, n_components, sign,
                    self.reducer.score_name, score_val, anchor_id, label,
                )
        return labelled


# Backwards-compatible alias
KPCALabeler = AxisLabeler
