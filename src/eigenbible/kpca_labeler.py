import logging

import numpy as np
from sklearn.decomposition import KernelPCA

from .chapter_text_resolver import ChapterTextResolver
from .label_results_store import LabelResultsStore
from .neighbourhood_strategy import NeighbourhoodStrategy
from .summarizer import ChapterSummarizer
from .vector_collection_reader import VectorCollectionReader

logger = logging.getLogger(__name__)


class KPCALabeler:
    def __init__(
        self,
        reader: VectorCollectionReader,
        text_resolver: ChapterTextResolver,
        strategy: NeighbourhoodStrategy,
        summarizer: ChapterSummarizer,
        results_store: LabelResultsStore,
        n_components: int,
        kernel: str,
    ):
        self.reader = reader
        self.text_resolver = text_resolver
        self.strategy = strategy
        self.summarizer = summarizer
        self.results_store = results_store
        self.n_components = n_components
        self.kernel = kernel

    def run(self) -> int:
        rows = self.reader.fetch_all()
        if not rows:
            raise ValueError(f"Collection '{self.reader.collection_name}' is empty")
        vectors = np.array([row["vector"] for row in rows])

        transformed, eigenvalues = self._reduce(vectors)
        n_components = transformed.shape[1]

        logger.info("Labelling %d components by summarizing each one's characteristic chapters", n_components)
        labelled = 0
        for component_index in range(n_components):
            neighbourhoods = self.strategy.neighbourhoods_for_component(component_index, transformed, rows)
            for sign, representative_idx, neighbour_rows in neighbourhoods:
                representative_row = rows[representative_idx]
                texts = [
                    self.text_resolver.read(self._bible_of(row), row["file_name"])
                    for row in neighbour_rows
                ]
                label = self.summarizer.label(texts)

                self.results_store.insert({
                    "component_index": component_index,
                    "eigenvalue": float(eigenvalues[component_index]),
                    "sign": sign,
                    "anchor_bible": self._bible_of(representative_row),
                    "anchor_file_name": representative_row["file_name"],
                    "label": label,
                    "kpca_vector": transformed[representative_idx].tolist(),
                })
                labelled += 1
                logger.info(
                    "[component %d/%d, %s] eigenvalue=%.4f anchor=%s/%s -> %s",
                    component_index + 1, n_components, sign, eigenvalues[component_index],
                    self._bible_of(representative_row), representative_row["file_name"], label,
                )
        return labelled

    def _reduce(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fits a Kernel PCA on the given vectors, returning (transformed, eigenvalues).

        An O(n^2..n^3) eigendecomposition, so it can take a while for a large collection -
        hence the before/after logging.
        """
        logger.info(
            "Fitting kernel PCA (kernel=%s, n_components=%d) on %d vectors of dimension %d - "
            "this is an O(n^2..n^3) eigendecomposition, so it may take a while for a large "
            "collection",
            self.kernel, self.n_components, *vectors.shape,
        )
        model = KernelPCA(n_components=self.n_components, kernel=self.kernel)
        transformed = model.fit_transform(vectors)
        logger.info("Kernel PCA fit complete")
        return transformed, model.eigenvalues_

    def _bible_of(self, row: dict) -> str:
        return self.reader.bible_of(row)
