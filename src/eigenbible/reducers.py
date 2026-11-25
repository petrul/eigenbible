"""Pluggable dimensionality reduction algorithms for dense vector datasets.

Separates numerical linear algebra (PCA, KernelPCA, and future NCA / LMNN / SVD)
from downstream neighbourhood selection and LLM labelling.
"""
import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.decomposition import KernelPCA, PCA

logger = logging.getLogger(__name__)


class DimensionalityReducer(ABC):
    """Reduces an (N, D) vector matrix to (N, K) projections + per-component scores."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def score_name(self) -> str: ...

    @abstractmethod
    def fit_transform(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


class PCAReducer(DimensionalityReducer):
    """Standard linear PCA returning projections and explained variance ratios."""

    def __init__(self, n_components: int, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state

    @property
    def name(self) -> str:
        return "pca"

    @property
    def score_name(self) -> str:
        return "explained_variance_ratio"

    def fit_transform(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        max_components = max(1, min(vectors.shape[0] - 1, vectors.shape[1]))
        k = min(self.n_components, max_components)
        logger.info("Fitting PCA (n_components=%d) on %d vectors of dim %d", k, *vectors.shape)
        model = PCA(n_components=k, random_state=self.random_state)
        transformed = model.fit_transform(vectors)
        logger.info(
            "PCA fit complete (cumulative explained variance: %.1f%%)",
            float(np.sum(model.explained_variance_ratio_)) * 100.0,
        )
        return transformed, np.asarray(model.explained_variance_ratio_, dtype=np.float64)


class KPCAReducer(DimensionalityReducer):
    """Kernel PCA returning nonlinear projections and kernel eigenvalues."""

    def __init__(self, n_components: int, kernel: str = "cosine", gamma: float | None = None):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma

    @property
    def name(self) -> str:
        return f"kpca_{self.kernel}"

    @property
    def score_name(self) -> str:
        return "eigenvalue"

    def fit_transform(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        max_components = max(1, vectors.shape[0] - 1)
        k = min(self.n_components, max_components)
        logger.info(
            "Fitting Kernel PCA (kernel=%s, n_components=%d) on %d vectors of dim %d",
            self.kernel, k, *vectors.shape,
        )
        model = KernelPCA(n_components=k, kernel=self.kernel, gamma=self.gamma)
        transformed = model.fit_transform(vectors)
        logger.info("Kernel PCA fit complete")
        return transformed, np.asarray(model.eigenvalues_, dtype=np.float64)


def build_reducer(
    method: str,
    n_components: int,
    kernel: str = "cosine",
    gamma: float | None = None,
) -> DimensionalityReducer:
    if method == "pca":
        return PCAReducer(n_components=n_components)
    if method == "kpca":
        return KPCAReducer(n_components=n_components, kernel=kernel, gamma=gamma)
    raise ValueError(f"Unknown reducer method: {method!r} (expected 'pca' or 'kpca')")
