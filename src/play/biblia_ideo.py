"""Exploratory scratch script: KernelPCA on synthetic concentric circles stored in FAISS."""
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA

from eigenbible.vector_store import FaissVectorStore

OUTPUT_DIR = Path("vector_data")
COLLECTION = "test_circles_kpca"


def save_vectors(vectors) -> None:
    store = FaissVectorStore(OUTPUT_DIR, COLLECTION, metric="l2")
    store.insert_batch([f"pt_{i}" for i in range(len(vectors))], vectors)
    store.close()


if __name__ == "__main__":
    X, _y = make_circles(n_samples=1000, factor=0.3, noise=0.05)

    kpca = KernelPCA(n_components=2, kernel="rbf", gamma=10, fit_inverse_transform=True)
    X_kpca = kpca.fit_transform(X)
    print(X_kpca)

    X_back = kpca.inverse_transform(X_kpca)
    print(X_back)

    save_vectors(X_kpca)

    plt.scatter(X_kpca[:, 0], X_kpca[:, 1], alpha=0.6, edgecolors="none")
    plt.show()
