"""Semantic clustering pipelines (HDBSCAN and BERTopic) over any VectorDataset.

Both algorithms consume the same `VectorDatasetReader` + `TextResolver` +
`Summarizer` + `ResultsStore` abstractions as the PCA/kPCA axis-labelling
pipeline, reusing precomputed embeddings (e.g. bge-m3 from FAISS) with zero
re-embedding.
"""
import logging
from typing import Any

import hdbscan
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer

from .chapter_text_resolver import InlineTextResolver, TextResolver
from .dataset import VectorDataset
from .kpca_labeler import _row_ref
from .label_results_store import ResultsStore
from .summarizer import Summarizer
from .vector_collection_reader import VectorDatasetReader

logger = logging.getLogger(__name__)


def _top_centroid_members(
    vectors: np.ndarray,
    member_indices: np.ndarray,
    k: int,
) -> list[int]:
    """Return up to `k` dataset indices in `member_indices` closest to the cluster centroid (cosine)."""
    sub = vectors[member_indices]
    norms = np.linalg.norm(sub, axis=1, keepdims=True)
    sub_norm = sub / np.where(norms == 0, 1.0, norms)
    centroid = sub_norm.mean(axis=0)
    c_norm = float(np.linalg.norm(centroid))
    if c_norm > 0:
        centroid = centroid / c_norm
    scores = sub_norm @ centroid
    order = np.argsort(-scores)[: min(k, len(member_indices))]
    return [int(member_indices[i]) for i in order]


def _reduce_for_clustering(
    vectors: np.ndarray,
    method: str,
    n_components: int,
    n_neighbors: int,
    random_state: int = 42,
) -> np.ndarray:
    if method == "none" or vectors.shape[1] <= n_components:
        return vectors
    n_samples = vectors.shape[0]
    target_dim = max(2, min(n_components, n_samples - 2, vectors.shape[1]))
    if method == "pca":
        return PCA(n_components=target_dim, random_state=random_state).fit_transform(vectors)
    if method == "umap":
        from umap import UMAP

        eff_neighbors = max(2, min(n_neighbors, n_samples - 1))
        return UMAP(
            n_components=target_dim,
            n_neighbors=eff_neighbors,
            min_dist=0.0,
            metric="cosine",
            random_state=random_state,
            compatibility_layout=True,
        ).fit_transform(vectors)
    raise ValueError(f"Unsupported pre-reduction method: {method!r} (expected 'umap', 'pca', or 'none')")


class HDBSCANClusterer:
    """Density-based semantic clustering (optional UMAP/PCA pre-reduction + HDBSCAN)
    with optional Ollama theme labelling per cluster.
    """

    def __init__(
        self,
        reader: VectorDatasetReader,
        results_store: ResultsStore,
        text_resolver: TextResolver | None = None,
        summarizer: Summarizer | None = None,
        min_cluster_size: int = 5,
        min_samples: int | None = None,
        cluster_selection_method: str = "eom",
        reduce_method: str = "umap",
        reduce_dim: int = 5,
        n_neighbors: int = 10,
        n_representatives: int = 5,
        random_state: int = 42,
    ):
        self.reader = reader
        self.results_store = results_store
        self.text_resolver = text_resolver or InlineTextResolver()
        self.summarizer = summarizer
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_method = cluster_selection_method
        self.reduce_method = reduce_method
        self.reduce_dim = reduce_dim
        self.n_neighbors = n_neighbors
        self.n_representatives = n_representatives
        self.random_state = random_state

    def run(self) -> dict[str, Any]:
        dataset = self.reader.fetch_dataset()
        if len(dataset) == 0:
            raise ValueError(f"Dataset '{self.reader.collection_name}' is empty")

        reduced = _reduce_for_clustering(
            dataset.vectors,
            method=self.reduce_method,
            n_components=self.reduce_dim,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
        )
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
            cluster_selection_method=self.cluster_selection_method,
        )
        labels = clusterer.fit_predict(reduced)
        return self._emit_clusters(dataset, labels, algorithm="hdbscan")

    def _emit_clusters(
        self,
        dataset: VectorDataset,
        labels: np.ndarray,
        algorithm: str,
        keywords_by_cluster: dict[int, list[str]] | None = None,
    ) -> dict[str, Any]:
        unique_ids = sorted(set(int(x) for x in labels), key=lambda cid: (cid == -1, cid))
        n_clusters = sum(1 for cid in unique_ids if cid != -1)
        n_noise = int(np.sum(labels == -1))
        logger.info(
            "%s found %d clusters (+ %d noise points out of %d) in '%s'",
            algorithm, n_clusters, n_noise, len(dataset), dataset.name,
        )

        for cid in unique_ids:
            member_indices = np.where(labels == cid)[0]
            rep_indices = _top_centroid_members(dataset.vectors, member_indices, self.n_representatives)
            rep_texts = [self.text_resolver.read_row(dataset.records[i]) for i in rep_indices]
            label = ""
            if cid != -1 and self.summarizer is not None and rep_texts:
                label = self.summarizer.label(rep_texts)

            representatives = [
                {
                    "row_idx": i,
                    "ref": _row_ref(dataset.records[i]),
                    "excerpt": rep_texts[pos][:240],
                }
                for pos, i in enumerate(rep_indices)
            ]
            record = {
                "dataset": dataset.name,
                "algorithm": algorithm,
                "cluster_id": cid,
                "size": int(len(member_indices)),
                "label": label,
                "keywords": (keywords_by_cluster or {}).get(cid, []),
                "representatives": representatives,
                "member_indices": [int(i) for i in member_indices],
            }
            self.results_store.insert(record)
            logger.info(
                "[%s cluster %d] size=%d label=%s",
                algorithm, cid, len(member_indices), label or "(unlabelled)",
            )

        return {
            "dataset": dataset.name,
            "algorithm": algorithm,
            "total_documents": len(dataset),
            "n_clusters": n_clusters,
            "n_noise": n_noise,
        }


class BERTopicClusterer(HDBSCANClusterer):
    """BERTopic topic modeling over precomputed embeddings (zero re-embedding),
    combining c-TF-IDF topic keywords with optional Ollama theme labels.
    """

    def __init__(
        self,
        reader: VectorDatasetReader,
        results_store: ResultsStore,
        text_resolver: TextResolver | None = None,
        summarizer: Summarizer | None = None,
        min_cluster_size: int = 5,
        min_samples: int | None = None,
        cluster_selection_method: str = "eom",
        reduce_dim: int = 5,
        n_neighbors: int = 10,
        n_representatives: int = 5,
        top_n_words: int = 8,
        random_state: int = 42,
    ):
        super().__init__(
            reader=reader,
            results_store=results_store,
            text_resolver=text_resolver,
            summarizer=summarizer,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
            reduce_method="umap",
            reduce_dim=reduce_dim,
            n_neighbors=n_neighbors,
            n_representatives=n_representatives,
            random_state=random_state,
        )
        self.top_n_words = top_n_words

    def run(self) -> dict[str, Any]:
        from bertopic import BERTopic
        from umap import UMAP

        dataset = self.reader.fetch_dataset()
        if len(dataset) == 0:
            raise ValueError(f"Dataset '{self.reader.collection_name}' is empty")

        docs = [self.text_resolver.read_row(row) for row in dataset.records]
        n_samples = len(docs)
        eff_dim = max(2, min(self.reduce_dim, n_samples - 2, dataset.dimension))
        eff_neighbors = max(2, min(self.n_neighbors, n_samples - 1))

        umap_model = UMAP(
            n_components=eff_dim,
            n_neighbors=eff_neighbors,
            min_dist=0.0,
            metric="cosine",
            random_state=self.random_state,
            compatibility_layout=True,
        )
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
            cluster_selection_method=self.cluster_selection_method,
            prediction_data=True,
        )
        vectorizer_model = CountVectorizer(ngram_range=(1, 2), min_df=1)

        topic_model = BERTopic(
            embedding_model=None,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            top_n_words=self.top_n_words,
            calculate_probabilities=False,
            verbose=False,
        )
        topics, _probs = topic_model.fit_transform(docs, embeddings=dataset.vectors)
        labels = np.asarray(topics, dtype=int)

        keywords_by_cluster: dict[int, list[str]] = {}
        for cid in sorted(set(int(x) for x in labels)):
            topic_words = topic_model.get_topic(cid)
            if topic_words:
                keywords_by_cluster[cid] = [word for word, _score in topic_words[: self.top_n_words]]

        return self._emit_clusters(
            dataset,
            labels,
            algorithm="bertopic",
            keywords_by_cluster=keywords_by_cluster,
        )
