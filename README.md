# The Eigenbible

Unsupervised spectral decomposition (**PCA**, **Kernel PCA**) and density/topic clustering (**HDBSCAN**, **BERTopic**) over literary vector databases — both parallel Bible translations and historical works from [`biblioteca.scriptorium.ro`](https://biblioteca.scriptorium.ro).

All datasets are stored locally as self-contained **FAISS** indexes (`index.faiss` + `records.jsonl` + `meta.json` + optional plain-text chapters) using 1024-D `bge-m3:567m-fp16` embeddings, making it effortless to load exact `(N, 1024)` `float32` matrices directly into NumPy / scikit-learn for linear algebra and metric-learning experiments (PCA, kPCA, NCA, LMNN, SVD, UMAP, HDBSCAN, BERTopic).

---

## Architecture

```text
  Qdrant (zmeu:6333) + biblioteca.scriptorium.ro  ──▶  src/download_biblioteca.py  ──┐
                                                                                     ▼
  Local Markdown chapters (biblia-*-capitole/)    ──▶  src/embed_biblia.py        ──▶  data/<work>/ (FAISS + JSONL + meta)
                                                                                          │
                                                             ┌────────────────────────────┴────────────────────────────┐
                                                             ▼                                                         ▼
                                                   src/kpca_label.py                                         src/cluster_topics.py
                                               (PCA / Kernel PCA + LLM axes)                                (HDBSCAN + BERTopic + LLM)
                                                             │                                                         │
                                                             └────────────────────────────┬────────────────────────────┘
                                                                                          ▼
                                                                             data/<work>/results/*.jsonl & *.md
```

### Key Design Choices
- **FAISS flat indexes (`IndexFlatIP` / `IndexFlatL2`)**: Because `bge-m3` vectors are unit-normalized ($\|v\|_2 = 1$), `IndexFlatIP` gives exact cosine kNN search *and* lossless `(N, D)` NumPy matrix reconstruction via `index.reconstruct_n(0, N)` without duplicating vector storage on disk.
- **No Milvus**: All storage, reading, merging, and search paths use **FAISS** (local) or **Qdrant** (remote on `zmeu.local:6333`).
- **SHA-256 verified downloads**: Qdrant's `biblioteca_paragraphs_bge_m3` collection stores `{opus_path, url, sha256}` + the 1024-D `embedding` vector; `download_biblioteca.py` fetches each paragraph's `.txt` from `biblioteca.scriptorium.ro`, verifies its SHA-256 digest against Qdrant bit-for-bit, orders paragraphs by `(section, paragraph_num)`, and re-aligns vectors identically.
- **Reusable abstractions**: `VectorDataset`, `VectorStore`, `VectorDatasetReader`, `NearestNeighbourSearch`, `TextResolver`, `DimensionalityReducer`, `NeighbourhoodStrategy`, `Summarizer`, and `ResultsStore` decouple storage from math and LLM labelling.

---

## Requirements & Setup

- **Python** `>=3.12` (managed with [uv](https://docs.astral.sh/uv/)).
- **Qdrant** at `http://zmeu.local:6333` (hardwired in `eigenbible.settings.QDRANT_URL`; collection `biblioteca_paragraphs_bge_m3`).
- **Ollama** at `http://zmeu.local:11434` (configured via `.env` `OLLAMA_URL`, defaults to `http://zmeu.local:11434`):
  - Default embedding model: `bge-m3:567m-fp16` (1024-D, matching Qdrant's `biblioteca_paragraphs_bge_m3`).
  - Default chat/labelling model: `qwen3.5:4b` (invoked with `think=False` so it outputs crisp labels directly).

```bash
uv sync
cp .env.example .env
```

---

## Usage

### 1. Download Biblioteca works from Qdrant into local FAISS datasets

Built-in presets in `src/eigenbible/qdrant_downloader.py`:
- `gesta_hungarorum` → `trad/gesta_hungarorum_la` (122 Latin paragraphs, 60 sections)
- `ureche_letopisetul` → `ureche/letopisetul_tarii_moldovei` (715 Romanian paragraphs, 244 sections)
- `miron_letopisetul` → `costin/letopisetul_tarii_moldovei` (623 Romanian paragraphs, 26 sections)
- `creanga_amintiri` → `creanga/amintiri_din_copilarie` (401 Romanian paragraphs, 5 sections)
- `london_white_fang` → `london/white_fang` (1,080 English paragraphs)
- `hugo_notre_dame` → `hugo/notre_dame_de_paris` (2,908 French paragraphs)
- `hugo_notre_dame_en` → `hugo/notre_dame_de_paris_2` (3,956 English paragraphs)

```bash
# Download all three preset works into data/<work>/
uv run python src/download_biblioteca.py
# or: rake download

# Download a single preset
uv run python src/download_biblioteca.py --work gesta_hungarorum

# Download any arbitrary opus_path from Qdrant
uv run python src/download_biblioteca.py --opus-path neculce/o_sama_de_cuvinte --name neculce
```

Each exported dataset under `data/<work>/` contains:
- `index.faiss` — flat FAISS index (`IndexFlatIP`, cosine)
- `records.jsonl` — one JSON record per row (`row_idx`, `id`, `opus_path`, `section`, `paragraph_num`, `url`, `sha256`, `sha256_verified`, `text`)
- `meta.json` — provenance & schema (`name`, `dimension`, `total_rows`, `metric`, `embedding_model`, `opus_path`, `sha256_mismatches`)
- `texts/` — human-readable per-section `.txt` files

### 2. Embed local Markdown corpora (Romanian Orthodox & French Darby Bibles)

```bash
# Embed both Bibles into local FAISS under data/ (default model: bge-m3:567m-fp16)
uv run python src/embed_biblia.py --backend faiss --data-dir data
# or: rake embed

# Also build the combined bilingual collection (no re-embedding)
uv run python src/embed_biblia.py --backend faiss --data-dir data --combine

# Embed into Qdrant or raw numpy memmap instead
uv run python src/embed_biblia.py --backend qdrant --combine
uv run python src/embed_biblia.py --backend local --data-dir vector_data
```

### 3. Discover & label latent axes (PCA & Kernel PCA)

```bash
# Linear PCA (top 8 components, both + and - poles labelled by Ollama)
uv run python src/kpca_label.py --dataset data/gesta_hungarorum --method pca -n 8 -k 6

# Cosine Kernel PCA
uv run python src/kpca_label.py --dataset data/gesta_hungarorum --method kpca --kernel cosine -n 8 -k 6

# Works identically on the chronicles and both Bibles
uv run python src/kpca_label.py --dataset data/ureche_letopisetul --method pca -n 10
uv run python src/kpca_label.py --dataset data/ureche_letopisetul --method kpca --kernel cosine -n 10
uv run python src/kpca_label.py --dataset data/miron_letopisetul --method pca -n 10
uv run python src/kpca_label.py --dataset data/biblia_ortodoxa_subcapitole --method kpca --kernel cosine -n 10
uv run python src/kpca_label.py --dataset data/biblia_darby_fr_subcapitole --method pca -n 10

# Alternative neighbourhood & summarization strategies
uv run python src/kpca_label.py --dataset data/gesta_hungarorum --method pca --strategy anchor-knn
uv run python src/kpca_label.py --dataset data/gesta_hungarorum --method pca --summarizer shuffled-lines
```

#### Axis-labelling strategies
- **`top-projection`** *(default)*: Ranks every passage along component $j$, takes the top $K$ positive and top $K$ negative projections, and asks the LLM to name each pole (`+` / `-`) separately. Purely algebraic on the projection matrix; reveals the thematic *contrast* defining each eigen-axis.
- **`anchor-knn`**: Picks the single most extreme passage on component $j$ and queries FAISS/Qdrant for its $K$ nearest neighbours in the original 1024-D embedding space. Produces one label per component.

### 4. Cluster topics (HDBSCAN & BERTopic)

Both algorithms reuse the precomputed `bge-m3` embeddings stored in FAISS (zero re-embedding, zero HuggingFace model downloads):

```bash
# Run both HDBSCAN and BERTopic (default: UMAP 5D cosine pre-reduction + HDBSCAN eom)
uv run python src/cluster_topics.py --dataset data/ureche_letopisetul --algorithm both --min-cluster-size 12
uv run python src/cluster_topics.py --dataset data/miron_letopisetul --algorithm both --min-cluster-size 12
uv run python src/cluster_topics.py --dataset data/biblia_ortodoxa_subcapitole --algorithm both --min-cluster-size 15
uv run python src/cluster_topics.py --dataset data/biblia_darby_fr_subcapitole --algorithm both --min-cluster-size 15

# Fine-grained leaf selection (recommended for small corpora or corpora with outlier colophons like Gesta)
uv run python src/cluster_topics.py --dataset data/gesta_hungarorum --algorithm both \
  --min-cluster-size 5 --cluster-selection-method leaf

# Fast numeric-only run without Ollama labelling
uv run python src/cluster_topics.py --dataset data/gesta_hungarorum --algorithm hdbscan --no-llm
```

### 5. Interactive NumPy / Linear Algebra Playground

Load any local FAISS dataset directly into NumPy for `PCA`, `KernelPCA`, `NCA`, `LMNN`, `SVD`, Procrustes alignment, or custom experiments:

```python
from eigenbible.vector_collection_reader import FaissCollectionReader

reader = FaissCollectionReader("data/gesta_hungarorum")
ds = reader.fetch_dataset()

X = ds.vectors                  # np.ndarray, shape (122, 1024), float32
texts = ds.texts()              # list[str]
index = reader.index            # faiss.IndexFlatIP for instant kNN

# Example: drop Wikisource 'about' colophon before SVD / NCA / LMNN
narrative = VectorDataset_or_filtered = ds.filter_by(sha256_verified=True)
```

---

## Project Layout

- **`src/eigenbible/`** — reusable library:
  - `settings.py` — `QDRANT_URL`, `QDRANT_COLLECTION`, `QDRANT_VECTOR_NAME`, `BIBLIOTECA_BASE_URL`, `OLLAMA_EMBED_MODEL`, `OLLAMA_URL`
  - `dataset.py` — `VectorDataset` container (`records`, `vectors`, `meta`, `filter_by`, `normalized_vectors`)
  - `vector_store.py` — `VectorStore` ABC (`FaissVectorStore`, `QdrantVectorStore`, `LocalDiskVectorStore`)
  - `vector_collection_reader.py` — `VectorDatasetReader` ABC (`FaissCollectionReader`, `QdrantCollectionReader`, `LocalDiskCollectionReader`)
  - `nearest_neighbour_search.py` — `NearestNeighbourSearch` ABC (`FaissNearestNeighbourSearch`, `QdrantNearestNeighbourSearch`)
  - `collection_merger.py` — `CollectionMerger` ABC (`FaissCollectionMerger`, `QdrantCollectionMerger`, `LocalDiskCollectionMerger`)
  - `chapter_text_resolver.py` — `TextResolver` ABC (`InlineTextResolver`, `ChapterTextResolver`)
  - `qdrant_downloader.py` — Qdrant + `biblioteca.scriptorium.ro` SHA-256-verified dataset exporter
  - `reducers.py` — `DimensionalityReducer` ABC (`PCAReducer`, `KPCAReducer`, `build_reducer`)
  - `neighbourhood_strategy.py` — `TopProjectionNeighbourhoodStrategy`, `AnchorKNNNeighbourhoodStrategy`
  - `summarizer.py` — `ChapterSummarizer` (alias `ExcerptSummarizer`), `ShuffledLinesSummarizer`
  - `label_results_store.py` — `ResultsStore` ABC (`JsonlResultsStore` + Markdown report renderers)
  - `kpca_labeler.py` — `AxisLabeler` (alias `KPCALabeler`)
  - `clusterer.py` — `HDBSCANClusterer`, `BERTopicClusterer`
  - `bibles.py`, `markdown_reader.py`, `embedder.py`, `importer.py` — Bible corpus registry and Ollama batch importer
- **Thin CLIs**: `src/download_biblioteca.py`, `src/embed_biblia.py`, `src/kpca_label.py`, `src/cluster_topics.py`, `src/tb-export-biblia-ortdx.py`
- **Tests**: `tests/test_eigenbible.py`, `tests/test_storage_and_qdrant.py`, `tests/test_downloader_and_analysis.py`

---

## Tests

```bash
rake test
# or: uv run python -m unittest discover -s tests -v
```

45 tests total (43 fast offline unit tests covering FAISS/Qdrant stores, readers, mergers, kNN search, URL parsing, SHA-256 downloader/alignment, PCA/kPCA reducers, `AxisLabeler`, HDBSCAN, BERTopic, and summarizers; plus 2 live-Ollama integration tests that auto-skip when Ollama is unreachable).
