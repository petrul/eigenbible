# The Eigenbible

kPCA on the Bible and such.

Embeds every chapter of two parallel bible translations, fits a Kernel PCA
over the embeddings, and asks an LLM to name the shared theme of each
resulting axis - a "what are the eigen-directions of meaning in this text"
experiment.

## Pipeline

```
tb-export-biblia-ortdx.py  ->  embed_biblia.py  ->  kpca_label.py
   (scrape chapters)          (chapter -> vector)   (vectors -> labelled axes)
```

1. **`tb-export-biblia-ortdx.py`** scrapes a bible translation off
   textbase.scriptorium.ro into per-chapter markdown files (already run;
   see `biblia-ortdx-capitole/` and `biblia-darby-fr-capitole/`).
2. **`embed_biblia.py`** embeds every chapter with an Ollama embedding model
   and stores the vectors - either in Milvus, or in a local disk
   memory-mapped format that needs no server (`--backend local`).
3. **`kpca_label.py`** pulls the vectors back out of a Milvus collection,
   fits a `KernelPCA`, and for each component asks an Ollama chat model to
   name the theme shared by the chapters that characterize that axis.

## Requirements

- Python `>=3.12` (`.python-version`), managed with [uv](https://docs.astral.sh/uv/).
- An [Ollama](https://ollama.com) server with an embedding-capable model
  (e.g. `qwen3-embedding:4b`, or the much smaller/faster `nomic-embed-text`
  for quick experiments) and a chat model for labelling (e.g. `qwen3.5:4b`).
- [Milvus](https://milvus.io) - only if you want `embed_biblia.py`'s default
  `--backend milvus`, or to run `kpca_label.py` at all (it currently only
  reads from Milvus, not the local disk backend).

```bash
uv sync
cp .env.example .env   # if starting fresh; fill in your OLLAMA_URL/MILVUS_URI
```

`OLLAMA_URL`/`MILVUS_URI` are never hardcoded in source - both scripts load
them from `.env` via `eigenbible.settings` (pydantic-settings).

## Usage

```bash
# embed both bibles into Milvus (default), then rebuild the combined collection
uv run python src/embed_biblia.py --combine
# or: rake embed

# embed into a local memory-mapped store instead - no Milvus needed
uv run python src/embed_biblia.py --backend local --local-dir ./vector_data

# just one bible, or skip straight to --combine
uv run python src/embed_biblia.py --bible darby
uv run python src/embed_biblia.py --bible none --combine

# fit a kernel PCA over a collection and label 50 components (default)
uv run python src/kpca_label.py --collection biblia_all_subcapitole

# fewer components, a different kernel/strategy, for a quick look
uv run python src/kpca_label.py -c biblia_ortodoxa_subcapitole -n 5 --strategy anchor-knn
```

Run either script with `--help` for the full option list (output format,
page/component counts, model/host overrides, etc).

### Labelling strategies (`kpca_label.py --strategy`)

Each kPCA component is a numeric axis; before it can be labelled, some
chapters have to be picked to represent it and read back to an LLM. The two
strategies differ in *which* chapters get picked, and in what they cost:

- **`top-projection`** (default). For each component, ranks every chapter by
  its score on that axis and takes the K highest-scoring and, separately,
  the K lowest-scoring. A kPCA axis is a contrast between two poles, not a
  single point - "K chapters that lean hardest one way" and "K that lean
  hardest the other way" - so this produces **two labels per component**,
  one per pole (`sign: "+"` / `"-"` in the results). Purely arithmetic on
  the projections `KernelPCA.fit_transform` already computed - no extra
  round-trip to Milvus, and the chapters are guaranteed to be the ones that
  actually define that specific axis.

- **`anchor-knn`**. Finds the single chapter that sits furthest out on the
  axis (in either direction) and asks Milvus for that chapter's K nearest
  neighbours - but in the *original* embedding space, not kPCA space. One
  Milvus search per component, **one label per component**. Cheaper on
  Ollama calls (half as many labelling requests as `top-projection` for the
  same K), but the neighbours are only "generally close to the anchor" in
  the full embedding, not necessarily the chapters that score highest on
  *this specific* axis - a weaker signal for the less-dominant/later
  components, where an axis's direction can diverge more from raw embedding
  similarity.

In short: `top-projection` is the more faithful/expensive default; try
`anchor-knn` for a quicker, rougher pass, or if you specifically want the
labelling to reflect embedding-space similarity rather than the kPCA
projection itself.

## Project layout

- **`embed_biblia.py`** / **`kpca_label.py`** - thin CLI entry points; the
  actual logic lives under `src/eigenbible/`:
  - `settings.py` - `.env`-backed `OLLAMA_URL`/`MILVUS_URI`
  - `bibles.py` - which bibles exist, their source dirs/collection names
  - `markdown_reader.py`, `embedder.py` - chapter files -> Ollama embeddings
  - `vector_store.py` - `MilvusVectorStore` / `LocalDiskVectorStore`
  - `collection_merger.py` - rebuilds the combined collection from the
    per-bible ones without re-embedding
  - `importer.py` - drives embed_biblia.py's per-chapter embed-and-store loop
  - `vector_collection_reader.py`, `chapter_text_resolver.py`,
    `nearest_neighbour_search.py` - kpca_label.py's Milvus/disk plumbing
  - `neighbourhood_strategy.py` - which chapters characterize a component
    (`top-projection` / `anchor-knn`)
  - `summarizer.py` - asks Ollama to name a component's shared theme
  - `label_results_store.py` - where labelled components get written
  - `kpca_labeler.py` - `KPCALabeler`, the kpca_label.py pipeline itself
- **`tb-export-biblia-ortdx.py`** - the scraper that produced
  `biblia-ortdx-capitole/`/`biblia-darby-fr-capitole/`
- **`src/play/`** - exploratory scratch scripts, not part of the pipeline
- **`biblia-ortdx-capitole/`**, **`biblia-darby-fr-capitole/`** - the two
  bible translations, one markdown file per chapter

## Tests

```bash
rake test
# or: uv run python -m unittest discover -s tests -v
```

Fast, fixture-based unit tests for the local disk vector store/merger, plus
a real (non-mocked) Ollama embedding call using a small model
(`nomic-embed-text`) so it stays quick to run routinely.
