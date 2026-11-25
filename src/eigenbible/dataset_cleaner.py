"""Filters structural headings (paragraph_num == 1), short fragments (< min_chars),
and boilerplate sections (e.g. 'about') from local FAISS VectorDatasets in place,
with zero network or re-embedding calls.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .vector_collection_reader import FaissCollectionReader
from .vector_store import FaissVectorStore

logger = logging.getLogger(__name__)


def should_keep_record(
    record: dict[str, Any],
    *,
    exclude_headings: bool = True,
    exclude_sections: Iterable[str] = ("about",),
    min_chars: int = 100,
) -> bool:
    """Return True if a Biblioteca paragraph record should be kept for semantic analysis."""
    if exclude_headings and record.get("paragraph_num") == 1:
        return False

    sec = str(record.get("section", "")).strip().lower()
    url = str(record.get("url", "")).strip().lower()
    for excluded in exclude_sections:
        ex = excluded.strip().strip("/").lower()
        if not ex:
            continue
        if sec == ex or sec.startswith(f"{ex}/") or f"/{ex}/" in url or url.endswith(f"/{ex}"):
            return False

    if min_chars > 0 and "text" in record:
        text = str(record.get("text") or "").strip()
        if len(text) < min_chars:
            return False

    return True


def filter_faiss_dataset(
    ds_dir: Path,
    *,
    exclude_headings: bool = True,
    exclude_sections: tuple[str, ...] = ("about",),
    min_chars: int = 100,
) -> dict[str, int]:
    """Filter an existing local FAISS dataset directory in place (idempotent)."""
    reader = FaissCollectionReader(ds_dir)
    ds = reader.fetch_dataset()

    keep_indices = [
        i
        for i, rec in enumerate(ds.records)
        if should_keep_record(
            rec,
            exclude_headings=exclude_headings,
            exclude_sections=exclude_sections,
            min_chars=min_chars,
        )
    ]
    before = len(ds.records)
    after = len(keep_indices)
    dropped = before - after

    if dropped == 0:
        logger.info("Dataset '%s' already clean (%d rows, 0 dropped).", ds_dir.name, before)
        return {"before": before, "after": after, "dropped": 0}

    kept_vectors = np.ascontiguousarray(ds.vectors[keep_indices], dtype=np.float32)
    kept_records = [ds.records[i] for i in keep_indices]
    file_names = [str(r.get("file_name", f"row_{i}")) for i, r in enumerate(kept_records)]
    extra_fields = [
        {k: v for k, v in r.items() if k not in ("file_name", "row_idx")}
        for r in kept_records
    ]

    meta = dict(ds.meta)
    meta["count"] = after
    meta["filtered"] = {
        "exclude_headings": exclude_headings,
        "exclude_sections": list(exclude_sections),
        "min_chars": min_chars,
        "before_count": before,
        "dropped_count": dropped,
    }

    # Remove old index files so FaissVectorStore writes a fresh, compact IndexFlatIP
    for fname in ("index.faiss", "records.jsonl", "meta.json"):
        p = ds_dir / fname
        if p.exists():
            p.unlink()

    store = FaissVectorStore(
        directory=ds_dir.parent,
        collection_name=ds_dir.name,
        metric=str(meta.get("metric", "cosine")),
        meta=meta,
    )
    store.insert_batch(file_names=file_names, vectors=kept_vectors, extra_fields=extra_fields)
    store.close()

    logger.info(
        "Filtered '%s': %d -> %d rows (dropped %d rows; exclude_headings=%s, min_chars=%d).",
        ds_dir.name,
        before,
        after,
        dropped,
        exclude_headings,
        min_chars,
    )
    return {"before": before, "after": after, "dropped": dropped}
