"""Downloads book paragraph vectors from Qdrant and their verified UTF-8 text
from biblioteca.scriptorium.ro into a self-contained local FAISS dataset.
"""
import hashlib
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests

from .dataset_cleaner import should_keep_record
from .settings import BIBLIOTECA_BASE_URL, OLLAMA_EMBED_MODEL, QDRANT_COLLECTION, QDRANT_URL, QDRANT_VECTOR_NAME
from .vector_collection_reader import QdrantCollectionReader
from .vector_store import FaissVectorStore

logger = logging.getLogger(__name__)

# Named presets for works on biblioteca.scriptorium.ro
BIBLIOTECA_WORKS: dict[str, str] = {
    "gesta_hungarorum": "trad/gesta_hungarorum_la",
    "ureche_letopisetul": "ureche/letopisetul_tarii_moldovei",
    "miron_letopisetul": "costin/letopisetul_tarii_moldovei",
    "creanga_amintiri": "creanga/amintiri_din_copilarie",
    "london_white_fang": "london/white_fang",
    "hugo_notre_dame": "hugo/notre_dame_de_paris",
    "hugo_notre_dame_en": "hugo/notre_dame_de_paris_2",
}

_PARA_RE = re.compile(r"^_(\d+)$")


def parse_biblioteca_url(url: str, opus_path: str) -> tuple[str, str, int]:
    """Extract `(url_path, section, paragraph_num)` from a Qdrant payload URL.

    Example:
        url       = 'http://server:8080/trad/gesta_hungarorum_la/de_duce_bycoriensy/_2'
        opus_path = 'trad/gesta_hungarorum_la'
        -> ('/trad/gesta_hungarorum_la/de_duce_bycoriensy/_2', 'de_duce_bycoriensy', 2)
    """
    path = urlparse(url).path.rstrip("/")
    prefix = "/" + opus_path.strip("/") + "/"
    rel = path[len(prefix):] if path.startswith(prefix) else path.lstrip("/")
    parts = rel.split("/")
    para_num = 0
    if parts:
        match = _PARA_RE.match(parts[-1])
        if match:
            para_num = int(match.group(1))
            section = "/".join(parts[:-1]) or "root"
        else:
            section = "/".join(parts)
    else:
        section = "root"
    return path, section, para_num


class BibliotecaTextFetcher:
    """Fetches exact paragraph `.txt` files from biblioteca.scriptorium.ro and verifies SHA-256."""

    USER_AGENT = "Mozilla/5.0 (compatible; eigenbible/1.0)"

    def __init__(
        self,
        base_url: str = BIBLIOTECA_BASE_URL,
        workers: int = 8,
        retries: int = 3,
        timeout: int = 20,
    ):
        self.base_url = base_url.rstrip("/")
        self.workers = workers
        self.retries = retries
        self.timeout = timeout

    def fetch_one(
        self,
        url_path: str,
        expected_sha256: str | None = None,
        allow_404: bool = False,
    ) -> tuple[str | None, bool]:
        target_url = f"{self.base_url}{url_path}.txt"
        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = requests.get(
                    target_url,
                    headers={"User-Agent": self.USER_AGENT},
                    timeout=self.timeout,
                )
                if allow_404 and resp.status_code == 404:
                    logger.warning("Skipping missing paragraph (404): %s", target_url)
                    return None, False
                resp.raise_for_status()
                raw = resp.content
                actual_sha = hashlib.sha256(raw).hexdigest()
                sha_ok = expected_sha256 is None or actual_sha == expected_sha256
                return raw.decode("utf-8"), sha_ok
            except Exception as exc:
                last_exc = exc
                time.sleep(0.4 * attempt)
        raise RuntimeError(f"Failed to fetch {target_url}: {last_exc}")

    def fetch_many(
        self,
        items: list[tuple[str, str | None]],
        allow_404: bool = False,
    ) -> list[tuple[str | None, bool]]:
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(lambda pair: self.fetch_one(pair[0], pair[1], allow_404=allow_404), items))


class BibliotecaDatasetExporter:
    """Exports one Biblioteca work from Qdrant + biblioteca.scriptorium.ro into a local FAISS directory."""

    def __init__(
        self,
        out_root: Path,
        qdrant_url: str = QDRANT_URL,
        collection_name: str = QDRANT_COLLECTION,
        vector_name: str = QDRANT_VECTOR_NAME,
        fetcher: BibliotecaTextFetcher | None = None,
    ):
        self.out_root = Path(out_root)
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.vector_name = vector_name
        self.fetcher = fetcher or BibliotecaTextFetcher()

    def export_work(
        self,
        work_name: str,
        opus_path: str,
        *,
        exclude_headings: bool = True,
        exclude_sections: tuple[str, ...] = ("about",),
        min_chars: int = 100,
    ) -> dict[str, Any]:
        logger.info("=== Exporting '%s' (opus_path='%s') from Qdrant ===", work_name, opus_path)
        reader = QdrantCollectionReader(
            url=self.qdrant_url,
            collection_name=self.collection_name,
            vector_name=self.vector_name,
            opus_path=opus_path,
        )
        dataset = reader.fetch_dataset()
        if len(dataset) == 0:
            raise ValueError(f"No points found in Qdrant '{self.collection_name}' for opus_path='{opus_path}'")

        # Attach parsed (url_path, section, paragraph_num) and sort into clean reading order
        enriched = []
        for idx, record in enumerate(dataset.records):
            url_path, section, para_num = parse_biblioteca_url(record["url"], opus_path)
            enriched.append((section, para_num, url_path, idx, record))
        enriched.sort(key=lambda item: (item[0], item[1], item[2]))

        fetch_jobs = [(url_path, record.get("sha256")) for _sec, _pnum, url_path, _idx, record in enriched]
        logger.info("Downloading %d paragraph texts from %s ...", len(fetch_jobs), self.fetcher.base_url)
        fetched = self.fetcher.fetch_many(fetch_jobs, allow_404=True)

        sha_mismatches = 0
        skipped_missing = 0
        skipped_filtered = 0
        kept_vectors: list[np.ndarray] = []
        file_names: list[str] = []
        extra_fields: list[dict[str, Any]] = []
        sections_text: dict[str, list[tuple[int, str]]] = {}

        for (section, para_num, url_path, orig_idx, record), (text, sha_ok) in zip(enriched, fetched):
            if text is None:
                skipped_missing += 1
                continue
            if not sha_ok:
                sha_mismatches += 1
            clean_text = text.strip()
            candidate = {
                "id": record["id"],
                "opus_path": opus_path,
                "section": section,
                "paragraph_num": para_num,
                "url": f"{self.fetcher.base_url}{url_path}",
                "sha256": record.get("sha256"),
                "sha256_verified": sha_ok,
                "text": clean_text,
            }
            if not should_keep_record(
                candidate,
                exclude_headings=exclude_headings,
                exclude_sections=exclude_sections,
                min_chars=min_chars,
            ):
                skipped_filtered += 1
                continue
            kept_vectors.append(dataset.vectors[orig_idx])
            rel_file = f"{section}/_{para_num}.txt"
            file_names.append(rel_file)
            extra_fields.append(candidate)
            sections_text.setdefault(section, []).append((para_num, clean_text))

        if not kept_vectors:
            raise ValueError(
                f"All fetched paragraphs were filtered out for '{work_name}' "
                f"(skipped_missing={skipped_missing}, skipped_filtered={skipped_filtered})."
            )
        ordered_vectors = np.asarray(kept_vectors, dtype=np.float32)

        store = FaissVectorStore(
            directory=self.out_root,
            collection_name=work_name,
            metric="cosine",
            meta={
                "source": "qdrant+biblioteca.scriptorium.ro",
                "qdrant_collection": self.collection_name,
                "opus_path": opus_path,
                "embedding_model": OLLAMA_EMBED_MODEL,
                "sha256_mismatches": sha_mismatches,
                "skipped_missing": skipped_missing,
                "skipped_filtered": skipped_filtered,
                "filtered": {
                    "exclude_headings": exclude_headings,
                    "exclude_sections": list(exclude_sections),
                    "min_chars": min_chars,
                },
            },
        )
        store.insert_batch(file_names=file_names, vectors=ordered_vectors, extra_fields=extra_fields)
        store.close()

        # Also materialize human-readable section text files under <work_dir>/texts/
        texts_dir = self.out_root / work_name / "texts"
        texts_dir.mkdir(parents=True, exist_ok=True)
        for section, paragraphs in sections_text.items():
            paragraphs.sort(key=lambda pair: pair[0])
            out_file = texts_dir / f"{section}.txt"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text("\n\n".join(p_text for _pnum, p_text in paragraphs) + "\n", encoding="utf-8")

        summary = {
            "work_name": work_name,
            "opus_path": opus_path,
            "paragraphs": len(file_names),
            "sections": len(sections_text),
            "dimension": int(ordered_vectors.shape[1]),
            "sha256_mismatches": sha_mismatches,
            "output_dir": str(self.out_root / work_name),
        }
        logger.info(
            "Exported '%s': %d paragraphs across %d sections (dim=%d, sha_mismatches=%d) -> %s",
            work_name, summary["paragraphs"], summary["sections"],
            summary["dimension"], sha_mismatches, summary["output_dir"],
        )
        return summary
