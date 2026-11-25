"""Persistence for axis-labelling and clustering results (local JSONL + Markdown)."""
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ResultsStore(ABC):
    """Checkpoints labelled components or clusters one record at a time."""

    @abstractmethod
    def insert(self, record: dict[str, Any]) -> None: ...


class JsonlResultsStore(ResultsStore):
    """Streams one JSON object per line to disk so long runs checkpoint cleanly."""

    LABEL_MAX_LENGTH = 4096

    def __init__(self, output_path: Path, overwrite: bool = True):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite and self.output_path.exists():
            self.output_path.unlink()
        self._records: list[dict[str, Any]] = []

    def insert(self, record: dict[str, Any]) -> None:
        item = dict(record)
        label = item.get("label")
        if isinstance(label, str) and len(label) > self.LABEL_MAX_LENGTH:
            item["label"] = label[: self.LABEL_MAX_LENGTH - 1] + "…"
        with self.output_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        self._records.append(item)
        logger.debug("Wrote result record to %s", self.output_path)

    @property
    def records(self) -> list[dict[str, Any]]:
        return list(self._records)

    @classmethod
    def read_all(cls, output_path: Path) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for line in Path(output_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def write_axes_markdown(self, md_path: Path, title: str) -> None:
        """Render labelled PCA/kPCA axes into a concise Markdown report."""
        lines = [f"# {title}", ""]
        by_component: dict[int, list[dict[str, Any]]] = {}
        for rec in self._records:
            by_component.setdefault(int(rec["component_index"]), []).append(rec)

        for comp_idx in sorted(by_component):
            entries = by_component[comp_idx]
            score = entries[0].get("eigenvalue", 0.0)
            score_name = entries[0].get("score_name", "eigenvalue")
            lines.append(f"## Component {comp_idx + 1} (`{score_name}={score:.4f}`)")
            lines.append("")
            for entry in entries:
                sign = entry.get("sign", "")
                anchor = entry.get("anchor_id") or entry.get("anchor_file_name", "")
                label = entry.get("label", "")
                lines.append(f"- **[{sign}] {label}** *(anchor: `{anchor}`)*")
                neighbours = entry.get("neighbour_ids") or []
                if neighbours:
                    lines.append(f"  - Top passages: {', '.join(f'`{n}`' for n in neighbours)}")
            lines.append("")

        Path(md_path).parent.mkdir(parents=True, exist_ok=True)
        Path(md_path).write_text("\n".join(lines), encoding="utf-8")
        logger.info("Wrote Markdown axes report to %s", md_path)

    def write_clusters_markdown(self, md_path: Path, title: str) -> None:
        """Render HDBSCAN / BERTopic clusters into a concise Markdown report."""
        lines = [f"# {title}", ""]
        for rec in self._records:
            cid = rec["cluster_id"]
            size = rec["size"]
            label = rec.get("label", "")
            keywords = rec.get("keywords") or []
            header = f"## Cluster {cid} ({size} paragraphs)" if cid != -1 else f"## Noise / Outliers (-1, {size} paragraphs)"
            lines.append(header)
            lines.append("")
            if label:
                lines.append(f"- **Theme**: {label}")
            if keywords:
                lines.append(f"- **Keywords (c-TF-IDF)**: {', '.join(keywords)}")
            for rep in rec.get("representatives") or []:
                ref = rep.get("ref", "")
                excerpt = rep.get("excerpt", "").replace("\n", " ")
                lines.append(f"  - `{ref}`: {excerpt}")
            lines.append("")

        Path(md_path).parent.mkdir(parents=True, exist_ok=True)
        Path(md_path).write_text("\n".join(lines), encoding="utf-8")
        logger.info("Wrote Markdown clusters report to %s", md_path)
