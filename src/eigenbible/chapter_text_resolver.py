"""Resolves a dataset record back to its raw text passage."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .bibles import BIBLES


class TextResolver(ABC):
    """Turns a row metadata dict into the underlying text passage."""

    @abstractmethod
    def read_row(self, row: dict[str, Any]) -> str: ...


class InlineTextResolver(TextResolver):
    """Reads text directly from a field stored in each record (e.g. FAISS records.jsonl)."""

    def __init__(self, field_name: str = "text"):
        self.field_name = field_name

    def read_row(self, row: dict[str, Any]) -> str:
        text = row.get(self.field_name)
        if text is None:
            raise KeyError(f"Record is missing inline text field '{self.field_name}': {row}")
        return str(text)


class ChapterTextResolver(TextResolver):
    """Maps a (bible, file_name) pair back to the chapter markdown on disk."""

    def __init__(self, base_dir: Path, default_bible: str | None = None):
        self.base_dir = Path(base_dir)
        self.default_bible = default_bible

    def read(self, bible: str, file_name: str) -> str:
        source_dirname, _collection = BIBLES[bible]
        return (self.base_dir / source_dirname / file_name).read_text(encoding="utf-8")

    def read_row(self, row: dict[str, Any]) -> str:
        bible = row.get("bible") or self.default_bible
        if not bible:
            raise KeyError(f"Cannot resolve bible for row without 'bible' field or default_bible: {row}")
        return self.read(bible, row["file_name"])
