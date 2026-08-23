from pathlib import Path

from .bibles import BIBLES


class ChapterTextResolver:
    """Maps a (bible, file_name) pair back to the chapter markdown on disk - Milvus only
    stores the vector and file_name, not the text itself."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    def read(self, bible: str, file_name: str) -> str:
        source_dirname, _collection = BIBLES[bible]
        return (self.base_dir / source_dirname / file_name).read_text(encoding="utf-8")
