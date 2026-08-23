import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MarkdownFileReader:
    def __init__(self, source_directory: Path):
        self.source_directory = source_directory

    def files(self) -> list[Path]:
        files = sorted(
            path for path in self.source_directory.rglob("*") if path.is_file()
        )
        logger.info("Found %d chapter files under %s", len(files), self.source_directory)
        return files

    def read(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def relative_name(self, path: Path) -> str:
        return path.relative_to(self.source_directory).as_posix()
