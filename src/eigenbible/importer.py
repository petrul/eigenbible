import logging

from .embedder import OllamaEmbedder
from .markdown_reader import MarkdownFileReader
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class BibliaEmbeddingImporter:
    def __init__(
        self,
        reader: MarkdownFileReader,
        embedder: OllamaEmbedder,
        vector_store: VectorStore,
        batch_size: int = 200,
    ):
        self.reader = reader
        self.embedder = embedder
        self.vector_store = vector_store
        self.batch_size = batch_size

    def run(self) -> int:
        imported = 0
        files = self.reader.files()
        batches = [files[i:i + self.batch_size] for i in range(0, len(files), self.batch_size)]
        logger.info(
            "Embedding chapters in batches of %d and storing each batch's vectors right away, "
            "so later a query can be embedded the same way and matched by similarity against "
            "these chapters",
            self.batch_size,
        )
        for batch_index, batch in enumerate(batches, start=1):
            file_names = [self.reader.relative_name(path) for path in batch]
            texts = [self.reader.read(path) for path in batch]
            vectors = self.embedder.embed_batch(texts)
            self.vector_store.insert_batch(file_names, vectors)
            imported += len(batch)
            logger.info(
                "[batch %d/%d] embedded and stored %d chapters (%d/%d total)",
                batch_index, len(batches), len(batch), imported, len(files),
            )
        return imported
