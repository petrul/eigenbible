import logging

from pymilvus import MilvusClient

from .bibles import COLLECTION_TO_BIBLE, COMBINED_COLLECTION

logger = logging.getLogger(__name__)


class VectorCollectionReader:
    """Pages through a Milvus collection built by embed_biblia.py and returns every row."""

    BATCH_SIZE = 1000

    def __init__(self, client: MilvusClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name
        self.has_bible_field = collection_name == COMBINED_COLLECTION

    def fetch_all(self) -> list[dict]:
        output_fields = ["id", "vector", "file_name"] + (["bible"] if self.has_bible_field else [])
        rows = []
        offset = 0
        while True:
            batch = self.client.query(
                collection_name=self.collection_name,
                filter="id >= 0",
                output_fields=output_fields,
                limit=self.BATCH_SIZE,
                offset=offset,
            )
            if not batch:
                break
            rows.extend(batch)
            offset += len(batch)
            logger.info("Fetched %d rows from '%s' so far", offset, self.collection_name)
        return rows

    def bible_of(self, row: dict) -> str:
        return row["bible"] if self.has_bible_field else COLLECTION_TO_BIBLE[self.collection_name]
