from pymilvus import MilvusClient


class MilvusNearestNeighbourSearch:
    def __init__(self, client: MilvusClient, collection_name: str, has_bible_field: bool):
        self.client = client
        self.collection_name = collection_name
        self.has_bible_field = has_bible_field
        self.client.load_collection(collection_name)

    def search(self, query_vector: list, k: int) -> list[dict]:
        output_fields = ["file_name"] + (["bible"] if self.has_bible_field else [])
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=k,
            output_fields=output_fields,
        )
        return [hit["entity"] for hit in results[0]]
