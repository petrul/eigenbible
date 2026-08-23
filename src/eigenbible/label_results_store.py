import logging

from pymilvus import DataType, MilvusClient

logger = logging.getLogger(__name__)


class LabelResultsStore:
    """Where each component's label lands, one row at a time, so a long run checkpoints its
    progress instead of losing everything if it's interrupted before the last component."""

    LABEL_MAX_LENGTH = 4096  # generous headroom - the prompt asks for a sentence but isn't always obeyed

    def __init__(self, client: MilvusClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name
        self.ready = False

    def ensure_collection(self, n_components: int) -> None:
        if self.ready:
            return
        if not self.client.has_collection(self.collection_name):
            logger.info(
                "Creating results collection '%s' (kpca_vector dim=%d)",
                self.collection_name, n_components,
            )
            schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="component_index", datatype=DataType.INT64)
            schema.add_field(field_name="eigenvalue", datatype=DataType.DOUBLE)
            schema.add_field(field_name="sign", datatype=DataType.VARCHAR, max_length=1)
            schema.add_field(field_name="anchor_bible", datatype=DataType.VARCHAR, max_length=64)
            schema.add_field(field_name="anchor_file_name", datatype=DataType.VARCHAR, max_length=1024)
            schema.add_field(field_name="label", datatype=DataType.VARCHAR, max_length=self.LABEL_MAX_LENGTH)
            schema.add_field(field_name="kpca_vector", datatype=DataType.FLOAT_VECTOR, dim=n_components)

            index_params = self.client.prepare_index_params()
            index_params.add_index(field_name="kpca_vector", index_type="AUTOINDEX", metric_type="L2")
            self.client.create_collection(
                collection_name=self.collection_name, schema=schema, index_params=index_params,
            )
            logger.info("Results collection '%s' created", self.collection_name)
        self.ready = True

    def insert(self, record: dict) -> None:
        self.ensure_collection(len(record["kpca_vector"]))
        # Belt and braces: the prompt asks for a short label but the model doesn't always
        # comply, and a VARCHAR overflow would otherwise crash the whole run on insert.
        if len(record["label"]) > self.LABEL_MAX_LENGTH:
            record = {**record, "label": record["label"][:self.LABEL_MAX_LENGTH - 1] + "…"}
        self.client.insert(collection_name=self.collection_name, data=[record])
        logger.debug("Stored label for component %d in '%s'", record["component_index"], self.collection_name)
