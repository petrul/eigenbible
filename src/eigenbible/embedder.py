import logging

import requests

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str):
        self.url = f"{base_url.rstrip('/')}/api/embed"
        self.model = model
        logger.info("Embedding with Ollama model '%s' at %s", model, self.url)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        logger.debug("Requesting embeddings for a batch of %d texts from Ollama", len(texts))
        response = requests.post(
            self.url,
            json={"model": self.model, "input": texts},
            timeout=600,
        )
        response.raise_for_status()
        embeddings = response.json().get("embeddings")
        if not embeddings or len(embeddings) != len(texts):
            raise ValueError("Ollama returned an unexpected embedding response")
        logger.debug("Got %d embeddings back, %d-dimensional each", len(embeddings), len(embeddings[0]))
        return embeddings
