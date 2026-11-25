"""App-wide settings and service endpoints.

Qdrant on zmeu and biblioteca.scriptorium.ro are hardwired as constants for now,
with Ollama URL still configurable via `.env` (via pydantic-settings) and
defaulting to zmeu so purely offline local-FAISS workflows never fail at import
time.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict

QDRANT_URL = "http://zmeu.local:6333"
QDRANT_COLLECTION = "biblioteca_paragraphs_bge_m3"
QDRANT_VECTOR_NAME = "embedding"
BIBLIOTECA_BASE_URL = "https://biblioteca.scriptorium.ro"
OLLAMA_EMBED_MODEL = "bge-m3:567m-fp16"
DEFAULT_OLLAMA_URL = "http://zmeu.local:11434"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_url: str = DEFAULT_OLLAMA_URL
    qdrant_url: str = QDRANT_URL
    ollama_embed_model: str = OLLAMA_EMBED_MODEL


settings = Settings()

OLLAMA_URL = settings.ollama_url
