"""App-wide settings, loaded from the environment / .env (via pydantic-settings,

already a project dependency for exactly this). Exposed both as a Settings
instance and as plain module-level constants (OLLAMA_URL, MILVUS_URI) for
existing callers (kpca_label.py) that just want the string values.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_url: str
    milvus_uri: str


settings = Settings()

OLLAMA_URL = settings.ollama_url
MILVUS_URI = settings.milvus_uri
