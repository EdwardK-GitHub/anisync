from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central application settings.

    We load from environment variables and from a local .env file in development.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: str = "development"
    secret_key: str
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/anisync"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    app_host: str = "127.0.0.1"
    app_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings accessor.

    Using a cached accessor avoids reparsing environment variables repeatedly.
    """
    return Settings()


settings = get_settings()