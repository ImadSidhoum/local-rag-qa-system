from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="local-rag-qa", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    data_dir: Path = Field(default=Path("/app/data"), alias="DATA_DIR")
    corpus_dir: Path = Field(default=Path("/app/data/corpus"), alias="CORPUS_DIR")
    chroma_dir: Path = Field(default=Path("/app/data/chroma"), alias="CHROMA_DIR")
    chroma_collection_name: str = Field(default="rag_chunks", alias="CHROMA_COLLECTION_NAME")
    index_manifest_path: Path = Field(default=Path("/app/data/index_manifest.json"), alias="INDEX_MANIFEST_PATH")
    chroma_anonymized_telemetry: bool = Field(default=False, alias="CHROMA_ANONYMIZED_TELEMETRY")
    chroma_product_telemetry_impl: str = Field(
        default="app.chroma_telemetry.NoOpProductTelemetry",
        alias="CHROMA_PRODUCT_TELEMETRY_IMPL",
    )
    chroma_telemetry_impl: str = Field(
        default="app.chroma_telemetry.NoOpProductTelemetry",
        alias="CHROMA_TELEMETRY_IMPL",
    )

    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    batch_size: int = Field(default=32, alias="BATCH_SIZE")
    chunk_size: int = Field(default=900, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")
    top_k: int = Field(default=4, alias="TOP_K")
    use_mmr: bool = Field(default=False, alias="USE_MMR")
    mmr_candidates: int = Field(default=12, alias="MMR_CANDIDATES")
    mmr_lambda: float = Field(default=0.6, alias="MMR_LAMBDA")
    min_similarity: float = Field(default=0.2, alias="MIN_SIMILARITY")

    ollama_base_url: str = Field(default="http://ollama:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.2:1b", alias="OLLAMA_MODEL")
    ollama_fallback_model: str = Field(default="llama3.2:1b", alias="OLLAMA_FALLBACK_MODEL")
    ollama_timeout_seconds: int = Field(default=240, alias="OLLAMA_TIMEOUT_SECONDS")
    ollama_auto_pull: bool = Field(default=True, alias="OLLAMA_AUTO_PULL")
    gen_temperature: float = Field(default=0.1, alias="GEN_TEMPERATURE")
    gen_max_tokens: int = Field(default=450, alias="GEN_MAX_TOKENS")
    query_rewrite_enabled: bool = Field(default=True, alias="QUERY_REWRITE_ENABLED")
    query_rewrite_max_tokens: int = Field(default=96, alias="QUERY_REWRITE_MAX_TOKENS")
    memory_enabled: bool = Field(default=True, alias="MEMORY_ENABLED")
    memory_max_turns: int = Field(default=6, alias="MEMORY_MAX_TURNS")

    langfuse_enabled: bool = Field(default=False, alias="LANGFUSE_ENABLED")
    langfuse_host: str = Field(default="http://localhost:3000", alias="LANGFUSE_HOST")
    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")

    random_seed: int = Field(default=42, alias="RANDOM_SEED")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    settings.corpus_dir.mkdir(parents=True, exist_ok=True)
    settings.index_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
