"""Environment-based application settings."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Typed settings loaded from environment variables."""

    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").lower()
    llm_model: str = os.getenv("LLM_MODEL", "llama3")
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    chroma_path: str = os.getenv("CHROMA_PATH", "chroma_persistent_storage")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "rag_documents")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: str = os.getenv("LOG_FILE", "rag_agent.log")
    enable_mlflow: bool = _env_bool("ENABLE_MLFLOW", True)
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow_experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "chat-with-pdf-rag")


settings = Settings()
