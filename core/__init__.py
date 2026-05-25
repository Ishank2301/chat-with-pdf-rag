"""Core package exports."""

from .config import settings
from .logger_config import setup_logging
from .embedding_manager import EmbeddingManager
from .semantic_chunker import SemanticChunker
from .document_loader import DocumentLoader
from .vector_db_manager import VectorDatabaseManager
from .rag_engine import RAGEngine
from .mlflow_tracker import MLflowTracker

__all__ = [
    "settings",
    "setup_logging",
    "EmbeddingManager",
    "SemanticChunker",
    "DocumentLoader",
    "VectorDatabaseManager",
    "RAGEngine",
    "MLflowTracker",
]

__version__ = "2.0.0"
