"""
2026 RAG Agent - Core package initialization.
Exports all core components for easy importing.
"""

from .logger_config import setup_logging
from .embedding_manager import EmbeddingManager
from .semantic_chunker import SemanticChunker
from .document_loader import DocumentLoader
from .vector_db_manager import VectorDatabaseManager
from .rag_engine import RAGEngine

__all__ = [
    "setup_logging",
    "EmbeddingManager",
    "SemanticChunker",
    "DocumentLoader",
    "VectorDatabaseManager",
    "RAGEngine",
]

__version__ = "2.0.0"
