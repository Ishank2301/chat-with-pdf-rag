"""
Embedding manager with batch processing and connection pooling.
Optimized for 2026 RAG Agent performance requirements.
"""
import logging
from typing import List, Dict, Optional
from functools import lru_cache
from langchain_ollama import OllamaEmbeddings
import tenacity

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages embeddings with batching and caching."""
    
    _instance = None
    _embeddings = None
    
    def __new__(cls):
        """Singleton pattern for connection pooling."""
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize embedding manager with singleton pattern."""
        if self._initialized:
            return
        
        self._embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self._cache = {}
        self._initialized = True
        logger.info("✅ EmbeddingManager initialized with connection pooling")
    
    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
        stop=tenacity.stop_after_attempt(3),
        reraise=True
    )
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with batching.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to embed per batch (default: 32)
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        total = len(texts)
        
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"🔄 Embedding batch [{i}/{total}] ({len(batch)} items)")
            
            try:
                batch_embeddings = self._embeddings.embed_documents(batch)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"❌ Batch embedding failed: {e}")
                raise
        
        logger.info(f"✅ Generated {len(embeddings)} embeddings")
        return embeddings
    
    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=10),
        stop=tenacity.stop_after_attempt(3),
        reraise=True
    )
    def embed_query(self, query: str, use_cache: bool = True) -> List[float]:
        """
        Generate embedding for a query with optional caching.
        
        Args:
            query: Query text to embed
            use_cache: Whether to use cached results (default: True)
            
        Returns:
            Embedding vector
        """
        query_hash = hash(query)
        
        if use_cache and query_hash in self._cache:
            logger.debug(f"📦 Using cached embedding for query")
            return self._cache[query_hash]
        
        try:
            embedding = self._embeddings.embed_query(query)
            if use_cache:
                self._cache[query_hash] = embedding
            return embedding
        except Exception as e:
            logger.error(f"❌ Query embedding failed: {e}")
            raise
    
    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self._cache.clear()
        logger.info("🗑️ Embedding cache cleared")
    
    @staticmethod
    def get_instance() -> "EmbeddingManager":
        """Get singleton instance."""
        return EmbeddingManager()
