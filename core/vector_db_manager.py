"""
Vector database manager with connection pooling and batch operations.
Optimized for 2026 RAG Agent performance.
"""
import logging
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class VectorDatabaseManager:
    """Manages Chroma vector database with connection pooling."""
    
    _instance = None
    _client = None
    _collection = None
    
    def __new__(cls):
        """Singleton pattern for database connection pooling."""
        if cls._instance is None:
            cls._instance = super(VectorDatabaseManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        path: str = "chroma_persistent_storage",
        collection_name: str = "rag_documents"
    ):
        """
        Initialize vector database manager with singleton pattern.
        
        Args:
            path: Path to Chroma persistent storage
            collection_name: Name of the collection
        """
        if self._initialized:
            return
        
        try:
            # Initialize Chroma client with persistence
            self._client = chromadb.PersistentClient(path=path)
            self._collection_name = collection_name
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "RAG documents collection"}
            )
            
            self._initialized = True
            logger.info(
                f"✅ VectorDatabaseManager initialized "
                f"(path={path}, collection={collection_name})"
            )
        except Exception as e:
            logger.error(f"❌ Failed to initialize VectorDatabaseManager: {e}")
            raise
    
    def upsert_batch(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict]] = None
    ) -> None:
        """
        Upsert multiple documents with embeddings in batch.
        
        Args:
            ids: List of document IDs
            documents: List of document texts
            embeddings: List of embedding vectors
            metadatas: Optional list of metadata dictionaries
        """
        if not ids or not documents or not embeddings:
            logger.warning("⚠️ Empty batch provided to upsert")
            return
        
        if not (len(ids) == len(documents) == len(embeddings)):
            logger.error("❌ Mismatched lengths in upsert_batch")
            raise ValueError("IDs, documents, and embeddings must have same length")
        
        try:
            batch_size = 100
            total = len(ids)
            
            for i in range(0, total, batch_size):
                end_idx = min(i + batch_size, total)
                
                batch_ids = ids[i:end_idx]
                batch_docs = documents[i:end_idx]
                batch_embeddings = embeddings[i:end_idx]
                batch_metadata = metadatas[i:end_idx] if metadatas else None
                
                logger.info(f"📤 Upserting batch [{i}/{total}] ({len(batch_ids)} items)")
                
                self._collection.upsert(
                    ids=batch_ids,
                    documents=batch_docs,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadata
                )
            
            logger.info(f"✅ Successfully upserted {total} documents")
        
        except Exception as e:
            logger.error(f"❌ Batch upsert failed: {e}")
            raise
    
    def query(
        self,
        query_embedding: List[float],
        n_results: int = 3,
        where: Optional[Dict] = None
    ) -> Dict[str, List]:
        """
        Query documents by embedding with optional filtering.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Optional metadata filter
            
        Returns:
            Query results with documents and distances
        """
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where
            )
            
            logger.info(f"🔍 Query returned {len(results.get('documents', [[]])[0])} results")
            return results
        
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            raise
    
    def get_by_ids(self, ids: List[str]) -> Dict:
        """
        Retrieve documents by IDs.
        
        Args:
            ids: List of document IDs
            
        Returns:
            Retrieved documents and metadata
        """
        try:
            results = self._collection.get(ids=ids)
            logger.info(f"📥 Retrieved {len(results.get('documents', []))} documents by ID")
            return results
        
        except Exception as e:
            logger.error(f"❌ Get by IDs failed: {e}")
            raise
    
    def delete_by_ids(self, ids: List[str]) -> None:
        """
        Delete documents by IDs.
        
        Args:
            ids: List of document IDs to delete
        """
        try:
            self._collection.delete(ids=ids)
            logger.info(f"🗑️ Deleted {len(ids)} documents")
        
        except Exception as e:
            logger.error(f"❌ Delete failed: {e}")
            raise
    
    def get_collection_stats(self) -> Dict:
        """
        Get collection statistics.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self._collection.count()
            return {
                "collection_name": self._collection_name,
                "document_count": count,
                "status": "healthy"
            }
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {e}")
            return {"status": "error", "message": str(e)}
    
    def clear_collection(self) -> None:
        """Clear all documents from collection."""
        try:
            self._client.delete_collection(name=self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name
            )
            logger.info(f"🗑️ Cleared collection: {self._collection_name}")
        
        except Exception as e:
            logger.error(f"❌ Failed to clear collection: {e}")
            raise
    
    @staticmethod
    def get_instance() -> "VectorDatabaseManager":
        """Get singleton instance."""
        return VectorDatabaseManager()
