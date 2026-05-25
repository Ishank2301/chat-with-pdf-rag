"""
2026 RAG Agent Orchestrator - Main application logic.
Coordinates all components for document processing and querying.
"""
import logging
from typing import List, Dict, Optional
from core import (
    setup_logging,
    EmbeddingManager,
    SemanticChunker,
    DocumentLoader,
    VectorDatabaseManager,
    RAGEngine
)

logger = logging.getLogger(__name__)


class RAGAgent:
    """Main RAG Agent orchestrator for 2026 workflows."""
    
    def __init__(self):
        """Initialize RAG Agent with all components."""
        setup_logging()
        
        self.embedding_manager = EmbeddingManager()
        self.semantic_chunker = SemanticChunker(chunk_size=1000, chunk_overlap=200)
        self.document_loader = DocumentLoader()
        self.vector_db = VectorDatabaseManager()
        self.rag_engine = RAGEngine()
        
        logger.info("🚀 RAG Agent 2026 initialized successfully")
    
    def ingest_documents(
        self,
        directory_path: str,
        use_parallel: bool = True,
        num_workers: int = 4
    ) -> Dict[str, any]:
        """
        Ingest and process documents from directory.
        
        Args:
            directory_path: Path to documents
            use_parallel: Use parallel loading
            num_workers: Number of parallel workers
            
        Returns:
            Ingestion statistics
        """
        logger.info(f"📂 Starting document ingestion from: {directory_path}")
        
        # Load documents
        if use_parallel:
            documents = self.document_loader.load_documents_parallel(
                directory_path,
                num_workers=num_workers
            )
        else:
            documents = self.document_loader.load_directory(directory_path)
        
        if not documents:
            logger.warning("⚠️ No documents loaded")
            return {"status": "error", "message": "No documents found"}
        
        # Chunk documents
        logger.info(f"✂️ Chunking {len(documents)} documents...")
        chunked_docs = self.semantic_chunker.chunk_documents(documents)
        
        if not chunked_docs:
            logger.error("❌ Failed to chunk documents")
            return {"status": "error", "message": "Chunking failed"}
        
        # Generate embeddings
        logger.info(f"🧠 Generating embeddings for {len(chunked_docs)} chunks...")
        texts = [doc["text"] for doc in chunked_docs]
        embeddings = self.embedding_manager.embed_batch(texts, batch_size=32)
        
        # Upsert to vector database
        logger.info("💾 Upserting to vector database...")
        ids = [doc["id"] for doc in chunked_docs]
        self.vector_db.upsert_batch(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=chunked_docs
        )
        
        stats = self.vector_db.get_collection_stats()
        logger.info(f"✅ Document ingestion complete - {stats['document_count']} total documents")
        
        return {
            "status": "success",
            "documents_loaded": len(documents),
            "chunks_created": len(chunked_docs),
            "total_in_database": stats['document_count']
        }
    
    def query_documents(
        self,
        question: str,
        n_results: int = 3,
        return_relevant_chunks: bool = False
    ) -> Dict[str, any]:
        """
        Query documents and generate answer.
        
        Args:
            question: User question
            n_results: Number of relevant chunks to retrieve
            return_relevant_chunks: Include relevant chunks in response
            
        Returns:
            Answer and optional metadata
        """
        logger.info(f"❓ Processing query: {question}")
        
        try:
            # Get query embedding
            query_embedding = self.embedding_manager.embed_query(
                question,
                use_cache=True
            )
            
            # Retrieve relevant chunks
            results = self.vector_db.query(query_embedding, n_results=n_results)
            relevant_chunks = results["documents"][0] if results["documents"] else []
            
            if not relevant_chunks:
                logger.warning("⚠️ No relevant documents found")
                return {
                    "status": "success",
                    "answer": "I couldn't find relevant information to answer your question.",
                    "confidence": 0.0
                }
            
            # Generate response
            answer = self.rag_engine.generate_response(
                question,
                relevant_chunks,
                max_length=3
            )
            
            result = {
                "status": "success",
                "question": question,
                "answer": answer,
                "num_sources": len(relevant_chunks)
            }
            
            if return_relevant_chunks:
                result["relevant_chunks"] = relevant_chunks
            
            logger.info(f"✅ Query processed - {len(relevant_chunks)} sources used")
            return result
        
        except Exception as e:
            logger.error(f"❌ Query processing failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def summarize_document(
        self,
        document_path: str,
        summary_length: str = "short"
    ) -> Dict[str, any]:
        """
        Summarize a single document.
        
        Args:
            document_path: Path to document
            summary_length: "short" or "detailed"
            
        Returns:
            Summary and metadata
        """
        logger.info(f"📄 Summarizing document: {document_path}")
        
        try:
            doc = self.document_loader.load_text_file(document_path)
            summary = self.rag_engine.summarize_documents(
                [doc["text"]],
                summary_length=summary_length
            )
            
            logger.info("✅ Document summarization complete")
            return {
                "status": "success",
                "document": document_path,
                "summary": summary
            }
        
        except Exception as e:
            logger.error(f"❌ Summarization failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def analyze_resume(self, resume_path: str) -> Dict[str, any]:
        """
        Analyze resume for ATS compatibility.
        
        Args:
            resume_path: Path to resume file
            
        Returns:
            ATS score, strengths, weaknesses, and suggestions
        """
        logger.info(f"📊 Analyzing resume: {resume_path}")
        
        try:
            resume_doc = self.document_loader.load_text_file(resume_path)
            analysis = self.rag_engine.analyze_resume_ats_score(resume_doc["text"])
            
            logger.info(f"✅ Resume analysis complete - ATS Score: {analysis['ats_score']}")
            return {
                "status": "success",
                "resume_file": resume_path,
                **analysis
            }
        
        except Exception as e:
            logger.error(f"❌ Resume analysis failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_database_stats(self) -> Dict[str, any]:
        """Get vector database statistics."""
        try:
            stats = self.vector_db.get_collection_stats()
            logger.info(f"📊 Database stats: {stats}")
            return stats
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {e}")
            return {"status": "error", "message": str(e)}
    
    def clear_database(self) -> Dict[str, str]:
        """Clear all documents from database."""
        try:
            self.vector_db.clear_collection()
            self.embedding_manager.clear_cache()
            logger.info("🗑️ Database and embedding cache cleared")
            return {"status": "success", "message": "Database cleared"}
        except Exception as e:
            logger.error(f"❌ Failed to clear database: {e}")
            return {"status": "error", "message": str(e)}
