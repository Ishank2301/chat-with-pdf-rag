"""Main application logic for document processing and querying."""

import logging
from typing import Dict

from core import (
    DocumentLoader,
    EmbeddingManager,
    RAGEngine,
    SemanticChunker,
    VectorDatabaseManager,
    setup_logging,
)

logger = logging.getLogger(__name__)


class RAGAgent:
    """Coordinates document loading, retrieval, and response generation."""

    def __init__(self):
        """Initialize the RAG agent components."""
        setup_logging()

        self.embedding_manager = EmbeddingManager()
        self.semantic_chunker = SemanticChunker(chunk_size=1000, chunk_overlap=200)
        self.document_loader = DocumentLoader()
        self.vector_db = VectorDatabaseManager()
        self.rag_engine = RAGEngine()

        logger.info("RAG agent initialized successfully")

    def ingest_documents(
        self,
        directory_path: str,
        use_parallel: bool = True,
        num_workers: int = 4,
    ) -> Dict[str, any]:
        """
        Ingest and process documents from directory.

        Args:
            directory_path: Path to documents.
            use_parallel: Whether to load files with a thread pool.
            num_workers: Number of worker threads.

        Returns:
            Ingestion statistics.
        """
        logger.info("Starting document ingestion from: %s", directory_path)

        if use_parallel:
            documents = self.document_loader.load_documents_parallel(
                directory_path,
                num_workers=num_workers,
            )
        else:
            documents = self.document_loader.load_directory(directory_path)

        if not documents:
            logger.warning("No documents loaded")
            return {"status": "error", "message": "No documents found"}

        logger.info("Chunking %s documents", len(documents))
        chunked_docs = self.semantic_chunker.chunk_documents(documents)

        if not chunked_docs:
            logger.error("Failed to chunk documents")
            return {"status": "error", "message": "Chunking failed"}

        logger.info("Generating embeddings for %s chunks", len(chunked_docs))
        texts = [doc["text"] for doc in chunked_docs]
        embeddings = self.embedding_manager.embed_batch(texts, batch_size=32)

        logger.info("Upserting chunks to vector database")
        ids = [doc["id"] for doc in chunked_docs]
        self.vector_db.upsert_batch(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=chunked_docs,
        )

        stats = self.vector_db.get_collection_stats()
        logger.info("Document ingestion complete: %s total documents", stats["document_count"])

        return {
            "status": "success",
            "documents_loaded": len(documents),
            "chunks_created": len(chunked_docs),
            "total_in_database": stats["document_count"],
        }

    def query_documents(
        self,
        question: str,
        n_results: int = 3,
        return_relevant_chunks: bool = False,
    ) -> Dict[str, any]:
        """
        Query documents and generate an answer.

        Args:
            question: User question.
            n_results: Number of relevant chunks to retrieve.
            return_relevant_chunks: Include retrieved chunks in the response.

        Returns:
            Answer and optional metadata.
        """
        logger.info("Processing query: %s", question)

        try:
            query_embedding = self.embedding_manager.embed_query(
                question,
                use_cache=True,
            )

            results = self.vector_db.query(query_embedding, n_results=n_results)
            relevant_chunks = results["documents"][0] if results["documents"] else []

            if not relevant_chunks:
                logger.warning("No relevant documents found")
                return {
                    "status": "success",
                    "answer": "I could not find relevant information to answer your question.",
                    "confidence": 0.0,
                }

            answer = self.rag_engine.generate_response(
                question,
                relevant_chunks,
                max_length=3,
            )

            result = {
                "status": "success",
                "question": question,
                "answer": answer,
                "num_sources": len(relevant_chunks),
            }

            if return_relevant_chunks:
                result["relevant_chunks"] = relevant_chunks

            logger.info("Query processed with %s sources", len(relevant_chunks))
            return result

        except Exception as e:
            logger.error("Query processing failed: %s", e)
            return {"status": "error", "message": str(e)}

    def summarize_document(
        self,
        document_path: str,
        summary_length: str = "short",
    ) -> Dict[str, any]:
        """
        Summarize a single document.

        Args:
            document_path: Path to document.
            summary_length: "short" or "detailed".

        Returns:
            Summary and metadata.
        """
        logger.info("Summarizing document: %s", document_path)

        try:
            doc = self.document_loader.load_text_file(document_path)
            summary = self.rag_engine.summarize_documents(
                [doc["text"]],
                summary_length=summary_length,
            )

            logger.info("Document summarization complete")
            return {
                "status": "success",
                "document": document_path,
                "summary": summary,
            }

        except Exception as e:
            logger.error("Summarization failed: %s", e)
            return {"status": "error", "message": str(e)}

    def analyze_resume(self, resume_path: str) -> Dict[str, any]:
        """
        Analyze resume for ATS compatibility.

        Args:
            resume_path: Path to resume file.

        Returns:
            ATS score, strengths, weaknesses, and suggestions.
        """
        logger.info("Analyzing resume: %s", resume_path)

        try:
            resume_doc = self.document_loader.load_text_file(resume_path)
            analysis = self.rag_engine.analyze_resume_ats_score(resume_doc["text"])

            logger.info("Resume analysis complete. ATS score: %s", analysis["ats_score"])
            return {
                "status": "success",
                "resume_file": resume_path,
                **analysis,
            }

        except Exception as e:
            logger.error("Resume analysis failed: %s", e)
            return {"status": "error", "message": str(e)}

    def get_database_stats(self) -> Dict[str, any]:
        """Get vector database statistics."""
        try:
            stats = self.vector_db.get_collection_stats()
            logger.info("Database stats: %s", stats)
            return stats
        except Exception as e:
            logger.error("Failed to get stats: %s", e)
            return {"status": "error", "message": str(e)}

    def clear_database(self) -> Dict[str, str]:
        """Clear all documents from database."""
        try:
            self.vector_db.clear_collection()
            self.embedding_manager.clear_cache()
            logger.info("Database and embedding cache cleared")
            return {"status": "success", "message": "Database cleared"}
        except Exception as e:
            logger.error("Failed to clear database: %s", e)
            return {"status": "error", "message": str(e)}
