"""Semantic text chunking utilities."""

import logging
from typing import Dict, List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class SemanticChunker:
    """Handles document chunking with semantic-aware separators."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        """
        Initialize semantic chunker.

        Args:
            chunk_size: Target size of each chunk.
            chunk_overlap: Overlap between chunks for context.
            separators: Custom separators for splitting.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        default_separators = [
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ]

        self.separators = separators or default_separators
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

        logger.info("Semantic chunker initialized with size %s and overlap %s", chunk_size, chunk_overlap)

    def chunk_text(self, text: str, doc_id: str = "document") -> List[Dict[str, str]]:
        """
        Split text into semantically meaningful chunks.

        Args:
            text: Text to chunk.
            doc_id: Document identifier for tracking.

        Returns:
            List of chunk dictionaries with id and text.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for document '%s'", doc_id)
            return []

        text = " ".join(text.split())
        chunks = self.splitter.split_text(text)

        chunked_docs = []
        valid_chunks = 0

        for i, chunk in enumerate(chunks, 1):
            if chunk.strip():
                chunked_docs.append({
                    "id": f"{doc_id}_chunk_{i}",
                    "text": chunk,
                    "chunk_index": i,
                    "original_doc_id": doc_id,
                })
                valid_chunks += 1

        logger.info(
            "Document '%s' split into %s chunks; removed %s empty chunks",
            doc_id,
            valid_chunks,
            len(chunks) - valid_chunks,
        )

        return chunked_docs

    def chunk_documents(
        self,
        documents: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        Split multiple documents into chunks.

        Args:
            documents: List of document dictionaries with id and text.

        Returns:
            Flattened list of all chunks.
        """
        all_chunks = []

        for doc in documents:
            doc_id = doc.get("id", "unknown")
            text = doc.get("text", "")

            chunks = self.chunk_text(text, doc_id)
            all_chunks.extend(chunks)

        logger.info("Processed %s documents into %s chunks", len(documents), len(all_chunks))
        return all_chunks

    def get_chunk_statistics(self, chunks: List[Dict[str, str]]) -> Dict:
        """
        Analyze chunk statistics.

        Args:
            chunks: List of chunks.

        Returns:
            Dictionary with statistical information.
        """
        if not chunks:
            return {"total_chunks": 0, "avg_size": 0, "max_size": 0, "min_size": 0}

        sizes = [len(chunk["text"]) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_size": sum(sizes) / len(sizes),
            "max_size": max(sizes),
            "min_size": min(sizes),
            "total_tokens_estimate": sum(sizes) // 4,
        }
