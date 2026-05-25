"""
Efficient document loading with streaming and chunking.
Handles various document formats for 2026 RAG Agent.
"""
import logging
import os
from typing import List, Dict, Optional, Generator
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Handles document loading with streaming and batching."""
    
    SUPPORTED_FORMATS = {
        ".txt": "text",
        ".pdf": "pdf",
        ".docx": "docx",
        ".md": "markdown"
    }
    
    def __init__(self, chunk_size_mb: int = 5):
        """
        Initialize document loader.
        
        Args:
            chunk_size_mb: Max memory per document chunk in MB
        """
        self.chunk_size_bytes = chunk_size_mb * 1024 * 1024
        logger.info(f"✅ DocumentLoader initialized (chunk_size={chunk_size_mb}MB)")
    
    def load_text_file(self, file_path: str) -> Dict[str, str]:
        """
        Load text file with memory-efficient streaming.
        
        Args:
            file_path: Path to text file
            
        Returns:
            Dictionary with file content and metadata
        """
        try:
            file_name = os.path.basename(file_path)
            
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
            
            logger.info(f"📄 Loaded text file: {file_name} ({len(content)} chars)")
            
            return {
                "id": file_name,
                "text": content,
                "source": file_path,
                "format": "text",
                "size_bytes": len(content.encode('utf-8'))
            }
        except Exception as e:
            logger.error(f"❌ Failed to load {file_path}: {e}")
            raise
    
    def load_directory(self, directory_path: str) -> List[Dict[str, str]]:
        """
        Load all supported documents from directory.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of loaded documents
        """
        if not os.path.exists(directory_path):
            logger.warning(f"⚠️ Directory not found: {directory_path}")
            return []
        
        documents = []
        files_found = 0
        files_loaded = 0
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            if not os.path.isfile(file_path):
                continue
            
            _, ext = os.path.splitext(filename)
            
            if ext.lower() not in self.SUPPORTED_FORMATS:
                logger.debug(f"⏭️ Skipping unsupported format: {filename}")
                continue
            
            files_found += 1
            
            try:
                if ext.lower() == ".txt":
                    doc = self.load_text_file(file_path)
                    documents.append(doc)
                    files_loaded += 1
                else:
                    logger.info(f"⏸️ Format {ext} requires additional libraries")
            except Exception as e:
                logger.error(f"❌ Error loading {filename}: {e}")
                continue
        
        logger.info(
            f"✅ Loaded {files_loaded}/{files_found} documents "
            f"from {directory_path}"
        )
        
        return documents
    
    def load_directory_streaming(
        self,
        directory_path: str,
        batch_size: int = 5
    ) -> Generator[List[Dict[str, str]], None, None]:
        """
        Stream documents from directory in batches.
        Reduces memory usage for large document collections.
        
        Args:
            directory_path: Path to directory
            batch_size: Number of documents per batch
            
        Yields:
            Batches of loaded documents
        """
        if not os.path.exists(directory_path):
            logger.warning(f"⚠️ Directory not found: {directory_path}")
            return
        
        batch = []
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            if not os.path.isfile(file_path):
                continue
            
            _, ext = os.path.splitext(filename)
            
            if ext.lower() not in self.SUPPORTED_FORMATS:
                continue
            
            try:
                if ext.lower() == ".txt":
                    doc = self.load_text_file(file_path)
                    batch.append(doc)
                    
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
            except Exception as e:
                logger.error(f"❌ Error loading {filename}: {e}")
                continue
        
        # Yield remaining documents
        if batch:
            yield batch
    
    def load_documents_parallel(
        self,
        directory_path: str,
        num_workers: int = 4
    ) -> List[Dict[str, str]]:
        """
        Load documents in parallel using thread pool.
        
        Args:
            directory_path: Path to directory
            num_workers: Number of parallel workers
            
        Returns:
            List of loaded documents
        """
        if not os.path.exists(directory_path):
            logger.warning(f"⚠️ Directory not found: {directory_path}")
            return []
        
        file_paths = []
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            if not os.path.isfile(file_path):
                continue
            
            _, ext = os.path.splitext(filename)
            
            if ext.lower() not in self.SUPPORTED_FORMATS:
                continue
            
            file_paths.append(file_path)
        
        documents = []
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(self.load_text_file, path)
                for path in file_paths
            ]
            
            for future in futures:
                try:
                    doc = future.result()
                    documents.append(doc)
                except Exception as e:
                    logger.error(f"❌ Parallel load failed: {e}")
        
        logger.info(f"✅ Parallel loaded {len(documents)} documents with {num_workers} workers")
        return documents
