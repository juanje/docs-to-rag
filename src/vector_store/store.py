"""
Vector storage and retrieval using FAISS.

This module provides efficient vector storage and similarity search
using FAISS (Facebook AI Similarity Search) for local vector operations.
"""

import json
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from src.config.settings import settings
from src.document_processor.chunker import TextChunk

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result of a vector similarity search."""

    chunk: TextChunk
    score: float
    rank: int


@dataclass
class VectorStoreStats:
    """Statistics about the vector store."""

    document_count: int
    chunk_count: int
    embedding_dimension: int
    db_size_mb: float
    index_type: str
    last_updated: str


class VectorStore:
    """
    FAISS-based vector store for embedding storage and retrieval.

    Provides efficient similarity search for RAG applications with
    complete local storage and no external dependencies.
    """

    def __init__(self, store_path: str | None = None):
        """
        Initialize the vector store.

        Args:
            store_path: Path to store the vector database
        """
        self.store_path = Path(store_path or settings.vector_db_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        # File paths
        self.index_path = self.store_path / "faiss_index.bin"
        self.chunks_path = self.store_path / "chunks.pkl"
        self.metadata_path = self.store_path / "metadata.json"

        # Internal state
        self.index: faiss.Index | None = None
        self.chunks: list[TextChunk] = []
        self.embedding_dimension: int | None = None
        self.is_loaded = False

        logger.info(f"VectorStore initialized at: {self.store_path}")

    def add_chunks(
        self, chunks: list[TextChunk], embeddings: list[list[float]]
    ) -> None:
        """
        Add text chunks and their embeddings to the vector store.

        Args:
            chunks: List of TextChunk objects
            embeddings: Corresponding embeddings for each chunk
        """
        if not chunks or not embeddings:
            logger.warning("No chunks or embeddings provided")
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch"
            )

        logger.info(f"Adding {len(chunks)} chunks to vector store")

        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Initialize index if needed
        if self.index is None:
            self.embedding_dimension = embeddings_array.shape[1]
            self._initialize_index(self.embedding_dimension)
            logger.info(
                f"Initialized FAISS index with dimension {self.embedding_dimension}"
            )

        # Validate embedding dimensions
        if embeddings_array.shape[1] != self.embedding_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dimension}, "
                f"got {embeddings_array.shape[1]}"
            )

        # Add to index
        self.index.add(embeddings_array)

        # Add chunks to our storage
        self.chunks.extend(chunks)

        logger.info(f"Added {len(chunks)} chunks. Total chunks: {len(self.chunks)}")

        # Verify summary chunks are properly indexed
        summary_count = sum(
            1
            for chunk in chunks
            if getattr(chunk, "metadata", {}).get("is_summary", False)
            or getattr(chunk, "chunk_type", "").endswith("_summary")
        )
        if summary_count > 0:
            logger.info(f"Indexed {summary_count} summary chunks successfully")

        # Verify FAISS indexing for summaries if any were added
        if summary_count > 0:
            self._verify_summary_indexing()

        # Save to disk
        self._save_to_disk()

    def search_similar(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[SearchResult]:
        """
        Search for similar chunks using vector similarity.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return

        Returns:
            List of SearchResult objects sorted by similarity score
        """
        if self.index is None or not self.chunks:
            logger.warning("Vector store is empty or not loaded")
            return []

        if len(query_embedding) != self.embedding_dimension:
            raise ValueError(
                f"Query embedding dimension ({len(query_embedding)}) doesn't match "
                f"store dimension ({self.embedding_dimension})"
            )

        # Convert to numpy array
        query_vector = np.array([query_embedding], dtype=np.float32)

        # Search in FAISS index
        scores, indices = self.index.search(query_vector, min(top_k, len(self.chunks)))

        # Convert results
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0], strict=False)):
            if idx >= 0 and idx < len(self.chunks):  # Valid index
                results.append(
                    SearchResult(chunk=self.chunks[idx], score=float(score), rank=rank)
                )

        logger.debug(f"Found {len(results)} similar chunks for query")
        return results

    def load_from_disk(self) -> bool:
        """
        Load vector store from disk.

        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            # Check if all required files exist
            if not all(
                path.exists()
                for path in [self.index_path, self.chunks_path, self.metadata_path]
            ):
                logger.info("Vector store files not found - starting with empty store")
                return False

            # Load metadata
            with open(self.metadata_path) as f:
                metadata = json.load(f)
                self.embedding_dimension = metadata.get("embedding_dimension")

            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))

            # Load chunks
            with open(self.chunks_path, "rb") as f:
                self.chunks = pickle.load(f)

            self.is_loaded = True
            logger.info(
                f"Loaded vector store: {len(self.chunks)} chunks, dimension {self.embedding_dimension}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}")
            return False

    def clear_all(self) -> None:
        """Clear all data from the vector store."""
        logger.info("Clearing all vector store data")

        # Clear in-memory data
        self.index = None
        self.chunks = []
        self.embedding_dimension = None
        self.is_loaded = False

        # Remove files from disk
        for file_path in [self.index_path, self.chunks_path, self.metadata_path]:
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Removed {file_path}")

        logger.info("Vector store cleared")

    def get_statistics(self) -> VectorStoreStats:
        """
        Get statistics about the vector store.

        Returns:
            VectorStoreStats with current store information
        """
        # Calculate database size
        db_size_bytes = 0
        for file_path in [self.index_path, self.chunks_path, self.metadata_path]:
            if file_path.exists():
                db_size_bytes += file_path.stat().st_size
        db_size_mb = db_size_bytes / (1024 * 1024)

        # Count unique documents
        unique_files = {chunk.source_file for chunk in self.chunks}

        # Get last update time
        last_updated = "Never"
        if self.metadata_path.exists():
            timestamp = self.metadata_path.stat().st_mtime
            last_updated = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

        # Determine index type
        index_type = "None"
        if self.index:
            index_type = type(self.index).__name__

        return VectorStoreStats(
            document_count=len(unique_files),
            chunk_count=len(self.chunks),
            embedding_dimension=self.embedding_dimension or 0,
            db_size_mb=db_size_mb,
            index_type=index_type,
            last_updated=last_updated,
        )

    def get_chunks_by_file(self, file_path: str) -> list[TextChunk]:
        """
        Get all chunks from a specific file.

        Args:
            file_path: Path to the source file

        Returns:
            List of chunks from that file
        """
        return [chunk for chunk in self.chunks if chunk.source_file == file_path]

    def get_all_source_files(self) -> list[str]:
        """
        Get list of all source files in the vector store.

        Returns:
            List of unique source file paths
        """
        return list({chunk.source_file for chunk in self.chunks})

    def _initialize_index(self, dimension: int) -> None:
        """
        Initialize FAISS index with appropriate settings.

        Args:
            dimension: Embedding vector dimension
        """
        # Use IndexFlatIP for inner product (cosine similarity)
        # This is simple and works well for most cases
        self.index = faiss.IndexFlatIP(dimension)
        self.embedding_dimension = dimension

        logger.debug(f"Initialized FAISS IndexFlatIP with dimension {dimension}")

    def _verify_summary_indexing(self) -> None:
        """Verify that summary chunks are properly indexed in FAISS."""
        if not self.index or not self.chunks:
            return

        # Find summary chunks using metadata
        summary_chunks = [
            chunk
            for chunk in self.chunks
            if (
                getattr(chunk, "metadata", {}).get("is_summary", False)
                or getattr(chunk, "chunk_type", "").endswith("_summary")
            )
        ]

        if not summary_chunks:
            return

        logger.info(
            f"Verifying FAISS indexing for {len(summary_chunks)} summary chunks..."
        )
        logger.info(f"FAISS index contains {self.index.ntotal} total vectors")

        # Basic verification - check that we have the right number of vectors
        expected_total = len(self.chunks)
        actual_total = self.index.ntotal

        if expected_total != actual_total:
            logger.error(
                f"FAISS indexing mismatch: expected {expected_total} vectors, got {actual_total}"
            )
        else:
            logger.info(
                f"FAISS indexing verification passed: {actual_total} vectors indexed correctly"
            )

    def _save_to_disk(self) -> None:
        """Save vector store to disk."""
        try:
            # Save FAISS index
            if self.index:
                faiss.write_index(self.index, str(self.index_path))

            # Save chunks
            with open(self.chunks_path, "wb") as f:
                pickle.dump(self.chunks, f)

            # Save metadata
            metadata = {
                "embedding_dimension": self.embedding_dimension,
                "chunk_count": len(self.chunks),
                "last_updated": time.time(),
                "version": "0.1.0",
            }
            with open(self.metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.debug("Vector store saved to disk")

        except Exception as e:
            logger.error(f"Failed to save vector store: {str(e)}")
            raise RuntimeError(f"Vector store save failed: {str(e)}") from e

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings for cosine similarity.

        Args:
            embeddings: Embedding vectors

        Returns:
            Normalized embeddings
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms
