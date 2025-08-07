"""
Vector store module for docs-to-rag.

This module handles embedding generation and vector similarity search
using Ollama for embeddings and FAISS for efficient storage and retrieval.
"""

from .embeddings import EmbeddingGenerator
from .store import VectorStore

__all__ = ["EmbeddingGenerator", "VectorStore"]
