"""
RAG (Retrieval-Augmented Generation) module.

This module implements the complete RAG pipeline including document retrieval
and answer generation using local Ollama models.
"""

from .generator import AnswerGenerator
from .pipeline import RAGPipeline
from .retriever import DocumentRetriever

__all__ = ["DocumentRetriever", "AnswerGenerator", "RAGPipeline"]
