"""
Embedding generation using Ollama local models.

This module provides embedding generation functionality using Ollama's
local embedding models for converting text to vector representations.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx
from langchain_ollama import OllamaEmbeddings

from src.config.settings import settings

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""

    embeddings: list[list[float]]
    texts: list[str]
    model_used: str
    generation_time: float
    total_tokens: int

    @property
    def dimension(self) -> int:
        """Get the dimension of the embeddings."""
        if not self.embeddings:
            return 0
        return len(self.embeddings[0]) if self.embeddings[0] else 0


class EmbeddingGenerator:
    """
    Local embedding generation using Ollama.

    Handles conversion of text chunks to vector embeddings using
    local Ollama models for complete privacy and control.
    """

    def __init__(self, model_name: str | None = None, force_multilingual: bool = False):
        """
        Initialize the embedding generator.

        Args:
            model_name: Name of the Ollama embedding model to use
            force_multilingual: Force use of multilingual model
        """
        # Choose model based on language settings
        if force_multilingual or (
            settings.auto_detect_language and settings.primary_language != "en"
        ):
            self.model_name = model_name or settings.embedding_model_multilingual
            self.is_multilingual = True
        else:
            self.model_name = model_name or settings.embedding_model
            self.is_multilingual = False

        self.base_url = settings.ollama_base_url
        self.timeout = settings.ollama_timeout

        # Initialize Ollama embeddings
        self.embeddings = OllamaEmbeddings(
            model=self.model_name, base_url=self.base_url
        )

        logger.info(
            f"EmbeddingGenerator initialized with model: {self.model_name} "
            f"(multilingual: {self.is_multilingual})"
        )

    async def generate_embeddings(self, texts: list[str]) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            EmbeddingResult with embeddings and metadata
        """
        if not texts:
            logger.warning("No texts provided for embedding generation")
            return EmbeddingResult([], [], self.model_name, 0.0, 0)

        logger.info(f"Generating embeddings for {len(texts)} texts")
        start_time = time.time()

        try:
            # Generate embeddings using Ollama
            embeddings = await self.embeddings.aembed_documents(texts)

            generation_time = time.time() - start_time
            total_tokens = sum(len(text.split()) for text in texts)  # Rough token count

            logger.info(
                f"Generated {len(embeddings)} embeddings in {generation_time:.2f}s "
                f"(~{total_tokens} tokens)"
            )

            return EmbeddingResult(
                embeddings=embeddings,
                texts=texts,
                model_used=self.model_name,
                generation_time=generation_time,
                total_tokens=total_tokens,
            )

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise RuntimeError(f"Embedding generation failed: {str(e)}") from e

    def generate_embeddings_sync(self, texts: list[str]) -> EmbeddingResult:
        """
        Synchronous wrapper for embedding generation.

        Args:
            texts: List of text strings to embed

        Returns:
            EmbeddingResult with embeddings and metadata
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.generate_embeddings(texts))

    async def generate_query_embedding(self, query: str) -> list[float]:
        """
        Generate embedding for a single query text.

        Args:
            query: Query string to embed

        Returns:
            Embedding vector as list of floats
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        logger.debug(f"Generating query embedding for: '{query[:50]}...'")

        try:
            embedding = await self.embeddings.aembed_query(query)
            logger.debug(f"Generated query embedding with {len(embedding)} dimensions")
            return embedding

        except Exception as e:
            logger.error(f"Failed to generate query embedding: {str(e)}")
            raise RuntimeError(f"Query embedding generation failed: {str(e)}") from e

    def generate_query_embedding_sync(self, query: str) -> list[float]:
        """
        Synchronous wrapper for query embedding generation.

        Args:
            query: Query string to embed

        Returns:
            Embedding vector as list of floats
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.generate_query_embedding(query))

    def check_ollama_connection(self) -> tuple[bool, str]:
        """
        Check if Ollama is accessible and the model is available.

        Returns:
            Tuple of (is_available, status_message)
        """
        try:
            # Check if Ollama is running
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{self.base_url}/api/tags")

                if response.status_code != 200:
                    return False, f"Ollama API returned status {response.status_code}"

                # Check if our model is available
                models_data = response.json()
                available_models = [
                    model["name"] for model in models_data.get("models", [])
                ]

                if self.model_name not in available_models:
                    return (
                        False,
                        f"Model {self.model_name} not found. Available: {available_models}",
                    )

                return True, f"Ollama connection OK, model {self.model_name} available"

        except httpx.ConnectError:
            return False, f"Cannot connect to Ollama at {self.base_url}"
        except Exception as e:
            return False, f"Error checking Ollama: {str(e)}"

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the current embedding model.

        Returns:
            Dictionary with model information
        """
        is_available, status = self.check_ollama_connection()

        info = {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "is_available": is_available,
            "status": status,
        }

        if is_available:
            try:
                # Try to get model details from Ollama
                with httpx.Client(timeout=10.0) as client:
                    response = client.post(
                        f"{self.base_url}/api/show", json={"name": self.model_name}
                    )
                    if response.status_code == 200:
                        model_data = response.json()
                        info.update(
                            {
                                "model_details": model_data.get("details", {}),
                                "model_size": model_data.get("size", "unknown"),
                            }
                        )
            except Exception as e:
                logger.debug(f"Could not get detailed model info: {e}")

        return info

    def get_embedding_dimension(self) -> int | None:
        """
        Get the dimension of embeddings produced by the model.

        Returns:
            Embedding dimension or None if cannot be determined
        """
        try:
            # Generate a test embedding to determine dimension
            test_embedding = self.generate_query_embedding_sync("test")
            return len(test_embedding)
        except Exception as e:
            logger.warning(f"Could not determine embedding dimension: {e}")
            return None
