"""
Document retrieval component for RAG pipeline.

This module handles the retrieval of relevant documents from the vector store
based on query similarity and provides context for answer generation.
"""

import logging
from dataclasses import dataclass
from typing import Any

from src.config.settings import settings
from src.document_processor.chunker import TextChunk
from src.vector_store import EmbeddingGenerator, VectorStore

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result of document retrieval."""

    query: str
    relevant_chunks: list[TextChunk]
    scores: list[float]
    retrieval_time: float
    total_retrieved: int


class DocumentRetriever:
    """
    Document retrieval system for RAG pipeline.

    Handles the retrieval of relevant document chunks based on query
    similarity using vector embeddings and FAISS search.
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
    ):
        """
        Initialize the document retriever.

        Args:
            vector_store: Vector store instance (creates new if None)
            embedding_generator: Embedding generator instance (creates new if None)
        """
        self.vector_store = vector_store or VectorStore()
        self.embedding_generator = embedding_generator or EmbeddingGenerator()

        # Load existing vector store if available
        if not self.vector_store.is_loaded:
            loaded = self.vector_store.load_from_disk()
            if loaded:
                logger.info("Loaded existing vector store")
            else:
                logger.info("Starting with empty vector store")

        logger.info("DocumentRetriever initialized")

    def retrieve_relevant_documents(
        self,
        query: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a given query.

        Args:
            query: User query string
            top_k: Number of documents to retrieve (uses config default if None)
            similarity_threshold: Minimum similarity score (uses config default if None)

        Returns:
            RetrievalResult with relevant chunks and metadata
        """
        import time

        if not query.strip():
            raise ValueError("Query cannot be empty")

        # Use adaptive parameters if not explicitly provided
        if top_k is None or similarity_threshold is None:
            total_chunks = (
                len(self.vector_store.chunks) if self.vector_store.chunks else 0
            )
            adaptive_top_k, adaptive_threshold = settings.get_adaptive_params(
                total_chunks
            )

            top_k = top_k or adaptive_top_k
            similarity_threshold = similarity_threshold or adaptive_threshold

            logger.info(
                f"Using adaptive parameters: top_k={top_k}, "
                f"threshold={similarity_threshold:.3f} (total_chunks={total_chunks})"
            )

        logger.info(f"Retrieving documents for query: '{query[:50]}...'")
        start_time = time.time()

        # Check if query is in Spanish and switch embedding model if needed
        original_generator = None
        if settings.auto_detect_language and settings.detect_spanish_content(query):
            logger.info("Detected Spanish query - using Spanish embedding model")
            from src.vector_store.embeddings import EmbeddingGenerator

            original_generator = self.embedding_generator
            self.embedding_generator = EmbeddingGenerator(force_multilingual=True)

        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_query_embedding_sync(
                query
            )

            # Search for similar documents
            search_results = self.vector_store.search_similar(query_embedding, top_k)

            # Filter by similarity threshold
            filtered_results = [
                result
                for result in search_results
                if result.score >= similarity_threshold
            ]

            retrieval_time = time.time() - start_time

            # Extract chunks and scores
            relevant_chunks = [result.chunk for result in filtered_results]
            scores = [result.score for result in filtered_results]

            logger.info(
                f"Retrieved {len(relevant_chunks)} relevant documents "
                f"in {retrieval_time:.2f}s (threshold: {similarity_threshold})"
            )

            # Apply hybrid retrieval strategy if summaries are available
            if settings.enable_summarization and relevant_chunks:
                relevant_chunks, scores = self._apply_hybrid_retrieval_strategy(
                    query, relevant_chunks, scores
                )

            # Debug: Show retrieved chunks if debug mode is enabled
            if settings.debug_mode and relevant_chunks:
                logger.info("=== DEBUG: Top retrieved chunks ===")
                for i, chunk in enumerate(relevant_chunks[:3]):  # Show top 3
                    score = scores[i] if i < len(scores) else "N/A"
                    chunk_type = getattr(chunk, "chunk_type", "original_text")
                    preview = chunk.content[:200].replace("\n", " ").strip()
                    logger.info(
                        f"Chunk {i + 1} [type:{chunk_type}] (score: {score:.3f}): {preview}..."
                    )
                logger.info("=== End debug chunks ===")

            return RetrievalResult(
                query=query,
                relevant_chunks=relevant_chunks,
                scores=scores,
                retrieval_time=retrieval_time,
                total_retrieved=len(search_results),
            )

        except Exception as e:
            logger.error(f"Document retrieval failed: {str(e)}")
            raise RuntimeError(f"Retrieval failed: {str(e)}") from e
        finally:
            # Restore original embedding generator if we switched it
            if original_generator is not None:
                self.embedding_generator = original_generator

    def get_context_text(
        self, chunks: list[TextChunk], max_length: int | None = None
    ) -> str:
        """
        Convert retrieved chunks into context text for generation.

        Args:
            chunks: List of relevant text chunks
            max_length: Maximum length of context text

        Returns:
            Formatted context text
        """
        if not chunks:
            return ""

        # Sort chunks by source file and position for better coherence
        sorted_chunks = sorted(chunks, key=lambda c: (c.source_file, c.start_pos))

        context_parts = []
        current_length = 0
        max_length = max_length or (settings.chunk_size * settings.top_k_retrieval * 2)

        for i, chunk in enumerate(sorted_chunks):
            # Add source information for first chunk from each file
            if i == 0 or chunk.source_file != sorted_chunks[i - 1].source_file:
                source_name = self._get_display_filename(chunk.source_file)
                header = f"\n--- From {source_name} ---\n"
                context_parts.append(header)
                current_length += len(header)

            # Check if adding this chunk would exceed max length
            chunk_text = chunk.content.strip()
            if current_length + len(chunk_text) > max_length and context_parts:
                logger.debug(f"Context truncated at {current_length} characters")
                break

            context_parts.append(chunk_text)
            current_length += len(chunk_text)

            # Add separator between chunks from same file
            if (
                i < len(sorted_chunks) - 1
                and chunk.source_file == sorted_chunks[i + 1].source_file
            ):
                separator = "\n\n"
                context_parts.append(separator)
                current_length += len(separator)

        context = "".join(context_parts).strip()
        logger.debug(
            f"Generated context text: {len(context)} characters from {len(chunks)} chunks"
        )

        return context

    def get_retrieval_stats(self) -> dict[str, Any]:
        """
        Get statistics about the retrieval system.

        Returns:
            Dictionary with retrieval system statistics
        """
        vector_stats = self.vector_store.get_statistics()
        embedding_info = self.embedding_generator.get_model_info()

        return {
            "vector_store": {
                "document_count": vector_stats.document_count,
                "chunk_count": vector_stats.chunk_count,
                "embedding_dimension": vector_stats.embedding_dimension,
                "db_size_mb": vector_stats.db_size_mb,
                "last_updated": vector_stats.last_updated,
            },
            "embedding_model": {
                "model_name": embedding_info["model_name"],
                "is_available": embedding_info["is_available"],
                "status": embedding_info["status"],
            },
            "retrieval_settings": {
                "top_k_retrieval": settings.top_k_retrieval,
                "similarity_threshold": settings.similarity_threshold,
            },
        }

    def check_system_ready(self) -> tuple[bool, list[str]]:
        """
        Check if the retrieval system is ready for queries.

        Returns:
            Tuple of (is_ready, list_of_issues)
        """
        issues = []

        # Check if vector store has data
        if not self.vector_store.chunks:
            issues.append("Vector store is empty - no documents indexed")

        # Check embedding model availability
        is_available, status = self.embedding_generator.check_ollama_connection()
        if not is_available:
            issues.append(f"Embedding model not available: {status}")

        # Check vector store consistency
        stats = self.vector_store.get_statistics()
        if stats.chunk_count != len(self.vector_store.chunks):
            issues.append("Vector store consistency issue detected")

        is_ready = len(issues) == 0
        return is_ready, issues

    def _get_display_filename(self, file_path: str) -> str:
        """Get a display-friendly filename from a path."""
        from pathlib import Path

        return Path(file_path).name

    def add_documents_to_store(
        self, chunks: list[TextChunk], embeddings: list[list[float]]
    ) -> None:
        """
        Add new documents to the vector store.

        Args:
            chunks: Text chunks to add
            embeddings: Corresponding embeddings
        """
        if not chunks or not embeddings:
            logger.warning("No chunks or embeddings provided for adding to store")
            return

        logger.info(f"Adding {len(chunks)} chunks to vector store")
        self.vector_store.add_chunks(chunks, embeddings)
        logger.info("Documents successfully added to vector store")

    def clear_vector_store(self) -> None:
        """Clear all documents from the vector store."""
        logger.info("Clearing vector store")
        self.vector_store.clear_all()
        logger.info("Vector store cleared")

    def _apply_hybrid_retrieval_strategy(
        self, query: str, chunks: list, scores: list[float]
    ) -> tuple[list, list[float]]:
        """
        Apply hybrid retrieval strategy that balances summaries and original content.

        Args:
            query: User query string
            chunks: Retrieved chunks
            scores: Similarity scores

        Returns:
            Tuple of reordered chunks and scores
        """
        if not chunks:
            return chunks, scores

        # Detect query type based on keywords
        query_lower = query.lower()

        # Keywords indicating need for summary/overview
        summary_keywords = {
            "es": [
                "resumen",
                "resúmen",
                "qué es",
                "cuál es",
                "idea principal",
                "tesis",
                "concepto",
                "explicar",
                "sobre qué",
                "de qué trata",
                "tema principal",
            ],
            "en": [
                "summary",
                "what is",
                "main idea",
                "thesis",
                "concept",
                "explain",
                "what about",
                "overview",
                "general",
                "overall",
            ],
        }

        # Keywords indicating need for specific details
        detail_keywords = {
            "es": [
                "cuánto",
                "qué porcentaje",
                "específicamente",
                "exactamente",
                "datos",
                "números",
                "mediciones",
                "paso a paso",
            ],
            "en": [
                "how much",
                "what percentage",
                "specifically",
                "exactly",
                "data",
                "numbers",
                "measurements",
                "step by step",
            ],
        }

        # Determine query preference
        wants_summary = any(
            kw in query_lower for kw_list in summary_keywords.values() for kw in kw_list
        )
        wants_details = any(
            kw in query_lower for kw_list in detail_keywords.values() for kw in kw_list
        )

        # FOR CONCEPTUAL QUERIES: Force inclusion of relevant summaries
        if wants_summary and not wants_details:
            # Get all summaries from the vector store that might be relevant
            all_summaries = self._get_relevant_summaries(query)
            if all_summaries:
                logger.info(
                    f"Found {len(all_summaries)} potentially relevant summaries for conceptual query"
                )

                # Combine FAISS results with forced summaries
                combined_chunks = all_summaries + chunks
                combined_scores = [0.85] * len(
                    all_summaries
                ) + scores  # Give summaries high artificial scores

                # Remove duplicates while preserving order
                seen_ids = set()
                final_chunks = []
                final_scores = []

                for chunk, score in zip(combined_chunks, combined_scores, strict=False):
                    chunk_id = getattr(chunk, "chunk_id", id(chunk))
                    if chunk_id not in seen_ids:
                        seen_ids.add(chunk_id)
                        final_chunks.append(chunk)
                        final_scores.append(score)
                        if len(final_chunks) >= len(chunks):
                            break

                # Log the hybrid composition
                final_count = min(len(chunks), len(final_chunks))
                summary_count = sum(
                    1
                    for chunk in final_chunks[:final_count]
                    if self._is_summary_chunk(chunk)
                )
                logger.info(
                    f"Forced hybrid strategy: {summary_count} summaries + {final_count - summary_count} original chunks"
                )

                return final_chunks[:final_count], final_scores[:final_count]

        # FALLBACK: Regular separation for non-conceptual or when no summaries found
        summary_chunks = []
        original_chunks = []
        summary_scores = []
        original_scores = []

        for i, chunk in enumerate(chunks):
            score = scores[i] if i < len(scores) else 0.0

            if self._is_summary_chunk(chunk):
                summary_chunks.append(chunk)
                summary_scores.append(score)
            else:
                original_chunks.append(chunk)
                original_scores.append(score)

        # Apply strategy based on query type
        if wants_summary and not wants_details:
            # AGGRESSIVE: Prioritize summaries heavily for conceptual questions
            logger.debug(
                "Query detected as conceptual - prioritizing summaries aggressively"
            )

            if summary_chunks:
                # Force summaries to the front regardless of similarity scores
                # Take ALL available summaries first, then fill with original content
                max_summaries = len(summary_chunks)  # Use ALL summaries available
                max_originals = max(0, len(chunks) - max_summaries)

                final_chunks = (
                    summary_chunks[:max_summaries] + original_chunks[:max_originals]
                )
                final_scores = (
                    summary_scores[:max_summaries] + original_scores[:max_originals]
                )

                logger.info(
                    f"Conceptual query boost: using {max_summaries} summaries + {max_originals} original chunks"
                )
            else:
                # No summaries available, use original content
                final_chunks = original_chunks[: len(chunks)]
                final_scores = original_scores[: len(chunks)]

        elif wants_details and not wants_summary:
            # Prioritize original content for specific questions
            logger.debug("Query detected as specific - prioritizing original content")

            # Take mostly original + one summary for context
            max_originals = min(
                len(original_chunks), max(len(chunks) - 1, len(chunks) * 3 // 4)
            )
            max_summaries = max(0, len(chunks) - max_originals)

            final_chunks = (
                original_chunks[:max_originals] + summary_chunks[:max_summaries]
            )
            final_scores = (
                original_scores[:max_originals] + summary_scores[:max_summaries]
            )

        else:
            # Summary-favored balanced approach
            logger.debug("Query detected as mixed - using summary-favored approach")

            if summary_chunks:
                # Include at least 50% summaries when available
                max_summaries = min(len(summary_chunks), max(1, len(chunks) // 2))
                max_originals = len(chunks) - max_summaries

                # Put summaries first to ensure they're included
                final_chunks = (
                    summary_chunks[:max_summaries] + original_chunks[:max_originals]
                )
                final_scores = (
                    summary_scores[:max_summaries] + original_scores[:max_originals]
                )

                logger.info(
                    f"Mixed query: using {max_summaries} summaries + {max_originals} original chunks"
                )
            else:
                final_chunks = original_chunks[: len(chunks)]
                final_scores = original_scores[: len(chunks)]

        if settings.debug_mode:
            summary_count = sum(
                1
                for chunk in final_chunks
                if "summary" in getattr(chunk, "chunk_type", "")
            )
            logger.info(
                f"Hybrid strategy: {summary_count} summaries + {len(final_chunks) - summary_count} original chunks"
            )

        return final_chunks, final_scores

    def _is_summary_chunk(self, chunk) -> bool:
        """Check if a chunk is a summary based on metadata and type."""
        chunk_type = getattr(chunk, "chunk_type", "original_text")
        metadata = getattr(chunk, "metadata", {})

        return (
            "summary" in chunk_type
            or metadata.get("is_summary", False)
            or metadata.get("generated_by") == "llm_summarizer"
            or chunk_type.endswith("_summary")
        )

    def _get_relevant_summaries(self, query: str) -> list:
        """Get summaries that might be relevant to the query by searching all summaries."""
        all_chunks = self.vector_store.chunks
        query_lower = query.lower()

        # Find all summary chunks
        summary_chunks = [
            chunk for chunk in all_chunks if self._is_summary_chunk(chunk)
        ]

        if not summary_chunks:
            return []

        # Simple relevance scoring based on content matching
        relevant_summaries = []

        for chunk in summary_chunks:
            content_lower = chunk.content.lower()

            # Extract key terms from query (clean and filter)
            import re

            # Remove punctuation and clean terms
            clean_query = re.sub(r"[¿?¡!.,;:]", "", query_lower)
            all_terms = clean_query.split()

            # Filter out common words and short terms
            stop_words = {
                "qué",
                "cuál",
                "cómo",
                "dónde",
                "cuándo",
                "por",
                "para",
                "que",
                "es",
                "el",
                "la",
                "los",
                "las",
                "un",
                "una",
                "del",
                "de",
                "al",
                "en",
                "con",
                "what",
                "how",
                "where",
                "when",
                "why",
                "is",
                "the",
                "of",
                "in",
                "and",
                "or",
            }
            query_terms = [
                term for term in all_terms if len(term) > 2 and term not in stop_words
            ]

            # Check if any query terms appear in the summary
            relevance_score = sum(1 for term in query_terms if term in content_lower)

            if relevance_score > 0:
                relevant_summaries.append(chunk)

        logger.info(
            f"Query terms: {query_terms}, found {len(relevant_summaries)} relevant summaries"
        )
        return relevant_summaries

    def get_source_files(self) -> list[str]:
        """
        Get list of all source files in the vector store.

        Returns:
            List of source file paths
        """
        return self.vector_store.get_all_source_files()
