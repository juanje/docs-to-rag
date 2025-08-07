"""
Complete RAG pipeline orchestrating retrieval and generation.

This module provides the main RAG pipeline that combines document retrieval
and answer generation into a seamless question-answering system.
"""

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

from src.config.settings import settings
from src.rag.generator import AnswerGenerator, GenerationResult
from src.rag.retriever import DocumentRetriever, RetrievalResult

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Complete result of a RAG pipeline execution."""

    question: str
    answer: str
    source_documents: list[str]
    relevance_scores: list[float]
    retrieval_time: float
    generation_time: float
    total_time: float
    context_length: int
    chunks_retrieved: int
    model_used: str


class RAGPipeline:
    """
    Complete RAG (Retrieval-Augmented Generation) pipeline.

    Orchestrates the entire process from query to answer, including
    document retrieval, context preparation, and answer generation.
    """

    def __init__(
        self,
        retriever: DocumentRetriever | None = None,
        generator: AnswerGenerator | None = None,
    ):
        """
        Initialize the RAG pipeline.

        Args:
            retriever: Document retriever instance (creates new if None)
            generator: Answer generator instance (creates new if None)
        """
        self.retriever = retriever or DocumentRetriever()
        self.generator = generator or AnswerGenerator()

        logger.info("RAG Pipeline initialized")

        # Verify components are ready
        self._check_system_readiness()

    def ask(self, question: str, **kwargs) -> RAGResult:
        """
        Process a question through the complete RAG pipeline.

        Args:
            question: User question
            **kwargs: Optional parameters (top_k, similarity_threshold, etc.)

        Returns:
            RAGResult with answer and metadata
        """
        if not question.strip():
            raise ValueError("Question cannot be empty")

        logger.info(f"Processing question: '{question[:50]}...'")
        start_time = time.time()

        # Step 1: Retrieve relevant documents
        retrieval_result = self.retriever.retrieve_relevant_documents(
            query=question,
            top_k=kwargs.get("top_k"),
            similarity_threshold=kwargs.get("similarity_threshold"),
        )

        # Step 2: Prepare context
        context = self.retriever.get_context_text(
            retrieval_result.relevant_chunks,
            max_length=kwargs.get("max_context_length"),
        )

        # Step 3: Generate answer
        generation_result = self.generator.generate_answer(question, context)

        total_time = time.time() - start_time

        # Step 4: Prepare result
        result = self._create_rag_result(
            question=question,
            retrieval_result=retrieval_result,
            generation_result=generation_result,
            total_time=total_time,
        )

        logger.info(f"RAG pipeline completed in {total_time:.2f}s")
        return result

    def ask_with_sources(self, question: str, **kwargs) -> dict[str, Any]:
        """
        Process a question and return detailed source information.

        Args:
            question: User question
            **kwargs: Optional parameters

        Returns:
            Dictionary with answer and detailed source information
        """
        result = self.ask(question, **kwargs)

        # Get detailed chunk information
        retrieval_result = self.retriever.retrieve_relevant_documents(question)
        chunk_details = []

        for i, chunk in enumerate(retrieval_result.relevant_chunks):
            chunk_details.append(
                {
                    "rank": i + 1,
                    "source_file": chunk.source_file,
                    "chunk_id": chunk.chunk_id,
                    "content_preview": chunk.content[:200] + "..."
                    if len(chunk.content) > 200
                    else chunk.content,
                    "relevance_score": retrieval_result.scores[i]
                    if i < len(retrieval_result.scores)
                    else 0.0,
                    "file_type": chunk.file_type,
                    "metadata": chunk.metadata,
                }
            )

        return {
            "answer": result.answer,
            "question": result.question,
            "chunks_used": chunk_details,
            "performance": {
                "total_time": result.total_time,
                "retrieval_time": result.retrieval_time,
                "generation_time": result.generation_time,
                "chunks_retrieved": result.chunks_retrieved,
                "context_length": result.context_length,
            },
            "model_info": {
                "chat_model": result.model_used,
                "embedding_model": self.retriever.embedding_generator.model_name,
            },
        }

    def batch_ask(self, questions: list[str], **kwargs) -> list[RAGResult]:
        """
        Process multiple questions through the RAG pipeline.

        Args:
            questions: List of user questions
            **kwargs: Optional parameters applied to all questions

        Returns:
            List of RAGResult objects
        """
        logger.info(f"Processing batch of {len(questions)} questions")

        results = []
        for i, question in enumerate(questions):
            try:
                logger.debug(f"Processing question {i + 1}/{len(questions)}")
                result = self.ask(question, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process question {i + 1}: {str(e)}")
                # Create error result
                error_result = RAGResult(
                    question=question,
                    answer=f"Error processing question: {str(e)}",
                    source_documents=[],
                    relevance_scores=[],
                    retrieval_time=0.0,
                    generation_time=0.0,
                    total_time=0.0,
                    context_length=0,
                    chunks_retrieved=0,
                    model_used=self.generator.model_name,
                )
                results.append(error_result)

        logger.info(f"Completed batch processing: {len(results)} results")
        return results

    def add_documents(self, documents: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Add new documents to the RAG system.

        Args:
            documents: List of document dictionaries from extractor

        Returns:
            Dictionary with processing results
        """
        from src.document_processor.chunker import TextChunker

        logger.info(f"Adding {len(documents)} documents to RAG system")

        # Step 0: Check for Spanish content and switch embedding model if needed
        spanish_documents = [doc for doc in documents if doc.get("is_spanish", False)]
        if spanish_documents and settings.auto_detect_language:
            spanish_ratio = len(spanish_documents) / len(documents)
            logger.info(
                f"Detected {len(spanish_documents)}/{len(documents)} Spanish documents ({spanish_ratio:.1%})"
            )

            # If majority is Spanish, switch to Spanish embeddings
            if spanish_ratio >= 0.5:
                from src.vector_store.embeddings import EmbeddingGenerator

                logger.info("Switching to Spanish embedding model for this batch")
                self.retriever.embedding_generator = EmbeddingGenerator(
                    force_multilingual=True
                )

        # Step 1: Chunk documents
        chunker = TextChunker()
        chunks = chunker.chunk_multiple_documents(documents)

        # Step 1.5: Generate hierarchical summaries if enabled
        if settings.enable_summarization:
            logger.info(
                "🧠 Generating hierarchical summaries for enhanced retrieval..."
            )
            from src.document_processor.summarizer import DocumentSummarizer

            summarizer = DocumentSummarizer()
            summary_count = 0

            for document in documents:
                try:
                    summary_chunks = summarizer.generate_all_summaries(document)
                    for summary_chunk in summary_chunks:
                        text_chunk = chunker.create_summary_chunk(summary_chunk)
                        chunks.append(text_chunk)
                        summary_count += 1
                except Exception as e:
                    logger.error(
                        f"Failed to generate summaries for {document.get('source_file', 'unknown')}: {str(e)}"
                    )

            logger.info(
                f"📚 Generated {summary_count} summary chunks ({len(chunks) - summary_count} original + {summary_count} summaries)"
            )

        if not chunks:
            logger.warning("No chunks created from documents")
            return {"success": False, "message": "No chunks created from documents"}

        # Step 2: Generate embeddings
        chunk_texts = [chunk.content for chunk in chunks]
        embedding_result = self.retriever.embedding_generator.generate_embeddings_sync(
            chunk_texts
        )

        # Step 3: Add to vector store
        self.retriever.add_documents_to_store(chunks, embedding_result.embeddings)

        # Step 4: Get statistics
        stats = self.get_system_stats()

        logger.info(
            f"Successfully added {len(chunks)} chunks from {len(documents)} documents"
        )

        return {
            "success": True,
            "documents_processed": len(documents),
            "chunks_created": len(chunks),
            "embedding_time": embedding_result.generation_time,
            "system_stats": stats,
        }

    def clear_knowledge_base(self) -> None:
        """Clear all documents from the knowledge base."""
        logger.info("Clearing knowledge base")
        self.retriever.clear_vector_store()
        logger.info("Knowledge base cleared")

    def get_system_stats(self) -> dict[str, Any]:
        """
        Get comprehensive system statistics.

        Returns:
            Dictionary with system statistics
        """
        retrieval_stats = self.retriever.get_retrieval_stats()
        generator_info = self.generator.get_model_info()

        return {
            "vector_store": retrieval_stats["vector_store"],
            "models": {
                "embedding": retrieval_stats["embedding_model"],
                "generation": generator_info,
            },
            "settings": {
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "top_k_retrieval": settings.top_k_retrieval,
                "similarity_threshold": settings.similarity_threshold,
                "temperature": settings.temperature,
                "max_tokens": settings.max_tokens_response,
            },
        }

    def check_readiness(self) -> dict[str, Any]:
        """
        Check if the RAG system is ready for queries.

        Returns:
            Dictionary with readiness status and any issues
        """
        # Check retriever
        retriever_ready, retriever_issues = self.retriever.check_system_ready()

        # Check generator
        generator_available, generator_status = (
            self.generator.check_model_availability()
        )

        all_issues = []
        if retriever_issues:
            all_issues.extend(retriever_issues)
        if not generator_available:
            all_issues.append(f"Generator model issue: {generator_status}")

        is_ready = retriever_ready and generator_available

        return {
            "is_ready": is_ready,
            "issues": all_issues,
            "components": {
                "retriever": {"ready": retriever_ready, "issues": retriever_issues},
                "generator": {
                    "available": generator_available,
                    "status": generator_status,
                },
            },
        }

    def _create_rag_result(
        self,
        question: str,
        retrieval_result: RetrievalResult,
        generation_result: GenerationResult,
        total_time: float,
    ) -> RAGResult:
        """Create a complete RAG result from component results."""
        # Extract source document names
        source_docs = list(
            {chunk.source_file for chunk in retrieval_result.relevant_chunks}
        )

        return RAGResult(
            question=question,
            answer=generation_result.answer,
            source_documents=source_docs,
            relevance_scores=retrieval_result.scores,
            retrieval_time=retrieval_result.retrieval_time,
            generation_time=generation_result.generation_time,
            total_time=total_time,
            context_length=len(generation_result.context_used),
            chunks_retrieved=len(retrieval_result.relevant_chunks),
            model_used=generation_result.model_used,
        )

    def _check_system_readiness(self) -> None:
        """Check if the system is ready and log any issues."""
        readiness = self.check_readiness()

        if readiness["is_ready"]:
            logger.info("RAG system is ready for queries")
        else:
            logger.warning("RAG system has issues:")
            for issue in readiness["issues"]:
                logger.warning(f"  - {issue}")

    def export_conversation(self, conversations: list[RAGResult]) -> dict[str, Any]:
        """
        Export conversation history for analysis or storage.

        Args:
            conversations: List of RAGResult objects

        Returns:
            Exportable conversation data
        """
        export_data = {
            "timestamp": time.time(),
            "system_info": self.get_system_stats(),
            "conversations": [],
        }

        for conv in conversations:
            export_data["conversations"].append(asdict(conv))

        return export_data
