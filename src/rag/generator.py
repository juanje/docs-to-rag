"""
Answer generation component for RAG pipeline.

This module handles answer generation using Ollama local models with
retrieved document context for accurate and contextual responses.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

from langchain.schema import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.config.settings import settings

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of answer generation."""

    query: str
    answer: str
    context_used: str
    model_used: str
    generation_time: float
    token_count: int


class AnswerGenerator:
    """
    Answer generation system using Ollama local models.

    Generates contextual answers based on retrieved documents using
    local LLMs for complete privacy and control.
    """

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ):
        """
        Initialize the answer generator.

        Args:
            model_name: Name of the Ollama chat model to use
            temperature: Temperature for text generation (0.0-1.0)
            top_p: Top-p sampling parameter (0.0-1.0)
        """
        self.model_name = model_name or settings.chat_model
        self.base_url = settings.ollama_base_url
        self.temperature = (
            temperature if temperature is not None else settings.temperature
        )
        self.top_p = top_p if top_p is not None else settings.top_p
        self.max_tokens = settings.max_tokens_response

        # Initialize Ollama chat model
        self.llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature,
            num_predict=self.max_tokens,  # Max tokens to generate
            top_p=self.top_p,
        )

        logger.info(
            f"AnswerGenerator initialized with model: {self.model_name} "
            f"(temp: {self.temperature}, top_p: {self.top_p})"
        )

    def generate_response(self, prompt: str, max_tokens: int | None = None) -> str:
        """
        Generate a response to a prompt (used for summarization).

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Generated response text
        """
        import time

        start_time = time.time()

        try:
            # Use instance-specific parameters or provided max_tokens
            effective_max_tokens = max_tokens or self.max_tokens

            # Create a temporary LLM with specific parameters if needed
            if max_tokens and max_tokens != self.max_tokens:
                temp_llm = ChatOllama(
                    model=self.model_name,
                    base_url=self.base_url,
                    temperature=self.temperature,
                    num_predict=effective_max_tokens,
                    top_p=self.top_p,
                )
                response = temp_llm.invoke([HumanMessage(content=prompt)])
            else:
                response = self.llm.invoke([HumanMessage(content=prompt)])

            generation_time = time.time() - start_time

            # Extract text content
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )

            logger.debug(
                f"Generated response in {generation_time:.2f}s (~{len(response_text)} chars)"
            )

            return response_text.strip()

        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            raise RuntimeError(f"Failed to generate response: {str(e)}") from e

    def generate_answer(self, query: str, context: str) -> GenerationResult:
        """
        Generate an answer based on query and retrieved context.

        Args:
            query: User query
            context: Retrieved document context

        Returns:
            GenerationResult with generated answer and metadata
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        logger.info(f"Generating answer for query: '{query[:50]}...'")
        start_time = time.time()

        try:
            # Build the prompt
            prompt = self._build_rag_prompt(query, context)

            # Create messages
            messages = [
                SystemMessage(content=self._get_system_prompt()),
                HumanMessage(content=prompt),
            ]

            # Generate response
            response = self.llm.invoke(messages)
            answer = response.content.strip()

            generation_time = time.time() - start_time

            # Rough token count estimation
            token_count = len(answer.split())

            logger.info(
                f"Generated answer in {generation_time:.2f}s (~{token_count} tokens)"
            )

            return GenerationResult(
                query=query,
                answer=answer,
                context_used=context,
                model_used=self.model_name,
                generation_time=generation_time,
                token_count=token_count,
            )

        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            raise RuntimeError(f"Generation failed: {str(e)}") from e

    def _get_system_prompt(self) -> str:
        """
        Get the system prompt for the RAG model.

        Returns:
            System prompt string
        """
        return """You are a helpful AI assistant that answers questions based on provided context.

Guidelines:
- Answer questions using ONLY the information provided in the context
- If the context doesn't contain enough information to answer the question, say so clearly
- Be concise but comprehensive in your responses
- Cite specific parts of the context when relevant
- If the question is unclear, ask for clarification
- Maintain a helpful and professional tone
- Format your response clearly and logically

Important:
- Do NOT make up information that isn't in the context
- Do NOT use knowledge from outside the provided context
- If you're unsure about something, acknowledge the uncertainty"""

    def _build_rag_prompt(self, query: str, context: str) -> str:
        """
        Build the RAG prompt combining query and context.

        Args:
            query: User query
            context: Retrieved document context

        Returns:
            Formatted prompt string
        """
        if not context.strip():
            return f"""I have a question but no relevant context was found in the documents.

Question: {query}

Please let me know that you don't have enough information to answer this question based on the available documents."""

        return f"""Please answer the following question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""

    def generate_summary(self, text: str, max_length: int | None = None) -> str:
        """
        Generate a summary of the provided text.

        Args:
            text: Text to summarize
            max_length: Maximum length of summary

        Returns:
            Generated summary
        """
        if not text.strip():
            return "No content to summarize."

        max_length = max_length or 200

        prompt = f"""Please provide a concise summary of the following text in no more than {max_length} words:

{text}

Summary:"""

        try:
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            return response.content.strip()

        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            return f"Error generating summary: {str(e)}"

    def check_model_availability(self) -> tuple[bool, str]:
        """
        Check if the chat model is available.

        Returns:
            Tuple of (is_available, status_message)
        """
        try:
            # Try a simple test generation
            test_messages = [HumanMessage(content="Hello, please respond with 'OK'")]
            response = self.llm.invoke(test_messages)

            if response and response.content:
                return True, f"Model {self.model_name} is working correctly"
            else:
                return False, f"Model {self.model_name} returned empty response"

        except Exception as e:
            return False, f"Model {self.model_name} error: {str(e)}"

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the current generation model.

        Returns:
            Dictionary with model information
        """
        is_available, status = self.check_model_availability()

        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": settings.top_p,
            "is_available": is_available,
            "status": status,
        }

    def adjust_parameters(
        self,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
    ) -> None:
        """
        Adjust generation parameters dynamically.

        Args:
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter (0.0 to 1.0)
        """
        if temperature is not None:
            if not 0.0 <= temperature <= 2.0:
                raise ValueError("Temperature must be between 0.0 and 2.0")
            self.temperature = temperature

        if max_tokens is not None:
            if max_tokens <= 0:
                raise ValueError("Max tokens must be positive")
            self.max_tokens = max_tokens

        if top_p is not None:
            if not 0.0 <= top_p <= 1.0:
                raise ValueError("Top_p must be between 0.0 and 1.0")

        # Recreate the LLM with new parameters
        self.llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature,
            num_predict=self.max_tokens,
            top_p=top_p or settings.top_p,
        )

        logger.info(
            f"Updated generation parameters: temp={self.temperature}, max_tokens={self.max_tokens}"
        )

    def generate_followup_questions(
        self, query: str, answer: str, num_questions: int = 3
    ) -> list[str]:
        """
        Generate follow-up questions based on a query and answer.

        Args:
            query: Original query
            answer: Generated answer
            num_questions: Number of follow-up questions to generate

        Returns:
            List of follow-up questions
        """
        prompt = f"""Based on this question and answer, generate {num_questions} relevant follow-up questions that a user might want to ask:

Original Question: {query}

Answer: {answer}

Please provide {num_questions} follow-up questions, one per line:"""

        try:
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)

            # Parse the response into individual questions
            questions = []
            for line in response.content.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("Follow-up"):
                    # Remove numbering and leading characters
                    question = line.lstrip("0123456789.- ").strip()
                    if question:
                        questions.append(question)

            return questions[:num_questions]  # Limit to requested number

        except Exception as e:
            logger.error(f"Follow-up question generation failed: {str(e)}")
            return []
