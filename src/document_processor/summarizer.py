"""
Document summarization for RAG enhancement.

This module generates hierarchical summaries (document, chapter, concept level)
to enrich the vector database with synthetic documents that improve retrieval
for conceptual and broad questions.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any

from src.config.settings import settings
from src.rag.generator import AnswerGenerator

logger = logging.getLogger(__name__)


@dataclass
class SummaryChunk:
    """A synthetic summary chunk for the vector database."""

    content: str
    source_file: str
    chunk_type: str  # 'document_summary', 'chapter_summary', 'concept_summary'
    level: str  # 'document', 'chapter', 'concept'
    chapter_number: int | None = None
    concept_name: str | None = None
    start_pos: int = 0
    end_pos: int = 0
    metadata: dict[str, Any] = None


class DocumentSummarizer:
    """
    Generates hierarchical summaries to enrich RAG retrieval.

    Creates synthetic documents at different levels:
    - Document level: Overall summary of the entire document
    - Chapter level: Summary of each identified chapter/section
    - Concept level: Focused summaries of key concepts
    """

    def __init__(self):
        """Initialize the summarizer with specialized LLM generator for summaries."""
        # Create specialized generator for summaries with optimized parameters
        self.generator = AnswerGenerator(
            model_name=settings.summarization_model,
            temperature=settings.summarization_temperature,
            top_p=settings.summarization_top_p,
        )

        # Define system prompt for faithful summarization
        if not settings.summarization_system_prompt:
            # Set default system prompt optimized for faithful summarization
            settings.summarization_system_prompt = (
                "You are an expert document analyst specialized in creating faithful, accurate summaries. "
                "Your task is to extract and synthesize the most important information while being completely "
                "faithful to the source material. Never add information not present in the original text. "
                "Focus on key concepts, main ideas, and conclusions. Be concise but comprehensive."
            )

        logger.info(
            f"DocumentSummarizer initialized with model: {settings.summarization_model} "
            f"(temp: {settings.summarization_temperature}, top_p: {settings.summarization_top_p})"
        )

    def generate_document_summary(self, document: dict[str, Any]) -> SummaryChunk:
        """
        Generate a high-level summary of the entire document.

        Args:
            document: Document dictionary from extractor

        Returns:
            SummaryChunk with document-level summary
        """
        content = document.get("plain_text", "")

        # Detect language for appropriate prompting
        is_spanish = document.get("is_spanish", False)

        if is_spanish:
            prompt = f"""
            Analiza este documento completo y genera un resumen ejecutivo que capture:
            
            1. La tesis o idea principal del documento
            2. Los conceptos clave que se desarrollan
            3. Las conclusiones principales
            4. El enfoque o metodología utilizada
            
            Documento:
            {content[:8000]}  # Limit for context window
            
            Genera un resumen ejecutivo de 200-300 palabras que permita entender de qué trata el documento sin leerlo completo.
            """
        else:
            prompt = f"""
            Analyze this complete document and generate an executive summary that captures:
            
            1. The main thesis or central idea
            2. Key concepts developed
            3. Main conclusions
            4. Approach or methodology used
            
            Document:
            {content[:8000]}
            
            Generate a 200-300 word executive summary that allows understanding what the document is about without reading it completely.
            """

        try:
            # Use optimized parameters for document summaries
            summary = self._generate_faithful_summary(
                prompt,
                max_tokens=settings.summarization_max_tokens_document,
                summary_type="document",
            )

            return SummaryChunk(
                content=summary,
                source_file=document.get("source_file", ""),
                chunk_type="document_summary",
                level="document",
                metadata={
                    "generated_by": "llm_summarizer",
                    "language": "es" if is_spanish else "en",
                    "summary_type": "document",
                    "original_length": len(content),
                    "summary_length": len(summary),
                    "is_summary": True,
                },
            )

        except Exception as e:
            logger.error(f"Failed to generate document summary: {str(e)}")
            return None

    def extract_chapters(
        self, content: str, is_spanish: bool = False
    ) -> list[dict[str, Any]]:
        """
        Identify chapters/sections in the document.

        Args:
            content: Full document text
            is_spanish: Whether content is in Spanish

        Returns:
            List of chapter dictionaries with content and metadata
        """
        chapters = []

        # Pattern for markdown headings and common chapter indicators
        if is_spanish:
            patterns = [
                r"^#+\s+.*",  # Markdown headings
                r"^\d+\.\s+.*",  # Numbered sections
                r"^Capítulo\s+\d+.*",  # Chapter X
                r"^Paso\s+\d+.*",  # Step X
                r"^Parte\s+\d+.*",  # Part X
            ]
        else:
            patterns = [
                r"^#+\s+.*",
                r"^\d+\.\s+.*",
                r"^Chapter\s+\d+.*",
                r"^Step\s+\d+.*",
                r"^Part\s+\d+.*",
            ]

        lines = content.split("\n")
        current_chapter = None
        current_content = []
        chapter_num = 0

        for i, line in enumerate(lines):
            line = line.strip()

            # Check if line matches any chapter pattern
            is_chapter_start = any(
                re.match(pattern, line, re.IGNORECASE) for pattern in patterns
            )

            if is_chapter_start and len(line) > 3:  # Avoid very short lines
                # Save previous chapter if exists
                if current_chapter and current_content:
                    chapters.append(
                        {
                            "title": current_chapter,
                            "number": chapter_num,
                            "content": "\n".join(current_content),
                            "start_line": getattr(
                                chapters[-1] if chapters else None, "end_line", 0
                            )
                            + 1
                            if chapters
                            else 0,
                            "end_line": i,
                        }
                    )

                # Start new chapter
                chapter_num += 1
                current_chapter = line
                current_content = []
            else:
                current_content.append(line)

        # Don't forget the last chapter
        if current_chapter and current_content:
            chapters.append(
                {
                    "title": current_chapter,
                    "number": chapter_num,
                    "content": "\n".join(current_content),
                    "start_line": chapters[-1]["end_line"] + 1 if chapters else 0,
                    "end_line": len(lines),
                }
            )

        logger.info(f"Extracted {len(chapters)} chapters/sections")
        return chapters

    def generate_chapter_summaries(
        self, document: dict[str, Any]
    ) -> list[SummaryChunk]:
        """
        Generate summaries for each chapter/section.

        Args:
            document: Document dictionary from extractor

        Returns:
            List of SummaryChunk objects for each chapter
        """
        content = document.get("plain_text", "")
        is_spanish = document.get("is_spanish", False)
        chapters = self.extract_chapters(content, is_spanish)

        if not chapters:
            logger.warning("No chapters detected in document")
            return []

        summaries = []

        for chapter in chapters:
            if len(chapter["content"].strip()) < 200:  # Skip very short chapters
                continue

            if is_spanish:
                prompt = f"""
                Resume este capítulo/sección identificando:
                
                1. El tema principal tratado
                2. Los puntos clave desarrollados
                3. Las conclusiones o insights importantes
                4. Cómo se relaciona con el tema general del documento
                
                Título: {chapter["title"]}
                
                Contenido:
                {chapter["content"][:4000]}
                
                Genera un resumen de 100-150 palabras que capture la esencia de esta sección.
                """
            else:
                prompt = f"""
                Summarize this chapter/section by identifying:
                
                1. Main topic covered
                2. Key points developed
                3. Important conclusions or insights
                4. How it relates to the document's overall theme
                
                Title: {chapter["title"]}
                
                Content:
                {chapter["content"][:4000]}
                
                Generate a 100-150 word summary that captures the essence of this section.
                """

            try:
                summary = self._generate_faithful_summary(
                    prompt,
                    max_tokens=settings.summarization_max_tokens_chapter,
                    summary_type="chapter",
                    source_content=chapter["content"],
                )

                summaries.append(
                    SummaryChunk(
                        content=f"[Capítulo: {chapter['title']}]\n\n{summary}"
                        if is_spanish
                        else f"[Chapter: {chapter['title']}]\n\n{summary}",
                        source_file=document.get("source_file", ""),
                        chunk_type="chapter_summary",
                        level="chapter",
                        chapter_number=chapter["number"],
                        metadata={
                            "generated_by": "llm_summarizer",
                            "language": "es" if is_spanish else "en",
                            "chapter_title": chapter["title"],
                            "original_length": len(chapter["content"]),
                            "summary_length": len(summary),
                        },
                    )
                )

            except Exception as e:
                logger.error(
                    f"Failed to generate summary for chapter '{chapter['title']}': {str(e)}"
                )
                continue

        logger.info(f"Generated {len(summaries)} chapter summaries")
        return summaries

    def _generate_faithful_summary(
        self, prompt: str, max_tokens: int, summary_type: str, source_content: str = ""
    ) -> str:
        """
        Generate a faithful summary with quality control and validation.

        Args:
            prompt: The prompt for summary generation
            max_tokens: Maximum tokens for the response
            summary_type: Type of summary (document, chapter, concept)
            source_content: Original content for validation (optional)

        Returns:
            Generated summary text
        """
        best_summary = None
        best_score = 0

        for attempt in range(settings.max_summary_retries + 1):
            try:
                # Enhanced prompt with system context
                enhanced_prompt = f"""
                {settings.summarization_system_prompt}
                
                CRITICAL REQUIREMENTS:
                1. Be completely faithful to the source material
                2. Do not add information not present in the original
                3. Focus on the most important concepts and conclusions
                4. Maintain the original meaning and context
                5. Be concise but comprehensive
                
                {prompt}
                
                Remember: Accuracy and faithfulness are more important than creativity.
                """

                summary = self.generator.generate_response(
                    enhanced_prompt, max_tokens=max_tokens
                )

                # Basic validation
                if not summary or len(summary.strip()) < 50:
                    logger.warning(
                        f"Generated summary too short (attempt {attempt + 1})"
                    )
                    continue

                # Quality validation if enabled
                if settings.enable_summary_validation:
                    quality_score = self._validate_summary_quality(
                        summary, source_content, summary_type, max_tokens
                    )

                    if quality_score > best_score:
                        best_summary = summary
                        best_score = quality_score

                    # If we got a high-quality summary, use it
                    if quality_score >= 0.8:  # 80% quality threshold
                        logger.debug(
                            f"High quality summary generated (score: {quality_score:.2f})"
                        )
                        return summary

                    logger.debug(
                        f"Summary quality score: {quality_score:.2f} (attempt {attempt + 1})"
                    )
                else:
                    # No validation, use first valid summary
                    return summary

            except Exception as e:
                logger.warning(
                    f"Summary generation attempt {attempt + 1} failed: {str(e)}"
                )
                if attempt == settings.max_summary_retries:
                    raise

        # Return best summary if validation was enabled, otherwise raise error
        if best_summary:
            logger.info(
                f"Using best summary from {settings.max_summary_retries + 1} attempts (score: {best_score:.2f})"
            )
            return best_summary
        else:
            raise RuntimeError(
                f"Failed to generate quality summary after {settings.max_summary_retries + 1} attempts"
            )

    def _validate_summary_quality(
        self, summary: str, source_content: str, summary_type: str, max_tokens: int
    ) -> float:
        """
        Validate the quality and faithfulness of a generated summary.

        Args:
            summary: Generated summary text
            source_content: Original source content
            summary_type: Type of summary being validated
            max_tokens: Maximum tokens allowed for this summary type

        Returns:
            Quality score between 0.0 and 1.0
        """
        if not settings.enable_summary_validation:
            return 1.0

        score = 0.0
        checks = 0

        # Basic length and structure checks
        if 50 <= len(summary.strip()) <= max_tokens * 4:  # Reasonable length
            score += 0.2
        checks += 1

        # Check for key indicators of good summaries
        summary_lower = summary.lower()

        # Should not contain phrases indicating uncertainty about source
        bad_phrases = [
            "i don't know",
            "not sure",
            "unclear",
            "cannot determine",
            "no se sabe",
            "no está claro",
            "no puedo determinar",
        ]
        if not any(phrase in summary_lower for phrase in bad_phrases):
            score += 0.2
        checks += 1

        # Should not start with meta-commentary
        meta_starts = [
            "here is",
            "this is a summary",
            "the following",
            "aquí está",
            "esto es un resumen",
            "lo siguiente",
        ]
        if not any(summary_lower.startswith(phrase) for phrase in meta_starts):
            score += 0.2
        checks += 1

        # For Spanish content, check Spanish indicators
        if source_content and settings.detect_spanish_content(source_content):
            spanish_indicators = ["el", "la", "de", "que", "en", "es", "son"]
            if any(word in summary_lower for word in spanish_indicators):
                score += 0.2
        else:
            # For English content, check English indicators
            english_indicators = ["the", "and", "of", "to", "in", "is", "are"]
            if any(word in summary_lower for word in english_indicators):
                score += 0.2
        checks += 1

        # Advanced faithfulness check (if enabled and source provided)
        if settings.summary_faithfulness_check and source_content:
            faithfulness_score = self._check_faithfulness(summary, source_content)
            score += faithfulness_score * 0.2
        else:
            score += 0.2  # Default to good if not checking
        checks += 1

        return score / checks if checks > 0 else 0.0

    def _check_faithfulness(self, summary: str, source_content: str) -> float:
        """
        Check if summary is faithful to source content using LLM.

        Args:
            summary: Generated summary
            source_content: Original source content

        Returns:
            Faithfulness score between 0.0 and 1.0
        """
        try:
            # Simple keyword overlap check (fast approximation)
            summary_words = set(summary.lower().split())
            source_words = set(source_content.lower().split())

            # Remove common stop words
            stop_words = {
                "el",
                "la",
                "de",
                "que",
                "y",
                "en",
                "un",
                "es",
                "se",
                "no",
                "te",
                "lo",
                "le",
                "the",
                "and",
                "of",
                "to",
                "a",
                "in",
                "is",
                "it",
                "you",
                "that",
                "he",
                "was",
            }

            summary_content = summary_words - stop_words
            source_content_words = source_words - stop_words

            if not source_content_words:
                return 0.5  # Cannot validate

            # Calculate overlap ratio
            overlap = len(summary_content & source_content_words)
            overlap_ratio = overlap / len(summary_content) if summary_content else 0

            # Good summaries should have 30-70% overlap with source (not too low, not too high)
            if 0.3 <= overlap_ratio <= 0.7:
                return min(1.0, overlap_ratio / 0.5)  # Normalize to 0-1
            elif overlap_ratio < 0.3:
                return overlap_ratio / 0.3  # Penalize low overlap
            else:
                return max(
                    0.5, 1.0 - (overlap_ratio - 0.7) / 0.3
                )  # Penalize too high overlap

        except Exception as e:
            logger.warning(f"Faithfulness check failed: {str(e)}")
            return 0.5  # Default to neutral if check fails

    def extract_key_concepts(self, document: dict[str, Any]) -> list[str]:
        """
        Identify key concepts in the document using LLM.

        Args:
            document: Document dictionary from extractor

        Returns:
            List of key concept names
        """
        content = document.get("plain_text", "")
        is_spanish = document.get("is_spanish", False)

        if is_spanish:
            prompt = f"""
            Analiza este documento e identifica los 5-8 conceptos clave más importantes.
            
            Devuelve solo una lista de conceptos, uno por línea, sin numeración ni explicaciones.
            Ejemplos de formato:
            - Sedentarismo
            - Sistema cardiovascular
            - Ejercicio físico
            
            Documento:
            {content[:6000]}
            
            Conceptos clave:
            """
        else:
            prompt = f"""
            Analyze this document and identify the 5-8 most important key concepts.
            
            Return only a list of concepts, one per line, without numbering or explanations.
            Format examples:
            - Sedentary lifestyle
            - Cardiovascular system
            - Physical exercise
            
            Document:
            {content[:6000]}
            
            Key concepts:
            """

        try:
            response = self._generate_faithful_summary(
                prompt,
                max_tokens=200,  # Keep same for concept extraction
                summary_type="concept_extraction",
            )

            # Extract concepts from response
            concepts = []
            for line in response.split("\n"):
                line = line.strip()
                if line and not line.startswith("#"):
                    # Remove bullets, dashes, numbers
                    concept = re.sub(r"^[-•*\d\.\)]\s*", "", line).strip()
                    if concept and len(concept) > 3:
                        concepts.append(concept)

            logger.info(f"Extracted {len(concepts)} key concepts")
            return concepts[:8]  # Limit to 8 concepts

        except Exception as e:
            logger.error(f"Failed to extract key concepts: {str(e)}")
            return []

    def generate_concept_summaries(
        self, document: dict[str, Any]
    ) -> list[SummaryChunk]:
        """
        Generate focused summaries for key concepts.

        Args:
            document: Document dictionary from extractor

        Returns:
            List of SummaryChunk objects for each concept
        """
        content = document.get("plain_text", "")
        is_spanish = document.get("is_spanish", False)
        concepts = self.extract_key_concepts(document)

        if not concepts:
            return []

        summaries = []

        for concept in concepts:
            if is_spanish:
                prompt = f"""
                Busca en este documento toda la información relacionada con "{concept}" y genera un resumen natural que explique este concepto.

                Documento:
                {content[:8000]}

                INSTRUCCIONES:
                - Comienza directamente explicando qué es "{concept}" según el documento
                - Incluye su definición, importancia y contexto
                - Usa entre 80-120 palabras
                - Escribe en español claro y natural
                - NO uses prefijos como "[Concepto:]" o similares
                - Escribe como si fuera una entrada de enciclopedia

                FORMATO: Comienza directamente con: "{concept} es..." o "{concept} se refiere a..."
                """
            else:
                prompt = f"""
                Search this document for all information related to "{concept}" and generate a natural summary explaining this concept.

                Document:
                {content[:8000]}

                INSTRUCTIONS:
                - Start directly by explaining what "{concept}" is according to the document
                - Include its definition, importance and context
                - Use 80-120 words
                - Write in clear and natural English
                - DO NOT use prefixes like "[Concept:]" or similar
                - Write like an encyclopedia entry

                FORMAT: Start directly with: "{concept} is..." or "{concept} refers to..."
                """

            try:
                # Extract relevant content for validation
                concept_content = content[:8000] if len(content) > 8000 else content

                summary = self._generate_faithful_summary(
                    prompt,
                    max_tokens=settings.summarization_max_tokens_concept,
                    summary_type="concept",
                    source_content=concept_content,
                )

                summaries.append(
                    SummaryChunk(
                        content=summary,  # Natural text without artificial prefixes
                        source_file=document.get("source_file", ""),
                        chunk_type="concept_summary",
                        level="concept",
                        concept_name=concept,
                        metadata={
                            "generated_by": "llm_summarizer",
                            "language": "es" if is_spanish else "en",
                            "concept_focus": concept,
                            "summary_length": len(summary),
                            "is_summary": True,
                            "summary_type": "concept",
                        },
                    )
                )

            except Exception as e:
                logger.error(
                    f"Failed to generate summary for concept '{concept}': {str(e)}"
                )
                continue

        logger.info(f"Generated {len(summaries)} concept summaries")
        return summaries

    def generate_all_summaries(self, document: dict[str, Any]) -> list[SummaryChunk]:
        """
        Generate all types of summaries for a document.

        Args:
            document: Document dictionary from extractor

        Returns:
            List of all generated summary chunks
        """
        all_summaries = []

        logger.info("Generating hierarchical summaries...")

        # 1. Document-level summary
        doc_summary = self.generate_document_summary(document)
        if doc_summary:
            all_summaries.append(doc_summary)

        # 2. Chapter-level summaries
        chapter_summaries = self.generate_chapter_summaries(document)
        all_summaries.extend(chapter_summaries)

        # 3. Concept-level summaries
        concept_summaries = self.generate_concept_summaries(document)
        all_summaries.extend(concept_summaries)

        logger.info(f"Generated {len(all_summaries)} total summary chunks")
        return all_summaries
