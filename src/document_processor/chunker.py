"""
Text chunking system for breaking documents into manageable pieces.

This module provides intelligent text chunking that respects document
structure and maintains semantic coherence.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config.settings import settings

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""

    content: str
    start_pos: int
    end_pos: int
    chunk_id: str
    source_file: str
    file_type: str
    metadata: dict[str, Any]

    # Enhanced metadata for hybrid retrieval
    chunk_type: str = "original_text"  # 'original_text', 'document_summary', 'chapter_summary', 'concept_summary'
    level: str = "chunk"  # 'chunk', 'document', 'chapter', 'concept'
    chapter_number: int | None = None
    concept_name: str | None = None
    language: str | None = None


class TextChunker:
    """
    Intelligent text chunking system.

    Breaks documents into manageable chunks while preserving semantic
    coherence and respecting document structure.
    """

    def __init__(self, chunk_size: int | None = None, overlap: int | None = None):
        """
        Initialize the text chunker.

        Args:
            chunk_size: Target size for chunks in characters
            overlap: Overlap size between chunks in characters
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.overlap = overlap or settings.chunk_overlap

        # Validate parameters
        if self.overlap >= self.chunk_size:
            raise ValueError("Overlap must be smaller than chunk size")

        logger.info(
            f"TextChunker initialized: size={self.chunk_size}, overlap={self.overlap}"
        )

    def chunk_document(self, document: dict[str, Any]) -> list[TextChunk]:
        """
        Chunk a document into smaller pieces.

        Args:
            document: Document dictionary from extractor

        Returns:
            List of TextChunk objects
        """
        content = document.get("plain_text", "")
        if not content:
            logger.warning(
                f"No text content found in document: {document.get('source_file', 'unknown')}"
            )
            return []

        document.get("file_type", "unknown")
        source_file = document.get("source_file", "unknown")

        logger.debug(f"Chunking document: {source_file} ({len(content)} chars)")

        # Get intelligent boundaries if available
        boundaries = self._get_intelligent_boundaries(document)

        if boundaries and len(boundaries) > 2:
            # Use structure-aware chunking
            chunks = self._chunk_with_boundaries(content, boundaries, document)
        else:
            # Fall back to simple sliding window
            chunks = self._chunk_sliding_window(content, document)

        logger.info(f"Created {len(chunks)} chunks for {source_file}")
        return chunks

    def chunk_multiple_documents(
        self, documents: list[dict[str, Any]]
    ) -> list[TextChunk]:
        """
        Chunk multiple documents.

        Args:
            documents: List of document dictionaries

        Returns:
            List of all chunks from all documents
        """
        all_chunks = []

        for doc in documents:
            try:
                chunks = self.chunk_document(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                source_file = doc.get("source_file", "unknown")
                logger.error(f"Failed to chunk document {source_file}: {str(e)}")

        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks

    def create_summary_chunk(
        self, summary_chunk, file_type: str = "synthetic"
    ) -> TextChunk:
        """
        Convert a SummaryChunk to a TextChunk for vector storage.

        Args:
            summary_chunk: SummaryChunk object from summarizer
            file_type: File type for compatibility

        Returns:
            TextChunk ready for vector storage
        """
        from pathlib import Path

        chunk_id = f"{Path(summary_chunk.source_file).stem}_{summary_chunk.chunk_type}_{hash(summary_chunk.content) % 10000}"

        # Merge metadata
        metadata = summary_chunk.metadata or {}
        metadata.update(
            {
                "generated_summary": True,
                "summary_level": summary_chunk.level,
                "chunk_type": summary_chunk.chunk_type,
            }
        )

        return TextChunk(
            content=summary_chunk.content,
            start_pos=summary_chunk.start_pos,
            end_pos=summary_chunk.end_pos,
            chunk_id=chunk_id,
            source_file=summary_chunk.source_file,
            file_type=file_type,
            metadata=metadata,
            chunk_type=summary_chunk.chunk_type,
            level=summary_chunk.level,
            chapter_number=summary_chunk.chapter_number,
            concept_name=summary_chunk.concept_name,
            language=metadata.get("language"),
        )

    def _get_intelligent_boundaries(self, document: dict[str, Any]) -> list[int] | None:
        """
        Get intelligent chunk boundaries based on document structure.

        Args:
            document: Document dictionary

        Returns:
            List of boundary positions or None
        """
        content = document.get("plain_text", "")
        file_type = document.get("file_type", "")

        # Try to get boundaries from parser if available
        if "structure" in document:
            return self._extract_boundaries_from_structure(
                content, document["structure"], file_type
            )

        # Fall back to generic boundary detection
        return self._find_generic_boundaries(content)

    def _extract_boundaries_from_structure(
        self, content: str, structure: dict[str, Any], file_type: str
    ) -> list[int]:
        """Extract boundaries based on document structure."""
        boundaries = [0]

        if file_type == "markdown" and "headers" in structure:
            # Use markdown headers for boundaries
            for header in structure["headers"]:
                if "position" in header and header["level"] <= 3:  # Only major headers
                    boundaries.append(header["position"])

        elif file_type == "html" and "headings" in structure:
            # Use HTML headings for boundaries
            for heading in structure["headings"]:
                if heading["level"] <= 3:  # h1, h2, h3
                    heading_text = heading["text"]
                    pos = content.find(heading_text)
                    if pos != -1:
                        boundaries.append(pos)

        # Sort and deduplicate
        boundaries = sorted(set(boundaries))

        # Add end position
        if boundaries[-1] != len(content):
            boundaries.append(len(content))

        return boundaries

    def _find_generic_boundaries(self, content: str) -> list[int]:
        """Find generic boundaries in text."""
        boundaries = [0]

        # Look for section breaks
        section_patterns = [
            r"\n\n[A-Z][^\\n]{20,}\n",  # Potential section headers
            r"\n\d+\.\s+[A-Z][^\\n]{10,}\n",  # Numbered sections
            r"\n[A-Z\s]{3,}\n",  # ALL CAPS potential headers
        ]

        for pattern in section_patterns:
            for match in re.finditer(pattern, content):
                boundaries.append(match.start())

        # Add paragraph boundaries for very long sections
        paragraphs = content.split("\n\n")
        current_pos = 0
        for paragraph in paragraphs:
            if len(paragraph) > self.chunk_size * 1.5:  # Very long paragraph
                boundaries.append(current_pos)
            current_pos += len(paragraph) + 2

        # Sort and deduplicate
        boundaries = sorted(set(boundaries))

        # Add end position
        if boundaries[-1] != len(content):
            boundaries.append(len(content))

        return boundaries

    def _chunk_with_boundaries(
        self, content: str, boundaries: list[int], document: dict[str, Any]
    ) -> list[TextChunk]:
        """Chunk text using intelligent boundaries."""
        chunks = []

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            section_text = content[start:end].strip()

            if not section_text:
                continue

            # If section is small enough, use as single chunk
            if len(section_text) <= self.chunk_size:
                chunk = self._create_chunk(
                    content=section_text,
                    start_pos=start,
                    end_pos=end,
                    chunk_index=len(chunks),
                    document=document,
                )
                chunks.append(chunk)
            else:
                # Split large sections using sliding window
                section_chunks = self._chunk_sliding_window_section(
                    section_text, start, len(chunks), document
                )
                chunks.extend(section_chunks)

        return chunks

    def _chunk_sliding_window(
        self, content: str, document: dict[str, Any]
    ) -> list[TextChunk]:
        """Chunk text using simple sliding window approach."""
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(content):
            # Calculate end position
            ideal_end = min(start + self.chunk_size, len(content))

            # Try to find a good breaking point near the end
            if ideal_end < len(content):
                end = self._find_break_point(content, start, ideal_end)
            else:
                end = ideal_end

            chunk_text = content[start:end].strip()

            if chunk_text:
                chunk = self._create_chunk(
                    content=chunk_text,
                    start_pos=start,
                    end_pos=end,
                    chunk_index=chunk_index,
                    document=document,
                )
                chunks.append(chunk)
                chunk_index += 1

            # Move start position with overlap
            # Ensure we make progress even if break point adjustment happened
            start = max(start + self.chunk_size // 2, end - self.overlap)

        return chunks

    def _chunk_sliding_window_section(
        self,
        section_text: str,
        section_start: int,
        base_chunk_index: int,
        document: dict[str, Any],
    ) -> list[TextChunk]:
        """Apply sliding window to a specific section."""
        chunks = []
        start = 0
        chunk_index = base_chunk_index

        while start < len(section_text):
            end = min(start + self.chunk_size, len(section_text))

            if end < len(section_text):
                end = self._find_break_point(section_text, start, end)

            chunk_text = section_text[start:end].strip()

            if chunk_text:
                chunk = self._create_chunk(
                    content=chunk_text,
                    start_pos=section_start + start,
                    end_pos=section_start + end,
                    chunk_index=chunk_index,
                    document=document,
                )
                chunks.append(chunk)
                chunk_index += 1

            start = max(start + 1, end - self.overlap)

        return chunks

    def _find_break_point(self, content: str, start: int, ideal_end: int) -> int:
        """Find a good breaking point near the ideal end position."""
        # Ensure we don't go beyond content length
        ideal_end = min(ideal_end, len(content))

        # First, search backwards from ideal_end for good break points
        # This ensures we never cut in the middle of a word
        search_range = min(200, ideal_end - start)  # Don't search beyond chunk start

        for i in range(ideal_end, max(start, ideal_end - search_range), -1):
            if i >= len(content):
                continue

            # Check for paragraph breaks (highest priority)
            if i >= 2 and content[i - 2 : i] == "\n\n":
                return i

            # Check for sentence ends
            if i >= 2 and content[i - 2 : i] == ". ":
                return i

            # Check for line breaks
            if i >= 1 and content[i - 1] == "\n":
                return i

            # Check for comma + space
            if i >= 2 and content[i - 2 : i] == ", ":
                return i

            # Check for any space (lowest priority but important to avoid breaking words)
            if i >= 1 and content[i - 1] == " ":
                return i

        # If no good break point found, search forward more extensively
        # to avoid cutting words in the middle
        for i in range(ideal_end, min(len(content), ideal_end + 200)):
            if content[i] == " ":
                return i + 1  # Break after the space

        # Last resort: return ideal_end (but this should be rare)
        return ideal_end

    def _create_chunk(
        self,
        content: str,
        start_pos: int,
        end_pos: int,
        chunk_index: int,
        document: dict[str, Any],
    ) -> TextChunk:
        """Create a TextChunk object with metadata."""
        source_file = document.get("source_file", "unknown")
        file_type = document.get("file_type", "unknown")

        # Generate unique chunk ID
        chunk_id = f"{Path(source_file).stem}_{chunk_index:04d}"

        # Collect relevant metadata
        metadata = {
            "chunk_index": chunk_index,
            "char_count": len(content),
            "start_position": start_pos,
            "end_position": end_pos,
            "source_metadata": document.get("metadata", {}),
        }

        return TextChunk(
            content=content,
            start_pos=start_pos,
            end_pos=end_pos,
            chunk_id=chunk_id,
            source_file=source_file,
            file_type=file_type,
            metadata=metadata,
        )

    def get_chunking_stats(self, chunks: list[TextChunk]) -> dict[str, Any]:
        """
        Get statistics about the chunking results.

        Args:
            chunks: List of text chunks

        Returns:
            Dictionary with chunking statistics
        """
        if not chunks:
            return {"total_chunks": 0}

        chunk_sizes = [len(chunk.content) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "total_characters": sum(chunk_sizes),
            "files_processed": len({chunk.source_file for chunk in chunks}),
            "file_types": list({chunk.file_type for chunk in chunks}),
        }
