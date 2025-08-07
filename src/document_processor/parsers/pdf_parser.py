"""
PDF document parser using Docling for robust extraction.

This parser handles PDF files with complex layouts, tables, and figures
using the Docling library for professional-grade document processing.
"""

from pathlib import Path
from typing import Any

try:
    from docling.document_converter import DocumentConverter

    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False


class PDFParser:
    """Parser for extracting content from PDF documents using Docling."""

    def __init__(self):
        """Initialize the PDF parser."""
        if not DOCLING_AVAILABLE:
            raise ImportError(
                "Docling is not available. Install with: pip install docling"
            )

        # Initialize Docling converter
        self.converter = DocumentConverter()

    def extract_content(self, file_path: str) -> dict[str, Any]:
        """
        Extract structured content from a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dictionary containing extracted content and metadata
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if not path.suffix.lower() == ".pdf":
            raise ValueError(f"File is not a PDF: {file_path}")

        try:
            # Convert PDF using Docling
            result = self.converter.convert(path)

            # Extract main content
            document = result.document

            # Get plain text
            plain_text = document.export_to_markdown()

            # Extract metadata
            metadata = self._extract_metadata(document)

            # Extract structural information
            structure = self._extract_structure(document)

            return {
                "source_file": str(path),
                "file_type": "pdf",
                "plain_text": plain_text,
                "metadata": metadata,
                "structure": structure,
                "size_bytes": path.stat().st_size,
                "page_count": self._get_page_count(document),
            }

        except Exception as e:
            raise RuntimeError(f"Failed to parse PDF {file_path}: {str(e)}") from e

    def _extract_metadata(self, document) -> dict[str, Any]:
        """
        Extract metadata from the Docling document.

        Args:
            document: Docling document object

        Returns:
            Dictionary with document metadata
        """
        metadata = {}

        # Extract available metadata fields
        if hasattr(document, "meta") and document.meta:
            doc_meta = document.meta

            # Common metadata fields
            if hasattr(doc_meta, "title") and doc_meta.title:
                metadata["title"] = doc_meta.title
            if hasattr(doc_meta, "author") and doc_meta.author:
                metadata["author"] = doc_meta.author
            if hasattr(doc_meta, "subject") and doc_meta.subject:
                metadata["subject"] = doc_meta.subject
            if hasattr(doc_meta, "creator") and doc_meta.creator:
                metadata["creator"] = doc_meta.creator
            if hasattr(doc_meta, "creation_date") and doc_meta.creation_date:
                metadata["creation_date"] = str(doc_meta.creation_date)

        return metadata

    def _extract_structure(self, document) -> dict[str, Any]:
        """
        Extract structural information from the PDF.

        Args:
            document: Docling document object

        Returns:
            Dictionary with structural information
        """
        structure = {"pages": [], "tables": [], "figures": [], "sections": []}

        try:
            # Iterate through document elements
            for element in document.body:
                if hasattr(element, "type"):
                    element_type = element.type

                    # Collect different types of content
                    if "table" in element_type.lower():
                        structure["tables"].append(
                            {
                                "type": element_type,
                                "content": str(element)[:200] + "..."
                                if len(str(element)) > 200
                                else str(element),
                            }
                        )
                    elif (
                        "figure" in element_type.lower()
                        or "image" in element_type.lower()
                    ):
                        structure["figures"].append(
                            {
                                "type": element_type,
                                "content": str(element)[:100] + "..."
                                if len(str(element)) > 100
                                else str(element),
                            }
                        )
                    elif (
                        "heading" in element_type.lower()
                        or "title" in element_type.lower()
                    ):
                        structure["sections"].append(
                            {"type": element_type, "content": str(element)}
                        )

        except Exception as e:
            # If structure extraction fails, continue with basic info
            structure["error"] = f"Structure extraction failed: {str(e)}"

        return structure

    def _get_page_count(self, document) -> int:
        """
        Get the number of pages in the document.

        Args:
            document: Docling document object

        Returns:
            Number of pages
        """
        try:
            if hasattr(document, "pages") and document.pages:
                return len(document.pages)
            else:
                # Fallback: estimate from content
                return 1
        except Exception:
            return 1

    def get_chunk_boundaries(self, content: str) -> list[int]:
        """
        Get intelligent chunk boundaries for PDF content.

        Args:
            content: Extracted text content

        Returns:
            List of character positions for chunk boundaries
        """
        boundaries = [0]

        # Look for section markers (often preserved in PDF extraction)
        section_patterns = [
            r"\n# .+\n",  # Markdown-style headers
            r"\n## .+\n",
            r"\n\d+\.\s*.+\n",  # Numbered sections
            r"\n[A-Z][A-Z\s]+\n",  # ALL CAPS headers
            r"\n\n[A-Z].{20,}\n",  # Potential section headers
        ]

        import re

        for pattern in section_patterns:
            for match in re.finditer(pattern, content):
                boundaries.append(match.start())

        # Add paragraph boundaries for very long sections
        paragraphs = content.split("\n\n")
        current_pos = 0
        for paragraph in paragraphs:
            if len(paragraph) > 500:  # Long paragraph
                boundaries.append(current_pos)
            current_pos += len(paragraph) + 2  # +2 for \n\n

        # Sort and deduplicate
        boundaries = sorted(set(boundaries))

        # Add end of document
        if boundaries[-1] != len(content):
            boundaries.append(len(content))

        return boundaries
