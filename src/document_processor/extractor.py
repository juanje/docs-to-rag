"""
Main document extraction coordinator.

This module orchestrates the extraction process for different document types
and provides a unified interface for processing documents.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config.settings import settings

from .parsers import HTMLParser, MarkdownParser, PDFParser

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of document processing operation."""

    processed: int
    failed: int
    chunks_created: int
    errors: list[str]
    documents: list[dict[str, Any]]


class DocumentExtractor:
    """
    Main document extraction coordinator.

    Handles different document types and provides a unified interface
    for extracting content from various file formats.
    """

    def __init__(self):
        """Initialize the document extractor with parsers."""
        logger.info("Initializing document parsers...")

        self.parsers = {
            ".md": MarkdownParser(),
            ".html": HTMLParser(),
        }
        logger.info("✓ Markdown and HTML parsers loaded")

        # Initialize PDF parser if available
        try:
            self.parsers[".pdf"] = PDFParser()
            logger.info("✓ PDF parser loaded (docling available)")
        except ImportError as e:
            logger.warning(f"⚠️ PDF parser not available: {e}")
            logger.warning("Install docling with: pip install docling")

        logger.info(
            f"DocumentExtractor ready with {len(self.parsers)} parsers: {list(self.parsers.keys())}"
        )

    def extract_document(self, file_path: str) -> dict[str, Any] | None:
        """
        Extract content from a single document.

        Args:
            file_path: Path to the document file

        Returns:
            Extracted content dictionary or None if extraction failed
        """
        path = Path(file_path)

        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return None

        # Get file extension
        extension = path.suffix.lower()

        if extension not in self.parsers:
            logger.warning(f"Unsupported file type: {extension} for file {file_path}")
            return None

        try:
            parser = self.parsers[extension]
            parser_name = parser.__class__.__name__
            logger.info(f"📄 Extracting content from {path.name} using {parser_name}")
            result = parser.extract_content(str(path))

            # Add processing metadata
            result["extracted_at"] = self._get_current_timestamp()
            result["processor_version"] = "0.1.0"

            # Detect language if auto-detection is enabled
            if settings.auto_detect_language:
                is_spanish = settings.detect_spanish_content(
                    result.get("plain_text", "")
                )
                result["detected_language"] = "es" if is_spanish else "en"
                result["is_spanish"] = is_spanish

                if is_spanish:
                    logger.info(
                        f"🇪🇸 Detected Spanish content in {path.name} → will use {settings.embedding_model_multilingual}"
                    )
                else:
                    logger.info(
                        f"🇺🇸 Detected English content in {path.name} → will use {settings.embedding_model}"
                    )

            logger.info(
                f"Successfully extracted {len(result['plain_text'])} characters from {path.name}"
            )
            return result

        except Exception as e:
            logger.error(f"Failed to extract content from {file_path}: {str(e)}")
            return None

    def process_directory(self, directory_path: str) -> ProcessingResult:
        """
        Process all supported documents in a directory recursively.

        Args:
            directory_path: Path to directory containing documents

        Returns:
            ProcessingResult with statistics and results
        """
        dir_path = Path(directory_path)

        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")

        logger.info(f"Processing documents in directory: {directory_path}")

        # Find all supported files
        supported_files = []
        for extension in settings.supported_extensions:
            pattern = f"**/*{extension}"
            files = list(dir_path.glob(pattern))
            supported_files.extend(files)

        logger.info(f"Found {len(supported_files)} supported files")

        # Process each file
        documents = []
        errors = []
        processed = 0
        failed = 0

        for file_path in supported_files:
            try:
                result = self.extract_document(str(file_path))
                if result:
                    documents.append(result)
                    processed += 1
                    logger.debug(f"✓ Processed: {file_path.name}")
                else:
                    failed += 1
                    error_msg = f"Failed to process: {file_path.name}"
                    errors.append(error_msg)
                    logger.warning(error_msg)

            except Exception as e:
                failed += 1
                error_msg = f"Error processing {file_path.name}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)

        logger.info(f"Processing complete: {processed} processed, {failed} failed")

        return ProcessingResult(
            processed=processed,
            failed=failed,
            chunks_created=0,  # Will be filled by chunker
            errors=errors,
            documents=documents,
        )

    def get_supported_extensions(self) -> list[str]:
        """
        Get list of supported file extensions.

        Returns:
            List of supported file extensions
        """
        return list(self.parsers.keys())

    def is_supported_file(self, file_path: str) -> bool:
        """
        Check if file type is supported.

        Args:
            file_path: Path to file

        Returns:
            True if file type is supported
        """
        extension = Path(file_path).suffix.lower()
        return extension in self.parsers

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime

        return datetime.now().isoformat()

    def get_parser_info(self) -> dict[str, Any]:
        """
        Get information about available parsers.

        Returns:
            Dictionary with parser information
        """
        info = {
            "available_parsers": list(self.parsers.keys()),
            "total_parsers": len(self.parsers),
            "parser_details": {},
        }

        for ext, parser in self.parsers.items():
            info["parser_details"][ext] = {
                "class_name": parser.__class__.__name__,
                "module": parser.__class__.__module__,
            }

        return info
