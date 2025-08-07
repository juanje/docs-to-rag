"""
Markdown document parser for extracting structured content.

This parser handles Markdown files with special attention to preserving
structure like headers, code blocks, and frontmatter metadata.
"""

import re
from pathlib import Path
from typing import Any

import markdown


class MarkdownParser:
    """Parser for extracting content from Markdown documents."""

    def __init__(self):
        """Initialize the Markdown parser with useful extensions."""
        self.md = markdown.Markdown(
            extensions=[
                "meta",  # For frontmatter
                "toc",  # Table of contents
                "tables",  # Table support
                "fenced_code",  # Code blocks
                "codehilite",  # Code highlighting
            ],
            extension_configs={
                "codehilite": {
                    "use_pygments": False,  # Avoid external dependency
                }
            },
        )

    def extract_content(self, file_path: str) -> dict[str, Any]:
        """
        Extract structured content from a Markdown file.

        Args:
            file_path: Path to the Markdown file

        Returns:
            Dictionary containing extracted content and metadata
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Markdown file not found: {file_path}")

        # Read file content
        with open(path, encoding="utf-8") as f:
            content = f.read()

        # Parse with markdown
        html_content = self.md.convert(content)

        # Extract metadata (frontmatter)
        metadata = getattr(self.md, "Meta", {})

        # Process metadata to simple strings
        processed_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, list) and len(value) == 1:
                processed_metadata[key] = value[0]
            else:
                processed_metadata[key] = value

        # Extract plain text (remove markdown formatting)
        plain_text = self._convert_to_plain_text(content)

        # Extract structural elements
        structure = self._extract_structure(content)

        return {
            "source_file": str(path),
            "file_type": "markdown",
            "raw_content": content,
            "plain_text": plain_text,
            "html_content": html_content,
            "metadata": processed_metadata,
            "structure": structure,
            "size_bytes": len(content.encode("utf-8")),
        }

    def _convert_to_plain_text(self, content: str) -> str:
        """
        Convert Markdown content to plain text.

        Args:
            content: Raw Markdown content

        Returns:
            Plain text with Markdown formatting removed
        """
        # Remove frontmatter
        content = re.sub(
            r"^---\n.*?\n---\n", "", content, flags=re.MULTILINE | re.DOTALL
        )

        # Remove headers (keep text)
        content = re.sub(r"^#{1,6}\s*", "", content, flags=re.MULTILINE)

        # Remove emphasis markers
        content = re.sub(r"\*\*(.*?)\*\*", r"\1", content)  # Bold
        content = re.sub(r"\*(.*?)\*", r"\1", content)  # Italic
        content = re.sub(r"__(.*?)__", r"\1", content)  # Bold
        content = re.sub(r"_(.*?)_", r"\1", content)  # Italic

        # Remove code blocks
        content = re.sub(r"```.*?\n(.*?)\n```", r"\1", content, flags=re.DOTALL)
        content = re.sub(r"`([^`]+)`", r"\1", content)  # Inline code

        # Remove links (keep text)
        content = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", content)

        # Remove images
        content = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", r"\1", content)

        # Remove horizontal rules
        content = re.sub(r"^---+$", "", content, flags=re.MULTILINE)

        # Remove list markers
        content = re.sub(r"^\s*[-*+]\s*", "", content, flags=re.MULTILINE)
        content = re.sub(r"^\s*\d+\.\s*", "", content, flags=re.MULTILINE)

        # Clean up extra whitespace
        content = re.sub(r"\n\s*\n\s*\n", "\n\n", content)
        content = content.strip()

        return content

    def _extract_structure(self, content: str) -> dict[str, Any]:
        """
        Extract structural information from Markdown content.

        Args:
            content: Raw Markdown content

        Returns:
            Dictionary with structural information
        """
        # Extract headers
        headers = []
        header_pattern = r"^(#{1,6})\s*(.+)$"
        for match in re.finditer(header_pattern, content, re.MULTILINE):
            level = len(match.group(1))
            text = match.group(2).strip()
            headers.append({"level": level, "text": text, "position": match.start()})

        # Extract code blocks
        code_blocks = []
        code_pattern = r"```(\w+)?\n(.*?)\n```"
        for match in re.finditer(code_pattern, content, re.DOTALL):
            language = match.group(1) or "text"
            code = match.group(2)
            code_blocks.append(
                {"language": language, "code": code, "position": match.start()}
            )

        # Extract links
        links = []
        link_pattern = r"\[([^\]]+)\]\(([^\)]+)\)"
        for match in re.finditer(link_pattern, content):
            text = match.group(1)
            url = match.group(2)
            links.append({"text": text, "url": url, "position": match.start()})

        return {
            "headers": headers,
            "code_blocks": code_blocks,
            "links": links,
            "total_headers": len(headers),
            "total_code_blocks": len(code_blocks),
            "total_links": len(links),
        }

    def get_chunk_boundaries(self, content: str) -> list[int]:
        """
        Get intelligent chunk boundaries based on Markdown structure.

        Args:
            content: Markdown content

        Returns:
            List of character positions for chunk boundaries
        """
        boundaries = [0]  # Start with beginning of document

        # Add boundaries at major headers (# and ##)
        header_pattern = r"^(#{1,2})\s*"
        for match in re.finditer(header_pattern, content, re.MULTILINE):
            boundaries.append(match.start())

        # Add boundaries at horizontal rules
        hr_pattern = r"^---+$"
        for match in re.finditer(hr_pattern, content, re.MULTILINE):
            boundaries.append(match.start())

        # Sort and deduplicate
        boundaries = sorted(set(boundaries))

        # Add end of document
        if boundaries[-1] != len(content):
            boundaries.append(len(content))

        return boundaries
