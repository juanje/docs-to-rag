"""
HTML document parser for extracting clean content.

This parser handles HTML files by removing scripts, styles, and other
non-content elements while preserving semantic structure.
"""

import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Comment


class HTMLParser:
    """Parser for extracting content from HTML documents."""

    def __init__(self):
        """Initialize the HTML parser."""
        self.soup = None

    def extract_content(self, file_path: str) -> dict[str, Any]:
        """
        Extract clean content from an HTML file.

        Args:
            file_path: Path to the HTML file

        Returns:
            Dictionary containing extracted content and metadata
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"HTML file not found: {file_path}")

        # Read file with encoding detection
        content = self._read_html_file(path)

        # Parse with BeautifulSoup
        self.soup = BeautifulSoup(content, "html.parser")

        # Clean the content
        self._clean_html()

        # Extract components
        plain_text = self._extract_text()
        metadata = self._extract_metadata()
        structure = self._extract_structure()

        return {
            "source_file": str(path),
            "file_type": "html",
            "raw_content": content,
            "plain_text": plain_text,
            "metadata": metadata,
            "structure": structure,
            "size_bytes": len(content.encode("utf-8")),
        }

    def _read_html_file(self, path: Path) -> str:
        """
        Read HTML file with proper encoding detection.

        Args:
            path: Path to HTML file

        Returns:
            HTML content as string
        """
        # Try common encodings
        encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]

        for encoding in encodings:
            try:
                with open(path, encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        # Fallback: read as binary and decode with errors='replace'
        with open(path, "rb") as f:
            return f.read().decode("utf-8", errors="replace")

    def _clean_html(self) -> None:
        """Remove unwanted elements from HTML soup."""
        if not self.soup:
            return

        # Remove script and style elements
        for script in self.soup(["script", "style"]):
            script.decompose()

        # Remove comments
        for comment in self.soup.find_all(
            string=lambda text: isinstance(text, Comment)
        ):
            comment.extract()

        # Remove common non-content elements
        unwanted_tags = [
            "nav",
            "header",
            "footer",
            "aside",
            "menu",
            "button",
            "form",
            "input",
            "select",
            "textarea",
        ]
        for tag in unwanted_tags:
            for element in self.soup.find_all(tag):
                element.decompose()

        # Remove elements with common non-content classes/ids
        unwanted_selectors = [
            '[class*="nav"]',
            '[class*="menu"]',
            '[class*="sidebar"]',
            '[class*="footer"]',
            '[class*="header"]',
            '[class*="ad"]',
            '[id*="nav"]',
            '[id*="menu"]',
            '[id*="sidebar"]',
            '[id*="footer"]',
            '[id*="header"]',
            '[id*="ad"]',
        ]
        for selector in unwanted_selectors:
            for element in self.soup.select(selector):
                element.decompose()

    def _extract_text(self) -> str:
        """
        Extract clean text content from HTML.

        Returns:
            Plain text with proper spacing
        """
        if not self.soup:
            return ""

        # Get text with spacing preserved
        text = self.soup.get_text(separator="\n", strip=True)

        # Clean up excessive whitespace
        text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)  # Multiple newlines
        text = re.sub(r"[ \t]+", " ", text)  # Multiple spaces/tabs
        text = text.strip()

        return text

    def _extract_metadata(self) -> dict[str, Any]:
        """
        Extract metadata from HTML head section.

        Returns:
            Dictionary with page metadata
        """
        metadata = {}

        if not self.soup:
            return metadata

        # Page title
        title_tag = self.soup.find("title")
        if title_tag:
            metadata["title"] = title_tag.get_text().strip()

        # Meta tags
        meta_tags = self.soup.find_all("meta")
        for meta in meta_tags:
            name = meta.get("name", "").lower()
            property_attr = meta.get("property", "").lower()
            content = meta.get("content", "")

            if name and content:
                metadata[name] = content
            elif property_attr and content:
                # OpenGraph and similar properties
                metadata[property_attr] = content

        # Language
        html_tag = self.soup.find("html")
        if html_tag and html_tag.get("lang"):
            metadata["language"] = html_tag.get("lang")

        return metadata

    def _extract_structure(self) -> dict[str, Any]:
        """
        Extract structural information from HTML.

        Returns:
            Dictionary with structural elements
        """
        structure = {
            "headings": [],
            "links": [],
            "images": [],
            "paragraphs": 0,
            "lists": 0,
            "tables": 0,
        }

        if not self.soup:
            return structure

        # Extract headings
        for i in range(1, 7):  # h1 to h6
            headings = self.soup.find_all(f"h{i}")
            for heading in headings:
                structure["headings"].append(
                    {
                        "level": i,
                        "text": heading.get_text().strip(),
                        "id": heading.get("id", ""),
                    }
                )

        # Extract links
        links = self.soup.find_all("a", href=True)
        for link in links:
            href = link.get("href", "")
            text = link.get_text().strip()
            if text and href:
                structure["links"].append(
                    {
                        "text": text,
                        "url": href,
                        "is_external": href.startswith(("http://", "https://")),
                    }
                )

        # Extract images
        images = self.soup.find_all("img", src=True)
        for img in images:
            src = img.get("src", "")
            alt = img.get("alt", "")
            if src:
                structure["images"].append({"src": src, "alt": alt})

        # Count structural elements
        structure["paragraphs"] = len(self.soup.find_all("p"))
        structure["lists"] = len(self.soup.find_all(["ul", "ol"]))
        structure["tables"] = len(self.soup.find_all("table"))

        return structure

    def get_chunk_boundaries(self, content: str) -> list[int]:
        """
        Get intelligent chunk boundaries based on HTML structure.

        Args:
            content: Plain text content

        Returns:
            List of character positions for chunk boundaries
        """
        boundaries = [0]

        if not self.soup:
            return boundaries

        # Find positions of major structural elements in the plain text
        headings = self.soup.find_all(["h1", "h2", "h3"])
        for heading in headings:
            heading_text = heading.get_text().strip()
            if heading_text in content:
                pos = content.find(heading_text)
                if pos != -1:
                    boundaries.append(pos)

        # Add boundaries at paragraph breaks for long content sections
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
