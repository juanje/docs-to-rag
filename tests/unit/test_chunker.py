"""Tests for text chunking functionality."""

import pytest

from src.document_processor.chunker import TextChunk, TextChunker
from src.document_processor.summarizer import SummaryChunk


@pytest.fixture
def sample_text_chunk():
    """Create a sample TextChunk for testing."""
    return TextChunk(
        content="Test content",
        source_file="test.md",
        file_type="markdown",
        start_pos=0,
        end_pos=12,
        chunk_id="test_1",
        chunk_type="original_text",
        level="chunk",
        metadata={},
    )


@pytest.fixture
def sample_summary_chunk():
    """Create a sample SummaryChunk for testing."""
    return SummaryChunk(
        content="This is a summary",
        source_file="test.md",
        chunk_type="document_summary",
        level="document",
        metadata={"type": "executive", "language": "en"},
    )


@pytest.fixture
def chunker():
    """Create a TextChunker instance."""
    return TextChunker()


@pytest.fixture
def custom_chunker():
    """Create a TextChunker with custom parameters."""
    return TextChunker(chunk_size=500, overlap=100)


# TextChunk tests


def test_text_chunk_creation_with_all_fields(sample_text_chunk):
    """Test creating a TextChunk with all fields."""
    chunk = sample_text_chunk

    assert chunk.content == "Test content"
    assert chunk.source_file == "test.md"
    assert chunk.start_pos == 0
    assert chunk.end_pos == 12
    assert chunk.chunk_id == "test_1"
    assert chunk.chunk_type == "original_text"
    assert chunk.level == "chunk"


def test_text_chunk_has_sensible_defaults():
    """Test TextChunk with default values."""
    chunk = TextChunk(
        content="Test content",
        source_file="test.md",
        file_type="markdown",
        start_pos=0,
        end_pos=12,
        chunk_id="test_default",
        metadata={},
    )

    assert chunk.chunk_type == "original_text"
    assert chunk.level == "chunk"
    assert chunk.chapter_number is None


# SummaryChunk tests


def test_summary_chunk_creation(sample_summary_chunk):
    """Test creating a SummaryChunk."""
    chunk = sample_summary_chunk

    assert chunk.content == "This is a summary"
    assert chunk.chunk_type == "document_summary"
    assert chunk.level == "document"
    assert chunk.metadata["type"] == "executive"


# TextChunker tests


def test_chunker_initialization_with_custom_parameters(custom_chunker):
    """Test chunker initialization with custom parameters."""
    assert custom_chunker.chunk_size == 500
    assert custom_chunker.overlap == 100


def test_chunker_has_reasonable_defaults(chunker):
    """Test chunker with default parameters."""
    # Check actual default values from implementation
    assert chunker.chunk_size > 0
    assert chunker.overlap >= 0
    assert chunker.overlap < chunker.chunk_size


def test_chunk_short_document_creates_single_chunk(chunker):
    """Test chunking a short document that doesn't need splitting."""
    document = {
        "source_file": "short.md",
        "file_type": "markdown",
        "plain_text": "This is a short document.",
        "metadata": {},
    }

    chunks = chunker.chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].content == "This is a short document."
    assert chunks[0].source_file == "short.md"
    assert chunks[0].start_pos == 0
    assert chunks[0].end_pos == 25


def test_chunk_long_document_creates_multiple_chunks(custom_chunker):
    """Test chunking a long document that needs splitting."""
    long_text = "This is a sentence. " * 100  # 2000 characters
    document = {
        "source_file": "long.md",
        "file_type": "markdown",
        "plain_text": long_text,
        "metadata": {},
    }

    chunks = custom_chunker.chunk_document(document)

    assert len(chunks) > 1
    assert all(chunk.source_file == "long.md" for chunk in chunks)
    assert all(len(chunk.content) <= 600 for chunk in chunks)  # chunk_size + margin


def test_chunk_empty_document_returns_empty_list(chunker):
    """Test chunking an empty document."""
    document = {
        "source_file": "empty.md",
        "file_type": "markdown",
        "plain_text": "",
        "metadata": {},
    }

    chunks = chunker.chunk_document(document)

    assert len(chunks) == 0


def test_chunk_document_missing_content_returns_empty_list(chunker):
    """Test chunking a document without plain_text field."""
    document = {"source_file": "missing.md", "file_type": "markdown", "metadata": {}}

    chunks = chunker.chunk_document(document)

    assert len(chunks) == 0


def test_chunk_multiple_documents(chunker):
    """Test chunking multiple documents."""
    documents = [
        {
            "source_file": "doc1.md",
            "plain_text": "First document content.",
            "metadata": {},
        },
        {
            "source_file": "doc2.md",
            "plain_text": "Second document content.",
            "metadata": {},
        },
    ]

    chunks = chunker.chunk_multiple_documents(documents)

    assert len(chunks) == 2
    assert chunks[0].source_file == "doc1.md"
    assert chunks[1].source_file == "doc2.md"
    assert "First document" in chunks[0].content
    assert "Second document" in chunks[1].content


def test_create_summary_chunk_from_summary_chunk(chunker, sample_summary_chunk):
    """Test creating a TextChunk from SummaryChunk."""
    text_chunk = chunker.create_summary_chunk(sample_summary_chunk)

    assert isinstance(text_chunk, TextChunk)
    assert text_chunk.content == "This is a summary"
    assert text_chunk.source_file == "test.md"
    assert text_chunk.chunk_type == "document_summary"
    assert "test_document_summary_" in text_chunk.chunk_id


def test_chunking_respects_sentence_boundaries():
    """Test that chunking tries to respect sentence boundaries."""
    # Create a document with clear sentence boundaries
    sentences = [
        "This is the first sentence. ",
        "This is the second sentence. ",
        "This is the third sentence. ",
        "This is the fourth sentence. ",
    ]
    long_text = "".join(sentences * 10)  # Repeat to make it long enough

    document = {"source_file": "sentences.md", "plain_text": long_text, "metadata": {}}

    chunker = TextChunker(chunk_size=200, overlap=50)
    chunks = chunker.chunk_document(document)

    # Verify chunks were created
    assert len(chunks) > 1

    # Most chunks should end at sentence boundaries (periods followed by space)
    sentence_boundary_count = 0
    for chunk in chunks[:-1]:  # Exclude last chunk
        if chunk.content.endswith(". ") or chunk.content.rstrip().endswith("."):
            sentence_boundary_count += 1

    # At least some chunks should respect sentence boundaries
    assert sentence_boundary_count > 0


def test_chunk_overlap_functionality():
    """Test that chunks have proper overlap."""
    # Create text that will definitely be split
    sentences = [
        "Sentence one. ",
        "Sentence two. ",
        "Sentence three. ",
        "Sentence four. ",
        "Sentence five. ",
    ]
    long_text = "".join(sentences * 10)  # Repeat to make it long

    document = {
        "source_file": "overlap_test.md",
        "plain_text": long_text,
        "metadata": {},
    }

    chunker = TextChunker(chunk_size=200, overlap=50)
    chunks = chunker.chunk_document(document)

    if len(chunks) > 1:
        # Check that there's some overlap between consecutive chunks
        chunk1_end = chunks[0].content[-50:]  # Last 50 chars of first chunk
        chunk2_start = chunks[1].content[:50]  # First 50 chars of second chunk

        # There should be some common content
        overlap_found = any(
            word in chunk2_start for word in chunk1_end.split() if len(word) > 3
        )
        assert overlap_found, "No overlap found between consecutive chunks"


@pytest.mark.parametrize(
    ("chunk_size", "overlap", "text_length", "expected_chunks"),
    [
        (100, 20, 50, 1),  # Short text, single chunk
        (100, 20, 300, 3),  # Medium text, multiple chunks
        (500, 100, 1500, 3),  # Longer chunks, fewer splits
    ],
)
def test_chunking_parameters_affect_output(
    chunk_size, overlap, text_length, expected_chunks
):
    """Test that different chunking parameters affect the output appropriately."""
    # Create text of specified length
    text = "Word " * (text_length // 5)  # Approximate target length

    document = {"source_file": "test.md", "plain_text": text, "metadata": {}}

    chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)
    chunks = chunker.chunk_document(document)

    # Should be close to expected number of chunks (allow some variance)
    # Chunking algorithm may create more chunks due to overlap and boundaries
    assert abs(len(chunks) - expected_chunks) <= 2


def test_chunks_preserve_metadata():
    """Test that chunks preserve document metadata."""
    document = {
        "source_file": "test.md",
        "file_type": "markdown",
        "plain_text": "Test content for metadata preservation.",
        "metadata": {"author": "test", "date": "2024-01-01"},
    }

    chunker = TextChunker()
    chunks = chunker.chunk_document(document)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.source_file == "test.md"
    assert chunk.file_type == "markdown"
    assert "author" in chunk.metadata["source_metadata"]
    assert "date" in chunk.metadata["source_metadata"]
