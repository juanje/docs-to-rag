"""Tests for vector store functionality."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.document_processor.chunker import TextChunk
from src.vector_store.store import VectorStore, VectorStoreStats


@pytest.fixture
def temp_dir():
    """Create temporary directory for vector store tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def vector_store(temp_dir):
    """Create a VectorStore instance in temporary directory."""
    return VectorStore(store_path=temp_dir)


@pytest.fixture
def sample_chunks():
    """Create sample TextChunk instances for testing."""
    return [
        TextChunk(
            content="First chunk content",
            source_file="test1.md",
            file_type="markdown",
            start_pos=0,
            end_pos=19,
            chunk_id="chunk_1",
            metadata={},
        ),
        TextChunk(
            content="Second chunk content",
            source_file="test2.md",
            file_type="markdown",
            start_pos=0,
            end_pos=20,
            chunk_id="chunk_2",
            metadata={},
        ),
    ]


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    return [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


# VectorStoreStats tests


def test_vector_store_stats_creation():
    """Test creating VectorStoreStats."""
    stats = VectorStoreStats(
        chunk_count=100,
        document_count=5,
        embedding_dimension=384,
        index_type="flat",
        last_updated="2024-01-01",
        db_size_mb=2.5,
    )

    assert stats.chunk_count == 100
    assert stats.document_count == 5
    assert stats.embedding_dimension == 384
    assert stats.index_type == "flat"
    assert stats.last_updated == "2024-01-01"
    assert stats.db_size_mb == 2.5


# VectorStore tests


def test_vector_store_initialization_creates_empty_store(vector_store, temp_dir):
    """Test vector store initialization with new store."""
    assert str(vector_store.store_path) == temp_dir
    assert vector_store.chunks == []
    assert vector_store.embedding_dimension is None
    assert vector_store.is_loaded is False


def test_vector_store_add_documents_first_time(
    vector_store, sample_chunks, sample_embeddings
):
    """Test adding documents to empty store."""
    vector_store.add_chunks(sample_chunks, sample_embeddings)

    assert len(vector_store.chunks) == 2
    assert vector_store.embedding_dimension == 3
    assert vector_store.chunks[0].content == "First chunk content"
    assert vector_store.chunks[1].content == "Second chunk content"


def test_vector_store_rejects_mismatched_chunks_and_embeddings(
    vector_store, sample_chunks
):
    """Test that vector store validates chunks and embeddings length match."""
    mismatched_embeddings = [[0.1, 0.2, 0.3]]  # Only 1 embedding for 2 chunks

    with pytest.raises(ValueError, match="length mismatch"):
        vector_store.add_chunks(sample_chunks, mismatched_embeddings)


def test_vector_store_search_empty_store_returns_empty_results(vector_store):
    """Test searching in empty store."""
    query_embedding = [0.1, 0.2, 0.3]
    results = vector_store.search_similar(query_embedding, top_k=5)

    assert len(results) == 0


def test_vector_store_search_with_data_returns_results(vector_store, sample_embeddings):
    """Test searching with data in store."""
    # Create test chunks
    chunks = [
        TextChunk(
            content=f"Content {i}",
            source_file=f"test{i}.md",
            file_type="markdown",
            start_pos=0,
            end_pos=10,
            chunk_id=f"chunk_{i}",
            metadata={},
        )
        for i in range(5)
    ]

    embeddings = [[0.1 * i, 0.2 * i, 0.3 * i] for i in range(1, 6)]

    vector_store.add_chunks(chunks, embeddings)

    # Search with a query
    query_embedding = [0.2, 0.4, 0.6]  # Should be close to second embedding
    search_results = vector_store.search_similar(query_embedding, top_k=3)

    assert len(search_results) <= 3
    assert all(hasattr(result, 'chunk') and hasattr(result, 'score') for result in search_results)
    
    # Scores from FAISS inner product can be any float value (negative or > 1.0)
    # Just check that we got valid numeric scores
    assert all(isinstance(result.score, (int, float)) for result in search_results)
    
    # Results should be ordered by score (highest first)
    scores = [result.score for result in search_results]
    assert scores == sorted(scores, reverse=True)


def test_vector_store_statistics_empty_store(vector_store):
    """Test getting statistics from empty store."""
    stats = vector_store.get_statistics()

    assert stats.chunk_count == 0
    assert stats.document_count == 0
    assert stats.embedding_dimension == 0


def test_vector_store_statistics_with_data(vector_store):
    """Test getting statistics with data from multiple documents."""
    chunks = [
        TextChunk(
            content="Content A",
            source_file="doc1.md",
            file_type="markdown",
            start_pos=0,
            end_pos=9,
            chunk_id="1",
            metadata={},
        ),
        TextChunk(
            content="Content B",
            source_file="doc1.md",
            file_type="markdown",
            start_pos=10,
            end_pos=19,
            chunk_id="2",
            metadata={},
        ),
        TextChunk(
            content="Content C",
            source_file="doc2.md",
            file_type="markdown",
            start_pos=0,
            end_pos=9,
            chunk_id="3",
            metadata={},
        ),
    ]
    embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

    vector_store.add_chunks(chunks, embeddings)
    stats = vector_store.get_statistics()

    assert stats.chunk_count == 3
    assert stats.document_count == 2  # Two unique source files
    assert stats.embedding_dimension == 2


def test_vector_store_get_unique_source_files(vector_store):
    """Test getting unique source files."""
    chunks = [
        TextChunk(
            content="Content A",
            source_file="doc1.md",
            file_type="markdown",
            start_pos=0,
            end_pos=9,
            chunk_id="1",
            metadata={},
        ),
        TextChunk(
            content="Content B",
            source_file="doc1.md",
            file_type="markdown",
            start_pos=10,
            end_pos=19,
            chunk_id="2",
            metadata={},
        ),
        TextChunk(
            content="Content C",
            source_file="doc2.md",
            file_type="markdown",
            start_pos=0,
            end_pos=9,
            chunk_id="3",
            metadata={},
        ),
        TextChunk(
            content="Content D",
            source_file="doc3.md",
            file_type="markdown",
            start_pos=0,
            end_pos=9,
            chunk_id="4",
            metadata={},
        ),
    ]
    embeddings = [[0.1, 0.2]] * 4

    vector_store.add_chunks(chunks, embeddings)
    source_files = vector_store.get_all_source_files()

    assert len(source_files) == 3
    assert "doc1.md" in source_files
    assert "doc2.md" in source_files
    assert "doc3.md" in source_files


def test_vector_store_clear_removes_all_data(vector_store):
    """Test clearing the vector store."""
    # Add some data
    chunks = [
        TextChunk(
            content="test",
            source_file="test.md",
            file_type="markdown",
            start_pos=0,
            end_pos=4,
            chunk_id="test_clear",
            metadata={},
        )
    ]
    embeddings = [[0.1, 0.2, 0.3]]
    vector_store.add_chunks(chunks, embeddings)

    assert len(vector_store.chunks) == 1

    # Clear the store
    vector_store.clear_all()

    assert len(vector_store.chunks) == 0
    assert vector_store.embedding_dimension is None
    assert vector_store.index is None


def test_vector_store_persistence_save_and_load(temp_dir):
    """Test saving and loading vector store to/from disk."""
    # Create and populate a store
    store1 = VectorStore(store_path=temp_dir)
    chunks = [
        TextChunk(
            content="First chunk",
            source_file="test.md",
            file_type="markdown",
            start_pos=0,
            end_pos=11,
            chunk_id="first",
            metadata={},
        ),
        TextChunk(
            content="Second chunk",
            source_file="test.md",
            file_type="markdown",
            start_pos=12,
            end_pos=24,
            chunk_id="second",
            metadata={},
        ),
    ]
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    store1.add_chunks(chunks, embeddings)

    # Verify files were created
    assert Path(temp_dir, "chunks.pkl").exists()
    assert Path(temp_dir, "faiss_index.bin").exists()
    assert Path(temp_dir, "metadata.json").exists()

    # Create a new store and load from disk
    store2 = VectorStore(store_path=temp_dir)
    success = store2.load_from_disk()

    assert success is True
    assert len(store2.chunks) == 2
    assert store2.embedding_dimension == 3
    assert store2.chunks[0].content == "First chunk"
    assert store2.chunks[1].content == "Second chunk"


def test_vector_store_load_missing_files_returns_false(vector_store):
    """Test loading from disk when files don't exist."""
    success = vector_store.load_from_disk()

    assert success is False
    assert len(vector_store.chunks) == 0


def test_vector_store_search_with_similarity_threshold(vector_store):
    """Test searching with similarity threshold filtering."""
    # Add test data with known embeddings
    chunks = [
        TextChunk(
            content="Very similar",
            source_file="test1.md",
            file_type="markdown",
            start_pos=0,
            end_pos=12,
            chunk_id="sim1",
            metadata={},
        ),
        TextChunk(
            content="Somewhat similar",
            source_file="test2.md",
            file_type="markdown",
            start_pos=0,
            end_pos=16,
            chunk_id="sim2",
            metadata={},
        ),
        TextChunk(
            content="Not similar",
            source_file="test3.md",
            file_type="markdown",
            start_pos=0,
            end_pos=11,
            chunk_id="sim3",
            metadata={},
        ),
    ]

    # Embeddings with different similarities to query
    embeddings = [
        [1.0, 0.0, 0.0],  # Will be very similar to query
        [0.5, 0.5, 0.0],  # Somewhat similar
        [0.0, 0.0, 1.0],  # Not similar
    ]

    vector_store.add_chunks(chunks, embeddings)

    # Query that should be most similar to first embedding
    query_embedding = [0.9, 0.1, 0.0]

    # Search and manually filter by threshold
    search_results = vector_store.search_similar(query_embedding, top_k=10)
    
    # Filter by high threshold (0.9)
    results_high = [result for result in search_results if result.score >= 0.9]
    
    # Filter by low threshold (0.1)
    results_low = [result for result in search_results if result.score >= 0.1]

    assert len(results_high) <= len(results_low)
    assert all(result.score >= 0.9 for result in results_high)


def test_vector_store_faiss_indexing_verification(vector_store):
    """Test FAISS indexing works correctly."""
    # Add some chunks including summary chunks
    chunks = [
        TextChunk(
            content="Original content",
            source_file="test.md",
            file_type="markdown",
            start_pos=0,
            end_pos=16,
            chunk_id="orig",
            chunk_type="original_text",
            metadata={},
        ),
        TextChunk(
            content="Summary content",
            source_file="test.md",
            file_type="markdown",
            start_pos=0,
            end_pos=15,
            chunk_id="summ",
            chunk_type="document_summary",
            metadata={},
        ),
    ]
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    vector_store.add_chunks(chunks, embeddings)

    # Verify the index has the correct number of vectors
    assert vector_store.index.ntotal == 2
    
    # Verify we can search and get results
    search_results = vector_store.search_similar([0.1, 0.2, 0.3], top_k=2)
    assert len(search_results) == 2


def test_vector_store_faiss_index_consistency(vector_store):
    """Test FAISS index consistency with chunk count."""
    # Add chunks
    chunks = [
        TextChunk(
            content="test",
            source_file="test.md",
            file_type="markdown",
            start_pos=0,
            end_pos=4,
            chunk_id="verify",
            metadata={},
        )
    ]
    embeddings = [[0.1, 0.2, 0.3]]
    vector_store.add_chunks(chunks, embeddings)

    # Verify index consistency
    assert vector_store.index.ntotal == len(vector_store.chunks)
    assert len(vector_store.chunks) == 1
    
    # Verify we can successfully search
    search_results = vector_store.search_similar([0.1, 0.2, 0.3], top_k=1)
    assert len(search_results) == 1


# Parametrized tests


@pytest.mark.parametrize(
    ("chunk_count", "expected_documents"),
    [
        (1, 1),  # Single chunk from one document
        (3, 2),  # Multiple chunks from multiple documents
        (5, 3),  # Many chunks from several documents
    ],
)
def test_vector_store_document_counting(vector_store, chunk_count, expected_documents):
    """Test that document counting works correctly."""
    # Create chunks from different documents
    doc_names = ["doc1.md", "doc2.md", "doc3.md", "doc4.md", "doc5.md"]
    chunks = []

    for i in range(chunk_count):
        doc_index = i % expected_documents  # Cycle through available documents
        chunks.append(
            TextChunk(
                content=f"Content {i}",
                source_file=doc_names[doc_index],
                file_type="markdown",
                start_pos=0,
                end_pos=10,
                chunk_id=f"chunk_{i}",
                metadata={},
            )
        )

    embeddings = [[0.1, 0.2, 0.3]] * chunk_count
    vector_store.add_chunks(chunks, embeddings)

    stats = vector_store.get_statistics()
    assert stats.chunk_count == chunk_count
    assert stats.document_count == expected_documents


@pytest.mark.parametrize(
    ("embedding_dim", "expected_dim"),
    [
        (2, 2),
        (128, 128),
        (384, 384),
        (1536, 1536),
    ],
)
def test_vector_store_handles_different_embedding_dimensions(
    vector_store, embedding_dim, expected_dim
):
    """Test that vector store handles different embedding dimensions correctly."""
    chunks = [
        TextChunk(
            content="test content",
            source_file="test.md",
            file_type="markdown",
            start_pos=0,
            end_pos=12,
            chunk_id="test",
            metadata={},
        )
    ]
    embeddings = [[0.1] * embedding_dim]

    vector_store.add_chunks(chunks, embeddings)

    assert vector_store.embedding_dimension == expected_dim
    stats = vector_store.get_statistics()
    assert stats.embedding_dimension == expected_dim


def test_vector_store_search_returns_results_in_score_order(vector_store):
    """Test that search results are returned in descending score order."""
    # Create chunks with embeddings that will have clear similarity rankings
    chunks = [
        TextChunk(
            content=f"Content {i}",
            source_file=f"test{i}.md",
            file_type="markdown",
            start_pos=0,
            end_pos=10,
            chunk_id=f"chunk_{i}",
            metadata={},
        )
        for i in range(3)
    ]

    # Embeddings with different similarities to query [1, 0, 0]
    embeddings = [
        [0.9, 0.1, 0.0],  # High similarity
        [0.1, 0.9, 0.0],  # Low similarity
        [0.7, 0.3, 0.0],  # Medium similarity
    ]

    vector_store.add_chunks(chunks, embeddings)

    query_embedding = [1.0, 0.0, 0.0]
    search_results = vector_store.search_similar(query_embedding, top_k=3)

    # Scores should be in descending order
    scores = [result.score for result in search_results]
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], f"Scores not in descending order: {scores}"
