"""Tests for CLI functionality."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from src.cli.commands import main


@pytest.fixture
def cli_runner():
    """Create a Click CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def mock_rag_pipeline():
    """Mock RAGPipeline class."""
    with patch("src.rag.pipeline.RAGPipeline", autospec=True) as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_document_extractor():
    """Mock DocumentExtractor class."""
    with patch("src.document_processor.extractor.DocumentExtractor", autospec=True) as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_chat_interface():
    """Mock ChatInterface class."""
    with patch("src.cli.chat.ChatInterface", autospec=True) as mock_class:
        mock_instance = Mock()
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_settings():
    """Mock settings object."""
    with patch("src.cli.commands.settings") as mock_settings:
        mock_settings.primary_language = "es"
        mock_settings.auto_detect_language = True
        mock_settings.embedding_model = "test-model"
        mock_settings.embedding_model_multilingual = "test-multilingual-model"
        mock_settings.enable_summarization = True
        mock_settings.generate_document_summaries = True
        mock_settings.generate_chapter_summaries = True
        mock_settings.generate_concept_summaries = True
        mock_settings.max_concepts_per_document = 8
        mock_settings.summarization_model = "test-summary-model"
        mock_settings.summarization_temperature = 0.1
        mock_settings.summarization_top_p = 0.8
        mock_settings.max_summary_retries = 2
        mock_settings.enable_summary_validation = True
        mock_settings.summary_faithfulness_check = True
        mock_settings.save_user_config = Mock()
        yield mock_settings


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Test Document\n\nTest content.")
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


# Basic CLI tests


def test_cli_help_command(cli_runner):
    """Test CLI help command shows usage information."""
    result = cli_runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "docs-to-rag" in result.output
    assert "Educational RAG system" in result.output


def test_cli_version_command(cli_runner):
    """Test CLI version command shows version."""
    result = cli_runner.invoke(main, ["--version"])

    assert result.exit_code == 0
    assert "0.1.0" in result.output


# Setup command tests


def test_setup_command_success(cli_runner, mock_rag_pipeline):
    """Test setup command with successful system check."""
    mock_rag_pipeline.check_readiness.return_value = {"is_ready": True, "issues": []}
    mock_rag_pipeline.get_system_stats.return_value = {
        "models": {
            "embedding": {"model_name": "test-embed", "is_available": True},
            "generation": {"model_name": "test-gen", "is_available": True},
        },
        "vector_store": {"document_count": 0, "chunk_count": 0},
    }

    result = cli_runner.invoke(main, ["setup"])

    assert result.exit_code == 0
    assert "Setup completed successfully" in result.output


def test_setup_command_with_system_issues(cli_runner, mock_rag_pipeline):
    """Test setup command when system has issues."""
    mock_rag_pipeline.check_readiness.return_value = {
        "is_ready": False,
        "issues": ["Ollama not running", "Model not found"],
    }

    result = cli_runner.invoke(main, ["setup"])

    assert result.exit_code == 1
    assert "System setup issues detected" in result.output
    assert "Ollama not running" in result.output


# Add command tests


def test_add_command_success(
    cli_runner, mock_document_extractor, mock_rag_pipeline, temp_file
):
    """Test add command with successful document processing."""
    mock_document_extractor.is_supported_file.return_value = True
    mock_document_extractor.extract_document.return_value = {
        "plain_text": "Test content",
        "source_file": temp_file,
    }

    mock_rag_pipeline.add_documents.return_value = {
        "success": True,
        "documents_processed": 1,
        "chunks_created": 2,
        "embedding_time": 1.5,
        "system_stats": {"vector_store": {"chunk_count": 2}},
    }

    result = cli_runner.invoke(main, ["add", temp_file])

    assert result.exit_code == 0
    assert "Successfully processed 1 documents" in result.output


def test_add_command_unsupported_file_type(
    cli_runner, mock_document_extractor, temp_file
):
    """Test add command with unsupported file type."""
    mock_document_extractor.is_supported_file.return_value = False

    result = cli_runner.invoke(main, ["add", temp_file])

    assert result.exit_code == 0
    assert "Unsupported file type" in result.output


def test_add_command_nonexistent_file(cli_runner):
    """Test add command with non-existent file."""
    result = cli_runner.invoke(main, ["add", "/nonexistent/file.md"])

    assert result.exit_code == 2  # Click error for invalid path


# List command tests


def test_list_command_empty_database(cli_runner, mock_rag_pipeline):
    """Test list command with no documents."""
    mock_rag_pipeline.retriever.get_source_files.return_value = []

    result = cli_runner.invoke(main, ["list"])

    assert result.exit_code == 0
    assert "No documents indexed yet" in result.output


def test_list_command_with_documents(cli_runner, mock_rag_pipeline):
    """Test list command showing indexed documents."""
    mock_rag_pipeline.retriever.get_source_files.return_value = [
        "/path/to/doc1.md",
        "/path/to/doc2.html",
    ]

    result = cli_runner.invoke(main, ["list"])

    assert result.exit_code == 0
    assert "Indexed Documents (2 files)" in result.output
    assert "doc1.md" in result.output
    assert "doc2.html" in result.output


# Clear command tests


def test_clear_command_with_confirmation(cli_runner, mock_rag_pipeline):
    """Test clear command with user confirmation."""
    result = cli_runner.invoke(main, ["clear"], input="y\n")

    assert result.exit_code == 0
    assert "Knowledge base cleared successfully" in result.output
    mock_rag_pipeline.clear_knowledge_base.assert_called_once()


def test_clear_command_with_rejection(cli_runner, mock_rag_pipeline):
    """Test clear command when user rejects."""
    result = cli_runner.invoke(main, ["clear"], input="n\n")

    assert result.exit_code == 1  # Click aborts
    mock_rag_pipeline.clear_knowledge_base.assert_not_called()


# Stats command tests


def test_stats_command_shows_system_information(cli_runner, mock_rag_pipeline):
    """Test stats command displays system information."""
    mock_rag_pipeline.get_system_stats.return_value = {
        "models": {
            "embedding": {"model_name": "test-embed", "is_available": True},
            "generation": {"model_name": "test-gen", "is_available": True},
        },
        "vector_store": {
            "document_count": 5,
            "chunk_count": 25,
            "embedding_dimension": 384,
            "db_size_mb": 2.5,
            "last_updated": "2024-01-01",
        },
        "settings": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "top_k_retrieval": 3,
            "similarity_threshold": 0.7,
            "temperature": 0.1,
            "max_tokens": 512,
        },
    }
    mock_rag_pipeline.check_readiness.return_value = {"is_ready": True, "issues": []}

    result = cli_runner.invoke(main, ["stats"])

    assert result.exit_code == 0
    assert "System Ready" in result.output


# Query command tests


def test_query_command_success(cli_runner, mock_rag_pipeline):
    """Test query command with successful response."""
    mock_rag_pipeline.check_readiness.return_value = {"is_ready": True, "issues": []}

    # Mock RAG result
    mock_result = Mock()
    mock_result.answer = "This is the answer to your question."
    mock_result.total_time = 2.5
    mock_result.chunks_retrieved = 3
    mock_rag_pipeline.ask.return_value = mock_result

    result = cli_runner.invoke(main, ["query", "What is this about?"])

    assert result.exit_code == 0
    assert "Answer:" in result.output
    assert "This is the answer" in result.output
    assert "2.50s" in result.output


def test_query_command_system_not_ready(cli_runner, mock_rag_pipeline):
    """Test query command when system is not ready."""
    mock_rag_pipeline.check_readiness.return_value = {
        "is_ready": False,
        "issues": ["Ollama not running"],
    }

    result = cli_runner.invoke(main, ["query", "What is this about?"])

    assert result.exit_code == 0
    assert "System not ready" in result.output
    assert "Ollama not running" in result.output


def test_query_command_with_debug_option(cli_runner, mock_rag_pipeline):
    """Test query command with debug option."""
    mock_rag_pipeline.check_readiness.return_value = {"is_ready": True, "issues": []}
    mock_result = Mock()
    mock_result.answer = "Test answer"
    mock_result.total_time = 1.0
    mock_result.chunks_retrieved = 2
    mock_rag_pipeline.ask.return_value = mock_result

    result = cli_runner.invoke(main, ["query", "test question", "--debug"])

    assert result.exit_code == 0
    assert "Test answer" in result.output
    mock_rag_pipeline.ask.assert_called()


# Config command tests


def test_config_command_show_current_configuration(cli_runner, mock_settings):
    """Test config command shows current configuration."""
    result = cli_runner.invoke(main, ["config"])

    assert result.exit_code == 0
    assert "Current Configuration" in result.output
    assert "Primary language: es" in result.output


def test_config_command_set_language(cli_runner, mock_settings):
    """Test config command sets language."""
    result = cli_runner.invoke(main, ["config", "--language", "fr"])

    assert result.exit_code == 0
    assert "Primary language set to: fr" in result.output
    mock_settings.save_user_config.assert_called_once()


def test_config_command_enable_summaries(cli_runner, mock_settings):
    """Test config command enables summaries."""
    result = cli_runner.invoke(main, ["config", "--enable-summaries"])

    assert result.exit_code == 0
    assert "Hierarchical summarization enabled" in result.output
    assert mock_settings.enable_summarization is True
    mock_settings.save_user_config.assert_called_once()


def test_config_command_disable_summaries(cli_runner, mock_settings):
    """Test config command disables summaries."""
    result = cli_runner.invoke(main, ["config", "--disable-summaries"])

    assert result.exit_code == 0
    assert "Hierarchical summarization disabled" in result.output
    assert mock_settings.enable_summarization is False
    mock_settings.save_user_config.assert_called_once()


def test_config_command_invalid_language(cli_runner, mock_settings):
    """Test config command with invalid language."""
    result = cli_runner.invoke(main, ["config", "--language", "invalid"])

    assert result.exit_code == 0
    assert "Unsupported language: invalid" in result.output
    assert "Supported: es, en, fr, de, it, pt" in result.output


# Inspect command tests


def test_inspect_command_shows_chunk_information(cli_runner, mock_rag_pipeline):
    """Test inspect command displays chunk information."""
    # Mock chunks
    mock_chunk1 = Mock()
    mock_chunk1.content = "This is the first chunk content for testing purposes."
    mock_chunk1.source_file = "test1.md"
    mock_chunk1.start_pos = 0
    mock_chunk1.end_pos = 52

    mock_chunk2 = Mock()
    mock_chunk2.content = "This is the second chunk content."
    mock_chunk2.source_file = "test2.md"
    mock_chunk2.start_pos = 0
    mock_chunk2.end_pos = 33

    mock_rag_pipeline.retriever.vector_store.chunks = [mock_chunk1, mock_chunk2]

    result = cli_runner.invoke(main, ["inspect", "--count", "2"])

    assert result.exit_code == 0
    assert "Inspecting chunks (total: 2)" in result.output
    assert "first chunk content" in result.output
    assert "second chunk content" in result.output


def test_inspect_command_with_search_filter(cli_runner, mock_rag_pipeline):
    """Test inspect command with search filtering."""
    mock_chunk = Mock()
    mock_chunk.content = "This chunk contains the search term."
    mock_chunk.source_file = "test.md"
    mock_chunk.start_pos = 0
    mock_chunk.end_pos = 37

    mock_rag_pipeline.retriever.vector_store.chunks = [mock_chunk]

    result = cli_runner.invoke(main, ["inspect", "--search", "search"])

    assert result.exit_code == 0
    assert "search term" in result.output


def test_inspect_command_empty_database(cli_runner, mock_rag_pipeline):
    """Test inspect command with empty database."""
    mock_rag_pipeline.retriever.vector_store.chunks = []

    result = cli_runner.invoke(main, ["inspect"])

    assert result.exit_code == 0
    assert "No chunks found" in result.output


# Chat command tests


def test_chat_command_starts_interactive_session(
    cli_runner, mock_rag_pipeline, mock_chat_interface
):
    """Test chat command starts interactive session."""
    mock_rag_pipeline.check_readiness.return_value = {"is_ready": True, "issues": []}

    result = cli_runner.invoke(main, ["chat"])

    assert result.exit_code == 0
    mock_chat_interface.start_interactive_session.assert_called_once()


def test_chat_command_system_not_ready(cli_runner, mock_rag_pipeline):
    """Test chat command when system is not ready."""
    mock_rag_pipeline.check_readiness.return_value = {
        "is_ready": False,
        "issues": ["No documents indexed"],
    }

    result = cli_runner.invoke(main, ["chat"])

    assert result.exit_code == 0
    assert "System not ready for chat" in result.output
    assert "No documents indexed" in result.output


# Integration tests


def test_cli_commands_handle_exceptions_gracefully(cli_runner, mock_rag_pipeline):
    """Test that CLI commands handle exceptions gracefully."""
    mock_rag_pipeline.side_effect = Exception("Connection failed")

    result = cli_runner.invoke(main, ["stats"])

    assert result.exit_code == 0
    assert "failed" in result.output.lower() or "error" in result.output.lower()


@pytest.mark.parametrize(
    ("command", "expected_in_output"),
    [
        # (["setup"], "Setting up docs-to-rag"),  # Disabled - complex mock setup required
        (["list"], "documents"),
        (["stats"], "system status"),
        (["inspect"], "chunks"),
    ],
)
def test_cli_commands_basic_functionality(
    cli_runner, mock_rag_pipeline, command, expected_in_output
):
    """Test that CLI commands execute and show expected output."""
    # Setup default mocks for commands that need them
    mock_rag_pipeline.check_readiness.return_value = {"is_ready": True, "issues": []}
    mock_rag_pipeline.retriever.get_source_files.return_value = []
    mock_rag_pipeline.retriever.vector_store.chunks = []
    mock_rag_pipeline.get_system_stats.return_value = {
        "models": {"embedding": {"model_name": "test", "is_available": True}},
        "vector_store": {"document_count": 0},
        "settings": {"chunk_size": 1000},
    }

    result = cli_runner.invoke(main, command)

    assert result.exit_code == 0
    assert expected_in_output.lower() in result.output.lower()
