"""Tests for configuration module."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config.settings import Settings, _LazySettings


@pytest.fixture
def temp_config_dir():
    """Create temporary directory for config tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def settings_instance():
    """Create a fresh Settings instance."""
    return Settings()


def test_settings_have_reasonable_defaults(settings_instance):
    """Test that settings have reasonable defaults."""
    assert settings_instance.chunk_size > 0
    assert settings_instance.chunk_overlap >= 0
    assert settings_instance.chunk_overlap < settings_instance.chunk_size
    assert settings_instance.top_k_retrieval > 0
    assert settings_instance.similarity_threshold > 0
    assert settings_instance.temperature >= 0
    assert settings_instance.embedding_model
    assert settings_instance.chat_model
    assert isinstance(settings_instance.enable_summarization, bool)


def test_settings_validation_with_valid_settings(settings_instance):
    """Test settings validation with valid settings."""
    is_valid, errors = settings_instance.is_valid()

    assert is_valid
    assert len(errors) == 0


def test_settings_validation_with_invalid_chunk_size():
    """Test settings validation with invalid chunk size."""
    settings = Settings()
    settings.chunk_size = -1

    is_valid, errors = settings.is_valid()

    assert not is_valid
    assert any("chunk_size must be positive" in error for error in errors)


def test_settings_validation_with_invalid_overlap():
    """Test settings validation with overlap larger than chunk size."""
    settings = Settings()
    settings.chunk_overlap = 2000  # Larger than default chunk_size

    is_valid, errors = settings.is_valid()

    assert not is_valid
    assert any("overlap" in error.lower() for error in errors)


def test_settings_validation_with_invalid_temperature():
    """Test settings validation with temperature out of range."""
    settings = Settings()
    settings.temperature = 3.0  # Out of range

    is_valid, errors = settings.is_valid()

    assert not is_valid
    assert len(errors) >= 1


@pytest.mark.parametrize(
    ("chunk_count", "expected_top_k_range"),
    [
        (50, (1, 4)),  # Small database
        (500, (2, 6)),  # Medium database
        (5000, (4, 10)),  # Large database
    ],
)
def test_get_adaptive_params_scales_with_database_size(
    settings_instance, chunk_count, expected_top_k_range
):
    """Test adaptive parameters scale appropriately with database size."""
    top_k, threshold = settings_instance.get_adaptive_params(chunk_count)

    assert expected_top_k_range[0] <= top_k <= expected_top_k_range[1]
    assert 0.1 <= threshold <= 1.0


def test_get_adaptive_params_small_database_uses_conservative_settings(
    settings_instance,
):
    """Test adaptive parameters for small database use conservative settings."""
    top_k, threshold = settings_instance.get_adaptive_params(50)

    assert top_k <= 3
    # Threshold should be close to default but may be slightly adjusted
    assert abs(threshold - settings_instance.similarity_threshold) <= 0.05


def test_get_adaptive_params_large_database_uses_aggressive_settings(settings_instance):
    """Test adaptive parameters for large database use more aggressive settings."""
    top_k, threshold = settings_instance.get_adaptive_params(10000)

    assert top_k > 3
    assert threshold < settings_instance.similarity_threshold


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Este es un documento en español que contiene muchas palabras comunes "
            "como el, la, de, que, y, en, un, es, se, no, te, lo, le y otras.",
            True,
        ),
        (
            "This is an English document that contains many common words "
            "like the, and, of, to, a, in, is, it, you, that, he, was and others.",
            False,
        ),
        ("Hello", False),  # Short text
        ("", False),  # Empty text
                    (
                "¡Hola! ¿Cómo estás? El día está muy bueno. "
                "Este es un texto en español que contiene palabras como el, la, de, que, y, en, un, es, se, no.",
                True,
            ),  # Clear Spanish
    ],
)
def test_detect_spanish_content(settings_instance, text, expected):
    """Test Spanish content detection with various texts."""
    result = settings_instance.detect_spanish_content(text)
    assert result == expected


@patch.dict(
    "os.environ",
    {
        "CHUNK_SIZE": "500",
        "EMBEDDING_MODEL": "custom-model",
        "CHAT_MODEL": "custom-chat-model",
    },
)
def test_environment_variables_override_defaults():
    """Test that environment variables override default settings."""
    settings = Settings.load()

    assert settings.chunk_size == 500
    assert settings.embedding_model == "custom-model"
    assert settings.chat_model == "custom-chat-model"


def test_save_and_load_user_config(temp_config_dir):
    """Test saving and loading user configuration."""
    settings = Settings()
    settings.project_root = temp_config_dir

    # Modify some settings
    settings.primary_language = "fr"
    settings.enable_summarization = False
    settings.chat_model = "custom-model"

    # Save configuration
    settings.save_user_config()

    # Verify file was created
    config_file = temp_config_dir / "config" / "user_config.json"
    assert config_file.exists()

    # Load configuration into new settings instance
    new_settings = Settings()
    new_settings.project_root = temp_config_dir
    new_settings.load_user_config()

    assert new_settings.primary_language == "fr"
    assert new_settings.enable_summarization is False
    assert new_settings.chat_model == "custom-model"


def test_save_user_config_creates_directory_if_missing(temp_config_dir):
    """Test that save_user_config creates config directory if it doesn't exist."""
    settings = Settings()
    settings.project_root = temp_config_dir
    settings.primary_language = "es"

    # Config directory shouldn't exist yet
    config_dir = temp_config_dir / "config"
    assert not config_dir.exists()

    # Save should create directory
    settings.save_user_config()

    assert config_dir.exists()
    assert (config_dir / "user_config.json").exists()


def test_load_user_config_handles_missing_file_gracefully(temp_config_dir):
    """Test that load_user_config handles missing config file gracefully."""
    settings = Settings()
    settings.project_root = temp_config_dir
    original_language = settings.primary_language

    # Try to load non-existent config
    settings.load_user_config()

    # Settings should remain unchanged
    assert settings.primary_language == original_language


def test_load_user_config_handles_invalid_json_gracefully(temp_config_dir):
    """Test that load_user_config handles invalid JSON gracefully."""
    settings = Settings()
    settings.project_root = temp_config_dir
    original_language = settings.primary_language

    # Create config directory and invalid JSON file
    config_dir = temp_config_dir / "config"
    config_dir.mkdir()
    config_file = config_dir / "user_config.json"
    config_file.write_text("invalid json content")

    # Load should not crash
    settings.load_user_config()

    # Settings should remain unchanged
    assert settings.primary_language == original_language


# Tests for lazy settings loading


def test_lazy_settings_delay_loading_until_accessed():
    """Test that lazy settings are not loaded until accessed."""
    lazy_settings = _LazySettings()

    # Settings should not be loaded yet
    assert lazy_settings._settings is None

    # Access an attribute to trigger loading
    _ = lazy_settings.chunk_size

    # Now settings should be loaded
    assert lazy_settings._settings is not None


def test_lazy_settings_delegate_attribute_access_correctly():
    """Test that lazy settings delegate attribute access correctly."""
    lazy_settings = _LazySettings()

    # This should trigger loading and return the value
    chunk_size = lazy_settings.chunk_size
    assert chunk_size > 0

    # Settings should now be loaded
    assert lazy_settings._settings is not None


def test_lazy_settings_allow_setting_attributes():
    """Test that lazy settings allow setting attributes."""
    lazy_settings = _LazySettings()

    # Set an attribute
    lazy_settings.chunk_size = 500

    # Verify it was set
    assert lazy_settings.chunk_size == 500
    assert lazy_settings._settings is not None


def test_lazy_settings_preserve_attribute_changes():
    """Test that lazy settings preserve attribute changes."""
    lazy_settings = _LazySettings()

    # Set multiple attributes
    lazy_settings.chunk_size = 800
    lazy_settings.temperature = 0.5

    # Verify both are preserved
    assert lazy_settings.chunk_size == 800
    assert lazy_settings.temperature == 0.5
