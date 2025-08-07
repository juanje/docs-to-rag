"""
Configuration settings for the docs-to-rag system.

This module defines all configurable parameters for document processing,
embedding generation, RAG pipeline, and system paths.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


@dataclass
class Settings:
    """Main system configuration with sensible defaults."""

    # === OLLAMA MODELS ===
    embedding_model: str = "nomic-embed-text:v1.5"  # Fast, efficient embeddings
    embedding_model_multilingual: str = (
        "jina/jina-embeddings-v2-base-es:latest"  # Spanish-specific embeddings
    )
    chat_model: str = "llama3.2:latest"  # Balanced speed/quality

    # === LANGUAGE SETTINGS ===
    auto_detect_language: bool = (
        True  # Auto-switch to multilingual embeddings for non-English
    )
    primary_language: str = "es"  # Primary language of documents (es, en, fr, etc.)
    spanish_detection_keywords: list[str] = field(
        default_factory=lambda: [
            "el",
            "la",
            "los",
            "las",
            "de",
            "del",
            "en",
            "con",
            "por",
            "para",
            "que",
            "se",
            "una",
            "un",
            "es",
            "son",
            "está",
            "están",
            "pero",
            "también",
            "muy",
            "más",
            "como",
            "cuando",
            "donde",
            "porque",
            "después",
            "antes",
            "durante",
            "entre",
            "sobre",
            "bajo",
            "desde",
        ]
    )
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: int = 120

    # === DOCUMENT PROCESSING ===
    chunk_size: int = 1000  # Text chunk size in characters
    chunk_overlap: int = 200  # Overlap between chunks
    supported_extensions: list[str] | None = None  # Will be set in __post_init__

    # === RAG PARAMETERS ===
    top_k_retrieval: int = 3  # Number of documents to retrieve (will be auto-adjusted)
    similarity_threshold: float = (
        0.7  # Minimum similarity score (will be auto-adjusted)
    )
    max_tokens_response: int = 512  # Maximum response tokens
    temperature: float = 0.1  # LLM temperature (deterministic)
    top_p: float = 0.9  # Nucleus sampling parameter

    # === ADAPTIVE RAG PARAMETERS ===
    enable_adaptive_params: bool = True  # Auto-adjust parameters based on DB size
    min_similarity_threshold: float = 0.3  # Minimum threshold for large databases
    max_top_k: int = 20  # Maximum chunks to retrieve for large databases
    adaptive_threshold_factor: float = (
        0.1  # How much to reduce threshold per 1000 chunks
    )

    # === DEBUG PARAMETERS ===
    debug_mode: bool = False  # Show detailed retrieval information
    show_similarity_scores: bool = False  # Display similarity scores in output

    # === DOCUMENT ENRICHMENT ===
    enable_summarization: bool = True  # Generate hierarchical summaries
    generate_document_summaries: bool = True  # Document-level summaries
    generate_chapter_summaries: bool = True  # Chapter/section summaries
    generate_concept_summaries: bool = True  # Key concept summaries
    max_concepts_per_document: int = 8  # Maximum number of concepts to extract

    # === SUMMARIZATION LLM SETTINGS ===
    summarization_model: str = (
        "llama3.2:latest"  # Dedicated model for summaries (can be different from chat)
    )
    summarization_temperature: float = (
        0.1  # Lower temperature for more faithful summaries
    )
    summarization_top_p: float = 0.8  # More focused sampling for summaries
    summarization_max_tokens_document: int = 400  # Document summary tokens
    summarization_max_tokens_chapter: int = 250  # Chapter summary tokens
    summarization_max_tokens_concept: int = 200  # Concept summary tokens
    summarization_system_prompt: str = ""  # System prompt for faithful summarization

    # === SUMMARIZATION QUALITY CONTROL ===
    enable_summary_validation: bool = True  # Validate summary quality
    summary_faithfulness_check: bool = True  # Check if summary is faithful to source
    max_summary_retries: int = 2  # Maximum retries for poor quality summaries

    # === PATHS ===
    project_root: Path | None = None  # Will be set in __post_init__
    documents_path: str = "data/documents"
    vector_db_path: str = "data/vector_db"

    # === LOGGING ===
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        """Initialize computed fields and apply environment overrides."""
        # Set default supported extensions
        if self.supported_extensions is None:
            self.supported_extensions = [".pdf", ".md", ".html"]

        # Set project root
        if self.project_root is None:
            self.project_root = Path(__file__).parent.parent

        # Apply environment variable overrides
        self._apply_env_overrides()

        # Ensure paths are absolute
        self._resolve_paths()

    def _apply_env_overrides(self) -> None:
        """Apply configuration overrides from environment variables."""
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", self.ollama_base_url)
        self.embedding_model = os.getenv("EMBEDDING_MODEL", self.embedding_model)
        self.chat_model = os.getenv("CHAT_MODEL", self.chat_model)

        # Process numeric environment variables
        if chunk_size := os.getenv("CHUNK_SIZE"):
            self.chunk_size = int(chunk_size)
        if chunk_overlap := os.getenv("CHUNK_OVERLAP"):
            self.chunk_overlap = int(chunk_overlap)
        if top_k := os.getenv("TOP_K_RETRIEVAL"):
            self.top_k_retrieval = int(top_k)

        self.log_level = os.getenv("LOG_LEVEL", self.log_level)

    def _resolve_paths(self) -> None:
        """Convert relative paths to absolute paths."""
        if not Path(self.documents_path).is_absolute():
            self.documents_path = str(self.project_root / self.documents_path)
        if not Path(self.vector_db_path).is_absolute():
            self.vector_db_path = str(self.project_root / self.vector_db_path)

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        Path(self.documents_path).mkdir(parents=True, exist_ok=True)
        Path(self.vector_db_path).mkdir(parents=True, exist_ok=True)

    def get_adaptive_params(self, total_chunks: int) -> tuple[int, float]:
        """
        Calculate adaptive RAG parameters based on database size.

        Args:
            total_chunks: Total number of chunks in the vector database

        Returns:
            Tuple of (adaptive_top_k, adaptive_threshold)
        """
        if not self.enable_adaptive_params:
            return self.top_k_retrieval, self.similarity_threshold

        # Calculate adaptive top_k
        if total_chunks <= 100:
            adaptive_top_k = min(3, total_chunks)
        elif total_chunks <= 1000:
            adaptive_top_k = min(5, total_chunks)
        elif total_chunks <= 5000:
            adaptive_top_k = min(10, total_chunks)
        else:
            # For very large databases, use more chunks
            adaptive_top_k = min(self.max_top_k, max(15, total_chunks // 1000))

        # Calculate adaptive similarity threshold
        # Reduce threshold for larger databases to be less restrictive
        chunks_in_thousands = total_chunks / 1000
        threshold_reduction = min(
            0.4, chunks_in_thousands * self.adaptive_threshold_factor
        )
        adaptive_threshold = max(
            self.min_similarity_threshold,
            self.similarity_threshold - threshold_reduction,
        )

        return adaptive_top_k, adaptive_threshold

    def detect_spanish_content(self, text: str) -> bool:
        """
        Detect if text content is primarily in Spanish.

        Args:
            text: Text content to analyze

        Returns:
            True if text appears to be in Spanish
        """
        if not text or len(text.strip()) < 50:
            return False

        # Convert to lowercase and split into words
        words = text.lower().split()
        if len(words) < 20:
            return False

        # Count Spanish keywords
        spanish_word_count = sum(
            1 for word in words if word in self.spanish_detection_keywords
        )
        spanish_ratio = spanish_word_count / len(words)

        # Consider it Spanish if more than 15% of words are Spanish keywords
        return spanish_ratio > 0.15

    @classmethod
    def load(cls) -> "Settings":
        """Load configuration with defaults and environment overrides."""
        # Load environment variables only when needed
        load_dotenv()

        settings = cls()
        # Load user configuration overrides
        settings.load_user_config()
        return settings

    def is_valid(self) -> tuple[bool, list[str]]:
        """
        Validate configuration settings.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Validate chunk settings
        if self.chunk_size <= 0:
            errors.append("chunk_size must be positive")
        if self.chunk_overlap >= self.chunk_size:
            errors.append("chunk_overlap must be less than chunk_size")

        # Validate RAG settings
        if self.top_k_retrieval <= 0:
            errors.append("top_k_retrieval must be positive")
        if not 0 <= self.temperature <= 2:
            errors.append("temperature must be between 0 and 2")
        if not 0 <= self.top_p <= 1:
            errors.append("top_p must be between 0 and 1")

        # Validate paths
        if not self.supported_extensions:
            errors.append("supported_extensions cannot be empty")

        return len(errors) == 0, errors

    def save_user_config(self) -> None:
        """Save user configuration overrides to local project config file."""
        import json

        # Use local config directory within the project
        config_dir = self.project_root / "config"
        config_dir.mkdir(exist_ok=True)

        # Save only the settings that can be modified via CLI
        user_config = {
            "primary_language": self.primary_language,
            "enable_summarization": self.enable_summarization,
            "enable_summary_validation": self.enable_summary_validation,
            "summarization_model": self.summarization_model,
            "chat_model": self.chat_model,
            "embedding_model": self.embedding_model,
            "embedding_model_multilingual": self.embedding_model_multilingual,
        }

        config_file = config_dir / "user_config.json"
        with open(config_file, "w") as f:
            json.dump(user_config, f, indent=2)

        # Use print instead of logger in case it's called during initialization
        print(f"Configuration saved to {config_file}")

    def load_user_config(self) -> None:
        """Load user configuration overrides from local project config file."""
        import json

        # Load from local config directory within the project
        config_file = self.project_root / "config" / "user_config.json"

        if config_file.exists():
            try:
                with open(config_file) as f:
                    user_config = json.load(f)

                # Apply user overrides
                for key, value in user_config.items():
                    if hasattr(self, key):
                        setattr(self, key, value)

                # Use print instead of logger to avoid import issues during initialization
                # logger.info(f"User configuration loaded from {config_file}")
            except Exception:
                # Silently fail during initialization
                pass


class _LazySettings:
    """Lazy-loading proxy for Settings that only loads when accessed."""

    def __init__(self) -> None:
        self._settings: Settings | None = None

    def _ensure_loaded(self) -> Settings:
        """Ensure settings are loaded and return them."""
        if self._settings is None:
            self._settings = Settings.load()
        return self._settings

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the actual settings object."""
        return getattr(self._ensure_loaded(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Handle setting attributes."""
        if name == "_settings":
            # Internal attribute
            super().__setattr__(name, value)
        else:
            # Delegate to the actual settings object
            setattr(self._ensure_loaded(), name, value)


# Global settings instance - lazy loaded
settings = _LazySettings()
