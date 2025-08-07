"""
CLI (Command Line Interface) module for docs-to-rag.

This module provides the command-line interface for interacting with
the RAG system, including document management and chat functionality.
"""

from .chat import ChatInterface
from .commands import main

__all__ = ["main", "ChatInterface"]
