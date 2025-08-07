"""
Main CLI commands for docs-to-rag.

This module defines all the command-line interface commands for managing
documents, querying the RAG system, and system administration.
"""

import logging
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Standard entry point for CLI commands
from src.config.settings import settings

# Initialize console for rich output
console = Console()

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0", prog_name="docs-to-rag")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose: bool) -> None:
    """
    docs-to-rag: Educational RAG system with local LLMs.

    Extract information from documents (PDF, Markdown, HTML) and create
    a searchable knowledge base using vector embeddings and RAG.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")


@main.command()
def setup() -> None:
    """Initialize the docs-to-rag system."""
    console.print("[bold blue]🚀 Setting up docs-to-rag system...[/bold blue]")

    try:
        # Step 1: Verify Ollama connection
        with console.status("[dim]🔍 Checking Ollama connection..."):
            from src.rag.pipeline import RAGPipeline

            rag_pipeline = RAGPipeline()
            readiness = rag_pipeline.check_readiness()

        if not readiness["is_ready"]:
            console.print("[red]❌ System setup issues detected:[/red]")
            for issue in readiness["issues"]:
                console.print(f"   • {issue}")

            console.print("\n[yellow]💡 Setup suggestions:[/yellow]")
            console.print("   • Make sure Ollama is running: [bold]ollama serve[/bold]")
            console.print("   • Download required models:")
            console.print(f"     [bold]ollama pull {settings.embedding_model}[/bold]")
            console.print(f"     [bold]ollama pull {settings.chat_model}[/bold]")

            sys.exit(1)

        # Step 2: Create directories
        console.print("[blue]📁 Creating directories...[/blue]")
        settings.ensure_directories()

        # Step 3: Display system info
        stats = rag_pipeline.get_system_stats()
        _display_system_info(stats)

        console.print("\n[green]✅ Setup completed successfully![/green]")
        console.print(
            "[dim]Use '[bold]docs-to-rag add /path/to/documents/[/bold]' "
            "to start adding documents.[/dim]"
        )

    except Exception as e:
        console.print(f"[red]❌ Setup failed: {str(e)}[/red]")
        sys.exit(1)


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option(
    "--recursive",
    "-r",
    is_flag=True,
    default=True,
    help="Process directories recursively",
)
def add(path: str, recursive: bool) -> None:
    """Process and add documents to the knowledge base."""
    console.print(f"[blue]📄 Processing documents from: {path}[/blue]")

    try:
        # Initialize components
        from src.document_processor.extractor import DocumentExtractor
        from src.rag.pipeline import RAGPipeline

        extractor = DocumentExtractor()
        rag_pipeline = RAGPipeline()

        # Check if path is file or directory
        path_obj = Path(path)
        if path_obj.is_file():
            if not extractor.is_supported_file(str(path_obj)):
                console.print(f"[red]❌ Unsupported file type: {path_obj.suffix}[/red]")
                return

            # Process single file
            with console.status(f"[dim]Processing {path_obj.name}..."):
                result = extractor.extract_document(str(path_obj))
                if result:
                    add_result = rag_pipeline.add_documents([result])
                else:
                    console.print(f"[red]❌ Failed to process {path_obj.name}[/red]")
                    return
        else:
            # Process directory
            with console.status("[dim]📄 Discovering and processing documents..."):
                processing_result = extractor.process_directory(path)

                if processing_result.processed == 0:
                    console.print("[yellow]⚠️ No supported documents found[/yellow]")
                    return

                # Add to RAG system
                add_result = rag_pipeline.add_documents(processing_result.documents)

        # Display results
        if add_result["success"]:
            console.print(
                f"[green]✅ Successfully processed "
                f"{add_result['documents_processed']} documents[/green]"
            )
            console.print(
                f"[blue]📊 Created {add_result['chunks_created']} text chunks[/blue]"
            )
            console.print(
                f"[blue]⏱️ Embedding generation took {add_result['embedding_time']:.2f}s[/blue]"
            )

            # Show updated stats
            stats = add_result["system_stats"]
            console.print(
                f"[dim]💾 Total chunks in database: {stats['vector_store']['chunk_count']}[/dim]"
            )
        else:
            console.print(
                f"[red]❌ Processing failed: {add_result.get('message', 'Unknown error')}[/red]"
            )

    except Exception as e:
        console.print(f"[red]❌ Error processing documents: {str(e)}[/red]")
        logger.error(f"Document processing error: {e}", exc_info=True)


@main.command("list")
def list_documents() -> None:
    """List all indexed documents."""
    try:
        from src.rag.pipeline import RAGPipeline

        rag_pipeline = RAGPipeline()
        source_files = rag_pipeline.retriever.get_source_files()

        if not source_files:
            console.print("[yellow]📝 No documents indexed yet[/yellow]")
            console.print("[dim]Use 'add' command to process documents[/dim]")
            return

        console.print(
            f"[blue]📚 Indexed Documents ({len(source_files)} files):[/blue]\n"
        )

        for i, file_path in enumerate(sorted(source_files), 1):
            file_name = Path(file_path).name
            console.print(f"  {i:2d}. {file_name}")
            console.print(f"      [dim]{file_path}[/dim]")

    except Exception as e:
        console.print(f"[red]❌ Error listing documents: {str(e)}[/red]")


@main.command()
@click.confirmation_option(
    prompt="Are you sure you want to clear all indexed documents?"
)
def clear() -> None:
    """Clear all indexed documents from the knowledge base."""
    try:
        with console.status("[dim]🗑️ Clearing knowledge base..."):
            from src.rag.pipeline import RAGPipeline

            rag_pipeline = RAGPipeline()
            rag_pipeline.clear_knowledge_base()

        console.print("[yellow]🗑️ Knowledge base cleared successfully[/yellow]")

    except Exception as e:
        console.print(f"[red]❌ Error clearing knowledge base: {str(e)}[/red]")


@main.command()
@click.argument("document_path")
@click.option("--multilingual", is_flag=True, help="Use multilingual embedding model")
def reprocess(document_path: str, multilingual: bool) -> None:
    """Clear database and reprocess a document with improved chunking."""
    try:
        if multilingual:
            console.print(
                "[blue]🌍 Reprocessing with multilingual embedding model[/blue]"
            )
            console.print(
                f"[dim]Using model: {settings.embedding_model_multilingual}[/dim]"
            )
        else:
            console.print(
                "[blue]🔄 Reprocessing document with improved chunking[/blue]"
            )

        console.print(f"[dim]Document: {document_path}[/dim]")

        # Clear existing database
        from src.rag.pipeline import RAGPipeline

        rag_pipeline = RAGPipeline()
        rag_pipeline.clear_knowledge_base()
        console.print("[yellow]✅ Cleared existing database[/yellow]")

        # Override embedding model if multilingual requested
        if multilingual:
            # Update settings temporarily
            original_model = settings.embedding_model
            settings.embedding_model = settings.embedding_model_multilingual
            console.print(
                f"[dim]Switched to multilingual model: {settings.embedding_model}[/dim]"
            )

        # Reprocess document
        from pathlib import Path

        from src.document_processor.extractor import DocumentExtractor

        extractor = DocumentExtractor()
        doc_path = Path(document_path)

        if doc_path.is_file():
            # Single file
            console.print(f"[dim]Processing single file: {doc_path.name}[/dim]")
            with console.status("[dim]🔄 Reprocessing document..."):
                doc_result = extractor.extract_document(str(doc_path))
                if doc_result:
                    result = rag_pipeline.add_documents([doc_result])
                else:
                    console.print(
                        f"[red]❌ Failed to extract content from {doc_path.name}[/red]"
                    )
                    return
        elif doc_path.is_dir():
            # Directory
            console.print(f"[dim]Processing directory: {doc_path}[/dim]")
            with console.status("[dim]🔄 Reprocessing documents..."):
                processing_result = extractor.process_directory(str(doc_path))
                if processing_result.processed == 0:
                    console.print("[yellow]⚠️ No supported documents found[/yellow]")
                    return
                result = rag_pipeline.add_documents(processing_result.documents)
        else:
            console.print(f"[red]❌ Path not found: {document_path}[/red]")
            return

        # Show results
        if result.get("success", False):
            console.print("[green]✅ Reprocessed successfully[/green]")
            console.print(
                f"[dim]Documents: {result.get('documents_processed', 0)}[/dim]"
            )
            console.print(f"[dim]Chunks: {result.get('chunks_created', 0)}[/dim]")
            console.print(f"[dim]Time: {result.get('embedding_time', 0):.2f}s[/dim]")
        else:
            console.print(
                f"[red]❌ Reprocessing failed: {result.get('message', 'Unknown error')}[/red]"
            )

        # Restore original model if we changed it
        if multilingual and "original_model" in locals():
            settings.embedding_model = original_model
            console.print(
                f"[dim]Restored original model: {settings.embedding_model}[/dim]"
            )

    except Exception as e:
        # Restore original model in case of error
        if multilingual and "original_model" in locals():
            settings.embedding_model = original_model
        console.print(f"[red]❌ Reprocessing failed: {str(e)}[/red]")


@main.command()
@click.option("--language", type=str, help="Set primary language (es, en, etc.)")
@click.option(
    "--enable-summaries", is_flag=True, help="Enable hierarchical summarization"
)
@click.option(
    "--disable-summaries", is_flag=True, help="Disable hierarchical summarization"
)
@click.option(
    "--enable-validation", is_flag=True, help="Enable summary quality validation"
)
@click.option(
    "--disable-validation", is_flag=True, help="Disable summary quality validation"
)
@click.option("--summary-model", type=str, help="Set model for summary generation")
def config(
    language: str,
    enable_summaries: bool,
    disable_summaries: bool,
    enable_validation: bool,
    disable_validation: bool,
    summary_model: str,
) -> None:
    """Configure system settings."""
    # Track if any changes were made
    changes_made = False

    if language:
        if language.lower() in ["es", "en", "fr", "de", "it", "pt"]:
            settings.primary_language = language.lower()
            console.print(
                f"[green]✅ Primary language set to: {language.lower()}[/green]"
            )

            if language.lower() == "es":
                console.print(
                    f"[dim]Spanish documents will use: {settings.embedding_model_multilingual}[/dim]"
                )
            else:
                console.print(
                    f"[dim]Non-Spanish documents will use: {settings.embedding_model}[/dim]"
                )
            changes_made = True
        else:
            console.print(f"[red]❌ Unsupported language: {language}[/red]")
            console.print("[dim]Supported: es, en, fr, de, it, pt[/dim]")

    if enable_summaries:
        settings.enable_summarization = True
        console.print("[green]✅ Hierarchical summarization enabled[/green]")
        console.print(
            "[dim]💡 New documents will generate summaries automatically[/dim]"
        )
        console.print("[dim]📚 This improves retrieval for conceptual questions[/dim]")
        changes_made = True

    if disable_summaries:
        settings.enable_summarization = False
        console.print("[yellow]⚠️ Hierarchical summarization disabled[/yellow]")
        console.print("[dim]🔄 Only original document chunks will be used[/dim]")
        changes_made = True

    if enable_validation:
        settings.enable_summary_validation = True
        settings.summary_faithfulness_check = True
        console.print("[green]✅ Summary quality validation enabled[/green]")
        console.print(
            "[dim]🔍 Summaries will be validated for faithfulness and quality[/dim]"
        )
        changes_made = True

    if disable_validation:
        settings.enable_summary_validation = False
        settings.summary_faithfulness_check = False
        console.print("[yellow]⚠️ Summary quality validation disabled[/yellow]")
        console.print("[dim]⚡ Faster summary generation, but no quality checks[/dim]")
        changes_made = True

    if summary_model:
        settings.summarization_model = summary_model
        console.print(f"[green]✅ Summary model set to: {summary_model}[/green]")
        console.print("[dim]🧠 New summaries will use this specialized model[/dim]")
        changes_made = True

    # Save configuration changes
    if changes_made:
        settings.save_user_config()
        console.print("[dim]💾 Configuration saved to ./config/user_config.json[/dim]")

    if not any(
        [
            language,
            enable_summaries,
            disable_summaries,
            enable_validation,
            disable_validation,
            summary_model,
        ]
    ):
        # Show current configuration
        console.print("[blue]📋 Current Configuration[/blue]")
        console.print(f"Primary language: {settings.primary_language}")
        console.print(f"Auto-detect language: {settings.auto_detect_language}")
        console.print(f"Default embedding model: {settings.embedding_model}")
        console.print(
            f"Multilingual embedding model: {settings.embedding_model_multilingual}"
        )
        console.print()
        console.print("[blue]📚 Document Enrichment[/blue]")
        status = "✅ Enabled" if settings.enable_summarization else "❌ Disabled"
        console.print(f"Hierarchical summarization: {status}")
        if settings.enable_summarization:
            console.print(
                f"  Document summaries: {'✅' if settings.generate_document_summaries else '❌'}"
            )
            console.print(
                f"  Chapter summaries: {'✅' if settings.generate_chapter_summaries else '❌'}"
            )
            console.print(
                f"  Concept summaries: {'✅' if settings.generate_concept_summaries else '❌'}"
            )
            console.print(f"  Max concepts: {settings.max_concepts_per_document}")
            console.print()
            console.print("[blue]🧠 Summary Generation Settings[/blue]")
            console.print(f"Summary model: {settings.summarization_model}")
            console.print(f"Temperature: {settings.summarization_temperature}")
            console.print(f"Top-p: {settings.summarization_top_p}")
            console.print(f"Max retries: {settings.max_summary_retries}")
            console.print()
            console.print("[blue]🔍 Quality Control[/blue]")
            validation_status = (
                "✅ Enabled" if settings.enable_summary_validation else "❌ Disabled"
            )
            console.print(f"Summary validation: {validation_status}")
            faithfulness_status = (
                "✅ Enabled" if settings.summary_faithfulness_check else "❌ Disabled"
            )
            console.print(f"Faithfulness checking: {faithfulness_status}")


@main.command()
@click.argument("document_path", required=False)
@click.option("--force", is_flag=True, help="Regenerate summaries even if they exist")
def enrich(document_path: str, force: bool) -> None:
    """Generate hierarchical summaries for existing documents to improve retrieval."""
    try:
        if not settings.enable_summarization:
            console.print("[red]❌ Hierarchical summarization is disabled[/red]")
            console.print(
                "[dim]Enable it with: docs-to-rag config --enable-summaries[/dim]"
            )
            return

        console.print(
            "[blue]🧠 Generating hierarchical summaries for enhanced retrieval...[/blue]"
        )

        if document_path:
            console.print(f"[dim]Document: {document_path}[/dim]")
        else:
            console.print("[dim]Processing all documents in knowledge base[/dim]")

        # Initialize components
        from src.document_processor.chunker import TextChunker
        from src.document_processor.extractor import DocumentExtractor
        from src.document_processor.summarizer import DocumentSummarizer
        from src.rag.pipeline import RAGPipeline

        rag_pipeline = RAGPipeline()
        extractor = DocumentExtractor()
        summarizer = DocumentSummarizer()
        chunker = TextChunker()

        # Check if system is ready
        readiness = rag_pipeline.check_readiness()
        if not readiness.get("is_ready", False):
            console.print("[red]❌ RAG system not ready[/red]")
            for issue in readiness.get("issues", []):
                console.print(f"[red]  - {issue}[/red]")
            return

        summary_count = 0
        processed_docs = 0

        if document_path:
            # Process specific document
            from pathlib import Path

            doc_path = Path(document_path)

            if not doc_path.exists():
                console.print(f"[red]❌ File not found: {document_path}[/red]")
                return

            with console.status(f"[dim]Generating summaries for {doc_path.name}..."):
                doc_result = extractor.extract_document(str(doc_path))
                if doc_result:
                    summary_chunks = summarizer.generate_all_summaries(doc_result)

                    # Convert to TextChunks and add to database
                    for summary_chunk in summary_chunks:
                        text_chunk = chunker.create_summary_chunk(summary_chunk)
                        # Add individual chunks to vector store
                        embedding_result = rag_pipeline.retriever.embedding_generator.generate_embeddings_sync(
                            [text_chunk.content]
                        )
                        rag_pipeline.retriever.add_documents_to_store(
                            [text_chunk], embedding_result.embeddings
                        )
                        summary_count += 1

                    processed_docs = 1
                else:
                    console.print(
                        f"[red]❌ Failed to extract content from {doc_path.name}[/red]"
                    )
                    return
        else:
            console.print(
                "[yellow]⚠️ Batch enrichment for all documents not yet implemented[/yellow]"
            )
            console.print("[dim]Please specify a document path for now[/dim]")
            return

        # Show results
        console.print("[green]✅ Enrichment completed successfully[/green]")
        console.print(f"[dim]Documents processed: {processed_docs}[/dim]")
        console.print(f"[dim]Summary chunks generated: {summary_count}[/dim]")
        console.print()
        console.print(
            "[blue]💡 Your knowledge base now has enhanced retrieval capabilities![/blue]"
        )
        console.print("[dim]Try asking conceptual questions like:[/dim]")
        console.print("[dim]  - '¿Cuál es la idea principal del documento?'[/dim]")
        console.print("[dim]  - '¿De qué trata este capítulo?'[/dim]")
        console.print("[dim]  - 'What is the main concept explained?'[/dim]")

    except Exception as e:
        console.print(f"[red]❌ Enrichment failed: {str(e)}[/red]")


@main.command()
def stats() -> None:
    """Show system statistics and information."""
    try:
        from src.rag.pipeline import RAGPipeline

        rag_pipeline = RAGPipeline()
        stats = rag_pipeline.get_system_stats()
        readiness = rag_pipeline.check_readiness()

        _display_detailed_stats(stats, readiness)

    except Exception as e:
        console.print(f"[red]❌ Error getting statistics: {str(e)}[/red]")


@main.command()
@click.argument("question")
@click.option("--top-k", type=int, help="Number of documents to retrieve")
@click.option("--threshold", type=float, help="Similarity threshold (0.0-1.0)")
@click.option("--show-sources", is_flag=True, help="Show source documents")
@click.option("--debug", is_flag=True, help="Show detailed retrieval information")
def query(
    question: str, top_k: int, threshold: float, show_sources: bool, debug: bool
) -> None:
    """Ask a single question to the RAG system."""
    try:
        from src.rag.pipeline import RAGPipeline

        rag_pipeline = RAGPipeline()

        # Check if system is ready
        readiness = rag_pipeline.check_readiness()
        if not readiness["is_ready"]:
            console.print("[red]❌ System not ready:[/red]")
            for issue in readiness["issues"]:
                console.print(f"   • {issue}")
            return

        console.print(f"[blue]❓ Question:[/blue] {question}")

        # Enable debug mode if requested
        if debug:
            settings.debug_mode = True
            settings.show_similarity_scores = True
            console.print("[dim]🔍 Debug mode enabled[/dim]")

        # Process query
        with console.status(
            "[dim]🔍 Searching knowledge base and generating answer..."
        ):
            kwargs = {}
            if top_k:
                kwargs["top_k"] = top_k
            if threshold:
                kwargs["similarity_threshold"] = threshold

            if show_sources:
                result = rag_pipeline.ask_with_sources(question, **kwargs)
                answer = result["answer"]
                sources_info = result["chunks_used"]
                performance = result["performance"]
            else:
                rag_result = rag_pipeline.ask(question, **kwargs)
                answer = rag_result.answer
                sources_info = []
                performance = {
                    "total_time": rag_result.total_time,
                    "chunks_retrieved": int(rag_result.chunks_retrieved),
                }

        # Display answer
        console.print("\n[bold green]🤖 Answer:[/bold green]")
        console.print(Panel(answer, border_style="green"))

        # Display performance info
        console.print(
            f"[dim]⏱️ Response time: {performance['total_time']:.2f}s | "
            f"📄 Chunks used: {performance['chunks_retrieved']}[/dim]"
        )

        # Display sources if requested
        if show_sources and sources_info:
            _display_sources(sources_info)

    except Exception as e:
        console.print(f"[red]❌ Error processing query: {str(e)}[/red]")


@main.command()
@click.option("--count", type=int, default=10, help="Number of chunks to inspect")
@click.option("--search", type=str, help="Search for chunks containing this text")
def inspect(count: int, search: str) -> None:
    """Inspect stored chunks to debug content quality."""
    try:
        from src.rag.pipeline import RAGPipeline

        rag_pipeline = RAGPipeline()

        # Access vector store chunks
        if not rag_pipeline.retriever.vector_store.chunks:
            console.print("[red]❌ No chunks found in vector store[/red]")
            return

        chunks = rag_pipeline.retriever.vector_store.chunks
        console.print(f"[blue]🔍 Inspecting chunks (total: {len(chunks)})[/blue]")

        # Filter chunks if search term provided
        if search:
            filtered_chunks = [
                chunk for chunk in chunks if search.lower() in chunk.content.lower()
            ]
            console.print(
                f"[dim]Found {len(filtered_chunks)} chunks containing '{search}'[/dim]"
            )
            chunks_to_show = filtered_chunks[:count]
        else:
            chunks_to_show = chunks[:count]

        for i, chunk in enumerate(chunks_to_show):
            console.print(f"\n[bold]--- Chunk {i + 1} ---[/bold]")
            console.print(f"[dim]Source: {chunk.source_file}[/dim]")
            console.print(f"[dim]Size: {len(chunk.content)} chars[/dim]")
            console.print(f"[dim]Position: {chunk.start_pos}-{chunk.end_pos}[/dim]")
            console.print("-" * 60)

            # Show content preview
            content_preview = chunk.content[:300]
            if len(chunk.content) > 300:
                content_preview += "..."
            console.print(content_preview)
            console.print("-" * 60)

    except Exception as e:
        console.print(f"[red]❌ Inspection failed: {str(e)}[/red]")


@main.command()
def chat() -> None:
    """Start interactive chat session."""
    try:
        from src.cli.chat import ChatInterface
        from src.rag.pipeline import RAGPipeline

        rag_pipeline = RAGPipeline()

        # Check if system is ready
        readiness = rag_pipeline.check_readiness()
        if not readiness["is_ready"]:
            console.print("[red]❌ System not ready for chat:[/red]")
            for issue in readiness["issues"]:
                console.print(f"   • {issue}")
            return

        # Start chat interface
        chat_interface = ChatInterface(rag_pipeline)
        chat_interface.start_interactive_session()

    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Chat session ended[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Chat error: {str(e)}[/red]")


def _display_system_info(stats: dict[str, Any]) -> None:
    """Display basic system information."""
    table = Table(title="System Information", border_style="blue")
    table.add_column("Component", style="cyan")
    table.add_column("Status", justify="left")

    # Models
    embedding_model = stats["models"]["embedding"]
    generation_model = stats["models"]["generation"]

    table.add_row(
        "Embedding Model",
        f"✅ {embedding_model['model_name']}"
        if embedding_model["is_available"]
        else f"❌ {embedding_model['status']}",
    )

    table.add_row(
        "Generation Model",
        f"✅ {generation_model['model_name']}"
        if generation_model["is_available"]
        else f"❌ {generation_model['status']}",
    )

    # Vector store
    vector_store = stats["vector_store"]
    table.add_row("Documents", str(vector_store["document_count"]))
    table.add_row("Text Chunks", str(vector_store["chunk_count"]))

    console.print(table)


def _display_detailed_stats(stats: dict[str, Any], readiness: dict[str, Any]) -> None:
    """Display detailed system statistics."""
    # System status
    status_panel = Panel(
        "🟢 System Ready" if readiness["is_ready"] else "🔴 System Issues Detected",
        title="System Status",
        border_style="green" if readiness["is_ready"] else "red",
    )
    console.print(status_panel)

    if not readiness["is_ready"]:
        console.print("[red]Issues:[/red]")
        for issue in readiness["issues"]:
            console.print(f"  • {issue}")
        console.print()

    # Vector store statistics
    vector_stats = stats["vector_store"]
    vector_table = Table(title="Knowledge Base", border_style="blue")
    vector_table.add_column("Metric", style="cyan")
    vector_table.add_column("Value", justify="right")

    vector_table.add_row("Documents", str(vector_stats["document_count"]))
    vector_table.add_row("Text Chunks", str(vector_stats["chunk_count"]))
    vector_table.add_row(
        "Embedding Dimension", str(vector_stats["embedding_dimension"])
    )
    vector_table.add_row("Database Size", f"{vector_stats['db_size_mb']:.1f} MB")
    vector_table.add_row("Last Updated", vector_stats["last_updated"])

    console.print(vector_table)

    # Model information
    models_table = Table(title="Models", border_style="green")
    models_table.add_column("Type", style="cyan")
    models_table.add_column("Model", style="green")
    models_table.add_column("Status", justify="center")

    embedding_model = stats["models"]["embedding"]
    generation_model = stats["models"]["generation"]

    models_table.add_row(
        "Embeddings",
        embedding_model["model_name"],
        "✅" if embedding_model["is_available"] else "❌",
    )
    models_table.add_row(
        "Generation",
        generation_model["model_name"],
        "✅" if generation_model["is_available"] else "❌",
    )

    console.print(models_table)

    # Settings
    settings_data = stats["settings"]
    settings_table = Table(title="Configuration", border_style="yellow")
    settings_table.add_column("Setting", style="cyan")
    settings_table.add_column("Value", justify="right")

    settings_table.add_row("Chunk Size", str(settings_data["chunk_size"]))
    settings_table.add_row("Chunk Overlap", str(settings_data["chunk_overlap"]))
    settings_table.add_row("Top-K Retrieval", str(settings_data["top_k_retrieval"]))
    settings_table.add_row(
        "Similarity Threshold", f"{settings_data['similarity_threshold']:.2f}"
    )
    settings_table.add_row("Temperature", f"{settings_data['temperature']:.2f}")
    settings_table.add_row("Max Tokens", str(settings_data["max_tokens"]))

    console.print(settings_table)


def _display_sources(sources_info: list[dict[str, Any]]) -> None:
    """Display source information for query results."""
    console.print(f"\n[blue]📄 Sources ({len(sources_info)} chunks):[/blue]")

    for source in sources_info:
        file_name = Path(source["source_file"]).name
        console.print(
            f"\n[bold]{source['rank']}. {file_name}[/bold] "
            f"[dim](score: {source['relevance_score']:.3f})[/dim]"
        )
        console.print(f"   [dim]{source['content_preview']}[/dim]")


if __name__ == "__main__":
    main()
