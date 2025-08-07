"""
Interactive chat interface for docs-to-rag.

This module provides a rich, interactive chat interface for querying
the RAG system with features like command handling and response formatting.
"""

import logging
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from src.rag.pipeline import RAGPipeline, RAGResult

# Set up logging
logger = logging.getLogger(__name__)


class ChatInterface:
    """
    Interactive chat interface for the RAG system.

    Provides a rich terminal interface for conversing with the
    local RAG system with command support and formatted output.
    """

    def __init__(self, rag_pipeline: RAGPipeline | None = None):
        """
        Initialize the chat interface.

        Args:
            rag_pipeline: RAG pipeline instance (creates new if None)
        """
        self.rag_pipeline = rag_pipeline or RAGPipeline()
        self.console = Console()
        self.conversation_history: list[RAGResult] = []

        # Chat commands
        self.commands = {
            "help": self._show_help,
            "stats": self._show_stats,
            "clear": self._clear_history,
            "sources": self._toggle_sources,
            "history": self._show_history,
            "exit": self._exit_chat,
            "quit": self._exit_chat,
        }

        # Settings
        self.show_sources = False
        self.running = True

        logger.debug("ChatInterface initialized")

    def start_interactive_session(self) -> None:
        """Start the interactive chat session."""
        self._display_welcome()

        while self.running:
            try:
                # Get user input
                user_input = self._get_user_input()

                if not user_input:
                    continue

                # Check for commands
                if user_input.startswith("/"):
                    self._handle_command(user_input[1:])
                    continue

                # Process as question
                self._process_question(user_input)

            except KeyboardInterrupt:
                self.console.print("\n[yellow]👋 Chat session ended[/yellow]")
                break
            except EOFError:
                self.console.print("\n[yellow]👋 Chat session ended[/yellow]")
                break
            except Exception as e:
                self.console.print(f"[red]❌ Error: {str(e)}[/red]")
                logger.error(f"Chat error: {e}", exc_info=True)

    def _display_welcome(self) -> None:
        """Display welcome message and instructions."""
        welcome_text = """
🤖 **Welcome to docs-to-rag Chat!**

Ask questions about your indexed documents and get AI-powered answers.

**Commands:**
• `/help` - Show this help message
• `/stats` - Show system statistics  
• `/sources` - Toggle source display
• `/history` - Show conversation history
• `/clear` - Clear conversation history
• `/exit` - Exit chat

**Tips:**
• Ask specific questions about your documents
• Use natural language - no special formatting needed
• Type `/sources` to see which documents were used for answers
        """

        self.console.print(
            Panel(
                welcome_text.strip(),
                title="docs-to-rag Chat",
                border_style="blue",
                padding=(1, 2),
            )
        )

        # Show system readiness
        readiness = self.rag_pipeline.check_readiness()
        if not readiness["is_ready"]:
            self.console.print("\n[red]⚠️ System Issues:[/red]")
            for issue in readiness["issues"]:
                self.console.print(f"   • {issue}")
            self.console.print()
        else:
            stats = self.rag_pipeline.get_system_stats()
            doc_count = stats["vector_store"]["document_count"]
            chunk_count = stats["vector_store"]["chunk_count"]
            self.console.print(
                f"[green]✅ System ready![/green] "
                f"[dim]{doc_count} documents, {chunk_count} chunks indexed[/dim]\n"
            )

    def _get_user_input(self) -> str:
        """Get user input with prompt."""
        return Prompt.ask("[bold blue]❓ Your question", console=self.console).strip()

    def _process_question(self, question: str) -> None:
        """Process a user question through the RAG pipeline."""
        self.console.print("\n[dim]🔍 Searching knowledge base...[/dim]")

        try:
            with self.console.status("[dim]💭 Generating answer..."):
                if self.show_sources:
                    result_dict = self.rag_pipeline.ask_with_sources(question)
                    answer = result_dict["answer"]
                    sources = result_dict["chunks_used"]
                    performance = result_dict["performance"]
                else:
                    rag_result = self.rag_pipeline.ask(question)
                    answer = rag_result.answer
                    sources = []
                    performance = {
                        "total_time": rag_result.total_time,
                        "chunks_retrieved": rag_result.chunks_retrieved,
                    }

            # Display answer
            self._display_answer(answer, performance)

            # Display sources if enabled
            if self.show_sources and sources:
                self._display_sources(sources)

            # Add to conversation history
            if not self.show_sources:
                self.conversation_history.append(rag_result)

        except Exception as e:
            self.console.print(f"[red]❌ Error processing question: {str(e)}[/red]")

    def _display_answer(self, answer: str, performance: dict[str, Any]) -> None:
        """Display the generated answer with formatting."""
        # Format answer as markdown if it contains formatting
        if any(marker in answer for marker in ["**", "*", "`", "#", "-", "1."]):
            formatted_answer = Markdown(answer)
        else:
            formatted_answer = answer

        self.console.print("\n[bold green]🤖 Answer:[/bold green]")
        self.console.print(
            Panel(formatted_answer, border_style="green", padding=(1, 2))
        )

        # Performance info
        total_time = performance.get("total_time", 0)
        chunks_retrieved = performance.get("chunks_retrieved", 0)
        self.console.print(
            f"[dim]⏱️ {total_time:.2f}s | 📄 {chunks_retrieved} chunks used[/dim]\n"
        )

    def _display_sources(self, sources: list[dict[str, Any]]) -> None:
        """Display source information."""
        self.console.print(f"[blue]📄 Sources ({len(sources)} chunks):[/blue]")

        for source in sources:
            from pathlib import Path

            file_name = Path(source["source_file"]).name
            score = source["relevance_score"]
            preview = source["content_preview"]

            self.console.print(
                f"\n[bold]{source['rank']}. {file_name}[/bold] "
                f"[dim](relevance: {score:.3f})[/dim]"
            )
            self.console.print(f"[dim]   {preview}[/dim]")

        self.console.print()

    def _handle_command(self, command: str) -> None:
        """Handle chat commands."""
        command = command.lower().strip()

        if command in self.commands:
            self.commands[command]()
        else:
            self.console.print(f"[red]❌ Unknown command: /{command}[/red]")
            self.console.print("[dim]Type /help for available commands[/dim]")

    def _show_help(self) -> None:
        """Show help information."""
        help_table = Table(title="Chat Commands", border_style="blue")
        help_table.add_column("Command", style="cyan", width=12)
        help_table.add_column("Description", style="white")

        help_table.add_row("/help", "Show this help message")
        help_table.add_row("/stats", "Display system statistics")
        help_table.add_row(
            "/sources",
            f"Toggle source display (currently: "
            f"{'ON' if self.show_sources else 'OFF'})",
        )
        help_table.add_row("/history", "Show conversation history")
        help_table.add_row("/clear", "Clear conversation history")
        help_table.add_row("/exit", "Exit the chat session")

        self.console.print(help_table)
        self.console.print()

    def _show_stats(self) -> None:
        """Show system statistics."""
        try:
            stats = self.rag_pipeline.get_system_stats()

            stats_table = Table(title="System Statistics", border_style="green")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="white", justify="right")

            # Vector store stats
            vector_store = stats["vector_store"]
            stats_table.add_row("Documents", str(vector_store["document_count"]))
            stats_table.add_row("Text Chunks", str(vector_store["chunk_count"]))
            stats_table.add_row("Database Size", f"{vector_store['db_size_mb']:.1f} MB")

            # Models
            embedding_model = stats["models"]["embedding"]["model_name"]
            generation_model = stats["models"]["generation"]["model_name"]
            stats_table.add_row("Embedding Model", embedding_model)
            stats_table.add_row("Generation Model", generation_model)

            # Conversation stats
            stats_table.add_row("Questions Asked", str(len(self.conversation_history)))

            self.console.print(stats_table)
            self.console.print()

        except Exception as e:
            self.console.print(f"[red]❌ Error getting stats: {str(e)}[/red]")

    def _clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
        self.console.print("[yellow]🗑️ Conversation history cleared[/yellow]\n")

    def _toggle_sources(self) -> None:
        """Toggle source display."""
        self.show_sources = not self.show_sources
        status = "enabled" if self.show_sources else "disabled"
        self.console.print(f"[blue]📄 Source display {status}[/blue]\n")

    def _show_history(self) -> None:
        """Show conversation history."""
        if not self.conversation_history:
            self.console.print("[yellow]📝 No conversation history yet[/yellow]\n")
            return

        self.console.print(
            f"[blue]📜 Conversation History ({len(self.conversation_history)} questions):[/blue]\n"
        )

        for i, result in enumerate(self.conversation_history, 1):
            self.console.print(f"[bold]{i}. {result.question}[/bold]")
            # Show first 100 chars of answer
            answer_preview = (
                result.answer[:100] + "..."
                if len(result.answer) > 100
                else result.answer
            )
            self.console.print(f"   [dim]{answer_preview}[/dim]")
            self.console.print(
                f"   [dim]({result.total_time:.2f}s, {result.chunks_retrieved} chunks)[/dim]\n"
            )

    def _exit_chat(self) -> None:
        """Exit the chat session."""
        self.running = False

        # Show session summary
        if self.conversation_history:
            total_questions = len(self.conversation_history)
            total_time = sum(r.total_time for r in self.conversation_history)
            avg_time = total_time / total_questions if total_questions > 0 else 0

            self.console.print("\n[blue]📊 Session Summary:[/blue]")
            self.console.print(f"   • Questions asked: {total_questions}")
            self.console.print(f"   • Total time: {total_time:.2f}s")
            self.console.print(f"   • Average response time: {avg_time:.2f}s")

        self.console.print("\n[yellow]👋 Thanks for using docs-to-rag![/yellow]")

    def batch_process(self, questions: list[str]) -> list[RAGResult]:
        """
        Process multiple questions in batch mode.

        Args:
            questions: List of questions to process

        Returns:
            List of RAGResult objects
        """
        self.console.print(f"[blue]📋 Processing {len(questions)} questions...[/blue]")

        results = []
        for i, question in enumerate(questions, 1):
            self.console.print(f"[dim]{i}/{len(questions)}: {question[:50]}...[/dim]")

            try:
                result = self.rag_pipeline.ask(question)
                results.append(result)
                self.conversation_history.append(result)
            except Exception as e:
                self.console.print(
                    f"[red]❌ Error processing question {i}: {str(e)}[/red]"
                )

        self.console.print(
            f"[green]✅ Batch processing completed: {len(results)} results[/green]"
        )
        return results
