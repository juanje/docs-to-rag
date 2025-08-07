# docs-to-rag

An educational project for learning document extraction and RAG (Retrieval-Augmented Generation) with local LLMs using Ollama. Features advanced hierarchical document enrichment for enhanced retrieval quality.

## 🎯 Overview

**docs-to-rag** is designed to teach the fundamentals of building a RAG system by:

- **Extracting content** from documents (PDF, Markdown, HTML)
- **Creating vector embeddings** using local Ollama models
- **Storing and searching** with FAISS vector database
- **Generating answers** using retrieved context and local LLMs
- **Enriching knowledge** with hierarchical document summarization for enhanced retrieval

This project goes beyond basic RAG by implementing **hierarchical document enrichment** that creates synthetic summaries at document, chapter, and concept levels, dramatically improving retrieval quality for conceptual questions.

All processing happens **locally** - no external APIs required!

## 🚀 Quick Start

### Prerequisites

1. **Python 3.10+**
2. **Ollama** installed and running
   ```bash
   # Install Ollama (see https://ollama.ai)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Start Ollama
   ollama serve
   
   # Download required models
   ollama pull nomic-embed-text:v1.5
   ollama pull llama3.2:latest
   ```

### Installation

```bash
# Clone the repository
git clone https://github.com/juanje/docs-to-rag.git
cd docs-to-rag

# Install dependencies with uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Or with pip
pip install -e .
```

After installation, the `docs-to-rag` command will be available in your PATH as a Python entry point.

### Setup and Usage

```bash
# 1. Initialize the system (check if everything is working)
docs-to-rag stats

# 2. Add documents to your knowledge base
docs-to-rag add ./path/to/your/documents/

# 3. Ask questions!
docs-to-rag query "What is the main topic of these documents?"

# 4. Start interactive chat
docs-to-rag chat
```

## 📚 Features

### Document Processing
- **PDF extraction** using Docling for robust text extraction
- **Markdown processing** with structure preservation
- **HTML content extraction** with cleaning and formatting
- **Intelligent chunking** that respects document structure

### Vector Operations
- **Local embeddings** using Ollama (nomic-embed-text)
- **FAISS vector storage** for efficient similarity search
- **Configurable similarity thresholds** and retrieval parameters

### RAG Pipeline
- **Context-aware retrieval** of relevant document chunks
- **Local answer generation** using Ollama LLMs
- **Source attribution** showing which documents were used
- **Performance metrics** for optimization and learning

### Hierarchical Summarization
- **Document-level summaries** for overall document understanding
- **Chapter-level summaries** for section-specific questions
- **Concept-level summaries** for key topic extraction
- **Enhanced retrieval** for conceptual and broad questions
- **Configurable summarization models** with quality validation

### CLI Interface
- **Simple commands** for document management
- **Interactive chat** with rich terminal interface
- **System statistics** and health monitoring
- **Configurable parameters** for experimentation

## 🎛️ Commands

### System Setup
```bash
docs-to-rag setup                     # Initialize system and check configuration
docs-to-rag --help                    # Show all available commands
docs-to-rag --version                 # Show version information
```

### Document Management
```bash
docs-to-rag add ./documents/          # Add documents
docs-to-rag list                      # List indexed documents  
docs-to-rag clear                     # Clear knowledge base
docs-to-rag stats                     # Show system statistics
docs-to-rag enrich ./document.pdf     # Generate hierarchical summaries
docs-to-rag enrich                    # Enrich all documents (planned)
```

### Advanced Operations
```bash
docs-to-rag reprocess ./doc.pdf       # Reprocess document with improved chunking
docs-to-rag reprocess ./doc.pdf --multilingual  # Use multilingual embedding model
docs-to-rag inspect                   # Inspect stored chunks for debugging
docs-to-rag inspect --search "term"   # Search for specific content in chunks
docs-to-rag config                    # Show/modify system configuration
```

### Querying
```bash
docs-to-rag query "Your question"                          # Single question (adaptive)
docs-to-rag query "Your question" --debug                  # With detailed info
docs-to-rag query "Your question" --top-k 20 --threshold 0.2  # Custom parameters
docs-to-rag chat                                           # Interactive chat session
```

### Document Enrichment
The `enrich` command generates hierarchical summaries to improve retrieval quality for conceptual questions:

```bash
# Enable summarization (required first time)
docs-to-rag config --enable-summaries

# Enrich specific document with summaries
docs-to-rag enrich ./path/to/document.pdf

# Configure summarization model
docs-to-rag config --summary-model llama3.2:latest

# Enable quality validation
docs-to-rag config --enable-validation
```

This creates three types of synthetic documents:
- **Document summaries**: Overall document content and main themes
- **Chapter summaries**: Section-specific summaries for focused retrieval  
- **Concept summaries**: Key concepts and definitions for better Q&A

These summaries are stored alongside original chunks and dramatically improve results for questions like:
- "What is this document about?"
- "Explain the main concepts"
- "Summarize the key findings"

### Advanced Operations

#### System Setup
- **`setup`**: Verifies Ollama connection, checks model availability, creates necessary directories, and provides troubleshooting guidance if issues are detected.

#### Document Reprocessing
- **`reprocess`**: Clears the database and reprocesses a specific document with improved chunking algorithms or different embedding models. Useful when:
  - You've updated the chunking logic
  - You want to switch between standard and multilingual embeddings
  - Document wasn't processed correctly the first time

#### System Debugging
- **`inspect`**: Shows the actual chunks stored in the vector database to help debug content quality, chunking behavior, and identify potential issues:
  ```bash
  docs-to-rag inspect --count 20        # Show first 20 chunks
  docs-to-rag inspect --search "python" # Find chunks containing "python"
  ```

#### Configuration Management
- **`config`**: View current system configuration or modify settings like language, summarization options, and model preferences. Changes are saved locally to `config/user_config.json`.

### Interactive Chat Commands
While in chat mode, you can use these special commands:
- `/help` - Show available commands and usage tips
- `/stats` - Display system statistics and database info
- `/sources` - Toggle source document display in responses
- `/history` - Show conversation history with timestamps
- `/clear` - Clear conversation history
- `/exit` or `/quit` - Exit chat session

These commands enhance the interactive experience by providing system information and controlling response formatting without leaving the chat session.

## 🧪 Testing

The project has a well-organized test suite separated into unit and integration tests:

### Quick Testing (Unit Tests - Fast)
```bash
# Run only unit tests (fast, ~4s)
uv run pytest tests/unit/

# Specific unit test file
uv run pytest tests/unit/test_config.py
```

### Complete Testing (All Tests)
```bash
# Run all tests (unit + integration)
uv run pytest tests/

# Run only integration tests (slower, real components)
uv run pytest tests/integration/
```

### Test Structure
- **`tests/unit/`**: Fast unit tests using mocks
- **`tests/integration/`**: Integration tests with real components

## 🔧 Configuration

The system can be configured via environment variables, CLI commands, or by editing `config/settings.py`:

```bash
# Example .env file
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text:v1.5
CHAT_MODEL=llama3.2:latest
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RETRIEVAL=3
```

### CLI Configuration

You can also configure the system using CLI commands. Settings are saved locally to `config/user_config.json`:

```bash
# Set primary language
docs-to-rag config --language es

# Enable/disable summarization
docs-to-rag config --enable-summaries
docs-to-rag config --disable-summaries

# Set specific models
docs-to-rag config --summary-model llama3.2:latest

# View current configuration
docs-to-rag config
```

### Key Parameters
- **Chunk Size**: Size of text chunks (default: 1000 characters)
- **Chunk Overlap**: Overlap between chunks (default: 200 characters)
- **Top-K Retrieval**: Number of chunks to retrieve (default: 3)
- **Temperature**: LLM creativity (default: 0.1 for factual responses)

## 📖 Learning Objectives

This project teaches:

1. **Document Processing**
   - Text extraction from different formats
   - Content cleaning and normalization
   - Intelligent text segmentation

2. **Vector Embeddings**
   - Converting text to numerical representations
   - Understanding semantic similarity
   - Local vs. cloud embedding models

3. **Vector Databases**
   - FAISS indexing and search
   - Similarity metrics and thresholds
   - Performance optimization

4. **RAG Implementation**
   - Retrieval-Augmented Generation concepts
   - Context window management
   - Prompt engineering for factual responses

5. **System Integration**
   - Modular architecture design
   - CLI application development
   - Local LLM integration

## 🏗️ Architecture

```
docs-to-rag/
├── src/
│   ├── document_processor/    # Text extraction and chunking
│   ├── vector_store/         # Embeddings and FAISS storage
│   ├── rag/                  # Retrieval and generation pipeline
│   └── cli/                  # Command-line interface
├── config/                   # Configuration management
├── data/                     # Documents and vector database
├── tests/                    # Test suite
└── docs-to-rag              # Main executable
```

## 🧪 Testing

```bash
# Run the test suite
pytest tests/

# Run with verbose output
pytest tests/ -v

# Test specific functionality
pytest tests/test_basic.py::TestDocumentProcessor
```

## 📋 Example Workflow

```bash
# 1. Setup (one time)
docs-to-rag setup
# ✅ Ollama connection verified
# ✅ Models available
# ✅ Directories created

# 2. Add your documentation
docs-to-rag add ./my_docs/
# 📄 Processing documents...
# ✅ 15 documents processed
# 📊 245 chunks created
# 💾 Vector database updated

# 3. Ask questions
docs-to-rag query "How do I install the software?"
# 🔍 Searching knowledge base...
# 🤖 Answer: To install the software, follow these steps...
# ⏱️ 2.3s | 📄 3 chunks used

# 4. Interactive chat
docs-to-rag chat
# 🤖 Welcome to docs-to-rag Chat!
# ❓ Your question: What are the main features?
# 🤖 Answer: The main features include...
```

## 🔧 Troubleshooting

### Common Issues

1. **"Ollama not accessible"**
   ```bash
   # Make sure Ollama is running
   ollama serve
   
   # Check if models are available
   ollama list
   ```

2. **"No documents indexed"**
   ```bash
   # Add documents first
   docs-to-rag add ./path/to/documents/
   
   # Check supported formats
   docs-to-rag stats
   ```

3. **"Docling not available"**
   ```bash
   # Install docling for PDF support
   uv pip install docling
   ```

### Performance Tips

- **Chunk Size**: Smaller chunks (500-800) for specific questions, larger chunks (1000-1500) for context
- **Top-K**: System auto-adjusts based on database size (3-20 chunks)
- **Similarity Threshold**: Auto-adapts from 0.7 (small DB) to 0.3 (large DB)
- **Models**: Use `llama3.2:latest` for speed, larger models like `llama3.1:8b` for better quality

### 🔧 RAG Optimization Guide

#### For Large Documents (10K+ chunks):
```bash
# Let adaptive parameters handle optimization automatically
docs-to-rag query "Your question" --debug

# For very specific searches, use lower threshold
docs-to-rag query "Specific term" --threshold 0.2

# For broader context, increase top-k
docs-to-rag query "Complex question" --top-k 25
```

#### 🌍 Language-Specific Optimization:
```bash
# For Spanish documents, reprocess with Spanish embeddings
docs-to-rag reprocess "documento.pdf" --multilingual

# Available embedding models:
# - nomic-embed-text:v1.5 (English, fast)
# - jina/jina-embeddings-v2-base-es:latest (Spanish, optimal)
# - mxbai-embed-large:latest (Multilingual, general)
```

#### Troubleshooting Poor Results:
1. **No results found**: Try `--threshold 0.1` to be less restrictive
2. **Too generic answers**: Use `--top-k 5` for more focused results  
3. **Missing context**: Increase `--top-k 20` for broader context
4. **Language issues**: Use `--multilingual` when reprocessing non-English docs
5. **Debug mode**: Always use `--debug` to see what's being retrieved
6. **Content quality**: Use `inspect --search "term"` to check chunk quality

## 🤝 Contributing

This is an educational project! Feel free to:
- Experiment with different parameters
- Add support for new document formats
- Improve the chunking strategies
- Enhance the chat interface
- Add evaluation metrics

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤖 AI Tools Disclaimer

This project was developed with the assistance of artificial intelligence tools:

**Tools used:**
- **Cursor**: Code editor with AI capabilities
- **Claude-4-Sonnet**: Anthropic's language model

**Division of responsibilities:**

**AI (Cursor + Claude-4-Sonnet)**:
- 🔧 Initial code prototyping
- 📝 Generation of examples and test cases
- 🐛 Assistance in debugging and error resolution
- 📚 Documentation and comments writing
- 💡 Technical implementation suggestions

**Human (Juanje Ojeda)**:
- 🎯 Specification of objectives and requirements
- 🔍 Critical review of code and documentation
- 💬 Iterative feedback and solution refinement
- 📋 Definition of project's educational structure
- ✅ Final validation of concepts and approaches

**Collaboration philosophy**: AI tools served as a highly capable technical assistant, while all design decisions, educational objectives, and project directions were defined and validated by the human.

---

**Author**: Juanje Ojeda (juanje@redhat.com)  
**Purpose**: Educational RAG system with local LLMs  
**Status**: Learning project - not for production use