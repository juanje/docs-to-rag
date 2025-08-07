# Sample Documentation

Welcome to the docs-to-rag sample documentation!

## Overview

This is a sample document to demonstrate the capabilities of the docs-to-rag system. This document will be processed, chunked, embedded, and made searchable through the RAG pipeline.

## Features

The docs-to-rag system includes:

- **Document Processing**: Extract text from PDF, Markdown, and HTML files
- **Vector Embeddings**: Convert text to vector representations using Ollama
- **Similarity Search**: Find relevant content using FAISS vector database
- **Local RAG**: Generate answers using local LLMs with retrieved context

## Installation

To get started with docs-to-rag:

1. Install Ollama and start the service
2. Download the required models
3. Install the project dependencies
4. Run the setup command

## Usage Examples

### Adding Documents
```bash
./docs-to-rag add ./my_documents/
```

### Asking Questions
```bash
./docs-to-rag query "How do I install the system?"
```

### Interactive Chat
```bash
./docs-to-rag chat
```

## Technical Details

The system uses:
- **Docling** for robust PDF processing
- **FAISS** for efficient vector similarity search
- **Ollama** for local embeddings and language models
- **Rich** for beautiful terminal interfaces

## Troubleshooting

If you encounter issues:

1. Check that Ollama is running: `ollama serve`
2. Verify models are downloaded: `ollama list`
3. Check system status: `./docs-to-rag stats`

## Conclusion

This sample document demonstrates how docs-to-rag can process and make content searchable. Try asking questions about the installation process, features, or troubleshooting steps!