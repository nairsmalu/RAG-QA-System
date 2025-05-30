# Document Q&A System

A Retrieval-Augmented Generation (RAG) system for querying PDF documents using open-source Large Language Models (LLMs) and vector databases. This project allows users to upload PDFs, process them into a vector store, and ask questions to retrieve accurate answers based on document content.

## Features
- **PDF Processing**: Extracts text from PDFs using `pdfplumber`and `PyPDF2`.
- **Vector Database**: Stores document embeddings in **ChromaDB** for efficient similarity search.
- **Embeddings**: Generates text embeddings using `sentence-transformers` (e.g., `paraphrase-MiniLM-L3-v2`).
- **RAG Pipeline**: Combines retrieval from **ChromaDB** with LLM generation using **Ollama** (e.g., `llama3.2:3b`).
- **Web Interface**: Built with **Streamlit** for uploading PDFs, selecting models, and querying documents.
- **CPU-Optimized**: Designed to run efficiently on CPU-only systems with lightweight models and batch processing.

## Tech Stack
- **LLM**: Local Ollama models (Llama3.2, Mistral, Qwen)
- **Embeddings**: `sentence-transformers` (paraphrase-MiniLM-L3-v2, all-MiniLM-L6-v2)
- **Vector Database**: ChromaDB (persistent local storage)
- **PDF Processing**: pdfplumber, PyPDF2, Tesseract OCR
- **Web Framework**: Streamlit
- **Text Processing**: spaCy for semantic chunking


