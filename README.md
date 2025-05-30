## Document Q&A System
A Retrieval-Augmented Generation (RAG) system for querying PDF documents using open-source Large Language Models (LLMs) and vector databases. This project allows users to upload PDFs, process them into a vector store, and ask questions to retrieve accurate answers based on document content.

## ✨ Features
- PDF Processing: Extracts text from PDFs using pdfplumber and PyPDF2
- Vector Database: Stores document embeddings in ChromaDB for efficient similarity search
- Embeddings: Generates text embeddings using sentence-transformers (e.g., paraphrase-MiniLM-L3-v2)
- RAG Pipeline: Combines retrieval from ChromaDB with LLM generation using Ollama (e.g., llama3.2:3b)
- Web Interface: Built with Streamlit for uploading PDFs, selecting models, and querying documents
- CPU-Optimized: Designed to run efficiently on CPU-only systems with lightweight models and batch processing

## 🛠️ Tech Stack

LLM: Local Ollama models (Llama3.2, Mistral, Qwen)
Embeddings: sentence-transformers (paraphrase-MiniLM-L3-v2, all-MiniLM-L6-v2)
Vector Database: ChromaDB (persistent local storage)
PDF Processing: pdfplumber, PyPDF2, Tesseract OCR
Web Framework: Streamlit
Text Processing: spaCy for semantic chunking

## 🚀 Installation
1. Clone the Repository
3. Set Up Virtual Environment
python -m venv rag_env
source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate
4. Install Dependencies
pip install -r requirements.txt
5. Install Tesseract OCR (for scanned PDFs)
sudo apt-get install tesseract-ocr poppler-utils  # On Ubuntu
6. Download spaCy Model
python -m spacy download en_core_web_sm
7. Run Ollama Server
ollama serve
ollama pull llama3.2:3b
8. Run the Application
streamlit run app.py

## 📖 Usage

Open the Streamlit app in your browser: http://localhost:8501
Select an LLM and embedding model from the sidebar
Upload one or more PDF documents
Ask questions related to the content of the uploaded documents
View answers with cited source excerpts

## 🧪 Example

Upload: A PDF document about machine learning
Question: "What is the main topic of the document?"
Answer: "The document discusses the fundamentals of machine learning, including supervised and unsupervised learning techniques." (with source excerpts)
