import requests
import json
from typing import List, Dict
from document_processor import DocumentProcessor
from vector_store import VectorStore

class OllamaLLM:
    def __init__(self, model_name: str = "llama3.2:3b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
    
    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate response using Ollama"""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7
            }
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()['response']
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def create_rag_prompt(self, question: str, context_docs: List[Dict]) -> str:
        """Create RAG prompt with context"""
        context = "\n\n".join([doc['document'] for doc in context_docs])
        
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer the question using only the information provided in the context
- If the answer is not in the context, say "I cannot find the answer in the provided documents"
- Be concise and accurate
- Cite relevant parts of the context when possible

Answer:"""
        
        return prompt

class RAGSystem:
    def __init__(self, model_name: str = "llama3.2:3b", embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.llm = OllamaLLM(model_name)
        self.vector_store = VectorStore(embedding_model_name=embedding_model_name)
        self.doc_processor = DocumentProcessor()
    
    def add_document(self, pdf_path: str, doc_name: str = None):
        """Add a PDF document to the system"""
        # Extract text
        text = self.doc_processor.extract_text_pdfplumber(pdf_path)
        cleaned_text = self.doc_processor.clean_text(text)
        
        # Chunk text
        chunks = self.doc_processor.chunk_text(cleaned_text)
        
        # Create metadata
        doc_name = doc_name or pdf_path.split('/')[-1]
        metadatas = [{"source": doc_name, "chunk_id": i} for i in range(len(chunks))]
        
        # Add to vector store
        self.vector_store.add_documents(chunks, metadatas)
        
        return len(chunks)
    
    def query(self, question: str, k: int = 5) -> Dict:
        """Query the RAG system"""
        # Retrieve relevant documents
        relevant_docs = self.vector_store.similarity_search(question, k=k)
        
        # Generate prompt
        prompt = self.llm.create_rag_prompt(question, relevant_docs)
        
        # Generate response
        response = self.llm.generate_response(prompt)
        
        return {
            "answer": response,
            "sources": relevant_docs,
            "prompt_used": prompt
        }