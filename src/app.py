import streamlit as st
import tempfile
import os
from document_processor import DocumentProcessor
from vector_store import VectorStore
from llm_interface import RAGSystem


# Page config
st.set_page_config(
    page_title="Document Q&A System",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = []

st.title("📚 Document Q&A System")
st.write("Upload PDFs and ask questions about their content using open-source AI models!")

# Sidebar for model selection
with st.sidebar:
    st.header("⚙️ Configuration")
    available_models = ["llama3.2:3b", "llama3.1:8b", "llama3.2:3b", "mistral:7b", "qwen2.5:7b"]
    available_embedding_models = ["paraphrase-MiniLM-L3-v2","all-MiniLM-L6-v2", "distiluse-base-multilingual-cased-v2"]
    selected_model = st.selectbox("Select LLM Model", available_models)
    selected_embedding_model = st.selectbox("Select Embedding Model", available_embedding_models)
    if st.button("Initialize System"):
        with st.spinner("Initializing RAG system..."):
            st.session_state.rag_system = RAGSystem(selected_model, selected_embedding_model)
        st.success("System initialized!")
    
    # System status
    if st.session_state.rag_system:
        st.success("✅ System Ready")
        st.write(f"Model: {selected_model}")
        st.write(f"Documents: {len(st.session_state.documents_processed)}")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📄 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True
    )
    
    if uploaded_files and st.session_state.rag_system:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.documents_processed:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Process document
                        num_chunks = st.session_state.rag_system.add_document(
                            tmp_path, 
                            uploaded_file.name
                        )
                        
                        st.success(f"✅ {uploaded_file.name} processed ({num_chunks} chunks)")
                        st.session_state.documents_processed.append(uploaded_file.name)
                        
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    
                    finally:
                        # Clean up temp file
                        os.unlink(tmp_path)
    
    # Display processed documents
    if st.session_state.documents_processed:
        st.subheader("Processed Documents")
        for doc in st.session_state.documents_processed:
            st.write(f"📄 {doc}")

with col2:
    st.header("❓ Ask Questions")
    
    if st.session_state.rag_system and st.session_state.documents_processed:
        question = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="What is the main topic discussed in the documents?"
        )
        
        col2a, col2b = st.columns([1, 3])
        with col2a:
            num_sources = st.slider("Number of sources", 1, 10, 5)
        
        if st.button("🔍 Ask Question", type="primary"):
            if question.strip():
                with st.spinner("Searching and generating answer..."):
                    try:
                        result = st.session_state.rag_system.query(question, k=num_sources)
                        
                        # Display answer
                        st.subheader("💡 Answer")
                        st.write(result["answer"])
                        
                        # Display sources
                        with st.expander("📚 View Sources"):
                            for i, source in enumerate(result["sources"], 1):
                                st.write(f"**Source {i}** (Distance: {source['distance']:.3f})")
                                st.write(f"*From: {source['metadata']['source']}*")
                                st.write(source["document"][:500] + "..." if len(source["document"]) > 500 else source["document"])
                                st.write("---")
                        
                        # Debug info
                        with st.expander("🔧 Debug Info"):
                            st.text_area("Prompt Used:", result["prompt_used"], height=200)
                            
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a question!")
    
    elif not st.session_state.rag_system:
        st.info("👈 Please initialize the system first")
    
    elif not st.session_state.documents_processed:
        st.info("👈 Please upload and process some documents first")

# Footer
st.markdown("---")
st.markdown("""
**Tech Stack:** Ollama + ChromaDB + Sentence Transformers + Streamlit
- 🤖 **LLM**: Local Ollama models (Llama, Mistral, Qwen)
- 🔍 **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- 🗄️ **Vector DB**: ChromaDB (persistent local storage)
- 📄 **PDF Processing**: pdfplumber + PyPDF2
""")
