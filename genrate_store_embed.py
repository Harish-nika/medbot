import streamlit as st
import fitz  # PyMuPDF for PDF processing
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
import re
import time

# Set directory to save embeddings and text chunks
EMBEDDING_DIR = "/home/harish/Agentic_AI/embeddings"
TEXT_CHUNKS_DIR = "/home/harish/Agentic_AI/text_chunks"  # Directory for text chunks
os.makedirs(EMBEDDING_DIR, exist_ok=True)
os.makedirs(TEXT_CHUNKS_DIR, exist_ok=True)

# Load SBERT model (using st.cache_resource)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Load the model before running the code
embedding_model = load_embedding_model()

# Streamlit UI
st.title("📚 MedBot - Medical AI Chatbot")
st.sidebar.header("Upload Medical PDFs")
uploaded_files = st.sidebar.file_uploader("Upload PDF Books", accept_multiple_files=True, type=["pdf"])

# Function to extract text from PDFs (using st.cache_data)
@st.cache_data
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join([page.get_text("text") for page in doc])
    return text

# Function to generate embeddings using SBERT (batch processing) (using st.cache_data)
@st.cache_data
def get_embeddings_batch(texts):
    embeddings = embedding_model.encode(texts, show_progress_bar=True, batch_size=16)  # Batch processing
    return embeddings

# Function to improve chunking based on topics and subtopics (headings)
def split_text_by_headings(text):
    heading_pattern = re.compile(r"^[A-Z][A-Za-z0-9\s\-]+:$")  # Simple heading pattern (e.g., "Introduction:")
    chunks = []
    current_chunk = []
    
    for line in text.split('\n'):
        if heading_pattern.match(line):  # If a heading is detected, start a new chunk
            if current_chunk:
                chunks.append("\n".join(current_chunk))
            current_chunk = [line]  # Start new chunk with heading
        else:
            current_chunk.append(line)  # Add line to current chunk
    
    if current_chunk:
        chunks.append("\n".join(current_chunk))  # Add remaining text
    
    return chunks

# Process uploaded PDFs
if uploaded_files:
    for file in uploaded_files:
        st.write(f"Processing: {file.name}")
        text = extract_text_from_pdf(file)
        
        # Split the text based on headings or topics/subtopics
        chunks = split_text_by_headings(text)
        
        # If chunks are too large, further split them using RecursiveCharacterTextSplitter
        all_chunks = []
        for chunk in chunks:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            sub_chunks = text_splitter.split_text(chunk)
            all_chunks.extend(sub_chunks)
        
        # Batch processing of embeddings
        start_time = time.time()
        embeddings = get_embeddings_batch(all_chunks)
        st.write(f"Embedding generation took {time.time() - start_time:.2f} seconds")
        
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # Save embeddings to .npy file
        save_path = os.path.join(EMBEDDING_DIR, f"{file.name}.npy")
        np.save(save_path, embeddings)
        st.success(f"Embeddings saved for {file.name}")
        
        # Save text chunks to .pkl file
        text_chunks_path = os.path.join(TEXT_CHUNKS_DIR, f"{file.name}_chunks.pkl")
        with open(text_chunks_path, 'wb') as f:
            pickle.dump(all_chunks, f)
        st.success(f"Text chunks saved for {file.name}")
