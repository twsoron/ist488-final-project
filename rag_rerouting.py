import streamlit as st
from openai import OpenAI
import sys
import chromadb
from pathlib import Path
import fitz

# Initialize ChromaDB
if 'VectorDB' not in st.session_state:
    chroma_client = chromadb.PersistentClient(path="./ChromaDB")
    st.session_state.VectorDB = chroma_client.get_or_create_collection(name="CourseCollection")

collection = st.session_state.VectorDB 

# Initialize OpenAI Client
if 'client' not in st.session_state:
    api_key = st.secrets["OPENAI_API_KEY"]
    st.session_state.client = OpenAI(api_key=api_key)

client = st.session_state.client

# Function to chunk text
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

# Get files metadata
def get_metadata(file_path):
    path = str(file_path).lower()
    filename = file_path.stem

    if "ist387_course_info" in path:
        return {"type": "syllabus", "source": filename}
    elif "ist387_hw_lab_code" in path:
        return {"type": "assignment", "source": filename}
    elif "ist387_notes_lecture" in path:
        return {"type": "concept", "source": filename}
    else:
        return {"type": "unknown", "source": filename}

# Extract text from PDF
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Add documents to collection
def add_to_collection(collection, text, file_name, metadata):
    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        response = client.embeddings.create(
            input=chunk,
            model="text-embedding-3-small"
        )

        embedding = response.data[0].embedding

        collection.add(
            documents=[chunk],
            ids=[f"{file_name}_{i}"],
            embeddings=[embedding],
            metadatas=[metadata]
        )
    
def load_all_pdfs(collection):
    folders = [
        "./hw_lab_code/",
        "./notes_lecture/",
        "./course_info/"
    ]

    for folder in folders:
        pdf_files = Path(folder).glob("*.pdf")

        for pdf_file in pdf_files:
            text = extract_text_from_pdf(pdf_file)
            metadata = get_metadata(pdf_file)

            add_to_collection(collection, text, pdf_file.stem, metadata)

# Populate collection with PDFs
if collection.count() == 0:
    load_all_pdfs(collection)

