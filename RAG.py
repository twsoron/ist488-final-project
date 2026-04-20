"""PDF ingest + chunking + indexing into ChromaDB for the IST 387 chatbot.

The pure functions (chunk_text, extract_text_from_pdf, get_metadata,
add_to_collection, load_all_pdfs) have no streamlit dependency so they
can be imported by standalone scripts (see rebuild_index.py).

Running under `streamlit run` also populates the collection on first load.
"""

import re
import unicodedata
from pathlib import Path

import chromadb
import fitz
from openai import OpenAI

# Per-type chunk sizing: syllabus facts are short; assignment problems are long.
CHUNK_SIZES_BY_TYPE = {
    "syllabus": 600,
    "assignment": 1500,
    "concept": 1200,
    "unknown": 1000,
}


def chunk_text(text, chunk_size=1200, overlap=200):
    """Sliding window that snaps each chunk boundary backward to the nearest
    paragraph/sentence/line/word break, so chunks never cut mid-word."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            for sep in ("\n\n", ". ", "\n", " "):
                split = text.rfind(sep, start + chunk_size // 2, end)
                if split != -1:
                    end = split + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


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


def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = "".join(page.get_text("text") for page in doc)
    return unicodedata.normalize("NFKC", text)


# Row separators for the syllabus schedule: "Lecture N/" markers or
# standalone no-lecture date rows ("March 12", "April 2", "April 23").
_SYLLABUS_ROW_SPLIT = re.compile(
    r"(?=Lecture\s+\d+\s*/|Lab 0:|^(?:March 12|April 2|April 23)\s*$)",
    re.MULTILINE,
)


def chunk_syllabus(text: str) -> list[str]:
    """Chunk the syllabus normally, except split the weekly schedule by row.

    Each schedule row (one week's lecture + reading + lab + HW) becomes its
    own chunk so queries like "when is HW 3 assigned" can retrieve a single
    self-contained row instead of a fragment of a 600-char window.
    """
    schedule_start = text.find("Week/ Date")
    if schedule_start < 0:
        return chunk_text(text, chunk_size=CHUNK_SIZES_BY_TYPE["syllabus"])

    schedule_end = len(text)
    for marker in ("SYRACUSE UNIVERSITY STUDENT POLICIES", "iSchool Values"):
        idx = text.find(marker, schedule_start)
        if idx > 0:
            schedule_end = min(schedule_end, idx)

    before = text[:schedule_start]
    schedule = text[schedule_start:schedule_end]
    after = text[schedule_end:]

    body_chunks = chunk_text(before, chunk_size=CHUNK_SIZES_BY_TYPE["syllabus"])
    body_chunks += chunk_text(after, chunk_size=CHUNK_SIZES_BY_TYPE["syllabus"])

    row_chunks = [
        "IST 387 Schedule — " + row.strip()
        for row in _SYLLABUS_ROW_SPLIT.split(schedule)
        if len(row.strip()) > 40
    ]
    return body_chunks + row_chunks


def add_to_collection(collection, client, text, file_name, metadata):
    if metadata["type"] == "syllabus":
        chunks = chunk_syllabus(text)
    else:
        chunk_size = CHUNK_SIZES_BY_TYPE.get(metadata["type"], 1000)
        chunks = chunk_text(text, chunk_size=chunk_size)

    prefix = f"Source: {metadata['source']} | Type: {metadata['type']}\n\n"

    for i, chunk in enumerate(chunks):
        response = client.embeddings.create(
            input=prefix + chunk,
            model="text-embedding-3-small",
        )
        embedding = response.data[0].embedding
        collection.add(
            documents=[chunk],
            ids=[f"{file_name}_{i}"],
            embeddings=[embedding],
            metadatas=[metadata],
        )


def load_all_pdfs(collection, client):
    folders = [
        "./ist387_hw_lab_code/",
        "./ist387_notes_lecture/",
        "./ist387_course_info/",
    ]
    for folder in folders:
        for pdf_file in Path(folder).glob("*.pdf"):
            text = extract_text_from_pdf(pdf_file)
            metadata = get_metadata(pdf_file)
            add_to_collection(collection, client, text, pdf_file.stem, metadata)
            print("Processing:", pdf_file.name)


# Streamlit-only module-level init (runs on `streamlit run`).
try:
    import streamlit as st

    chroma_client = chromadb.PersistentClient(path="./ChromaDB")
    collection = chroma_client.get_or_create_collection(name="CourseCollection")

    if "client" not in st.session_state:
        st.session_state.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    client = st.session_state.client

    if collection.count() == 0:
        load_all_pdfs(collection, client)

    print("TOTAL DOCS:", collection.count())
except Exception:
    pass
