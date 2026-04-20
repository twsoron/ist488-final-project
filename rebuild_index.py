"""Rebuild ChromaDB from PDFs using RAG.py's pure functions.

Run with:
    OPENAI_API_KEY=... python rebuild_index.py
"""

import os
import sys

import chromadb
from openai import OpenAI

from RAG import load_all_pdfs


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY env var.", file=sys.stderr)
        sys.exit(1)

    chroma_client = chromadb.PersistentClient(path="./ChromaDB")

    try:
        chroma_client.delete_collection("CourseCollection")
        print("Deleted existing collection.")
    except Exception as exc:
        print(f"(No existing collection to delete: {exc})")

    collection = chroma_client.create_collection(name="CourseCollection")
    client = OpenAI(api_key=api_key)

    load_all_pdfs(collection, client)
    print(f"\nTOTAL DOCS: {collection.count()}")


if __name__ == "__main__":
    main()
