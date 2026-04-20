"""Integration test: ChromaDB retrieval -> ZeroEntropy rerank.

Run with:
    OPENAI_API_KEY=... ZEROENTROPY_API_KEY=... python test_reranker.py
"""

import os
import sys

import chromadb
from openai import OpenAI

from reranker import rerank

QUERIES = [
    "How do I use the apply function in R to get row means?",
    "When is homework 3 due?",
    "How do I make a scatter plot with ggplot2?",
]

TOP_K_RETRIEVE = 10
TOP_N_RERANK = 5


def embed(client: OpenAI, text: str) -> list[float]:
    resp = client.embeddings.create(input=text, model="text-embedding-3-small")
    return resp.data[0].embedding


def main() -> None:
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    chroma_client = chromadb.PersistentClient(path="./ChromaDB")
    collection = chroma_client.get_or_create_collection(name="CourseCollection")

    print(f"Collection size: {collection.count()} chunks\n")

    for query in QUERIES:
        print("=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        query_emb = embed(openai_client, query)
        results = collection.query(query_embeddings=[query_emb], n_results=TOP_K_RETRIEVE)

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        print(f"\n--- VECTOR RETRIEVAL (top {TOP_K_RETRIEVE}) ---")
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
            print(f"{i}. [dist={dist:.3f}] ({meta.get('source')}) {doc[:100].strip()}...")

        candidates = [{"text": d, **m} for d, m in zip(docs, metas)]
        reranked = rerank(query, candidates, top_n=TOP_N_RERANK)

        print(f"\n--- RERANKED (top {TOP_N_RERANK}) ---")
        for i, r in enumerate(reranked, 1):
            moved = r["original_index"] + 1
            arrow = "" if moved == i else f"  (was #{moved})"
            print(f"{i}. [score={r['relevance_score']:.4f}] ({r.get('source')}){arrow}")
            print(f"   {r['text'][:140].strip()}...")
        print()


if __name__ == "__main__":
    try:
        main()
    except KeyError as e:
        print(f"Missing env var: {e}", file=sys.stderr)
        sys.exit(1)
