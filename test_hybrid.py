"""Test hybrid (BM25 + vector) retrieval -> ZeroEntropy rerank.

Run with:
    OPENAI_API_KEY=... ZEROENTROPY_API_KEY=... python test_hybrid.py
"""

import os
import sys

import chromadb
from openai import OpenAI

from hybrid_search import HybridIndex
from reranker import rerank

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QUERIES = [
    "How do I use the apply function in R to get row means?",
    "When is homework 3 due?",
    "How do I make a scatter plot with ggplot2?",
    "When is HW 3 assigned?",
    "geom_boxplot syntax",
]

FETCH_K = 20
TOP_K_FUSED = 10
TOP_N_RERANK = 5


def embed(client: OpenAI, text: str) -> list[float]:
    resp = client.embeddings.create(input=text, model="text-embedding-3-small")
    return resp.data[0].embedding


def main() -> None:
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    chroma_client = chromadb.PersistentClient(path="./ChromaDB")
    collection = chroma_client.get_or_create_collection(name="CourseCollection")

    print(f"Building BM25 index over {collection.count()} chunks...")
    index = HybridIndex(collection)
    print("Done.\n")

    for query in QUERIES:
        print("=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        q_emb = embed(openai_client, query)
        fused = index.hybrid_retrieve(
            query, q_emb, top_k=TOP_K_FUSED, fetch_k=FETCH_K
        )

        print(f"\n--- HYBRID FUSED (top {TOP_K_FUSED}) ---")
        for i, r in enumerate(fused, 1):
            src_tag = "+".join(r["sources"])
            print(
                f"{i}. [rrf={r['rrf_score']:.4f}] [{src_tag:12}] ({r.get('source')}) "
                f"{r['text'][:90].strip()}..."
            )

        reranked = rerank(query, fused, top_n=TOP_N_RERANK)

        print(f"\n--- RERANKED (top {TOP_N_RERANK}) ---")
        for i, r in enumerate(reranked, 1):
            orig_rank = r["original_index"] + 1
            move = "" if orig_rank == i else f"  (was #{orig_rank})"
            print(f"{i}. [score={r['relevance_score']:.4f}] ({r.get('source')}){move}")
            print(f"   {r['text'][:140].strip()}...")
        print()


if __name__ == "__main__":
    try:
        main()
    except KeyError as e:
        print(f"Missing env var: {e}", file=sys.stderr)
        sys.exit(1)
