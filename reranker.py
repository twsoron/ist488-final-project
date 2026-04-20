"""ZeroEntropy reranker for the IST 387 chatbot RAG pipeline.

Sits between retrieval (BM25 + vector) and generation: takes a query plus the
candidate chunks pulled from the vector DB and reorders them by true relevance
so the generator sees the best context first.
"""

from __future__ import annotations

import os
from typing import Any, Iterable

import requests

ZEROENTROPY_RERANK_URL = "https://api.zeroentropy.dev/v1/models/rerank"
DEFAULT_MODEL = "zerank-1"
DEFAULT_LATENCY = "fast"
DEFAULT_TIMEOUT = 30


class RerankerError(RuntimeError):
    pass


def _resolve_api_key(api_key: str | None) -> str:
    if api_key:
        return api_key
    env_key = os.environ.get("ZEROENTROPY_API_KEY")
    if env_key:
        return env_key
    try:
        import streamlit as st  # type: ignore

        return st.secrets["ZEROENTROPY_API_KEY"]
    except Exception as exc:
        raise RerankerError(
            "No ZeroEntropy API key found. Pass api_key=..., set "
            "ZEROENTROPY_API_KEY env var, or add it to Streamlit secrets."
        ) from exc


def _normalize_documents(documents: Iterable[Any]) -> tuple[list[str], list[dict]]:
    """Split documents into (texts, metadata) so we can reattach metadata after reranking."""
    texts: list[str] = []
    meta: list[dict] = []
    for doc in documents:
        if isinstance(doc, str):
            texts.append(doc)
            meta.append({})
        elif isinstance(doc, dict):
            text = doc.get("text") or doc.get("content")
            if text is None:
                raise RerankerError(
                    f"Document dict missing 'text' or 'content' key: {doc!r}"
                )
            texts.append(text)
            meta.append({k: v for k, v in doc.items() if k not in ("text", "content")})
        else:
            raise RerankerError(f"Unsupported document type: {type(doc)}")
    return texts, meta


def rerank(
    query: str,
    documents: Iterable[Any],
    top_n: int | None = None,
    model: str = DEFAULT_MODEL,
    latency: str = DEFAULT_LATENCY,
    api_key: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[dict]:
    """Rerank retrieved chunks by relevance to the query.

    Args:
        query: The student query.
        documents: Either a list of strings, or dicts with a "text" (or "content")
            key. Any other dict keys (source, type, page, chunk_id, ...) are
            preserved on the returned items.
        top_n: Keep only the top N results. None keeps all.
        model: ZeroEntropy model id — one of zerank-1, zerank-1-small, zerank-2.
        latency: "fast" or "slow".
        api_key: Overrides env/secrets lookup.
        timeout: HTTP timeout in seconds.

    Returns:
        List of dicts sorted by relevance_score (desc):
            {"text": str, "relevance_score": float, "original_index": int, **metadata}
    """
    texts, meta = _normalize_documents(documents)

    if not texts:
        return []
    if len(texts) == 1:
        return [{"text": texts[0], "relevance_score": 1.0, "original_index": 0, **meta[0]}]

    payload: dict[str, Any] = {
        "model": model,
        "query": query,
        "documents": texts,
        "latency": latency,
    }
    if top_n is not None:
        payload["top_n"] = top_n

    headers = {
        "Authorization": f"Bearer {_resolve_api_key(api_key)}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            ZEROENTROPY_RERANK_URL, json=payload, headers=headers, timeout=timeout
        )
    except requests.RequestException as exc:
        raise RerankerError(f"ZeroEntropy request failed: {exc}") from exc

    if response.status_code != 200:
        raise RerankerError(
            f"ZeroEntropy returned {response.status_code}: {response.text}"
        )

    data = response.json()
    results = data.get("results", [])

    reranked: list[dict] = []
    for item in results:
        idx = item["index"]
        reranked.append(
            {
                "text": texts[idx],
                "relevance_score": item["relevance_score"],
                "original_index": idx,
                **meta[idx],
            }
        )
    return reranked


def rerank_safe(
    query: str,
    documents: Iterable[Any],
    top_n: int | None = None,
    **kwargs: Any,
) -> list[dict]:
    """Same as rerank() but never raises — falls back to input order on error.

    Use this in production paths where a reranker outage shouldn't drop the whole
    RAG response. Logs to stderr so failures are visible.
    """
    import sys

    docs_list = list(documents)
    try:
        return rerank(query, docs_list, top_n=top_n, **kwargs)
    except RerankerError as exc:
        print(f"[reranker] falling back to original order: {exc}", file=sys.stderr)
        texts, meta = _normalize_documents(docs_list)
        fallback = [
            {"text": t, "relevance_score": 0.0, "original_index": i, **m}
            for i, (t, m) in enumerate(zip(texts, meta))
        ]
        return fallback[:top_n] if top_n else fallback


if __name__ == "__main__":
    # Smoke test — run with: python reranker.py
    # Requires ZEROENTROPY_API_KEY in env.
    sample_query = "How do I use the apply function in R to get row means?"
    sample_docs = [
        {
            "text": "ggplot2 is used for data visualization. geom_point creates scatter plots.",
            "source": "Lecture_5_slides.pdf",
            "type": "concept",
        },
        {
            "text": "apply(mat, 1, mean) computes the mean across rows of a matrix. "
                    "The second argument (MARGIN) is 1 for rows, 2 for columns.",
            "source": "Functions_and_Subsetting.pdf",
            "type": "concept",
        },
        {
            "text": "Homework 3 is due Friday. Submit via Blackboard before 11:59pm.",
            "source": "IST387_Syllabus.pdf",
            "type": "syllabus",
        },
        {
            "text": "sapply and lapply iterate over lists. sapply simplifies the output "
                    "to a vector where possible.",
            "source": "Functions_and_Subsetting.pdf",
            "type": "concept",
        },
    ]

    ranked = rerank(sample_query, sample_docs, top_n=3)
    print(f"Query: {sample_query}\n")
    for i, r in enumerate(ranked, 1):
        print(f"{i}. [{r['relevance_score']:.4f}] ({r.get('source')}) {r['text'][:90]}...")
