"""BM25 + vector hybrid retrieval, fused with Reciprocal Rank Fusion.

Vector search is strong on semantic similarity ("scatter plot" ≈ "geom_point").
BM25 is strong on exact keywords ("HW 3", "apply", specific filenames).
Fusing both gives better recall than either alone.

Usage:
    index = HybridIndex(collection)
    results = index.hybrid_retrieve(query, query_embedding, top_k=10)
"""

from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class HybridIndex:
    """In-memory BM25 index backed by a ChromaDB collection.

    Rebuild by constructing a new instance; cheap for small collections.
    """

    def __init__(self, collection):
        data = collection.get(include=["documents", "metadatas"])
        self.ids: list[str] = data["ids"]
        self.docs: list[str] = data["documents"]
        self.metas: list[dict] = data["metadatas"]
        self.collection = collection
        # Tokenize doc + source filename so BM25 matches on "hw3" etc.
        tokenized = [
            tokenize(f"{m.get('source', '')} {d}") for d, m in zip(self.docs, self.metas)
        ]
        self.bm25 = BM25Okapi(tokenized)

    def _matches_filter(self, meta: dict, where: dict | None) -> bool:
        if not where:
            return True
        return all(meta.get(k) == v for k, v in where.items())

    def search_bm25(
        self, query: str, top_k: int, where: dict | None = None
    ) -> list[tuple[str, str, dict, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in ranked:
            if self._matches_filter(self.metas[i], where):
                out.append((self.ids[i], self.docs[i], self.metas[i], float(scores[i])))
                if len(out) >= top_k:
                    break
        return out

    def search_vector(
        self, query_embedding: list[float], top_k: int, where: dict | None = None
    ) -> list[tuple[str, str, dict, float]]:
        kwargs = {"query_embeddings": [query_embedding], "n_results": top_k}
        if where:
            kwargs["where"] = where
        r = self.collection.query(**kwargs)
        return list(zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0]))

    def hybrid_retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 10,
        fetch_k: int = 20,
        rrf_k: int = 60,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve fetch_k from each source, fuse with RRF, return top_k.

        RRF formula: score = sum over sources of 1 / (rrf_k + rank).
        k=60 is the canonical default (Cormack et al. 2009).
        """
        bm25 = self.search_bm25(query, fetch_k, where=where)
        vec = self.search_vector(query_embedding, fetch_k, where=where)

        scores: dict[str, float] = {}
        sources: dict[str, set[str]] = {}
        payload: dict[str, tuple[str, dict]] = {}

        for rank, (doc_id, doc, meta, _) in enumerate(bm25):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            sources.setdefault(doc_id, set()).add("bm25")
            payload[doc_id] = (doc, meta)

        for rank, (doc_id, doc, meta, _) in enumerate(vec):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            sources.setdefault(doc_id, set()).add("vector")
            payload[doc_id] = (doc, meta)

        ordered_ids = sorted(scores, key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {
                "id": doc_id,
                "text": payload[doc_id][0],
                "rrf_score": scores[doc_id],
                "sources": sorted(sources[doc_id]),
                **payload[doc_id][1],
            }
            for doc_id in ordered_ids
        ]
