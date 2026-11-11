"""Hybrid retrieval: BM25 and optional dense vectors, merged by rank, optionally reranked."""
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from .text import tokens

RRF_K = 60
CANDIDATES = 30


class Embedder(Protocol):
    def __call__(self, texts: list[str]) -> np.ndarray: ...


Reranker = Callable[[str, list[str]], list[float]]


@dataclass
class Hit:
    chunk: dict
    score: float
    rerank: float | None = None


class BM25:
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = [Counter(d) for d in documents]
        self.lengths = [len(d) for d in documents]
        self.mean = sum(self.lengths) / max(len(documents), 1) or 1.0
        self.k1, self.b = k1, b
        df = Counter(t for d in self.docs for t in d)
        n = len(documents)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for doc, length in zip(self.docs, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / self.mean)
            out.append(sum(self.idf.get(t, 0.0) * doc[t] * (self.k1 + 1) / (doc[t] + norm) for t in query if t in doc))
        return out


def _ranked(scores, limit: int) -> list[int]:
    order = np.argsort(-np.asarray(scores), kind="stable")[:limit]
    return [int(i) for i in order if scores[i] > 0]


class Index:
    def __init__(self, chunks: list[dict], vectors: np.ndarray | None = None,
                 embed: Embedder | None = None, rerank: Reranker | None = None):
        self.chunks, self.vectors, self.embed, self.rerank = chunks, vectors, embed, rerank
        self.bm25 = BM25([tokens(self._searchable(c)) for c in chunks])

    @staticmethod
    def _searchable(chunk: dict) -> str:
        return f"{chunk['doc'].rsplit('.', 1)[0]} {chunk['text']}"

    @classmethod
    def build(cls, chunks: list[dict], embed: Embedder | None = None, rerank: Reranker | None = None) -> "Index":
        vectors = embed([cls._searchable(c) for c in chunks]) if embed and chunks else None
        return cls(chunks, vectors, embed, rerank)

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "chunks.json").write_text(json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8")
        if self.vectors is not None:
            np.save(folder / "vectors.npy", self.vectors)

    @classmethod
    def load(cls, folder: Path, embed: Embedder | None = None, rerank: Reranker | None = None) -> "Index":
        chunks = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        vectors = np.load(folder / "vectors.npy") if (folder / "vectors.npy").exists() else None
        return cls(chunks, vectors, embed if vectors is not None else None, rerank)

    def search(self, query: str, k: int = 8) -> list[Hit]:
        if not self.chunks:
            return []
        rankings = [_ranked(self.bm25.scores(tokens(query)), CANDIDATES)]
        if self.embed is not None:
            q = self.embed([query])[0]
            rankings.append(_ranked(self.vectors @ q, CANDIDATES))
        merged: dict[int, float] = {}
        for ranking in rankings:
            for rank, i in enumerate(ranking):
                merged[i] = merged.get(i, 0.0) + 1 / (RRF_K + rank + 1)
        hits = [Hit(self.chunks[i], s) for i, s in sorted(merged.items(), key=lambda x: -x[1])[:CANDIDATES]]
        if self.rerank is not None and hits:
            for hit, s in zip(hits, self.rerank(query, [h.chunk["text"] for h in hits])):
                hit.rerank = s
            hits.sort(key=lambda h: -h.rerank)
        return hits[:k]
