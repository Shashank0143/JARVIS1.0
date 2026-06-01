"""Simple local RAG index with Vector space models and Fuzzy token search."""

from __future__ import annotations

import json
import math
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .text import chunk_text, tokenize


@dataclass
class DocumentChunk:
    id: str
    source: str
    title: str
    text: str


class RagStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.documents: list[DocumentChunk] = []
        self._vectors: list[dict[str, float]] = []  # Normalized TF-IDF vector embeddings
        self._doc_freq: Counter[str] = Counter()

    def add_text(self, text: str, *, source: str, title: str = "") -> int:
        added = 0
        for chunk in chunk_text(text):
            doc = DocumentChunk(
                id=str(uuid.uuid4()),
                source=source,
                title=title or source,
                text=chunk,
            )
            self.documents.append(doc)
            added += 1
        self._rebuild()
        return added

    def search(self, query: str, *, limit: int = 5) -> list[tuple[float, DocumentChunk]]:
        if not self.documents:
            return []

        # Tokenize query and build TF-IDF vector for query
        query_raw_counts = Counter(tokenize(query))
        if not query_raw_counts:
            return []

        total_docs = len(self.documents)
        query_vector: dict[str, float] = {}
        for token, count in query_raw_counts.items():
            idf = math.log((1 + total_docs) / (1 + self._doc_freq[token])) + 1.0
            query_vector[token] = count * idf

        # Normalize query vector
        query_norm = math.sqrt(sum(v * v for v in query_vector.values())) or 1.0
        for token in query_vector:
            query_vector[token] /= query_norm

        scores: list[tuple[float, DocumentChunk]] = []
        for doc, doc_vec in zip(self.documents, self._vectors):
            score = 0.0
            # Perform hybrid vector matching with fuzzy fallback
            for q_token, q_weight in query_vector.items():
                if q_token in doc_vec:
                    # Exact Match / Dense Cosine similarity component
                    score += q_weight * doc_vec[q_token]
                else:
                    # Fuzzy Match component using Jaccard trigram similarity
                    best_match_token = None
                    best_sim = 0.0
                    for d_token in doc_vec:
                        sim = self._fuzzy_match_score(q_token, d_token)
                        if sim > best_sim:
                            best_sim = sim
                            best_match_token = d_token
                    
                    if best_sim >= 0.55 and best_match_token is not None:
                        # Add fuzzy matching score with a slight discount penalty (0.8)
                        score += q_weight * doc_vec[best_match_token] * best_sim * 0.8

            if score > 0.0:
                scores.append((score, doc))

        scores.sort(key=lambda item: item[0], reverse=True)
        return scores[:limit]

    @staticmethod
    def _trigrams(word: str) -> set[str]:
        padded = f"^{word}$"
        return {padded[i : i + 3] for i in range(len(padded) - 2)}

    @staticmethod
    def _fuzzy_match_score(word1: str, word2: str) -> float:
        # Ignore extremely short words for fuzzy matching to avoid false positives
        if len(word1) < 3 or len(word2) < 3:
            return 0.0
        if abs(len(word1) - len(word2)) > 3:
            return 0.0
        tg1 = RagStore._trigrams(word1)
        tg2 = RagStore._trigrams(word2)
        intersection = len(tg1.intersection(tg2))
        union = len(tg1.union(tg2))
        if union == 0:
            return 0.0
        jaccard = intersection / union
        return jaccard

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(document) for document in self.documents]
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RagStore":
        store = cls(path)
        if path.exists():
            rows = json.loads(path.read_text(encoding="utf-8"))
            store.documents = [DocumentChunk(**row) for row in rows]
            store._rebuild()
        return store

    def _rebuild(self) -> None:
        self._doc_freq = Counter()
        raw_vectors = []
        for document in self.documents:
            vec = Counter(tokenize(document.text))
            raw_vectors.append(vec)
            self._doc_freq.update(vec.keys())

        total_docs = len(self.documents)
        self._vectors = []
        for vec in raw_vectors:
            tf_idf_vec = {}
            for token, tf in vec.items():
                idf = math.log((1 + total_docs) / (1 + self._doc_freq[token])) + 1.0
                tf_idf_vec[token] = tf * idf
            
            # Normalize vector for exact cosine similarity
            norm = math.sqrt(sum(v * v for v in tf_idf_vec.values())) or 1.0
            normalized_vec = {token: val / norm for token, val in tf_idf_vec.items()}
            self._vectors.append(normalized_vec)
