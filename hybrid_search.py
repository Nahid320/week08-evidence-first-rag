"""Persistent Chroma semantic retrieval + in-memory BM25 + rank fusion.

Design choices for teaching:
* The PDF and incident notes are embedded once and stored persistently in Chroma.
* BM25 is rebuilt from Chroma's stored documents when a search process starts.
* The SAME Chroma IDs identify results in both retrieval lists.
* Reciprocal Rank Fusion combines RANKS, never raw cosine distances and BM25 scores.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "chroma_data"
COLLECTION_NAME = "week7_hybrid_documents"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def tokenize(text: str) -> list[str]:
    """Lowercase while keeping codes such as INV-7X4 or GW-12B intact."""
    return re.findall(r"[a-z0-9]+(?:[-_/][a-z0-9]+)*", text.casefold())


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
    top_k: int | None = None,
) -> list[dict]:
    """Rank fusion; results are dicts with id, text and metadata.

    A document present in both ranked lists gets two contributions.
    `k` here is a *smoothing constant*, not the number of returned results.
    """
    if k <= 0 or (top_k is not None and top_k < 1):
        raise ValueError("k and top_k must be positive")
    scores: dict[str, float] = defaultdict(float)
    records: dict[str, dict] = {}
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for i, ranked in enumerate(result_lists):
        seen_in_list: set[str] = set()
        for rank, hit in enumerate(ranked, start=1):
            doc_id = hit["id"]
            if doc_id in seen_in_list:
                continue
            seen_in_list.add(doc_id)
            scores[doc_id] += 1.0 / (k + rank)
            records.setdefault(doc_id, hit)
            ranks[doc_id][f"search_{i+1}"] = rank
    ids = sorted(scores, key=lambda x: (-scores[x], x))
    if top_k is not None:
        ids = ids[:top_k]
    return [
        {**records[doc_id], "rrf_score": scores[doc_id], "ranks": ranks[doc_id]}
        for doc_id in ids
    ]


def keyword_ranking(records: list[dict], query: str, limit: int) -> list[dict]:
    """BM25 on text stored in Chroma; returns only positive-score matches."""
    if limit < 1:
        raise ValueError("limit must be positive")
    if not records or not tokenize(query):
        return []
    from rank_bm25 import BM25Okapi
    tokenized = [tokenize(item["text"]) for item in records]
    if not any(tokenized):
        return []
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(tokenize(query))
    order = sorted(range(len(records)), key=lambda i: (-float(scores[i]), records[i]["id"]))
    return [
        {**records[i], "bm25_score": float(scores[i])}
        for i in order if float(scores[i]) > 0.0
    ][:limit]


def open_collection(db_path: Path = DB_PATH):
    import chromadb
    client = chromadb.PersistentClient(path=str(db_path))
    return client, client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )


def embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_ID)


def index_records(records: list[dict], db_path: Path = DB_PATH, overwrite: bool = False) -> int:
    import chromadb
    if not records:
        raise ValueError("Cannot index zero documents")
    client = chromadb.PersistentClient(path=str(db_path))
    if overwrite:
        try:
            client.delete_collection(name=COLLECTION_NAME)
        except Exception as exc:
            # Only ignore a genuinely absent collection, never storage/permission failures.
            if "does not exist" not in str(exc).lower() and "not found" not in str(exc).lower():
                raise
    _, collection = open_collection(db_path)
    if collection.count():
        raise ValueError("Collection already contains documents. Use index --overwrite to rebuild it.")
    model = embedding_model()
    embeddings = model.encode_document(
        [item["text"] for item in records], normalize_embeddings=True
    )
    collection.add(
        ids=[item["id"] for item in records],
        documents=[item["text"] for item in records],
        metadatas=[item["metadata"] for item in records],
        embeddings=embeddings.tolist(),
    )
    return collection.count()


class HybridSearcher:
    def __init__(self, db_path: Path = DB_PATH):
        _, self.collection = open_collection(db_path)
        if self.collection.count() == 0:
            raise ValueError("Database is empty. Run `python app.py index` first.")
        stored = self.collection.get(include=["documents", "metadatas"])
        self.records = [
            {"id": id_, "text": text, "metadata": metadata}
            for id_, text, metadata in zip(
                stored["ids"], stored["documents"], stored["metadatas"]
            )
        ]
        self._model = None  # Keyword-only search does not load Hugging Face models.

    def semantic(self, query: str, limit: int) -> list[dict]:
        if limit < 1:
            raise ValueError("limit must be positive")
        if self._model is None:
            self._model = embedding_model()
        vector = self._model.encode_query([query], normalize_embeddings=True)
        output = self.collection.query(
            query_embeddings=vector.tolist(),
            n_results=min(limit, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"id": id_, "text": text, "metadata": metadata,
             "distance": float(distance)}
            for id_, text, metadata, distance in zip(
                output["ids"][0], output["documents"][0],
                output["metadatas"][0], output["distances"][0]
            )
        ]

    def keyword(self, query: str, limit: int) -> list[dict]:
        return keyword_ranking(self.records, query, limit)

    def search(self, query: str, mode: str = "hybrid", top_k: int = 3,
               candidates: int | None = None) -> list[dict]:
        if not query.strip():
            raise ValueError("Query cannot be blank")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        limit = min(len(self.records), candidates or max(top_k * 4, 12))
        if mode == "semantic":
            return self.semantic(query, limit)[:top_k]
        if mode == "keyword":
            return self.keyword(query, limit)[:top_k]
        if mode == "hybrid":
            return reciprocal_rank_fusion(
                [self.semantic(query, limit), self.keyword(query, limit)],
                k=60, top_k=top_k,
            )
        raise ValueError(f"Unknown mode: {mode}")
