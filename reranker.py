"""Second-stage relevance scoring of first-stage hybrid retrieval candidates.

No heavyweight dependencies are imported when this module is imported: tests can
run offline without downloading the CrossEncoder model.
"""
from __future__ import annotations

import math
from typing import Any

RERANK_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"


def load_cross_encoder(model_id: str = RERANK_MODEL_ID):
    """Download (on first use) and load an MS MARCO CrossEncoder."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_id)


def rerank_candidates(
    question: str,
    candidates: list[dict[str, Any]],
    cross_encoder: Any,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Score (question, passage) pairs and sort by descending model score.

    Returns results with original ID, source metadata and first-stage RRF score.
    Scores may be negative logits; do not interpret them as probabilities.
    """
    if not question.strip():
        raise ValueError("Question must not be blank")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if not candidates:
        return []
    ids = [hit["id"] for hit in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("Candidate IDs must be unique")

    pairs = [
        (question, (str(hit.get("metadata", {}).get("section", "")) + "\n" + hit["text"]).strip())
        for hit in candidates
    ]
    scores = cross_encoder.predict(pairs, batch_size=16, show_progress_bar=False)
    if len(scores) != len(candidates):
        raise ValueError("Cross-Encoder returned an unexpected number of scores")

    ranked = []
    for previous_rank, (hit, score) in enumerate(zip(candidates, scores), 1):
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("Cross-Encoder returned a non-finite score")
        ranked.append({
            **hit,
            "cross_score": value,
            "candidate_rank": previous_rank,
        })
    # Stable tie-break: preserve the first-stage rank for equal scores.
    ranked.sort(key=lambda item: (-item["cross_score"], item["candidate_rank"]))
    return ranked[:top_k]


def retrieve_and_rerank(
    searcher: Any,
    question: str,
    cross_encoder: Any,
    top_k: int = 3,
    candidate_count: int = 12,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (hybrid candidate list, final reranked passages)."""
    if top_k < 1 or candidate_count < top_k:
        raise ValueError("Require candidate_count >= top_k >= 1")
    candidates = searcher.search(
        question,
        mode="hybrid",
        top_k=candidate_count,
        candidates=candidate_count,
    )
    return candidates, rerank_candidates(question, candidates, cross_encoder, top_k)
