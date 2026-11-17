"""Lightweight quote and source-ID checks for optional LLM-generated answers.

These checks confirm literal evidence existence, NOT logical entailment.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field


class SupportedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    statement: str
    source_id: str
    quote: str


class GroundedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answerable: bool
    claims: list[SupportedClaim] = Field(default_factory=list)


def canonical_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def verify_claims(result: GroundedResponse, sources: dict[str, dict]) -> tuple[bool, str]:
    """Reject unsupported source IDs and fabricated quotations.

    This intentionally does not assert that quotes imply their claim statements.
    """
    if not result.answerable:
        return False, "ABSTAINED"
    if not result.claims:
        return False, "VERIFICATION_FAILED: no sourced claims"
    for claim in result.claims:
        if not claim.statement.strip() or not claim.quote.strip():
            return False, "VERIFICATION_FAILED: empty statement or quote"
        if claim.source_id not in sources:
            return False, f"VERIFICATION_FAILED: unknown source {claim.source_id}"
        if canonical_text(claim.quote) not in canonical_text(sources[claim.source_id]["text"]):
            return False, f"VERIFICATION_FAILED: quote not found in {claim.source_id}"
    return True, "VERIFIED_QUOTES_ONLY"


def source_context(passages: list[dict]) -> tuple[str, dict[str, dict]]:
    """Make [S1], [S2] references only from the FINAL reranked passages."""
    sources = {}
    blocks = []
    for position, hit in enumerate(passages, 1):
        source_id = f"S{position}"
        sources[source_id] = hit
        meta = hit["metadata"]
        loc = f"page {meta['page']}" if meta["page"] > 0 else "incident note"
        blocks.append(
            f"[{source_id}] {meta['source']} ({loc}; {meta['section']})\n{hit['text']}"
        )
    return "\n\n".join(blocks), sources
