"""Reproducible relevance evaluation for Week 7's Advanced RAG pipeline.

Offline metrics use human-provided target passage IDs in the bundled synthetic
corpus. A correct citation identifier or quotation is NOT proof of entailment.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = ROOT / "data" / "evaluation_questions.json"
METHODS = ("semantic", "bm25", "hybrid", "reranked")
METRICS = ("precision", "recall", "hit", "mrr")


def load_cases(path: Path = DEFAULT_CASES, *, known_ids: set[str] | None = None) -> list[dict]:
    """Load labeled questions and catch stale/broken annotations explicitly."""
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation dataset must be a nonempty JSON list")
    seen = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str):
            raise ValueError("Each case needs a string ID")
        case_id = case["id"]
        if not case_id.strip() or case_id in seen:
            raise ValueError(f"Blank or duplicate case ID: {case_id!r}")
        seen.add(case_id)
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError(f"Case {case_id} has no question")
        relevant = case.get("relevant_ids")
        if not isinstance(relevant, list) or any(not isinstance(x, str) for x in relevant):
            raise ValueError(f"Case {case_id} needs a list of relevant_ids")
        if len(relevant) != len(set(relevant)):
            raise ValueError(f"Duplicate relevant IDs in {case_id}")
        if not isinstance(case.get("answerable"), bool):
            raise ValueError(f"Case {case_id} needs boolean answerable")
        if case["answerable"] != bool(relevant):
            raise ValueError(f"Case {case_id}: answerable must agree with relevant_ids")
        if known_ids is not None:
            missing = set(relevant) - known_ids
            if missing:
                raise ValueError(
                    f"Case {case_id} refers to unknown IDs: {sorted(missing)}. "
                    "Index/corpus version may have changed; review labels."
                )
    return cases


def measure(ranked_ids: list[str], relevant_ids: list[str], k: int) -> dict[str, float]:
    """Precision, Recall, Hit, Reciprocal Rank at k for one answerable query.

    The precision denominator is k, consistently across queries. Recall and
    reciprocal rank require at least one manually labeled relevant passage.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if not relevant_ids:
        raise ValueError("Unanswerable questions have no recall denominator")
    if len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("Retrieval results contain duplicate IDs")
    gold = set(relevant_ids)
    top = ranked_ids[:k]
    found = sum(doc_id in gold for doc_id in top)
    first = next((rank for rank, doc_id in enumerate(top, 1) if doc_id in gold), None)
    return {
        "precision": found / k,
        "recall": found / len(gold),
        "hit": float(first is not None),
        "mrr": 1.0 / first if first is not None else 0.0,
    }


def evaluate_retrieval(
    searcher: Any,
    cross_encoder: Any,
    cases: list[dict],
    *,
    top_k: int = 3,
    candidates: int = 12,
) -> dict:
    """Compare the SAME labeled questions using four retrieval pipelines."""
    if not 1 <= top_k <= candidates:
        raise ValueError("Require candidates >= top_k >= 1")
    from reranker import rerank_candidates

    known = {record["id"] for record in searcher.records}
    for case in cases:
        if set(case["relevant_ids"]) - known:
            raise ValueError(f"Gold IDs not present in the indexed corpus: {case['id']}")

    totals = {name: defaultdict(float) for name in METHODS}
    per_query = []
    answerable = [case for case in cases if case["answerable"]]
    if not answerable:
        raise ValueError("At least one answerable case is required for retrieval metrics")

    for case in answerable:
        question = case["question"]
        semantic = searcher.search(question, mode="semantic", top_k=top_k, candidates=candidates)
        bm25 = searcher.search(question, mode="keyword", top_k=top_k, candidates=candidates)
        hybrid_candidates = searcher.search(question, mode="hybrid", top_k=candidates, candidates=candidates)
        reranked = rerank_candidates(question, hybrid_candidates, cross_encoder, top_k=top_k)
        rankings = {
            "semantic": semantic,
            "bm25": bm25,
            "hybrid": hybrid_candidates[:top_k],
            "reranked": reranked,
        }
        results = {}
        for method, hits in rankings.items():
            ids = [hit["id"] for hit in hits]
            scores = measure(ids, case["relevant_ids"], top_k)
            for metric, value in scores.items():
                totals[method][metric] += value
            results[method] = {"ids": ids, "metrics": scores}

        candidate_ids = {hit["id"] for hit in hybrid_candidates}
        per_query.append({
            "id": case["id"],
            "question": question,
            "relevant_ids": case["relevant_ids"],
            "hybrid_candidate_recall": len(candidate_ids.intersection(case["relevant_ids"])) / len(case["relevant_ids"]),
            "results": results,
        })
    n = len(answerable)
    summary = {
        name: {m: totals[name][m] / n for m in METRICS}
        for name in METHODS
    }
    return {
        "settings": {"top_k": top_k, "candidates": candidates},
        "query_counts": {"answerable": n, "unanswerable_excluded": len(cases) - n},
        "summary": summary,
        "reranking_minus_hybrid": {m: summary["reranked"][m] - summary["hybrid"][m] for m in METRICS},
        "average_hybrid_candidate_recall": sum(q["hybrid_candidate_recall"] for q in per_query) / n,
        "per_query": per_query,
        "limitations": (
            "Synthetic corpus, manually labeled single-passage targets; results are not "
            "a production benchmark. An apparently wrong answer may still cite a valid quote."
        ),
    }


def answer_status(text: str) -> str:
    """Parse the explicit status emitted by ask_llm(), not the answer content."""
    first = text.splitlines()[0].split()[0] if text.strip() else ""
    if first in ("ABSTAINED", "VERIFIED_QUOTES_ONLY"):
        return first
    if first.startswith("VERIFICATION_FAILED"):
        return "VERIFICATION_FAILED"
    return "UNKNOWN"


def evaluate_answers(
    searcher: Any,
    cross_encoder: Any,
    cases: list[dict],
    generate: Callable[[str, list[dict]], str],
    *,
    max_cases: int = 4,
    top_k: int = 3,
    candidates: int = 12,
) -> dict:
    """Optional live LLM behavior check (requires separate user opt-in).

    Tests unanswerable cases FIRST, even with a small max_cases. These are
    behavioral checks, NOT verification of logical claim-to-quote entailment.
    """
    if max_cases < 1:
        raise ValueError("max_cases must be >= 1")
    from reranker import retrieve_and_rerank

    ordered = sorted(cases, key=lambda c: c["answerable"])
    selected = ordered[:max_cases]
    details = []
    for case in selected:
        _, hits = retrieve_and_rerank(
            searcher, case["question"], cross_encoder,
            top_k=top_k, candidate_count=candidates,
        )
        answer = generate(case["question"], hits)
        status = answer_status(answer)
        expected = "VERIFIED_QUOTES_ONLY" if case["answerable"] else "ABSTAINED"
        details.append({
            "id": case["id"],
            "question": case["question"],
            "expected_answerable": case["answerable"],
            "status": status,
            "expected_status": expected,
            "status_matches_expectation": status == expected,
            "answer": answer,
            "human_evidence_review": None,
        })
    return {
        "cases_run": len(details),
        "statuses_matching_expectation": sum(x["status_matches_expectation"] for x in details),
        "cases": details,
        "warning": "Status agreement and quote validity are not proof of factual correctness or entailment.",
    }


def write_report(report: dict, output: Path) -> tuple[Path, Path]:
    """Save machine-readable report plus per-query rankings and metrics."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    csv_path = output.with_name(output.stem + "_per_query.csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question_id", "question", "method", "retrieved_ids", "gold_ids", *METRICS])
        writer.writeheader()
        for item in report["per_query"]:
            for name in METHODS:
                result = item["results"][name]
                writer.writerow({
                    "question_id": item["id"],
                    "question": item["question"],
                    "method": name,
                    "retrieved_ids": ";".join(result["ids"]),
                    "gold_ids": ";".join(item["relevant_ids"]),
                    **result["metrics"],
                })
    return output, csv_path
