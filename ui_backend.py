"""Streamlit-independent helpers for a local/session-isolated PDF RAG interface.

No heavy models are imported when the module loads.
User-uploaded PDFs are treated as untrusted, non-sensitive demo input.
"""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re

from corpus import build_records
from chunkers import build_chunks
from reliability import answer_safely
from reranker import rerank_candidates

MAX_FILES = 3
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_TOTAL_PAGES = 50
MAX_QUESTION_CHARS = 1200


def safe_filename(name: str) -> str:
    """Convert untrusted upload names into a short, displayable filename."""
    leaf = str(name).replace("\\", "/").split("/")[-1]
    if not leaf.lower().endswith(".pdf"):
        raise ValueError("Only .pdf documents are supported")
    stem = re.sub(r"[^a-zA-Z0-9_.-]+", "_", leaf[:-4]).strip("._-")[:70]
    return (stem or "document") + ".pdf"


def upload_signature(uploads: list[tuple[str, bytes]]) -> str:
    """Stable digest to detect uploads changed since indexing, without storing bytes in state."""
    h = sha256(b"week08-pdf-corpus")
    for name, content in sorted(uploads, key=lambda x: (x[0], sha256(x[1]).hexdigest())):
        h.update(safe_filename(name).encode("utf-8"))
        h.update(sha256(content).digest())
    return h.hexdigest()


def build_upload_records(
    uploads: list[tuple[str, bytes]], workspace: Path,
    max_files: int = MAX_FILES, max_file_bytes: int = MAX_FILE_BYTES,
    max_total_pages: int = MAX_TOTAL_PAGES,
) -> list[dict]:
    """Validate, extract and chunk uploaded PDFs without mixing them with sample corpus.

    Files are saved under an isolated temporary workspace, never a shared app directory.
    No OCR, no antivirus scan, and no full layout parsing are provided here.
    """
    from pypdf import PdfReader

    if not uploads:
        raise ValueError("Upload at least one PDF")
    if len(uploads) > max_files:
        raise ValueError(f"Limit: {max_files} PDF files")
    workspace = Path(workspace)
    outdir = workspace / "uploads"
    outdir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    seen: set[tuple[str, str]] = set()
    total_pages = 0
    for name, content in uploads:
        safe = safe_filename(name)
        if not isinstance(content, bytes) or not content:
            raise ValueError(f"Empty or invalid PDF: {safe}")
        if len(content) > max_file_bytes:
            raise ValueError(f"{safe} exceeds the {max_file_bytes // (1024*1024)} MB limit")
        if not content.lstrip().startswith(b"%PDF-"):
            raise ValueError(f"{safe} is not a PDF")
        digest = sha256(content).hexdigest()
        if (safe, digest) in seen:
            continue
        seen.add((safe, digest))
        try:
            reader = PdfReader(BytesIO(content), strict=False)
            if reader.is_encrypted:
                raise ValueError(f"{safe} is encrypted; use an unencrypted PDF")
            total_pages += len(reader.pages)
            if total_pages > max_total_pages:
                raise ValueError(f"The total PDF page limit is {max_total_pages}")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Could not read PDF: {safe}") from exc

        path = outdir / f"{digest[:16]}_{safe}"
        path.write_bytes(content)
        try:
            chunks = build_chunks(path, "structured", max_words=120, overlap=25)
        except Exception as exc:
            raise ValueError(f"Could not extract PDF text from {safe}") from exc
        finally:
            # We persist extracted chunks in Chroma, not the original uploaded PDF.
            # Note: Chroma itself retains the chunk text for this session.
            path.unlink(missing_ok=True)
        for chunk in chunks:
            # Chunk IDs are based on file bytes and chunk content, not solely the filename.
            record_id = "PDF-" + sha256(f"{digest}:{chunk.id}".encode()).hexdigest()[:24]
            results.append({
                "id": record_id,
                "text": chunk.text,
                "metadata": {
                    "source": safe,
                    "page": int(chunk.page),
                    "section": str(chunk.section),
                    "kind": "pdf",
                },
            })

    if not results:
        raise ValueError("No selectable text found. Scanned PDFs need OCR (not included).")
    if len(results) > 1500:
        raise ValueError("Too many chunks for this demo (maximum 1500).")
    ids = [r["id"] for r in results]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs; try fewer or distinct PDFs")
    return results


def sample_records() -> list[dict]:
    return build_records()


def retrieve(question: str, searcher, reranker, top_k: int = 3,
             candidates: int = 12) -> dict:
    """Run hybrid retrieval + rerank using already loaded models."""
    if not question.strip() or len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"Enter a question (maximum {MAX_QUESTION_CHARS} characters)")
    if top_k < 1 or candidates < top_k:
        raise ValueError("Require candidates >= top_k >= 1")
    n = len(searcher.records)
    if n == 0:
        return {"candidates": [], "hits": []}
    limit = min(candidates, n)
    all_hits = searcher.search(question, mode="hybrid", top_k=limit, candidates=limit)
    hits = rerank_candidates(question, all_hits, reranker, top_k=min(top_k, len(all_hits)))
    return {"candidates": all_hits, "hits": hits}


def answer(question: str, hits: list[dict], generator) -> dict:
    return answer_safely(question, hits, generator)
