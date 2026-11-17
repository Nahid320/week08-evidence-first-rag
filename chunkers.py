"""PDF extraction and two transparent, testable chunking strategies."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Iterable


@dataclass(frozen=True)
class Block:
    source: str
    page: int
    section: str
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    page: int
    section: str
    mode: str
    text: str


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def word_count(text: str) -> int:
    return len(text.split())


def is_heading(text: str) -> bool:
    """Conservative heuristic; PDF often doesn't encode semantic headings."""
    text = normalize(text)
    if not text or word_count(text) > 12 or text.endswith(('.', '!', '?', ':')):
        return False
    if re.match(r"^\d+(?:\.\d+)*\s+[A-Z]", text):
        return True
    letters = [ch for ch in text if ch.isalpha()]
    return len(letters) >= 4 and text.upper() == text


def extract_blocks(path: str | Path) -> list[Block]:
    """Preserve page numbers, detect blank-line separated blocks and headings.

    This is NOT a general layout parser and does not perform OCR.
    """
    from pypdf import PdfReader

    pdf = Path(path)
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf}")

    reader = PdfReader(str(pdf))
    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported by this example.")

    section = "Untitled section"
    blocks: list[Block] = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text(extraction_mode="layout") or ""
        # Only fall back if layout extraction was empty.
        if not text.strip():
            text = page.extract_text() or ""
        for part in re.split(r"\n\s*\n+", text):
            clean = normalize(part)
            if not clean:
                continue
            # Ignore common standalone page-number footers.
            if re.fullmatch(r"(?:Page\s+)?\d+", clean, re.IGNORECASE):
                continue
            if is_heading(clean):
                section = clean
                continue
            blocks.append(Block(pdf.name, page_num, section, clean))
    return blocks


def _make_chunk(source: str, page: int, section: str, mode: str, body: str) -> Chunk:
    # Include section name in embedding input. Body is still identifiable by metadata.
    text = f"Section: {section}\n{body}" if mode == "structured" else body
    key = f"{source}|{page}|{section}|{mode}|{text}"
    ident = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    return Chunk(id=ident, source=source, page=page, section=section, mode=mode, text=text)


def fixed_word_chunks(blocks: list[Block], max_words: int = 120, overlap: int = 25) -> list[Chunk]:
    """Baseline: fixed-size overlapping word windows *within each page*."""
    if max_words < 1 or not (0 <= overlap < max_words):
        raise ValueError("Require max_words > 0 and 0 <= overlap < max_words")
    grouped: dict[tuple[str, int], list[str]] = {}
    for block in blocks:
        grouped.setdefault((block.source, block.page), []).append(block.text)
    result: list[Chunk] = []
    for (source, page), paragraphs in grouped.items():
        words = " ".join(paragraphs).split()
        stride = max_words - overlap
        for pos in range(0, len(words), stride):
            window = words[pos:pos + max_words]
            if not window:
                break
            result.append(_make_chunk(source, page, "Fixed-size window", "fixed", " ".join(window)))
            if pos + max_words >= len(words):
                break
    return result


def split_long_paragraph(text: str, max_words: int) -> list[str]:
    """Prefer sentence boundaries; fall back to word boundaries for long sentences."""
    if max_words < 1:
        raise ValueError("max_words must be positive")
    text = normalize(text)
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    pieces: list[str] = []
    current: list[str] = []
    current_n = 0
    for sentence in sentences:
        tokens = sentence.split()
        if current and current_n + len(tokens) > max_words:
            pieces.append(" ".join(current))
            current, current_n = [], 0
        while len(tokens) > max_words:
            pieces.append(" ".join(tokens[:max_words]))
            tokens = tokens[max_words:]
        if tokens:
            current.append(" ".join(tokens))
            current_n += len(tokens)
    if current:
        pieces.append(" ".join(current))
    return pieces


def structured_chunks(blocks: list[Block], max_words: int = 120) -> list[Chunk]:
    """Keep paragraphs together when possible, and do not cross sections/pages.

    Long paragraphs are split at sentence boundaries when possible. No overlap is
    added here, making the comparison with fixed overlapping chunks explicit.
    """
    if max_words < 1:
        raise ValueError("max_words must be positive")
    out: list[Chunk] = []
    current_key: tuple[str, int, str] | None = None
    current: list[str] = []
    n_words = 0

    def flush() -> None:
        nonlocal current, n_words
        if current_key is not None and current:
            source, page, section = current_key
            out.append(_make_chunk(source, page, section, "structured", "\n\n".join(current)))
        current, n_words = [], 0

    for block in blocks:
        key = (block.source, block.page, block.section)
        if key != current_key:
            flush()
            current_key = key
        for piece in split_long_paragraph(block.text, max_words):
            n = word_count(piece)
            if current and n_words + n > max_words:
                flush()
            current.append(piece)
            n_words += n
    flush()
    return out


def build_chunks(pdf: str | Path, mode: str, max_words: int = 120, overlap: int = 25) -> list[Chunk]:
    blocks = extract_blocks(pdf)
    if mode == "fixed":
        return fixed_word_chunks(blocks, max_words=max_words, overlap=overlap)
    if mode == "structured":
        return structured_chunks(blocks, max_words=max_words)
    raise ValueError(f"Unsupported mode: {mode}")
