"""Prepare chunks from the Week 7 Session 1 PDF plus fictional maintenance logs."""
from __future__ import annotations

import json
from pathlib import Path

from chunkers import build_chunks

ROOT = Path(__file__).resolve().parent
DEFAULT_PDF = ROOT / "sample_pdfs" / "industrial_energy_report.pdf"
DEFAULT_LOGS = ROOT / "data" / "maintenance_incidents.json"


def build_records(pdf_path: Path = DEFAULT_PDF, log_path: Path = DEFAULT_LOGS) -> list[dict]:
    chunks = build_chunks(pdf_path, "structured", max_words=120, overlap=25)
    result = [
        {
            "id": "PDF-" + ch.id,
            "text": ch.text,
            "metadata": {
                "source": ch.source, "page": ch.page,
                "section": ch.section, "kind": "pdf"
            },
        }
        for ch in chunks
    ]
    incidents = json.loads(Path(log_path).read_text(encoding="utf-8"))
    for incident in incidents:
        result.append({
            "id": incident["id"],
            "text": incident["text"],
            "metadata": {
                "source": Path(log_path).name, "page": -1,
                "section": incident["section"], "kind": "incident"
            },
        })
    ids = [item["id"] for item in result]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate document IDs; check corpus inputs.")
    if not result:
        raise ValueError("Cannot index an empty corpus.")
    return result
