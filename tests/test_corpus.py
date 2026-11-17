import json
import tempfile
import unittest
from pathlib import Path

from corpus import build_records, DEFAULT_PDF, DEFAULT_LOGS


class CorpusTests(unittest.TestCase):
    def test_fictional_pdf_exists(self):
        self.assertTrue(DEFAULT_PDF.exists())

    def test_logs_have_unique_ids(self):
        notes = json.loads(DEFAULT_LOGS.read_text(encoding="utf-8"))
        self.assertEqual(len(set(n["id"] for n in notes)), len(notes))

    def test_corpus_contains_pdf_and_log(self):
        records = build_records()
        self.assertGreater(len(records), 12)
        self.assertIn("pdf", {r["metadata"]["kind"] for r in records})
        self.assertIn("incident", {r["metadata"]["kind"] for r in records})

    def test_retrievable_equipment_code(self):
        records = build_records()
        matching = [r for r in records if r["id"] == "LOG-001"]
        self.assertEqual(len(matching), 1)
        self.assertIn("INV-7X4", matching[0]["text"])

    def test_pdf_page_numbers(self):
        records = build_records()
        self.assertTrue(all(r["metadata"]["page"] >= 1 for r in records if r["metadata"]["kind"] == "pdf"))

    def test_ids_unique(self):
        records = build_records()
        self.assertEqual(len({r["id"] for r in records}), len(records))

    def test_duplicate_incident_rejected(self):
        notes = json.loads(DEFAULT_LOGS.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "duplicate.json"
            path.write_text(json.dumps(notes + [notes[0]]), encoding="utf-8")
            with self.assertRaises(ValueError):
                build_records(log_path=path)


if __name__ == "__main__":
    unittest.main()
