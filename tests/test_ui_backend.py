import tempfile
import unittest
from pathlib import Path

from ui_backend import (build_upload_records, retrieve, safe_filename,
                        sample_records, upload_signature)


class TestFileValidation(unittest.TestCase):
    def test_safe_filename_strips_paths(self):
        self.assertEqual(safe_filename("..\\folder\\hello.pdf"), "hello.pdf")
    def test_reject_extension(self):
        with self.assertRaises(ValueError): safe_filename("payload.txt")
    def test_reject_empty_uploads(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "at least one"):
                build_upload_records([], Path(d))
    def test_reject_non_pdf(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "not a PDF"):
                build_upload_records([("fake.pdf", b"not pdf")], Path(d))
    def test_reject_too_many(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "Limit"):
                build_upload_records([("x.pdf", b"%PDF-x")] * 4, Path(d))
    def test_reject_too_large(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "exceeds"):
                build_upload_records([("x.pdf", b"%PDF-" + b"X" * 100)], Path(d), max_file_bytes=30)
    def test_fingerprint_is_deterministic(self):
        x = [("A.pdf", b"aa"), ("B.pdf", b"bb")]
        self.assertEqual(upload_signature(x), upload_signature(list(reversed(x))))
        self.assertNotEqual(upload_signature(x), upload_signature([("A.pdf", b"new")]))
    def test_sample_records_have_sources(self):
        data = sample_records()
        self.assertTrue(data)
        self.assertTrue(all("source" in r["metadata"] for r in data))
    def test_real_sample_pdf_can_be_uploaded(self):
        sample = Path(__file__).resolve().parents[1] / "sample_pdfs" / "industrial_energy_report.pdf"
        with tempfile.TemporaryDirectory() as d:
            rows = build_upload_records([(sample.name, sample.read_bytes())], Path(d))
            self.assertTrue(rows)
            self.assertTrue(all(row["metadata"]["page"] >= 1 for row in rows))
            self.assertTrue(all(row["metadata"]["source"] == sample.name for row in rows))
    def test_scanned_pdf_with_no_text_would_be_rejected(self):
        # Input is deliberately invalid PDF, exercising robust rejection, not OCR.
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                build_upload_records([("example.pdf", b"%PDF-not-a-valid-document")], Path(d))


class StubSearcher:
    records = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    def search(self, question, mode, top_k, candidates):
        return [{"id": "a", "text": "alpha", "metadata": {"section": "one", "source": "x.pdf", "page": 1}},
                {"id": "b", "text": "beta", "metadata": {"section": "two", "source": "x.pdf", "page": 2}},
                {"id": "c", "text": "gamma", "metadata": {"section": "three", "source": "x.pdf", "page": 3}}][:top_k]


class FakeCrossEncoder:
    def predict(self, pairs, **kwargs):
        return [2.0, 8.0, 1.0][:len(pairs)]


class TestRetrieval(unittest.TestCase):
    def test_reranks(self):
        out = retrieve("Which?", StubSearcher(), FakeCrossEncoder(), top_k=2, candidates=3)
        self.assertEqual([h["id"] for h in out["hits"]], ["b", "a"])
    def test_invalid_question(self):
        with self.assertRaises(ValueError):
            retrieve(" ", StubSearcher(), FakeCrossEncoder())
    def test_invalid_counts(self):
        with self.assertRaises(ValueError):
            retrieve("Which?", StubSearcher(), FakeCrossEncoder(), top_k=3, candidates=2)


if __name__ == "__main__":
    unittest.main()
