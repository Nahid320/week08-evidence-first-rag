import math
import unittest
from unittest.mock import patch

from reranker import rerank_candidates, retrieve_and_rerank, RERANK_MODEL_ID
from app import main, parser


def hit(doc_id, text):
    return {"id": doc_id, "text": text, "metadata": {"section": "test", "source": "sample", "page": 1}, "rrf_score": 0.02}


class FakeEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.pairs = None
        self.kwargs = None

    def predict(self, pairs, **kwargs):
        self.pairs = pairs
        self.kwargs = kwargs
        return self.scores


class FakeSearcher:
    def __init__(self, records):
        self.records = records
        self.args = None

    def search(self, question, mode, top_k, candidates):
        self.args = (question, mode, top_k, candidates)
        return self.records[:top_k]


class RerankingTests(unittest.TestCase):
    def setUp(self):
        self.docs = [hit("A", "an unrelated paragraph"), hit("B", "inverter was overheating"), hit("C", "cost estimates")]

    def test_model_id(self):
        self.assertEqual(RERANK_MODEL_ID, "cross-encoder/ms-marco-MiniLM-L6-v2")

    def test_pair_construction(self):
        model = FakeEncoder([0.3, 0.2, 0.1])
        rerank_candidates("What happened?", self.docs, model)
        self.assertEqual(model.pairs[0], ("What happened?", "test\nan unrelated paragraph"))
        self.assertEqual(len(model.pairs), 3)

    def test_reordering(self):
        model = FakeEncoder([-3, 8.2, 1])
        out = rerank_candidates("What caused the issue?", self.docs, model)
        self.assertEqual([r["id"] for r in out], ["B", "C", "A"])
        self.assertEqual(out[0]["candidate_rank"], 2)
        self.assertEqual(out[0]["rrf_score"], self.docs[1]["rrf_score"])

    def test_negative_logits_not_dropped(self):
        out = rerank_candidates("a", self.docs, FakeEncoder([-3, -5, -2]))
        self.assertEqual(out[0]["id"], "C")

    def test_tied_scores_keep_initial_order(self):
        out = rerank_candidates("a", self.docs, FakeEncoder([0.3, 0.3, 0.2]))
        self.assertEqual([r["id"] for r in out], ["A", "B", "C"])

    def test_top_k(self):
        out = rerank_candidates("a", self.docs, FakeEncoder([0.2, 0.5, 0.1]), top_k=2)
        self.assertEqual([r["id"] for r in out], ["B", "A"])

    def test_no_candidates(self):
        self.assertEqual(rerank_candidates("a", [], FakeEncoder([])), [])

    def test_blank_question(self):
        with self.assertRaisesRegex(ValueError, "blank"):
            rerank_candidates("  ", self.docs, FakeEncoder([1, 2, 3]))

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            rerank_candidates("a", self.docs, FakeEncoder([1, 2, 3]), top_k=0)

    def test_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            rerank_candidates("a", [self.docs[0], self.docs[0]], FakeEncoder([1, 2]))

    def test_wrong_number_scores(self):
        with self.assertRaisesRegex(ValueError, "unexpected"):
            rerank_candidates("a", self.docs, FakeEncoder([1]))

    def test_nonfinite_score(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            rerank_candidates("a", self.docs, FakeEncoder([1, math.nan, 2]))

    def test_preserves_metadata(self):
        out = rerank_candidates("a", self.docs, FakeEncoder([0, 1, -1]), top_k=1)
        self.assertEqual(out[0]["metadata"]["page"], 1)
        self.assertEqual(out[0]["id"], "B")

    def test_first_stage_only_candidates(self):
        searcher = FakeSearcher(self.docs)
        candidates, output = retrieve_and_rerank(searcher, "what", FakeEncoder([0, 2, 1]), top_k=2, candidate_count=3)
        self.assertEqual(len(candidates), 3)
        self.assertEqual(searcher.args, ("what", "hybrid", 3, 3))
        self.assertEqual([o["id"] for o in output], ["B", "C"])

    def test_candidate_count_smaller_than_k(self):
        with self.assertRaisesRegex(ValueError, "candidate_count"):
            retrieve_and_rerank(FakeSearcher(self.docs), "a", FakeEncoder([1, 2, 3]), top_k=3, candidate_count=2)

    def test_cli_demo(self):
        with patch("builtins.print") as printer:
            self.assertEqual(main(["demo"]), 0)
        self.assertTrue(any("['B', 'A', 'C']" in str(call) for call in printer.call_args_list))

    def test_cli_parse(self):
        args = parser().parse_args(["compare", "--query", "a", "--top-k", "4", "--candidates", "9"])
        self.assertEqual((args.top_k, args.candidates), (4, 9))


if __name__ == "__main__":
    unittest.main()
