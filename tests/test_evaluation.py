"""Offline evaluation tests: no OpenAI calls, HF downloads, or Chroma installs."""
import json
import tempfile
import unittest
from pathlib import Path

from corpus import build_records
from evaluation import (DEFAULT_CASES, answer_status, evaluate_answers,
                        evaluate_retrieval, load_cases, measure, write_report)


class EvaluationMetricTests(unittest.TestCase):
    def test_all_four_metrics(self):
        output = measure(["x", "a", "b"], ["a", "b", "c", "d"], 3)
        self.assertAlmostEqual(output["precision"], 2/3)
        self.assertAlmostEqual(output["recall"], 0.5)
        self.assertEqual(output["hit"], 1.0)
        self.assertAlmostEqual(output["mrr"], 0.5)

    def test_no_hits(self):
        self.assertEqual(measure(["x", "y"], ["a"], 3),
                         {"precision": 0.0, "recall": 0.0, "hit": 0.0, "mrr": 0.0})

    def test_precision_denominator_k(self):
        self.assertEqual(measure(["a"], ["a"], 3)["precision"], 1/3)

    def test_mrr_only_first_relevant_rank(self):
        self.assertEqual(measure(["a", "b"], ["a", "b"], 2)["mrr"], 1)

    def test_truncated_at_k(self):
        self.assertEqual(measure(["x", "a"], ["a"], 1)["hit"], 0)

    def test_invalid_k(self):
        with self.assertRaises(ValueError): measure(["a"], ["a"], 0)

    def test_unanswerable_not_scored_with_recall(self):
        with self.assertRaises(ValueError): measure([], [], 3)

    def test_duplicate_retrieval_ids_rejected(self):
        with self.assertRaises(ValueError): measure(["a", "a"], ["a"], 3)


class DatasetTests(unittest.TestCase):
    def test_real_corpus_ids_exist(self):
        ids = {r["id"] for r in build_records()}
        cases = load_cases(DEFAULT_CASES, known_ids=ids)
        self.assertEqual(len(cases), 14)
        self.assertEqual(sum(x["answerable"] for x in cases), 12)
        self.assertEqual(sum(not x["answerable"] for x in cases), 2)

    def test_unknown_gold_fails(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"questions.json"
            path.write_text(json.dumps([{
                "id":"q", "question":"Something?", "answerable":True,
                "relevant_ids":["missing"]
            }]))
            with self.assertRaisesRegex(ValueError, "unknown"):
                load_cases(path, known_ids={"known"})

    def test_duplicate_question_ids_fail(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"questions.json"
            path.write_text(json.dumps([
                {"id":"q","question":"a","answerable":True,"relevant_ids":["A"]},
                {"id":"q","question":"b","answerable":True,"relevant_ids":["A"]},
            ]))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_cases(path)

    def test_unanswerable_must_have_empty_gold(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"questions.json"
            path.write_text(json.dumps([
                {"id":"q","question":"a","answerable":False,"relevant_ids":["A"]},
            ]))
            with self.assertRaisesRegex(ValueError, "agree"):
                load_cases(path)


class MockEncoder:
    def predict(self, pairs, **kwargs):
        # Always place the text of C ahead of B and A.
        return [{"doc A":1.0,"doc B":2.0,"doc C":3.0}[p[1].split("\n")[-1]] for p in pairs]


class MockSearcher:
    records = [
        {"id": "A", "text": "doc A", "metadata": {"section":""}},
        {"id": "B", "text": "doc B", "metadata": {"section":""}},
        {"id": "C", "text": "doc C", "metadata": {"section":""}},
    ]
    def search(self, question, mode, top_k, candidates):
        by_id = {r["id"]: r for r in self.records}
        order = {"semantic": ["B","A","C"], "keyword": ["A","C","B"], "hybrid": ["B","A","C"]}[mode]
        return [by_id[x] for x in order[:top_k]]


class EvaluationPipelineTests(unittest.TestCase):
    def setUp(self):
        self.searcher=MockSearcher()
        self.encoder=MockEncoder()
        self.cases=[
            {"id":"q1","question":"Find C","answerable":True,"relevant_ids":["C"]},
            {"id":"q2","question":"CEO salary?","answerable":False,"relevant_ids":[]},
            {"id":"q3","question":"Stock price?","answerable":False,"relevant_ids":[]},
        ]

    def test_metrics_aggregation_and_improvement(self):
        report=evaluate_retrieval(self.searcher,self.encoder,self.cases,top_k=2,candidates=3)
        self.assertEqual(report['query_counts'], {'answerable':1,'unanswerable_excluded':2})
        self.assertEqual(report['summary']['hybrid']['hit'],0.0)
        self.assertEqual(report['summary']['reranked']['hit'],1.0)
        self.assertEqual(report['summary']['reranked']['mrr'],1.0)
        self.assertEqual(report['summary']['bm25']['hit'],1.0)
        self.assertEqual(report['average_hybrid_candidate_recall'],1.0)

    def test_report_contains_all_four_methods(self):
        report=evaluate_retrieval(self.searcher,self.encoder,self.cases,top_k=2,candidates=3)
        self.assertEqual(set(report['summary']), {'semantic','bm25','hybrid','reranked'})
        self.assertEqual(report['per_query'][0]['results']['reranked']['ids'], ['C','B'])

    def test_missing_label_fails_before_running(self):
        bad=[{"id":"x","question":"Foo","answerable":True,"relevant_ids":["Z"]}]
        with self.assertRaisesRegex(ValueError, "not present"):
            evaluate_retrieval(self.searcher,self.encoder,bad,top_k=2,candidates=3)

    def test_no_answerable_fails(self):
        with self.assertRaisesRegex(ValueError, "At least one"):
            evaluate_retrieval(self.searcher,self.encoder,[self.cases[1]],top_k=2,candidates=3)

    def test_invalid_candidate_count_fails(self):
        with self.assertRaisesRegex(ValueError, "candidates"):
            evaluate_retrieval(self.searcher,self.encoder,self.cases,top_k=4,candidates=3)

    def test_save_json_and_csv(self):
        report=evaluate_retrieval(self.searcher,self.encoder,self.cases,top_k=2,candidates=3)
        with tempfile.TemporaryDirectory() as d:
            json_path, csv_path=write_report(report,Path(d)/'evaluation.json')
            self.assertTrue(json_path.is_file())
            self.assertTrue(csv_path.is_file())
            self.assertIn('reranking_minus_hybrid', json.loads(json_path.read_text()))
            self.assertIn('reranked',csv_path.read_text())

    def test_answer_statuses(self):
        self.assertEqual(answer_status('ABSTAINED\nNo evidence.'),'ABSTAINED')
        self.assertEqual(answer_status('VERIFIED_QUOTES_ONLY (not logical proof)'),'VERIFIED_QUOTES_ONLY')
        self.assertEqual(answer_status('VERIFICATION_FAILED: bad quote'),'VERIFICATION_FAILED')
        self.assertEqual(answer_status('surprise'), 'UNKNOWN')

    def test_answer_evaluation_runs_unanswerables_first(self):
        seen=[]
        def generate(question, hits):
            seen.append(question)
            return 'ABSTAINED\nNo evidence.'
        result=evaluate_answers(self.searcher,self.encoder,self.cases,generate,
                               max_cases=2,top_k=2,candidates=3)
        self.assertEqual(result['cases_run'],2)
        self.assertEqual(result['statuses_matching_expectation'],2)
        self.assertEqual(seen,['CEO salary?','Stock price?'])

    def test_answer_evaluation_can_detect_status_failures(self):
        result=evaluate_answers(self.searcher,self.encoder,self.cases,
                lambda q,h:'VERIFIED_QUOTES_ONLY\nUnsupported answer',
                max_cases=3,top_k=2,candidates=3)
        self.assertEqual(result['statuses_matching_expectation'],1)
        self.assertIsNone(result['cases'][0]['human_evidence_review'])


if __name__ == '__main__':
    unittest.main()

class CLIIntegrationTests(unittest.TestCase):
    def test_evaluate_cli_end_to_end_with_mock_models(self):
        """Exercise the real CLI, table rendering and JSON/CSV saving offline."""
        from unittest.mock import patch
        import app
        searcher = MockSearcher()
        cases = [{"id":"q1", "question":"Find C", "answerable":True, "relevant_ids":["C"]},
                 {"id":"q2", "question":"No evidence", "answerable":False, "relevant_ids":[]}]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/"evaluation.json"
            with patch.object(app, "HybridSearcher", return_value=searcher), \
                 patch.object(app, "load_cross_encoder", return_value=MockEncoder()), \
                 patch.object(app, "load_cases", return_value=cases):
                exitcode = app.main(["evaluate", "--top-k", "2", "--candidates", "3", "--out", str(out)])
            self.assertEqual(exitcode,0)
            self.assertTrue(out.exists())
            self.assertTrue(out.with_name("evaluation_per_query.csv").exists())
            self.assertEqual(json.loads(out.read_text())['query_counts']['unanswerable_excluded'],1)

    def test_cli_with_llm_checks_unanswerable_first(self):
        from unittest.mock import patch
        import app
        cases = [{"id":"a", "question":"Find C", "answerable":True, "relevant_ids":["C"]},
                 {"id":"x", "question":"No evidence", "answerable":False, "relevant_ids":[]}]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/"evaluation.json"
            with patch.object(app, "HybridSearcher", return_value=MockSearcher()), \
                 patch.object(app, "load_cross_encoder", return_value=MockEncoder()), \
                 patch.object(app, "load_cases", return_value=cases), \
                 patch.object(app, "ask_llm", return_value="ABSTAINED\nNo evidence."):
                exitcode=app.main(["evaluate","--top-k","2","--candidates","3",
                                    "--with-llm","--max-llm-cases","1","--out",str(out)])
            self.assertEqual(exitcode,0)
            report=json.loads(out.read_text())
            self.assertEqual(report['answer_checks']['cases'][0]['id'],'x')
