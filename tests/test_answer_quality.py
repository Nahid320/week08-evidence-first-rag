"""No network, embedding models, or API key required."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from answer_quality import (
    ClaimVerdict, JudgeVerdict, evaluate_quality, inspect_case, judge_case,
    load_quality_cases, normalize, save_report,
)


class AnswerQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_quality_cases()
        cls.by_id = {r['id']: r for r in cls.cases}

    def test_ten_examples_available(self):
        self.assertEqual(len(self.cases), 10)

    def test_whitespace_normalization(self):
        self.assertEqual(normalize('HELLO   world\n'), 'hello world')

    def test_real_quote_passes(self):
        x = inspect_case(self.by_id['qa01'])
        self.assertTrue(x['all_citations_valid'])

    def test_valid_quote_does_not_prove_true_claim(self):
        x = inspect_case(self.by_id['qa02'])
        self.assertTrue(x['all_citations_valid'])
        self.assertEqual(x['citations'][0]['human_verdict'], 'contradicted')

    def test_fabricated_quote_fails(self):
        x = inspect_case(self.by_id['qa05'])
        self.assertFalse(x['all_citations_valid'])

    def test_unknown_source_fails(self):
        x = inspect_case(self.by_id['qa06'])
        self.assertFalse(x['citations'][0]['source_exists'])

    def test_correct_abstention(self):
        x = inspect_case(self.by_id['qa07'])
        self.assertTrue(x['abstention_decision_correct'])
        self.assertIsNone(x['all_citations_valid'])

    def test_incorrect_non_abstention(self):
        self.assertFalse(inspect_case(self.by_id['qa08'])['abstention_decision_correct'])

    def test_incorrect_abstention(self):
        self.assertFalse(inspect_case(self.by_id['qa09'])['abstention_decision_correct'])

    def test_two_claims(self):
        self.assertEqual(len(inspect_case(self.by_id['qa10'])['citations']),2)

    def test_baseline_report_has_no_faithfulness(self):
        report = evaluate_quality(self.cases)
        self.assertEqual(report['summary']['cases'], 10)
        self.assertIsNone(report['summary']['judge_faithfulness'])
        self.assertEqual(report['summary']['abstention_decision_accuracy'], 0.8)

    def test_report_saves_json(self):
        report = evaluate_quality(self.cases)
        with tempfile.TemporaryDirectory() as d:
            path = save_report(report, Path(d) / 'nested' / 'report.json')
            self.assertEqual(json.loads(path.read_text())['summary']['cases'], 10)

    def test_duplicate_case_id_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'cases.json'
            p.write_text(json.dumps([self.cases[0], self.cases[0]]))
            with self.assertRaisesRegex(ValueError, 'unique'):
                load_quality_cases(p)

    def test_abstention_with_claim_rejected(self):
        case = json.loads(json.dumps(self.cases[0]))
        case['abstained'] = True
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'c.json';p.write_text(json.dumps([case]))
            with self.assertRaisesRegex(ValueError, 'abstention'):
                load_quality_cases(p)

    def test_empty_data_rejected(self):
        with self.assertRaisesRegex(ValueError, 'No cases'):
            evaluate_quality([])

    def test_judge_mock_supported(self):
        sample = self.by_id['qa01']
        mock = Mock()
        mock.responses.parse.return_value.output_parsed = JudgeVerdict(
            answers_question=True, claim_verdicts=[ClaimVerdict(
                claim_id='C1', verdict='supported', explanation='Source directly reports repeated alarms'
            )]
        )
        output = judge_case(sample, client=mock, model='test-model')
        self.assertEqual(output['faithfulness'], 1.0)
        self.assertEqual(output['agreement_with_human_labels'], 1.0)
        call = mock.responses.parse.call_args.kwargs
        self.assertEqual(call['text_format'], JudgeVerdict)
        self.assertNotIn('human_verdict', call['input'])

    def test_judge_mock_contradiction(self):
        sample = self.by_id['qa02']
        mock = Mock()
        mock.responses.parse.return_value.output_parsed = JudgeVerdict(
            answers_question=True, claim_verdicts=[ClaimVerdict(
                claim_id='C1', verdict='contradicted', explanation='Source says loading was reduced'
            )]
        )
        output = judge_case(sample, client=mock)
        self.assertEqual(output['faithfulness'], 0.0)
        self.assertEqual(output['agreement_with_human_labels'], 1.0)

    def test_judge_mismatched_ids_rejected(self):
        mock=Mock()
        mock.responses.parse.return_value.output_parsed = JudgeVerdict(
            answers_question=True, claim_verdicts=[ClaimVerdict(
                claim_id='NOT_C1', verdict='supported', explanation='x'
            )]
        )
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            judge_case(self.by_id['qa01'], client=mock)

    def test_judge_refusal_or_incomplete_rejected(self):
        mock=Mock()
        mock.responses.parse.return_value.output_parsed = None
        with self.assertRaisesRegex(RuntimeError, 'no parsed'):
            judge_case(self.by_id['qa01'], client=mock)

    def test_aggregate_mock_judgments(self):
        mock=Mock()
        mock.responses.parse.return_value.output_parsed = JudgeVerdict(
            answers_question=True, claim_verdicts=[ClaimVerdict(
                claim_id='C1', verdict='supported', explanation='Direct support'
            )]
        )
        report=evaluate_quality([self.by_id['qa01'], self.by_id['qa07']], with_judge=True, client=mock)
        self.assertEqual(report['summary']['judge_faithfulness'], 1.0)
        self.assertEqual(mock.responses.parse.call_count, 1)  # Never judge abstention

    def test_cli_offline_without_api_key(self):
        from app import main
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(main(['quality', '--out', str(Path(d)/'out.json')]), 0)
            self.assertTrue((Path(d)/'out.json').exists())

    def test_cli_invalid_limit(self):
        from app import main
        self.assertEqual(main(['quality','--max-cases','0']),1)


if __name__ == '__main__':
    unittest.main()
