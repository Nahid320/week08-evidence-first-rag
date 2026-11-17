import unittest
from unittest.mock import patch

from hybrid_search import tokenize, reciprocal_rank_fusion, keyword_ranking
from app import main, parser


def hit(item):
    return {"id": item, "text": item, "metadata": {}}


class FusionTests(unittest.TestCase):
    def test_tokenize_equipment_ids(self):
        self.assertEqual(tokenize("INV-7X4, GW-12B and MTR-3C"),
                         ["inv-7x4", "gw-12b", "and", "mtr-3c"])

    def test_tokenize_empty(self):
        self.assertEqual(tokenize("!@#$"), [])

    def test_both_rankers_help(self):
        result = reciprocal_rank_fusion([
            [hit("A"), hit("B"), hit("C")],
            [hit("B"), hit("D"), hit("A")],
        ])
        self.assertEqual([r["id"] for r in result[:2]], ["B", "A"])
        self.assertEqual(len(result[0]["ranks"]), 2)

    def test_single_list_order(self):
        self.assertEqual([r["id"] for r in reciprocal_rank_fusion([[hit("C"), hit("B")]])], ["C", "B"])

    def test_rrf_formula(self):
        results = reciprocal_rank_fusion([[hit("A")], [hit("A")]])
        self.assertAlmostEqual(results[0]["rrf_score"], 2 / 61)

    def test_rrf_no_duplicate_contribution_per_ranker(self):
        self.assertAlmostEqual(reciprocal_rank_fusion([[hit("A"), hit("A")]])[0]["rrf_score"], 1 / 61)

    def test_rrf_top_k(self):
        self.assertEqual(len(reciprocal_rank_fusion([[hit("A"), hit("B")]], top_k=1)), 1)

    def test_invalid_rrf_k(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([[hit("A")]], k=0)

    def test_invalid_top_k(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([[hit("A")]], top_k=0)

    def test_empty_rrf(self):
        self.assertEqual(reciprocal_rank_fusion([[], []]), [])

    def test_keyword_empty_query(self):
        self.assertEqual(keyword_ranking([hit("A")], "", 2), [])

    def test_keyword_empty_records(self):
        self.assertEqual(keyword_ranking([], "abc", 2), [])

    def test_bad_keyword_limit(self):
        with self.assertRaises(ValueError):
            keyword_ranking([hit("A")], "abc", 0)

    def test_cli_demo(self):
        with patch("builtins.print") as printer:
            code = main(["demo"])
        self.assertEqual(code, 0)
        self.assertGreaterEqual(printer.call_count, 3)

    def test_parser_mode(self):
        args = parser().parse_args(["compare", "--query", "abc", "--top-k", "3"])
        self.assertEqual(args.top_k, 3)


if __name__ == "__main__":
    unittest.main()
