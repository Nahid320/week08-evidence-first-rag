import unittest

from grounding import GroundedResponse, SupportedClaim, canonical_text, source_context, verify_claims


class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.hits = [
            {"id": "LOG-001", "text": "INV-7X4 had repeated thermal alarms during commissioning.",
             "metadata": {"source": "maintenance_incidents.json", "page": -1, "section": "Thermal alarms"}},
            {"id": "PDF-001", "text": "The team inspected the cable tray and reduced loading.",
             "metadata": {"source": "industrial_energy_report.pdf", "page": 3, "section": "Testing"}},
        ]
        self.context, self.sources = source_context(self.hits)

    def claim(self, statement="The inverter had thermal alarms.", source="S1", quote="repeated thermal alarms"):
        return SupportedClaim(statement=statement, source_id=source, quote=quote)

    def test_source_ids_match_final_reranked_passages(self):
        self.assertIn("[S1] maintenance_incidents.json", self.context)
        self.assertIn("[S2] industrial_energy_report.pdf (page 3", self.context)
        self.assertEqual(self.sources["S1"]["id"], "LOG-001")

    def test_verified_quote(self):
        r = GroundedResponse(answerable=True, claims=[self.claim()])
        self.assertEqual(verify_claims(r, self.sources), (True, "VERIFIED_QUOTES_ONLY"))

    def test_space_and_case_normalized(self):
        r = GroundedResponse(answerable=True, claims=[self.claim(quote="REPEATED\nTHERMAL   ALARMS")])
        self.assertTrue(verify_claims(r, self.sources)[0])

    def test_unknown_source(self):
        r = GroundedResponse(answerable=True, claims=[self.claim(source="S3")])
        self.assertFalse(verify_claims(r, self.sources)[0])

    def test_invented_quote(self):
        r = GroundedResponse(answerable=True, claims=[self.claim(quote="the inverter exploded")])
        self.assertFalse(verify_claims(r, self.sources)[0])

    def test_abstention(self):
        r = GroundedResponse(answerable=False, claims=[])
        self.assertEqual(verify_claims(r, self.sources), (False, "ABSTAINED"))

    def test_empty_claims_fail(self):
        r = GroundedResponse(answerable=True, claims=[])
        self.assertFalse(verify_claims(r, self.sources)[0])

    def test_empty_quote_fails(self):
        r = GroundedResponse(answerable=True, claims=[self.claim(quote=" ")])
        self.assertFalse(verify_claims(r, self.sources)[0])

    def test_empty_statement_fails(self):
        r = GroundedResponse(answerable=True, claims=[self.claim(statement="")])
        self.assertFalse(verify_claims(r, self.sources)[0])

    def test_no_sources(self):
        text, sources = source_context([])
        self.assertEqual(text, "")
        self.assertEqual(sources, {})

    def test_basic_whitespace(self):
        self.assertEqual(canonical_text(" Some  TEXT\n "), "some text")


if __name__ == "__main__":
    unittest.main()
