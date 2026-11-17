import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from grounding import GroundedResponse, SupportedClaim
from llm_adapter import (
    MAX_OUTPUT_TOKENS, IncompleteResponseError, UnparsedResponseError,
    check_parsed_response, request_grounded_answer,
)
from reliability import answer_safely, public_error


class LLMAdapterTests(unittest.TestCase):
    def test_limit_is_2000(self):
        self.assertEqual(MAX_OUTPUT_TOKENS, 2000)

    def test_completed_parsed_response(self):
        answer = GroundedResponse(answerable=False, claims=[])
        self.assertIs(check_parsed_response(SimpleNamespace(status="completed", output_parsed=answer)), answer)

    def test_incomplete_response_rejected(self):
        with self.assertRaises(IncompleteResponseError):
            check_parsed_response(SimpleNamespace(status="incomplete", output_parsed=None))

    def test_missing_parsed_response_rejected(self):
        with self.assertRaises(UnparsedResponseError):
            check_parsed_response(SimpleNamespace(status="completed", output_parsed=None))

    def test_invalid_parsed_type_rejected(self):
        with self.assertRaises(UnparsedResponseError):
            check_parsed_response(SimpleNamespace(status="completed", output_parsed="hello"))

    def test_request_passes_bounded_tokens_and_no_storage(self):
        client = Mock()
        parsed = GroundedResponse(answerable=False, claims=[])
        client.responses.parse.return_value = SimpleNamespace(status="completed", output_parsed=parsed)
        self.assertEqual(request_grounded_answer(client, "test-model", "Q", "[S1] Evidence"), parsed)
        kwargs = client.responses.parse.call_args.kwargs
        self.assertEqual(kwargs["max_output_tokens"], 2000)
        self.assertFalse(kwargs["store"])
        self.assertIs(kwargs["text_format"], GroundedResponse)
        self.assertIn("[S1] Evidence", kwargs["input"])

    def test_safe_user_message_when_incomplete(self):
        msg = public_error(IncompleteResponseError("hidden prompt text"))
        self.assertIn("INCOMPLETE_RESPONSE", msg)
        self.assertNotIn("hidden prompt", msg)

    def test_safe_user_message_when_unparsed(self):
        msg = public_error(UnparsedResponseError("private evidence"))
        self.assertIn("UNPARSEABLE_RESPONSE", msg)
        self.assertNotIn("private evidence", msg)

    def test_older_safety_wrapper_remains_compatible(self):
        hits = [{"id": "x", "text": "Some data", "metadata": {"source": "f.pdf", "page": 1, "section": "S"}}]
        result = answer_safely("Question", hits, lambda *_: GroundedResponse(answerable=False, claims=[]))
        self.assertEqual(result["status"], "ABSTAINED")


if __name__ == "__main__":
    unittest.main()
