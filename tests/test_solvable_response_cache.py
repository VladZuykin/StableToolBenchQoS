import json
import unittest

from scripts.build_solvable_response_cache import (
    build_messages,
    cache_keys,
    canonical_input,
    parse_model_output,
    request_example_count,
    stable_seed,
    target_example_count,
)


class SolvableResponseCacheTests(unittest.TestCase):
    def test_request_example_count_supports_single_example_tail_mode(self):
        self.assertEqual(request_example_count(3, None), 3)
        self.assertEqual(request_example_count(3, 1), 1)
        self.assertEqual(request_example_count(1, 2), 1)
        self.assertEqual(request_example_count(0, 1), 0)

    def test_canonical_input_accepts_python_repr_and_json(self):
        self.assertEqual(
            canonical_input("{'b': 2, 'a': 1}"),
            canonical_input('{"a":1,"b":2}'),
        )

    def test_parser_normalizes_parameters_and_rejects_duplicates(self):
        output = json.dumps(
            {
                "examples": [
                    {"input": {"User ID": "42"}, "response": {"ok": True}},
                    {"input": {"user_id": "42"}, "response": {"ok": False}},
                ]
            }
        )

        examples, rejected = parse_model_output(
            output,
            required_parameters=[{"name": "User ID"}],
            existing_keys=set(),
            limit=3,
        )

        self.assertEqual(len(examples), 1)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(next(iter(examples.values()))["response"], {"ok": True})

    def test_cache_keys_deduplicate_different_serializations(self):
        keys = cache_keys(
            {"{'a': 1, 'b': 2}": {}},
            {'{"b":2,"a":1}': {}},
        )
        self.assertEqual(len(keys), 1)

    def test_endpoint_seed_is_stable_and_endpoint_specific(self):
        self.assertEqual(stable_seed(42, "a/b/c"), stable_seed(42, "a/b/c"))
        self.assertNotEqual(stable_seed(42, "a/b/c"), stable_seed(42, "a/b/d"))

    def test_parameterless_endpoint_needs_only_one_example(self):
        record = {
            "api": {"required_parameters": [], "optional_parameters": []}
        }
        self.assertEqual(target_example_count(record, 3), 1)

    def test_parameterized_endpoint_uses_configured_minimum(self):
        record = {
            "api": {
                "required_parameters": [{"name": "query"}],
                "optional_parameters": [],
            }
        }
        self.assertEqual(target_example_count(record, 3), 3)

    def test_retry_attempt_changes_prompt_and_seed_material(self):
        document = {"api_id": "a/b/c"}
        first = build_messages(document, 2, [], attempt=1)
        second = build_messages(document, 2, [], attempt=2)
        self.assertNotEqual(first, second)
        self.assertNotEqual(
            stable_seed(42, "a/b/c\x1f1"),
            stable_seed(42, "a/b/c\x1f2"),
        )


if __name__ == "__main__":
    unittest.main()
