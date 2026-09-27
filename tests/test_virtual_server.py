import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

from fastapi.testclient import TestClient


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = REPOSITORY_ROOT / "server"


def load_server_module():
    previous_directory = Path.cwd()
    sys.path.insert(0, str(SERVER_DIR))
    try:
        os.chdir(SERVER_DIR)
        specification = importlib.util.spec_from_file_location(
            "virtual_server_main", SERVER_DIR / "main.py"
        )
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        return module
    finally:
        os.chdir(previous_directory)


server = load_server_module()


class VirtualServerTests(unittest.TestCase):
    def test_exact_cache_match_is_excluded_from_llm_examples(self):
        payload = {
            "category": "Artificial Intelligence/Machine Learning",
            "tool_name": "ai_content_detector_v2",
            "api_name": "chat_gpt_detector_for_ai_content_detector_v2",
            "tool_input": {"text": "same input"},
            "strip": "",
            "toolbench_key": "",
        }
        calls = []

        cache = {
            "{'text': 'same input'}": {
                "error": "",
                "response": {"source": "cache"},
            },
            "{'text': 'different input'}": {
                "error": "",
                "response": {"source": "other_cache_example"},
            },
        }
        original_load_cache = server.load_cache_examples
        original_generator = server.fake_response_function_chat
        original_write_log = server.write_log
        original_qos_enabled = server.QOS_SIMULATOR.enabled
        try:
            server.load_cache_examples = lambda *paths: cache
            server.QOS_SIMULATOR.enabled = False
            server.write_log = lambda **kwargs: None

            def generate(examples, tool_input, api_doc):
                calls.append(examples)
                return json.dumps(
                    {"error": "", "response": {"source": "llm"}}
                )

            server.fake_response_function_chat = generate
            response = TestClient(server.app).post("/virtual", json=payload)
        finally:
            server.load_cache_examples = original_load_cache
            server.fake_response_function_chat = original_generator
            server.write_log = original_write_log
            server.QOS_SIMULATOR.enabled = original_qos_enabled

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["response"]["source"], "llm")
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 1)
        self.assertEqual(
            calls[0][0][1]["response"]["source"], "other_cache_example"
        )

    def test_qos_failure_skips_llm_generator(self):
        payload = {
            "category": "Artificial Intelligence/Machine Learning",
            "tool_name": "ai_content_detector_v2",
            "api_name": "chat_gpt_detector_for_ai_content_detector_v2",
            "tool_input": {"text": "test"},
            "strip": "",
            "toolbench_key": "",
        }
        api_id = (
            "Artificial_Intelligence_Machine_Learning/"
            "ai_content_detector_v2/chat_gpt_detector"
        )
        profiles = {
            api_id: {
                "simulation": {
                    "latency_distribution": "lognormal",
                    "latency_log_sigma": 0.0,
                    "profiles": {
                        "normal": {
                            "success_rate": 0.0,
                            "expected_latency_ms": 1.0,
                            "cost_per_call_units": 0.0005,
                        }
                    },
                }
            }
        }
        original_simulator = server.QOS_SIMULATOR
        original_generator = server.fake_response_function_chat
        original_write_log = server.write_log
        generator_calls = []
        try:
            server.QOS_SIMULATOR = server.QoSSimulator(
                profiles, enabled=True, sleep_enabled=False
            )
            server.fake_response_function_chat = lambda *args: generator_calls.append(1)
            server.write_log = lambda **kwargs: None
            response = TestClient(server.app).post("/virtual", json=payload)
        finally:
            server.QOS_SIMULATOR = original_simulator
            server.fake_response_function_chat = original_generator
            server.write_log = original_write_log

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["error"], "API not working error...")
        self.assertFalse(response.json()["qos"]["succeeded"])
        self.assertEqual(generator_calls, [])


if __name__ == "__main__":
    unittest.main()
