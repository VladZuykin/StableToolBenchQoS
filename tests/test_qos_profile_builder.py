import json
from pathlib import Path
import unittest

from scripts.build_qos_profiles import (
    build_empirical_pool,
    build_scenarios,
    generate_baseline,
    query_api_id,
    select_empirical_donor,
    validate_generation_config,
)


CONFIG = json.loads(
    Path("configs/qos_generation_v1.json").read_text(encoding="utf-8")
)


def catalog_record(api_id, *, latency, service, success):
    category, tool, _ = api_id.split("/", 2)
    return {
        "api_id": api_id,
        "category": category,
        "tool_id": tool,
        "qos": {
            "avgLatency": latency,
            "avgServiceLevel": service,
            "avgSuccessRate": success,
        },
    }


class QoSProfileBuilderTests(unittest.TestCase):
    def test_normalizes_solvable_api_id_like_the_runtime(self):
        self.assertEqual(
            query_api_id(
                {
                    "category_name": "Artificial Intelligence/Machine Learning",
                    "tool_name": "Example Tool",
                    "api_name": "id",
                }
            ),
            "Artificial_Intelligence_Machine_Learning/example_tool/is_id",
        )

    def test_empirical_pool_keeps_one_complete_record_per_tool(self):
        rows = [
            catalog_record("Data/complete/one", latency=100, service=99, success=98),
            catalog_record("Data/complete/two", latency=100, service=99, success=98),
            catalog_record("Data/incomplete/api", latency=None, service=None, success=None),
        ]

        pool = build_empirical_pool(rows)

        self.assertEqual([row["api_id"] for row in pool], ["Data/complete/one"])

    def test_donor_selection_is_deterministic_and_excludes_own_tool(self):
        own = catalog_record("Data/target/source", latency=100, service=99, success=98)
        first = catalog_record("Data/first/api", latency=200, service=95, success=90)
        second = catalog_record("Sports/second/api", latency=300, service=85, success=80)

        selected_a = select_empirical_donor(
            "Data/target/request", [own, first, second], 42
        )
        selected_b = select_empirical_donor(
            "Data/target/request", [own, first, second], 42
        )

        self.assertEqual(selected_a["api_id"], selected_b["api_id"])
        self.assertNotEqual(selected_a["api_id"], own["api_id"])

    def test_baseline_uses_donor_success_and_latency_but_synthetic_cost(self):
        validate_generation_config(CONFIG)
        donor = catalog_record(
            "Data/donor/api", latency=250, service=90, success=80
        )

        baseline = generate_baseline("Data/target/api", donor, 42, CONFIG)

        self.assertEqual(baseline["success_rate"], 0.72)
        self.assertEqual(baseline["expected_latency_ms"], 250.0)
        self.assertGreaterEqual(baseline["cost_per_call_units"], 0.00005)
        self.assertLessEqual(baseline["cost_per_call_units"], 0.05)

    def test_scenarios_apply_configured_factors(self):
        baseline = {
            "success_rate": 0.8,
            "expected_latency_ms": 100.0,
            "cost_per_call_units": 0.01,
            "latency_log_sigma": 0.1,
        }

        profiles = build_scenarios(baseline, CONFIG)

        self.assertEqual(profiles["normal"]["success_rate"], 0.8)
        self.assertEqual(profiles["degraded"]["expected_latency_ms"], 200.0)
        self.assertEqual(profiles["outage"]["success_rate"], 0.008)
        self.assertEqual(profiles["outage"]["cost_per_call_units"], 0.01)


if __name__ == "__main__":
    unittest.main()
