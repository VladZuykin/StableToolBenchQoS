import math
import random
import unittest

from server.qos_simulator import QOS_FAILURE_ERROR, QoSSimulator


API_ID = "Category/tool/api"


def profile(*, success_rate=1.0, latency_ms=100.0, sigma=0.0, cost=0.25):
    return {
        API_ID: {
            "simulation": {
                "latency_distribution": "lognormal",
                "latency_log_sigma": sigma,
                "profiles": {
                    "normal": {
                        "success_rate": success_rate,
                        "expected_latency_ms": latency_ms,
                        "cost_per_call_units": cost,
                    }
                },
            }
        }
    }


class QoSSimulatorTests(unittest.TestCase):
    def test_disabled_simulator_preserves_response(self):
        response = {"error": "", "response": "ok"}
        simulator = QoSSimulator(profile(), enabled=False)
        self.assertIs(simulator.apply(API_ID, response), response)

    def test_success_adds_qos_and_sleeps_for_sampled_latency(self):
        sleeps = []
        simulator = QoSSimulator(
            profile(), enabled=True, sleep_function=sleeps.append
        )

        result = simulator.apply(API_ID, {"error": "", "response": "ok"})

        self.assertEqual(result["response"], "ok")
        self.assertTrue(result["qos"]["succeeded"])
        self.assertEqual(result["qos"]["success_rate"], 1.0)
        self.assertEqual(result["qos"]["latency_ms"], 100.0)
        self.assertEqual(result["qos"]["cost_units"], 0.25)
        self.assertEqual(sleeps, [0.1])

    def test_llm_time_is_deducted_from_simulated_latency(self):
        sleeps = []
        simulator = QoSSimulator(
            profile(latency_ms=100.0), enabled=True, sleep_function=sleeps.append
        )
        qos = simulator.prepare_call(API_ID)

        simulator.wait_for_latency(qos, elapsed_seconds=0.04)

        self.assertAlmostEqual(sleeps[0], 0.06)

    def test_failed_call_replaces_response_with_tool_error(self):
        simulator = QoSSimulator(
            profile(success_rate=0.0), enabled=True, sleep_enabled=False
        )

        result = simulator.apply(API_ID, {"error": "", "response": "secret"})

        self.assertEqual(result["error"], QOS_FAILURE_ERROR)
        self.assertEqual(result["response"], "")
        self.assertFalse(result["qos"]["succeeded"])
        self.assertEqual(result["qos"]["cost_units"], 0.25)

    def test_prepared_failure_does_not_allow_response_generation(self):
        simulator = QoSSimulator(
            profile(success_rate=0.0), enabled=True, sleep_enabled=False
        )

        qos = simulator.prepare_call(API_ID)

        self.assertFalse(simulator.should_generate_response(qos))
        result = simulator.resolve({}, qos)
        self.assertEqual(result["error"], QOS_FAILURE_ERROR)

    def test_sequence_is_reproducible_for_same_seed(self):
        first = QoSSimulator(
            profile(success_rate=0.5, sigma=0.2),
            enabled=True,
            seed=123,
            sleep_enabled=False,
        )
        second = QoSSimulator(
            profile(success_rate=0.5, sigma=0.2),
            enabled=True,
            seed=123,
            sleep_enabled=False,
        )

        first_samples = [first.sample(API_ID) for _ in range(5)]
        second_samples = [second.sample(API_ID) for _ in range(5)]

        self.assertEqual(first_samples, second_samples)

    def test_outcomes_do_not_depend_on_latency_sigma(self):
        fixed_latency = QoSSimulator(
            profile(success_rate=0.5, sigma=0.0),
            enabled=True,
            seed=123,
            sleep_enabled=False,
        )
        jittered_latency = QoSSimulator(
            profile(success_rate=0.5, sigma=0.5),
            enabled=True,
            seed=123,
            sleep_enabled=False,
        )

        fixed_outcomes = [
            fixed_latency.sample(API_ID)["succeeded"] for _ in range(20)
        ]
        jittered_outcomes = [
            jittered_latency.sample(API_ID)["succeeded"] for _ in range(20)
        ]

        self.assertEqual(fixed_outcomes, jittered_outcomes)

    def test_lognormal_parameterization_preserves_arithmetic_mean(self):
        rng = random.Random(7)
        expected = 250.0
        samples = [
            QoSSimulator._sample_latency_ms(rng, expected, 0.3)
            for _ in range(100_000)
        ]
        sample_mean = math.fsum(samples) / len(samples)
        self.assertAlmostEqual(sample_mean, expected, delta=2.0)

    def test_missing_profile_returns_explicit_error(self):
        simulator = QoSSimulator(profile(), enabled=True, sleep_enabled=False)

        result = simulator.apply("Category/tool/missing", {"error": "", "response": "ok"})

        self.assertIn("QoS profile not found", result["error"])
        self.assertFalse(result["qos"]["profile_found"])

    def test_v2_success_probability_is_still_supported(self):
        legacy = profile()
        legacy_profile = legacy[API_ID]["simulation"]["profiles"]["normal"]
        legacy_profile["end_to_end_success_probability"] = legacy_profile.pop(
            "success_rate"
        )
        simulator = QoSSimulator(legacy, enabled=True, sleep_enabled=False)

        result = simulator.apply(API_ID, {"error": "", "response": "ok"})

        self.assertTrue(result["qos"]["succeeded"])
        self.assertEqual(result["qos"]["success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
