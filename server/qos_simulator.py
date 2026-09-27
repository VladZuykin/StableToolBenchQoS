"""Runtime QoS simulation for StableToolBench virtual API calls."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
from threading import Lock
import time
from typing import Any, Callable


QOS_FAILURE_ERROR = "API not working error..."


def parse_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def load_profiles(path: Path) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            api_id = record.get("api_id")
            if not isinstance(api_id, str) or not api_id:
                raise ValueError(f"Missing api_id at {path}:{line_number}")
            if api_id in profiles:
                raise ValueError(f"Duplicate QoS profile for {api_id}")
            profiles[api_id] = record
    return profiles


class QoSSimulator:
    """Draw deterministic per-call outcomes from generated API QoS profiles."""

    def __init__(
        self,
        profiles: dict[str, dict[str, Any]] | None = None,
        *,
        enabled: bool = False,
        scenario: str = "normal",
        seed: int = 42,
        sleep_enabled: bool = True,
        sleep_function: Callable[[float], None] = time.sleep,
    ) -> None:
        self.profiles = profiles or {}
        self.enabled = enabled
        self.scenario = scenario
        self.seed = seed
        self.sleep_enabled = sleep_enabled
        self._sleep = sleep_function
        self._call_counts: defaultdict[str, int] = defaultdict(int)
        self._lock = Lock()

        if self.enabled:
            for api_id, record in self.profiles.items():
                available = (record.get("simulation") or {}).get("profiles") or {}
                if self.scenario not in available:
                    raise ValueError(
                        f"QoS scenario {self.scenario!r} is missing for {api_id}"
                    )

    @classmethod
    def from_file(
        cls,
        path: Path,
        **kwargs: Any,
    ) -> "QoSSimulator":
        enabled = parse_bool(kwargs.get("enabled"), default=False)
        kwargs["enabled"] = enabled
        if not enabled:
            return cls(**kwargs)
        if not path.is_file():
            raise FileNotFoundError(f"QoS profiles file does not exist: {path}")
        return cls(load_profiles(path), **kwargs)

    def _next_call_index(self, api_id: str) -> int:
        with self._lock:
            call_index = self._call_counts[api_id]
            self._call_counts[api_id] += 1
        return call_index

    def _rng_for_dimension(
        self, api_id: str, call_index: int, dimension: str
    ) -> random.Random:
        material = (
            f"{self.seed}\x1f{api_id}\x1f{call_index}\x1f{dimension}".encode("utf-8")
        )
        digest = hashlib.sha256(material).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    @staticmethod
    def _sample_latency_ms(
        rng: random.Random,
        expected_latency_ms: float,
        log_sigma: float,
    ) -> float:
        if not math.isfinite(expected_latency_ms) or expected_latency_ms <= 0:
            raise ValueError("expected_latency_ms must be positive")
        if not math.isfinite(log_sigma) or log_sigma < 0:
            raise ValueError("latency_log_sigma must be non-negative")
        if log_sigma == 0:
            return expected_latency_ms

        # Python accepts the mean and standard deviation of the underlying
        # normal distribution. This conversion makes expected_latency_ms the
        # arithmetic mean of the resulting log-normal distribution.
        log_mu = math.log(expected_latency_ms) - (log_sigma**2) / 2.0
        return rng.lognormvariate(log_mu, log_sigma)

    def sample(self, api_id: str) -> dict[str, Any]:
        record = self.profiles.get(api_id)
        if record is None:
            return {
                "api_id": api_id,
                "profile": self.scenario,
                "profile_found": False,
                "succeeded": False,
                "success_rate": None,
                "latency_ms": 0.0,
                "cost_units": 0.0,
                "call_index": None,
            }

        simulation = record["simulation"]
        profile = simulation["profiles"][self.scenario]
        distribution = simulation["latency_distribution"]
        if distribution != "lognormal":
            raise ValueError(
                f"Unsupported latency distribution for {api_id}: {distribution}"
            )
        # end_to_end_success_probability is accepted for profiles-v2 so that
        # existing local experiments remain readable after the terminology
        # changed from pass probability to success rate.
        success_rate_value = profile.get(
            "success_rate", profile.get("end_to_end_success_probability")
        )
        if success_rate_value is None:
            raise ValueError(f"Missing success rate for {api_id}")
        success_rate = float(success_rate_value)
        expected_latency_ms = float(profile["expected_latency_ms"])
        log_sigma = float(simulation["latency_log_sigma"])
        cost_units = float(profile["cost_per_call_units"])
        if not 0.0 <= success_rate <= 1.0:
            raise ValueError(f"Invalid success rate for {api_id}: {success_rate}")
        if not math.isfinite(cost_units) or cost_units < 0:
            raise ValueError(f"Invalid cost for {api_id}: {cost_units}")

        call_index = self._next_call_index(api_id)
        latency_rng = self._rng_for_dimension(api_id, call_index, "latency")
        outcome_rng = self._rng_for_dimension(api_id, call_index, "outcome")
        latency_ms = self._sample_latency_ms(
            latency_rng, expected_latency_ms, log_sigma
        )
        succeeded = outcome_rng.random() < success_rate

        return {
            "api_id": api_id,
            "profile": self.scenario,
            "profile_found": True,
            "succeeded": succeeded,
            "success_rate": success_rate,
            "latency_ms": round(latency_ms, 4),
            "expected_latency_ms": expected_latency_ms,
            "latency_distribution": distribution,
            "latency_log_sigma": log_sigma,
            "cost_units": cost_units,
            "call_index": call_index,
        }

    def prepare_call(self, api_id: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        return self.sample(api_id)

    def wait_for_latency(
        self,
        qos: dict[str, Any] | None,
        *,
        elapsed_seconds: float = 0.0,
    ) -> None:
        if not self.sleep_enabled or qos is None:
            return
        remaining_seconds = max(0.0, qos["latency_ms"] / 1000.0 - elapsed_seconds)
        if remaining_seconds > 0:
            self._sleep(remaining_seconds)

    @staticmethod
    def should_generate_response(qos: dict[str, Any] | None) -> bool:
        return qos is None or (
            qos["profile_found"] and qos["succeeded"]
        )

    def resolve(
        self,
        response: dict[str, Any] | str,
        qos: dict[str, Any] | None,
    ) -> dict[str, Any] | str:
        if qos is None:
            return response

        if isinstance(response, str):
            response = json.loads(response)
        if not isinstance(response, dict):
            raise TypeError("Tool response must be a JSON object")

        if not qos["profile_found"]:
            return {
                "error": f"QoS profile not found for {qos['api_id']}",
                "response": "",
                "qos": qos,
            }
        if not qos["succeeded"]:
            return {"error": QOS_FAILURE_ERROR, "response": "", "qos": qos}

        result = deepcopy(response)
        result["qos"] = qos
        return result

    def apply(self, api_id: str, response: dict[str, Any] | str) -> dict[str, Any] | str:
        qos = self.prepare_call(api_id)
        self.wait_for_latency(qos)
        return self.resolve(response, qos)
