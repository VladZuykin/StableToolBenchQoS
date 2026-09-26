"""Build reproducible per-API QoS profiles without modifying relation clusters."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean, median
from typing import Any


PROFILE_VERSION = "qos-profiles-v2"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def probability(percent: Any) -> float | None:
    if not numeric(percent):
        return None
    return min(1.0, max(0.0, float(percent) / 100.0))


def stable_rng(seed: int, api_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}\x1f{api_id}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def bounded(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, value))


def scenario(
    availability: float,
    success: float,
    latency_ms: float,
    cost_units: float,
    *,
    availability_factor: float,
    success_factor: float,
    latency_factor: float,
) -> dict[str, float]:
    scenario_availability = bounded(availability * availability_factor)
    scenario_success = bounded(success * success_factor)
    return {
        "availability_probability": round(scenario_availability, 8),
        "success_given_available_probability": round(scenario_success, 8),
        "end_to_end_success_probability": round(
            scenario_availability * scenario_success, 8
        ),
        "expected_latency_ms": round(max(1.0, latency_ms * latency_factor), 4),
        "cost_per_call_units": cost_units,
    }


def reference_cost(policy: dict[str, Any]) -> float:
    """Comparison-only cost; actual cost depends on the current quota state."""
    included = int(policy["included_calls"])
    upfront = float(policy["upfront_cost_units"])
    overage = policy.get("overage_cost_per_call_units")
    if upfront > 0 and included > 0:
        return upfront / included
    if overage is not None:
        return float(overage)
    return 0.0


def summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": median(values),
        "mean": mean(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/catalog/tools.jsonl"),
        help="Audited StableToolBench catalog containing observed QoS",
    )
    parser.add_argument(
        "--api-to-cluster",
        type=Path,
        default=Path("data/relation_graph/v16_migrated/api_to_cluster.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/qos/v1"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--monetization-config",
        type=Path,
        default=Path("configs/qos_monetization_v1.json"),
    )
    parser.add_argument(
        "--latency-jitter-min",
        type=float,
        default=0.05,
        help="Minimum simulated log-normal latency sigma",
    )
    parser.add_argument(
        "--latency-jitter-max",
        type=float,
        default=0.25,
        help="Maximum simulated log-normal latency sigma",
    )
    args = parser.parse_args()
    if args.latency_jitter_min < 0 or args.latency_jitter_max < args.latency_jitter_min:
        parser.error("invalid latency jitter range")

    cluster_by_api = {
        row["api_id"]: row["cluster_id"] for row in load_jsonl(args.api_to_cluster)
    }
    catalog = {row["api_id"]: row for row in load_jsonl(args.catalog)}
    monetization = json.loads(args.monetization_config.read_text(encoding="utf-8"))
    tier_templates = monetization["tiers"]
    missing = sorted(set(cluster_by_api) - set(catalog))
    if missing:
        raise SystemExit(
            f"{len(missing)} graph APIs are absent from the QoS catalog; first: {missing[0]}"
        )

    records = []
    pricing_counts: Counter[str] = Counter()
    latencies = []
    availabilities = []
    successes = []
    for api_id in sorted(cluster_by_api):
        source = catalog[api_id]
        qos = source.get("qos") or {}
        latency = qos.get("avgLatency")
        availability = probability(qos.get("avgServiceLevel"))
        success = probability(qos.get("avgSuccessRate"))
        popularity = qos.get("popularityScore")
        if not numeric(latency) or availability is None or success is None:
            raise SystemExit(f"Incomplete QoS for graph API: {api_id}")
        latency = max(1.0, float(latency))
        pricing = str(source.get("pricing") or "UNKNOWN").upper()
        rng = stable_rng(args.seed, api_id)
        templates = tier_templates.get(pricing, tier_templates["UNKNOWN"])
        policy = dict(templates[rng.randrange(len(templates))])
        cost_units = reference_cost(policy)
        jitter = rng.uniform(args.latency_jitter_min, args.latency_jitter_max)
        profiles = {
            "normal": scenario(
                availability, success, latency, cost_units,
                availability_factor=1.0, success_factor=1.0, latency_factor=1.0,
            ),
            "degraded": scenario(
                availability, success, latency, cost_units,
                availability_factor=0.75, success_factor=0.85, latency_factor=2.0,
            ),
            "outage": scenario(
                availability, success, latency, cost_units,
                availability_factor=0.05, success_factor=0.20, latency_factor=4.0,
            ),
        }
        records.append({
            "api_id": api_id,
            "cluster_id": cluster_by_api[api_id],
            "profile_version": PROFILE_VERSION,
            "seed": args.seed,
            "source": {
                "catalog": str(args.catalog),
                "qos_level": "tool",
                "avg_latency_ms": float(qos["avgLatency"]),
                "avg_service_level_percent": float(qos["avgServiceLevel"]),
                "avg_success_rate_percent": float(qos["avgSuccessRate"]),
                "popularity_score": float(popularity) if numeric(popularity) else None,
                "pricing": pricing,
            },
            "simulation": {
                "latency_distribution": "lognormal",
                "latency_log_sigma": round(jitter, 8),
                "reference_cost_per_call_units": round(cost_units, 10),
                "monetization": policy,
                "profiles": profiles,
            },
        })
        pricing_counts[pricing] += 1
        latencies.append(latency)
        availabilities.append(availability)
        successes.append(success)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "api_qos_profiles.jsonl"
    temporary = output_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(output_path)

    metadata = {
        "profile_version": PROFILE_VERSION,
        "seed": args.seed,
        "api_count": len(records),
        "cluster_mapping": str(args.api_to_cluster),
        "source_catalog": str(args.catalog),
        "monetization_config": str(args.monetization_config),
        "monetization_version": monetization["version"],
        "currency": monetization["currency"],
        "scenario_factors": {
            "normal": {"availability": 1.0, "success": 1.0, "latency": 1.0},
            "degraded": {"availability": 0.75, "success": 0.85, "latency": 2.0},
            "outage": {"availability": 0.05, "success": 0.20, "latency": 4.0},
        },
        "latency_jitter_sigma_range": [args.latency_jitter_min, args.latency_jitter_max],
        "pricing_distribution": dict(sorted(pricing_counts.items())),
        "observed_summary": {
            "latency_ms": summary(latencies),
            "availability_probability": summary(availabilities),
            "success_given_available_probability": summary(successes),
        },
        "notes": [
            "Observed QoS is supplied by StableToolBench at tool level and repeated for its APIs.",
            "Cost is a normalized experimental unit, not a monetary price.",
            "Actual call cost is stateful and depends on included calls, overage, hard limits, and spending caps.",
            "Degraded and outage profiles are synthetic deterministic transformations.",
        ],
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Profiles: {output_path}")


if __name__ == "__main__":
    main()
