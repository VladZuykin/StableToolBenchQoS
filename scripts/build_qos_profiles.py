"""Build reproducible empirical-synthetic QoS profiles for solvable APIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
from statistics import mean, median
from typing import Any


PROFILE_VERSION = "qos-profiles-v5"
RESERVED_NAMES = {"from", "class", "return", "false", "true", "id", "and"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def numeric(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def standardize(value: str) -> str:
    result = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9_]", "_", value or "")
    result = re.sub(r"_+", "_", result).strip("_").lower()
    if result and result[0].isdigit():
        result = "get_" + result
    return result


def change_name(value: str) -> str:
    return "is_" + value if value in RESERVED_NAMES else value


def standardize_category(value: str) -> str:
    result = (value or "").replace(" ", "_").replace(",", "_").replace("/", "_")
    return re.sub(r"_+", "_", result)


def query_api_id(api: dict[str, Any]) -> str:
    category = standardize_category(str(api.get("category_name") or ""))
    tool = standardize(str(api.get("tool_name") or ""))
    name = change_name(standardize(str(api.get("api_name") or "")))
    return f"{category}/{tool}/{name}"


def tool_key(api_id: str) -> str:
    return "/".join(api_id.split("/")[:2])


def collect_solvable_api_ids(query_roots: list[Path]) -> tuple[set[str], int]:
    api_ids: set[str] = set()
    query_count = 0
    for root in query_roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(root.rglob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                continue
            for query in data:
                if not isinstance(query, dict) or not isinstance(query.get("api_list"), list):
                    continue
                query_count += 1
                for api in query["api_list"]:
                    if isinstance(api, dict):
                        api_ids.add(query_api_id(api))
    return api_ids, query_count


def stable_rng(seed: int, api_id: str, dimension: str) -> random.Random:
    material = f"{seed}\x1f{api_id}\x1f{dimension}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def bounded(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def positive_number(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if not numeric(value):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    lower_ok = value >= 0 if allow_zero else value > 0
    if not lower_ok:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def validate_generation_config(config: dict[str, Any]) -> None:
    empirical = config.get("empirical_bootstrap") or {}
    synthetic = config.get("synthetic") or {}
    scenarios = config.get("scenarios") or {}

    success = empirical.get("success_rate") or {}
    if success.get("formula") != "service_level_times_success_rate":
        raise ValueError(
            "empirical_bootstrap.success_rate.formula must be "
            "'service_level_times_success_rate'"
        )
    success_min = positive_number(
        success.get("min"), "success_rate.min", allow_zero=True
    )
    success_max = positive_number(success.get("max"), "success_rate.max")
    if not 0 <= success_min < success_max <= 1:
        raise ValueError("success_rate bounds must satisfy 0 <= min < max <= 1")

    latency = empirical.get("expected_latency_ms") or {}
    if latency.get("field") != "avgLatency":
        raise ValueError("empirical expected latency field must be avgLatency")
    latency_min = positive_number(latency.get("min"), "expected_latency_ms.min")
    latency_max = positive_number(latency.get("max"), "expected_latency_ms.max")
    if latency_min >= latency_max:
        raise ValueError("expected_latency_ms.min must be less than max")

    cost = synthetic.get("cost_per_call_units") or {}
    if cost.get("distribution") != "lognormal":
        raise ValueError("cost_per_call_units.distribution must be 'lognormal'")
    positive_number(cost.get("median"), "cost_per_call_units.median")
    positive_number(cost.get("log_sigma"), "cost_per_call_units.log_sigma")
    cost_min = positive_number(
        cost.get("min"), "cost_per_call_units.min", allow_zero=True
    )
    cost_max = positive_number(cost.get("max"), "cost_per_call_units.max")
    if cost_min >= cost_max:
        raise ValueError("cost_per_call_units.min must be less than max")

    jitter = synthetic.get("latency_log_sigma") or {}
    if jitter.get("distribution") != "uniform":
        raise ValueError("latency_log_sigma.distribution must be 'uniform'")
    jitter_min = positive_number(
        jitter.get("min"), "latency_log_sigma.min", allow_zero=True
    )
    jitter_max = positive_number(jitter.get("max"), "latency_log_sigma.max")
    if jitter_min > jitter_max:
        raise ValueError("latency_log_sigma.min must not exceed max")

    required_scenarios = {"normal", "degraded", "outage"}
    if set(scenarios) != required_scenarios:
        raise ValueError(f"scenarios must be exactly {sorted(required_scenarios)}")
    for scenario_name, values in scenarios.items():
        for factor in ("success_rate_factor", "latency_factor", "cost_factor"):
            positive_number(
                values.get(factor),
                f"scenarios.{scenario_name}.{factor}",
                allow_zero=factor != "latency_factor",
            )


def build_empirical_pool(catalog_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one complete observed-QoS record per StableToolBench tool."""
    donors_by_tool: dict[str, dict[str, Any]] = {}
    for row in catalog_rows:
        qos = row.get("qos") or {}
        if (
            numeric(qos.get("avgLatency"))
            and float(qos["avgLatency"]) > 0
            and numeric(qos.get("avgServiceLevel"))
            and numeric(qos.get("avgSuccessRate"))
        ):
            donors_by_tool.setdefault(tool_key(row["api_id"]), row)
    return sorted(donors_by_tool.values(), key=lambda row: row["api_id"])


def select_empirical_donor(
    api_id: str,
    donors: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    if not donors:
        raise ValueError("QoS catalog has no complete empirical donor")
    # Avoid assigning an endpoint its own tool-level metrics. The resulting
    # value is a bootstrap sample from the population, not a direct lookup.
    candidates = [row for row in donors if tool_key(row["api_id"]) != tool_key(api_id)]
    if not candidates:
        candidates = donors
    rng = stable_rng(seed, api_id, "empirical_qos_donor")
    return candidates[rng.randrange(len(candidates))]


def sample_lognormal(rng: random.Random, config: dict[str, Any]) -> float:
    value = rng.lognormvariate(
        math.log(float(config["median"])),
        float(config["log_sigma"]),
    )
    return bounded(value, float(config["min"]), float(config["max"]))


def generate_baseline(
    api_id: str,
    donor: dict[str, Any],
    seed: int,
    config: dict[str, Any],
) -> dict[str, float]:
    donor_qos = donor["qos"]
    empirical = config["empirical_bootstrap"]
    success_config = empirical["success_rate"]
    success_rate = (
        bounded(float(donor_qos["avgServiceLevel"]) / 100.0, 0.0, 1.0)
        * bounded(float(donor_qos["avgSuccessRate"]) / 100.0, 0.0, 1.0)
    )
    success_rate = bounded(
        success_rate,
        float(success_config["min"]),
        float(success_config["max"]),
    )

    latency_config = empirical["expected_latency_ms"]
    expected_latency_ms = bounded(
        float(donor_qos["avgLatency"]),
        float(latency_config["min"]),
        float(latency_config["max"]),
    )

    synthetic = config["synthetic"]
    cost_per_call_units = sample_lognormal(
        stable_rng(seed, api_id, "cost_per_call_units"),
        synthetic["cost_per_call_units"],
    )
    jitter = synthetic["latency_log_sigma"]
    latency_log_sigma = stable_rng(seed, api_id, "latency_log_sigma").uniform(
        float(jitter["min"]),
        float(jitter["max"]),
    )
    return {
        "success_rate": round(success_rate, 8),
        "expected_latency_ms": round(expected_latency_ms, 4),
        "cost_per_call_units": round(cost_per_call_units, 10),
        "latency_log_sigma": round(latency_log_sigma, 8),
    }


def build_scenarios(
    baseline: dict[str, float],
    config: dict[str, Any],
) -> dict[str, dict[str, float]]:
    profiles = {}
    for name, factors in config["scenarios"].items():
        profiles[name] = {
            "success_rate": round(
                bounded(
                    baseline["success_rate"] * float(factors["success_rate_factor"]),
                    0.0,
                    1.0,
                ),
                8,
            ),
            "expected_latency_ms": round(
                max(
                    1.0,
                    baseline["expected_latency_ms"] * float(factors["latency_factor"]),
                ),
                4,
            ),
            "cost_per_call_units": round(
                max(
                    0.0,
                    baseline["cost_per_call_units"] * float(factors["cost_factor"]),
                ),
                10,
            ),
        }
    return profiles


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


def empirical_values(donors: list[dict[str, Any]]) -> dict[str, list[float]]:
    success_rates = []
    latencies = []
    for donor in donors:
        qos = donor["qos"]
        success_rates.append(
            bounded(float(qos["avgServiceLevel"]) / 100.0, 0.0, 1.0)
            * bounded(float(qos["avgSuccessRate"]) / 100.0, 0.0, 1.0)
        )
        latencies.append(float(qos["avgLatency"]))
    return {"success_rate": success_rates, "expected_latency_ms": latencies}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-root",
        action="append",
        type=Path,
        help=(
            "Solvable query root; repeat to use several roots. Defaults to "
            "solvable_queries and solvable_queries_example."
        ),
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/catalog/tools.jsonl"),
        help="Catalog used only to estimate the empirical QoS distribution.",
    )
    parser.add_argument(
        "--api-to-cluster",
        type=Path,
        help=(
            "Optional API-to-cluster JSONL. When supplied, graph APIs are added "
            "to the solvable target set and receive cluster_id values."
        ),
    )
    parser.add_argument(
        "--generation-config",
        type=Path,
        default=Path("configs/qos_generation_v1.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/qos/v5"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    query_roots = args.query_root or [
        Path("solvable_queries"),
        Path("solvable_queries_example"),
    ]
    solvable_api_ids, query_count = collect_solvable_api_ids(query_roots)
    cluster_by_api = (
        {
            row["api_id"]: row["cluster_id"]
            for row in load_jsonl(args.api_to_cluster)
        }
        if args.api_to_cluster
        else {}
    )
    target_api_ids = set(cluster_by_api) | solvable_api_ids

    catalog_rows = load_jsonl(args.catalog)
    donors = build_empirical_pool(catalog_rows)
    if not donors:
        raise SystemExit("Catalog contains no complete tool-level QoS records")
    config_bytes = args.generation_config.read_bytes()
    config = json.loads(config_bytes)
    validate_generation_config(config)

    records = []
    generated_success_rates = []
    generated_latencies = []
    generated_costs = []
    generated_sigmas = []
    for api_id in sorted(target_api_ids):
        donor = select_empirical_donor(api_id, donors, args.seed)
        baseline = generate_baseline(api_id, donor, args.seed, config)
        donor_qos = donor["qos"]
        records.append(
            {
                "api_id": api_id,
                "cluster_id": cluster_by_api.get(api_id),
                "profile_version": PROFILE_VERSION,
                "seed": args.seed,
                "source": {
                    "kind": "empirical_bootstrap",
                    "catalog": str(args.catalog),
                    "donor_api_id": donor["api_id"],
                    "donor_tool_id": tool_key(donor["api_id"]),
                    "uses_endpoint_own_observed_qos": False,
                    "donor_avg_latency_ms": float(donor_qos["avgLatency"]),
                    "donor_avg_service_level_percent": float(
                        donor_qos["avgServiceLevel"]
                    ),
                    "donor_avg_success_rate_percent": float(
                        donor_qos["avgSuccessRate"]
                    ),
                },
                "simulation": {
                    "latency_distribution": "lognormal",
                    "latency_log_sigma": baseline["latency_log_sigma"],
                    "baseline": {
                        "success_rate": baseline["success_rate"],
                        "expected_latency_ms": baseline["expected_latency_ms"],
                        "cost_per_call_units": baseline["cost_per_call_units"],
                    },
                    "profiles": build_scenarios(baseline, config),
                },
            }
        )
        generated_success_rates.append(baseline["success_rate"])
        generated_latencies.append(baseline["expected_latency_ms"])
        generated_costs.append(baseline["cost_per_call_units"])
        generated_sigmas.append(baseline["latency_log_sigma"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "api_qos_profiles.jsonl"
    temporary = output_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(output_path)

    empirical = empirical_values(donors)
    metadata = {
        "profile_version": PROFILE_VERSION,
        "seed": args.seed,
        "api_count": len(records),
        "solvable_api_count": len(solvable_api_ids),
        "solvable_profiles_written": len(solvable_api_ids & target_api_ids),
        "solvable_query_count": query_count,
        "solvable_query_roots": [str(path) for path in query_roots],
        "graph_api_count": len(cluster_by_api),
        "graph_solvable_overlap": len(set(cluster_by_api) & solvable_api_ids),
        "cluster_mapping": str(args.api_to_cluster) if args.api_to_cluster else None,
        "target_union_api_count": len(target_api_ids),
        "target_fingerprint_sha256": hashlib.sha256(
            "\n".join(sorted(target_api_ids)).encode("utf-8")
        ).hexdigest(),
        "generation": {
            "kind": "empirical_bootstrap",
            "config": str(args.generation_config),
            "config_version": config["version"],
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "catalog": str(args.catalog),
            "empirical_donor_tool_count": len(donors),
            "uses_endpoint_own_observed_qos": False,
            "success_and_latency_share_donor": True,
            "cost_source": "synthetic_lognormal",
        },
        "empirical_pool_summary": {
            "success_rate": summary(empirical["success_rate"]),
            "expected_latency_ms": summary(empirical["expected_latency_ms"]),
        },
        "generated_baseline_summary": {
            "success_rate": summary(generated_success_rates),
            "expected_latency_ms": summary(generated_latencies),
            "cost_per_call_units": summary(generated_costs),
            "latency_log_sigma": summary(generated_sigmas),
        },
        "notes": [
            "StableToolBench tool-level QoS defines only the empirical source distribution.",
            "Each target API samples another tool as a deterministic bootstrap donor.",
            "Success rate equals donor service level times donor conditional success rate.",
            "Expected latency comes from the same donor to preserve their empirical relationship.",
            "Per-call latency is log-normal around the generated expected latency.",
            "Cost is synthetic because StableToolBench contains no numeric price distribution.",
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
