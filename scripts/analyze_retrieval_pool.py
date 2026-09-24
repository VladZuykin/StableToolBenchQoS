"""Analyze a retrieval pool and create a reproducible stratified pilot sample."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
from tqdm.auto import tqdm


STRATA = (
    "qwen_mutual_top3",
    "qwen_mutual_rank4_10",
    "qwen_mutual_rank11_30",
    "qwen_oneway_top3",
    "qwen_oneway_rank4_10",
    "qwen_oneway_rank11_30",
    "bm25_only_top10",
    "bm25_only_rank11_30",
)


def load_catalog(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            api_id = record["api_id"]
            if api_id in result:
                raise ValueError(f"Duplicate api_id on catalog line {line_number}: {api_id}")
            result[api_id] = record
    return result


def load_pairs(path: Path, show_progress: bool) -> list[dict[str, Any]]:
    pairs = []
    with path.open(encoding="utf-8") as source:
        for line in tqdm(source, desc="Reading retrieval pairs", unit="pair", disable=not show_progress):
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def dense_method(retrieval: dict[str, Any]) -> str | None:
    methods = [method for method in retrieval if method.startswith("dense:")]
    if len(methods) > 1:
        raise ValueError(f"Expected at most one dense method, got: {methods}")
    return methods[0] if methods else None


def evidence_directions(evidence: Iterable[dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (item["source_api_id"], item["candidate_api_id"])
        for item in evidence
    }


def parameter_signature(parameters: Iterable[dict[str, Any]]) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (
                str(parameter.get("name") or "").strip().lower(),
                str(parameter.get("type") or "UNKNOWN").strip().upper(),
            )
            for parameter in parameters
            if isinstance(parameter, dict)
        )
    )


def pair_features(pair: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> dict[str, Any]:
    left = catalog[pair["left_api_id"]]
    right = catalog[pair["right_api_id"]]
    retrieval = pair["retrieval"]
    dense = dense_method(retrieval)
    dense_evidence = retrieval.get(dense, []) if dense else []
    bm25_evidence = retrieval.get("bm25", [])
    dense_ranks = [int(item["rank"]) for item in dense_evidence]
    dense_scores = [float(item["score"]) for item in dense_evidence]
    bm25_ranks = [int(item["rank"]) for item in bm25_evidence]
    bm25_scores = [float(item["score"]) for item in bm25_evidence]
    dense_reciprocal = len(evidence_directions(dense_evidence)) == 2
    bm25_reciprocal = len(evidence_directions(bm25_evidence)) == 2
    required_left = parameter_signature(left.get("required_parameters") or [])
    required_right = parameter_signature(right.get("required_parameters") or [])
    optional_left = parameter_signature(left.get("optional_parameters") or [])
    optional_right = parameter_signature(right.get("optional_parameters") or [])
    left_names = {name for name, _ in required_left + optional_left if name}
    right_names = {name for name, _ in required_right + optional_right if name}
    return {
        "dense_method": dense,
        "qwen_present": bool(dense_evidence),
        "qwen_reciprocal": dense_reciprocal,
        "qwen_best_rank": min(dense_ranks) if dense_ranks else None,
        "qwen_worst_rank": max(dense_ranks) if dense_ranks else None,
        "qwen_similarity": max(dense_scores) if dense_scores else None,
        "bm25_present": bool(bm25_evidence),
        "bm25_reciprocal": bm25_reciprocal,
        "bm25_best_rank": min(bm25_ranks) if bm25_ranks else None,
        "bm25_worst_rank": max(bm25_ranks) if bm25_ranks else None,
        "bm25_max_score": max(bm25_scores) if bm25_scores else None,
        "same_category": left.get("category") == right.get("category"),
        "same_tool": (
            left.get("category") == right.get("category")
            and left.get("tool_id") == right.get("tool_id")
        ),
        "same_required_signature": required_left == required_right,
        "same_optional_signature": optional_left == optional_right,
        "shared_parameter_names": sorted(left_names & right_names),
    }


def assign_stratum(features: dict[str, Any]) -> str:
    if features["qwen_present"]:
        best_rank = int(features["qwen_best_rank"])
        prefix = "qwen_mutual" if features["qwen_reciprocal"] else "qwen_oneway"
        if best_rank <= 3:
            return f"{prefix}_top3"
        if best_rank <= 10:
            return f"{prefix}_rank4_10"
        return f"{prefix}_rank11_30"
    best_rank = int(features["bm25_best_rank"])
    if best_rank <= 10:
        return "bm25_only_top10"
    return "bm25_only_rank11_30"


def quantiles(values: Iterable[float]) -> dict[str, float] | None:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return None
    return {
        "min": float(array.min()),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def stable_rng(seed: int, name: str) -> random.Random:
    suffix = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big")
    return random.Random(seed ^ suffix)


def stratified_sample(
    enriched_pairs: list[dict[str, Any]],
    sample_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in enriched_pairs:
        by_stratum[pair["sampling_stratum"]].append(pair)
    available_strata = [name for name in STRATA if by_stratum[name]]
    if not available_strata:
        return [], {}
    target_size = min(sample_size, len(enriched_pairs))
    base, remainder = divmod(target_size, len(available_strata))
    selected = []
    selected_ids = set()
    requested: dict[str, int] = {}
    for position, name in enumerate(available_strata):
        quota = base + (1 if position < remainder else 0)
        requested[name] = quota
        candidates = sorted(by_stratum[name], key=lambda item: item["pair_id"])
        stable_rng(seed, name).shuffle(candidates)
        for pair in candidates[:quota]:
            selected.append(pair)
            selected_ids.add(pair["pair_id"])

    missing = target_size - len(selected)
    if missing:
        remaining = sorted(
            (pair for pair in enriched_pairs if pair["pair_id"] not in selected_ids),
            key=lambda item: item["pair_id"],
        )
        stable_rng(seed, "redistribution").shuffle(remaining)
        selected.extend(remaining[:missing])
    selected.sort(key=lambda item: (item["sampling_stratum"], item["pair_id"]))
    return selected, requested


def write_pilot_jsonl(pairs: list[dict[str, Any]], path: Path, seed: int) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for pair in pairs:
            row = dict(pair)
            row["sampling"] = {"stratum": row.pop("sampling_stratum"), "seed": seed}
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_review_csv(
    pairs: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    fieldnames = [
        "pair_id", "sampling_stratum", "left_api_id", "left_tool_name", "left_api_name",
        "left_api_description", "left_required_parameters", "left_optional_parameters",
        "right_api_id", "right_tool_name", "right_api_name", "right_api_description",
        "right_required_parameters", "right_optional_parameters", "qwen_reciprocal",
        "qwen_best_rank", "qwen_worst_rank", "qwen_similarity", "bm25_reciprocal",
        "bm25_best_rank", "bm25_worst_rank", "bm25_max_score", "same_category",
        "same_tool", "same_required_signature", "same_optional_signature",
        "shared_parameter_names", "manual_relation", "manual_direction", "comment",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            left = catalog[pair["left_api_id"]]
            right = catalog[pair["right_api_id"]]
            features = pair["analysis_features"]
            writer.writerow(
                {
                    "pair_id": pair["pair_id"],
                    "sampling_stratum": pair["sampling_stratum"],
                    "left_api_id": pair["left_api_id"],
                    "left_tool_name": left.get("tool_name"),
                    "left_api_name": left.get("api_name"),
                    "left_api_description": left.get("api_description"),
                    "left_required_parameters": json_cell(left.get("required_parameters") or []),
                    "left_optional_parameters": json_cell(left.get("optional_parameters") or []),
                    "right_api_id": pair["right_api_id"],
                    "right_tool_name": right.get("tool_name"),
                    "right_api_name": right.get("api_name"),
                    "right_api_description": right.get("api_description"),
                    "right_required_parameters": json_cell(right.get("required_parameters") or []),
                    "right_optional_parameters": json_cell(right.get("optional_parameters") or []),
                    "qwen_reciprocal": features["qwen_reciprocal"],
                    "qwen_best_rank": features["qwen_best_rank"],
                    "qwen_worst_rank": features["qwen_worst_rank"],
                    "qwen_similarity": features["qwen_similarity"],
                    "bm25_reciprocal": features["bm25_reciprocal"],
                    "bm25_best_rank": features["bm25_best_rank"],
                    "bm25_worst_rank": features["bm25_worst_rank"],
                    "bm25_max_score": features["bm25_max_score"],
                    "same_category": features["same_category"],
                    "same_tool": features["same_tool"],
                    "same_required_signature": features["same_required_signature"],
                    "same_optional_signature": features["same_optional_signature"],
                    "shared_parameter_names": json_cell(features["shared_parameter_names"]),
                    "manual_relation": "",
                    "manual_direction": "",
                    "comment": "",
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-dir", type=Path, default=Path("data/retrieval"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/retrieval_analysis"))
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_size < 1:
        raise SystemExit("--sample-size must be positive")
    catalog_path = args.retrieval_dir / "catalog.jsonl"
    pairs_path = args.retrieval_dir / "pooled_pairs.jsonl"
    if not catalog_path.is_file() or not pairs_path.is_file():
        raise SystemExit(f"Expected catalog.jsonl and pooled_pairs.jsonl in {args.retrieval_dir}")

    show_progress = not args.no_progress
    catalog = load_catalog(catalog_path)
    pairs = load_pairs(pairs_path, show_progress)
    enriched_pairs = []
    stratum_counts: Counter[str] = Counter()
    dense_scores: list[float] = []
    bm25_scores: list[float] = []
    for pair in tqdm(pairs, desc="Deriving pair features", unit="pair", disable=not show_progress):
        features = pair_features(pair, catalog)
        stratum = assign_stratum(features)
        enriched = dict(pair)
        enriched["analysis_features"] = features
        enriched["sampling_stratum"] = stratum
        enriched_pairs.append(enriched)
        stratum_counts[stratum] += 1
        if features["qwen_similarity"] is not None:
            dense_scores.append(features["qwen_similarity"])
        if features["bm25_max_score"] is not None:
            bm25_scores.append(features["bm25_max_score"])

    sample, requested = stratified_sample(enriched_pairs, args.sample_size, args.seed)
    selected_counts = Counter(pair["sampling_stratum"] for pair in sample)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = args.output_dir / "retrieval_analysis.json"
    pilot_path = args.output_dir / "pilot_pairs.jsonl"
    review_path = args.output_dir / "pilot_review.csv"
    analysis = {
        "retrieval_dir": str(args.retrieval_dir),
        "catalog_apis": len(catalog),
        "unique_pairs": len(pairs),
        "seed": args.seed,
        "requested_sample_size": args.sample_size,
        "actual_sample_size": len(sample),
        "strata_order": list(STRATA),
        "stratum_population_counts": dict(sorted(stratum_counts.items())),
        "stratum_requested_quotas": requested,
        "stratum_selected_counts": dict(sorted(selected_counts.items())),
        "qwen_similarity_quantiles": quantiles(dense_scores),
        "bm25_max_score_quantiles": quantiles(bm25_scores),
        "outputs": {
            "pilot_pairs": str(pilot_path),
            "pilot_review": str(review_path),
        },
    }
    analysis_path.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_pilot_jsonl(sample, pilot_path, args.seed)
    write_review_csv(sample, catalog, review_path)
    print(f"Unique pairs analyzed: {len(pairs)}")
    print(f"Pilot pairs selected: {len(sample)}")
    print(f"Analysis: {analysis_path}")
    print(f"Pilot JSONL: {pilot_path}")
    print(f"Review CSV: {review_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
