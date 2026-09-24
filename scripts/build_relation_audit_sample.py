"""Build a deterministic targeted sample for auditing strong pair relations."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Callable


STOPWORDS = {
    "a", "an", "and", "api", "by", "for", "from", "get", "in", "of", "on",
    "or", "the", "to", "with",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def tokens(value: Any) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
        if len(token) > 1 and token not in STOPWORDS
    }


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 0.0


def parameter_signature(api: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    parameters = (api.get("required_parameters") or []) + (api.get("optional_parameters") or [])
    return tuple(sorted(
        (str(item.get("name", "")).lower(), str(item.get("type", "")).lower())
        for item in parameters if isinstance(item, dict)
    ))


def features(pair: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> dict[str, Any]:
    left = catalog[pair["left_api_id"]]
    right = catalog[pair["right_api_id"]]
    left_name = tokens(left.get("api_name"))
    right_name = tokens(right.get("api_name"))
    left_desc = tokens(left.get("api_description"))
    right_desc = tokens(right.get("api_description"))
    analysis = pair.get("analysis_features") or {}
    name_similarity = jaccard(left_name, right_name)
    description_similarity = jaccard(left_desc, right_desc)
    left_all = left_name | left_desc
    right_all = right_name | right_desc
    containment = max(
        len(left_all & right_all) / max(1, len(left_all)),
        len(left_all & right_all) / max(1, len(right_all)),
    )
    same_name = str(left.get("api_name", "")).strip().lower() == str(right.get("api_name", "")).strip().lower()
    same_description = str(left.get("api_description", "")).strip().lower() == str(right.get("api_description", "")).strip().lower()
    scope_words = {"all", "list", "latest", "recent", "multiple", "single", "one", "history", "details", "summary"}
    scope_contrast = bool((left_name ^ right_name) & scope_words)
    qwen_similarity = analysis.get("qwen_similarity") or 0.0
    return {
        "same_name": same_name,
        "same_description": same_description,
        "same_signature": parameter_signature(left) == parameter_signature(right),
        "same_tool": analysis.get("same_tool", False),
        "qwen_reciprocal": analysis.get("qwen_reciprocal", False),
        "qwen_similarity": float(qwen_similarity),
        "name_similarity": name_similarity,
        "description_similarity": description_similarity,
        "containment": containment,
        "scope_contrast": scope_contrast,
    }


def score_duplicate(f: dict[str, Any]) -> float:
    return (
        5 * f["same_description"] + 3 * f["same_name"] + 2 * f["same_signature"]
        + f["qwen_reciprocal"] + f["qwen_similarity"]
    )


def score_interchangeable(f: dict[str, Any]) -> float:
    exact_penalty = 5 if f["same_description"] else 0
    return (
        3 * f["name_similarity"] + 2 * f["description_similarity"]
        + 1.5 * f["same_signature"] + f["qwen_reciprocal"]
        + f["qwen_similarity"] - exact_penalty
    )


def score_contains(f: dict[str, Any]) -> float:
    return (
        3 * f["containment"] + 2 * f["scope_contrast"] + 1.5 * f["same_tool"]
        + f["qwen_reciprocal"] + f["qwen_similarity"]
    )


def score_boundary(f: dict[str, Any]) -> float:
    disagreement = 1 if 0.25 <= f["name_similarity"] <= 0.75 else 0
    return f["qwen_similarity"] + f["description_similarity"] + disagreement + 0.5 * f["same_tool"]


def select(
    candidates: list[dict[str, Any]],
    count: int,
    used: set[str],
    scorer: Callable[[dict[str, Any]], float],
    target: str,
) -> list[dict[str, Any]]:
    ranked = sorted(
        (pair for pair in candidates if pair["pair_id"] not in used),
        key=lambda pair: (-scorer(pair["audit_features"]), pair["pair_id"]),
    )
    result = []
    for pair in ranked[:count]:
        row = dict(pair)
        row["audit_sampling"] = {
            "target": target,
            "heuristic_score": round(scorer(pair["audit_features"]), 6),
            "note": "Candidate for auditing; not a ground-truth relation label.",
        }
        result.append(row)
        used.add(pair["pair_id"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=Path("data/retrieval_analysis/pilot_pairs.jsonl"))
    parser.add_argument("--catalog", type=Path, default=Path("data/retrieval/catalog.jsonl"))
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/pilot_pair_annotations.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/retrieval_analysis/relation_audit_pairs.jsonl"))
    parser.add_argument("--per-target", type=int, default=5)
    args = parser.parse_args()

    pairs = load_jsonl(args.pairs)
    catalog = {row["api_id"]: row for row in load_jsonl(args.catalog)}
    completed = {
        row["pair_id"] for row in load_jsonl(args.annotations)
    } if args.annotations.exists() else set()
    candidates = []
    for pair in pairs:
        if pair["pair_id"] in completed:
            continue
        row = dict(pair)
        row["audit_features"] = features(pair, catalog)
        candidates.append(row)

    used: set[str] = set()
    selected = []
    for target, scorer in (
        ("probable_duplicate", score_duplicate),
        ("probable_interchangeable", score_interchangeable),
        ("probable_contains", score_contains),
        ("hard_boundary", score_boundary),
    ):
        selected.extend(select(candidates, args.per_target, used, scorer, target))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for row in selected:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Selected: {len(selected)}")
    print("Targets: " + json.dumps(Counter(row["audit_sampling"]["target"] for row in selected), sort_keys=True))
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
