"""Join pair annotations with API documentation into a human-review CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def as_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def load_existing_human_fields(path: Path, fields: list[str]) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = csv.DictReader(source)
        return {
            row["pair_id"]: {field: row.get(field, "") for field in fields}
            for row in rows
            if row.get("pair_id")
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/pilot_pair_annotations.jsonl"))
    parser.add_argument("--catalog", type=Path, default=Path("data/retrieval/catalog.jsonl"))
    parser.add_argument("--pilot-pairs", type=Path, default=Path("data/retrieval_analysis/pilot_pairs.jsonl"))
    parser.add_argument("--audit-pairs", type=Path, default=Path("data/retrieval_analysis/relation_audit_pairs.jsonl"))
    parser.add_argument("--reviews", type=Path, default=Path("data/annotations/human_pair_reviews.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/annotations/pilot_annotation_review.csv"))
    args = parser.parse_args()

    catalog = {row["api_id"]: row for row in load_jsonl(args.catalog)}
    pairs = {
        row["pair_id"]: row
        for path in (args.pilot_pairs, args.audit_pairs)
        for row in load_jsonl(path)
    }
    annotations = load_jsonl(args.annotations)
    annotations.sort(key=lambda record: (
        0 if (pairs.get(record["pair_id"], {}).get("audit_sampling") or {}).get("target") else 1,
        0 if record["annotation"].get("needs_human_review") else 1,
        record["pair_id"],
    ))
    human_fields = [
        "human_relation", "human_direction", "human_relation_correct",
        "human_comment",
    ]
    existing_human = load_existing_human_fields(args.output, human_fields)
    for review in load_jsonl(args.reviews):
        existing_human[review["pair_id"]] = {
            field: review.get(field, "") for field in human_fields
        }
    fields = [
        "pair_id", "prompt_version", "retrieval_stratum", "audit_target",
        "left_api_id", "left_tool_name", "left_api_name", "left_api_description",
        "left_required_parameters", "left_optional_parameters",
        "right_api_id", "right_tool_name", "right_api_name", "right_api_description",
        "right_required_parameters", "right_optional_parameters",
        "llm_relation", "llm_direction", "llm_confidence", "needs_human_review",
        "reasoning", "evidence", *human_fields,
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for record in annotations:
            annotation = record["annotation"]
            pair = pairs.get(record["pair_id"], {})
            left = catalog[record["left_api_id"]]
            right = catalog[record["right_api_id"]]
            row = {
                "pair_id": record["pair_id"],
                "prompt_version": record["provenance"]["prompt_version"],
                "retrieval_stratum": (pair.get("sampling") or {}).get("stratum", ""),
                "audit_target": (pair.get("audit_sampling") or {}).get("target", ""),
                "left_api_id": record["left_api_id"],
                "left_tool_name": left.get("tool_name", ""),
                "left_api_name": left.get("api_name", ""),
                "left_api_description": left.get("api_description", ""),
                "left_required_parameters": as_json(left.get("required_parameters") or []),
                "left_optional_parameters": as_json(left.get("optional_parameters") or []),
                "right_api_id": record["right_api_id"],
                "right_tool_name": right.get("tool_name", ""),
                "right_api_name": right.get("api_name", ""),
                "right_api_description": right.get("api_description", ""),
                "right_required_parameters": as_json(right.get("required_parameters") or []),
                "right_optional_parameters": as_json(right.get("optional_parameters") or []),
                "llm_relation": annotation["relation"],
                "llm_direction": annotation["direction"],
                "llm_confidence": annotation["confidence"],
                "needs_human_review": annotation["needs_human_review"],
                "reasoning": annotation["reasoning"],
                "evidence": as_json(annotation["evidence"]),
            }
            row.update(existing_human.get(record["pair_id"], {}))
            writer.writerow(row)
    print(f"Rows: {len(annotations)}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
