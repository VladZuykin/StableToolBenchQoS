"""Build a human-review queue from direct relation-graph decisions."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def prompt_version(row: dict[str, Any]) -> str:
    provenance = row.get("provenance") or {}
    return str(
        provenance.get("prompt_version")
        or (provenance.get("source_provenance") or {}).get("prompt_version")
        or "unknown"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decisions",
        type=Path,
        default=Path("data/relation_graph/v16_migrated/pair_decisions.jsonl"),
    )
    parser.add_argument(
        "--existing-reviews",
        action="append",
        type=Path,
        help="JSONL review log to exclude; may be repeated.",
    )
    parser.add_argument(
        "--relation",
        action="append",
        choices=("interchangeable", "contains", "different_capability"),
        help="Relation to include; may be repeated. Defaults to interchangeable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/annotations/graph_review_queue.jsonl"),
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    review_paths = args.existing_reviews or [
        Path("data/annotations/human_pair_reviews.jsonl"),
        Path("data/annotations/human_graph_reviews.jsonl"),
    ]
    reviewed = {
        row["pair_id"]
        for path in review_paths
        for row in load_jsonl(path)
        if row.get("pair_id")
    }
    relations = set(args.relation or ["interchangeable"])
    selected = []
    seen = set()
    for row in load_jsonl(args.decisions):
        pair_id = row.get("pair_id")
        annotation = row.get("annotation") or {}
        if (
            row.get("status") != "ok"
            or row.get("decision_source") not in {"llm", "migrated_seed"}
            or annotation.get("relation") not in relations
            or not pair_id
            or pair_id in reviewed
            or pair_id in seen
        ):
            continue
        item = dict(row)
        provenance = dict(item.get("provenance") or {})
        provenance["prompt_version"] = prompt_version(item)
        item["provenance"] = provenance
        selected.append(item)
        seen.add(pair_id)

    selected.sort(
        key=lambda row: (
            0 if row["annotation"].get("needs_human_review") else 1,
            float(row["annotation"].get("confidence") or 0),
            row["pair_id"],
        )
    )
    eligible = len(selected)
    if args.limit is not None:
        selected = selected[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for row in selected:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    print(json.dumps({
        "decisions": str(args.decisions),
        "relations": sorted(relations),
        "review_logs": [str(path) for path in review_paths],
        "already_reviewed_pair_count": len(reviewed),
        "eligible_pair_count": eligible,
        "output_pair_count": len(selected),
        "output": str(args.output),
        "relation_counts": dict(Counter(
            row["annotation"]["relation"] for row in selected
        )),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
