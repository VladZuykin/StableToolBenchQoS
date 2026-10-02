"""Create a relation-decision log with latest human reviews applied as overrides."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


DIRECT_SOURCES = {"llm", "migrated_seed", "human"}
VERSION = "human-relation-overrides-v1"


def canonical_relation(value: str) -> str:
    if value == "exact_duplicate":
        return "interchangeable"
    if value == "uncertain":
        return "different_capability"
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decisions",
        type=Path,
        default=Path("data/relation_graph/v16_migrated/pair_decisions.jsonl"),
    )
    parser.add_argument(
        "--reviews",
        action="append",
        type=Path,
        help="Human-review JSONL; may be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/relation_graph/v16_human"),
    )
    args = parser.parse_args()

    review_paths = args.reviews or [
        Path("data/annotations/human_pair_reviews.jsonl"),
        Path("data/annotations/human_graph_reviews.jsonl"),
    ]
    latest = {}
    event_count = 0
    for path in review_paths:
        for review in load_jsonl(path):
            event_count += 1
            latest[review["pair_id"]] = {**review, "_source_file": str(path)}

    rows = load_jsonl(args.decisions)
    matched = set()
    changed = confirmed = 0
    relation_changes = Counter()
    output_rows = []
    for row in rows:
        review = latest.get(row.get("pair_id"))
        if (
            review is None
            or row.get("status") != "ok"
            or row.get("decision_source") not in DIRECT_SOURCES
        ):
            output_rows.append(row)
            continue

        matched.add(row["pair_id"])
        old_annotation = row["annotation"]
        old_relation = old_annotation["relation"]
        old_direction = old_annotation.get("direction", "none")
        new_relation = canonical_relation(review["human_relation"])
        new_direction = (
            review.get("human_direction", "none")
            if new_relation == "contains"
            else "none"
        )
        if (old_relation, old_direction) == (new_relation, new_direction):
            confirmed += 1
        else:
            changed += 1
            relation_changes[f"{old_relation}->{new_relation}"] += 1

        updated = dict(row)
        annotation = dict(old_annotation)
        annotation["relation"] = new_relation
        annotation["direction"] = new_direction
        annotation["needs_human_review"] = False
        updated["annotation"] = annotation
        updated["decision_source"] = "human"
        provenance = dict(updated.get("provenance") or {})
        provenance["human_override"] = {
            "version": VERSION,
            "review_file": review["_source_file"],
            "reviewer": review.get("reviewer", ""),
            "reviewed_at": review.get("reviewed_at"),
            "human_comment": review.get("human_comment", ""),
            "original_relation": old_relation,
            "original_direction": old_direction,
        }
        updated["provenance"] = provenance
        output_rows.append(updated)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = args.output_dir / "pair_decisions.jsonl"
    with decisions_path.open("w", encoding="utf-8", newline="\n") as output:
        for row in output_rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    metadata = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_decisions": str(args.decisions),
        "review_files": [str(path) for path in review_paths],
        "source_decision_count": len(rows),
        "review_event_count": event_count,
        "unique_review_pair_count": len(latest),
        "matched_review_count": len(matched),
        "confirmed_count": confirmed,
        "changed_count": changed,
        "relation_changes": dict(sorted(relation_changes.items())),
        "unmatched_review_count": len(set(latest) - matched),
        "output": str(decisions_path),
    }
    (args.output_dir / "human_override_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
