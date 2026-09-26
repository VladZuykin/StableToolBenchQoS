"""Seed a new relation-graph run with safe direct decisions from an older prompt."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path


SAFE_RELATIONS = {"interchangeable", "contains"}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/relation_graph/pair_decisions.jsonl"),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("data/relation_graph/v16_migrated"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.source)
    direct = [
        row for row in rows
        if row.get("status") == "ok" and row.get("decision_source") == "llm"
    ]
    safe = [row for row in direct if row["annotation"]["relation"] in SAFE_RELATIONS]
    reconsider = [
        row for row in direct if row["annotation"]["relation"] == "different_capability"
    ]
    ignored = [
        row for row in direct if row["annotation"]["relation"] not in SAFE_RELATIONS | {"different_capability"}
    ]
    counts = Counter(row["annotation"]["relation"] for row in direct)

    print(f"Source direct LLM decisions: {len(direct)}")
    print("Source classes: " + json.dumps(dict(sorted(counts.items())), sort_keys=True))
    print(f"Reusable direct decisions: {len(safe)}")
    print(f"To reconsider with the new prompt: {len(reconsider)}")
    print(f"Ignored unknown classes: {len(ignored)}")
    print(f"Source inferred decisions not copied: {len(rows) - len(direct)}")
    print(f"Destination: {args.run_dir}")
    if args.dry_run:
        return

    output = args.run_dir / "pair_decisions.jsonl"
    errors = args.run_dir / "errors.jsonl"
    checkpoint = args.run_dir / "checkpoint.json"
    existing = [path for path in (output, errors, checkpoint) if path.exists()]
    if existing:
        raise SystemExit(
            "Destination is not empty; refusing to overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    migrated_at = datetime.now(timezone.utc).isoformat()
    with output.open("w", encoding="utf-8", newline="\n") as destination:
        for row in safe:
            migrated = {
                "decision_id": row["decision_id"],
                "pair_id": row["pair_id"],
                "left_api_id": row["left_api_id"],
                "right_api_id": row["right_api_id"],
                "status": "ok",
                "decision_source": "migrated_seed",
                "annotation": row["annotation"],
                "provenance": {
                    "migration_rule": "reuse_interchangeable_and_contains",
                    "migrated_at": migrated_at,
                    "source_file": str(args.source),
                    "source_decision_id": row["decision_id"],
                    "source_provenance": row.get("provenance"),
                },
                "request_count": 0,
            }
            destination.write(json.dumps(migrated, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Written migrated seeds: {len(safe)}")


if __name__ == "__main__":
    main()
