"""Rebuild the exact pair set referenced by an existing annotation JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/pilot_pair_annotations.jsonl"))
    parser.add_argument("--pilot-pairs", type=Path, default=Path("data/retrieval_analysis/pilot_pairs.jsonl"))
    parser.add_argument("--audit-pairs", type=Path, default=Path("data/retrieval_analysis/relation_audit_pairs.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/retrieval_analysis/reannotation_80_pairs.jsonl"))
    args = parser.parse_args()

    annotations = load_jsonl(args.annotations)
    pairs = {
        row["pair_id"]: row
        for path in (args.pilot_pairs, args.audit_pairs)
        for row in load_jsonl(path)
    }
    pair_ids = []
    seen = set()
    for annotation in annotations:
        pair_id = annotation["pair_id"]
        if pair_id not in seen:
            pair_ids.append(pair_id)
            seen.add(pair_id)
    missing = [pair_id for pair_id in pair_ids if pair_id not in pairs]
    if missing:
        raise KeyError(f"Pairs absent from input pools: {missing}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for pair_id in pair_ids:
            output.write(json.dumps(pairs[pair_id], ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Pairs: {len(pair_ids)}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
