"""Compare prompt-regression annotations with their expected relations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("data/retrieval_analysis/v15_regression_pairs.jsonl"),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/annotations/v15_regression_annotations.jsonl"),
    )
    parser.add_argument(
        "--errors",
        type=Path,
        default=Path("data/annotations/v15_regression_errors.jsonl"),
    )
    args = parser.parse_args()

    annotations = {
        row["pair_id"]: row for row in load_jsonl(args.annotations) if row.get("status") == "ok"
    }
    errors = {}
    for row in load_jsonl(args.errors):
        errors.setdefault(row["pair_id"], []).append(row)

    passed = failed = missing = 0
    for pair in load_jsonl(args.pairs):
        pair_id = pair["pair_id"]
        expected = pair["regression_expected"]
        annotation = annotations.get(pair_id)
        if annotation is None:
            missing += 1
            status = "ERROR" if pair_id in errors else "MISSING"
            detail = errors[pair_id][-1]["error"] if pair_id in errors else "no annotation"
            print(f"{status:7} {expected['case']}: {detail}")
            continue
        actual = annotation["annotation"]
        matches = (
            actual["relation"] == expected["relation"]
            and actual["direction"] == expected["direction"]
        )
        if matches:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"
        print(
            f"{status:7} {expected['case']}: "
            f"expected={expected['relation']}/{expected['direction']} "
            f"actual={actual['relation']}/{actual['direction']}"
        )
    total = passed + failed + missing
    print(f"\nPassed: {passed}/{total}; failed: {failed}; missing/errors: {missing}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
