"""Estimate a Qwen candidate-filter threshold from human-reviewed API pairs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEWS = ROOT / "data/annotations/human_pair_reviews.jsonl"
DEFAULT_PAIRS = ROOT / "data/retrieval/pooled_pairs.jsonl"
QWEN_PREFIX = "dense:Qwen/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    return parser.parse_args()


def canonical_relation(value: str) -> str:
    if value == "exact_duplicate":
        return "interchangeable"
    return value


def load_latest_reviews(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                record = json.loads(line)
                latest[record["pair_id"]] = record
    return latest


def qwen_score(pair: dict[str, Any]) -> float | None:
    scores = [
        float(item["score"])
        for method, items in pair.get("retrieval", {}).items()
        if method.startswith(QWEN_PREFIX)
        for item in items
    ]
    return max(scores) if scores else None


def load_scores(
    path: Path, reviewed_ids: set[str]
) -> tuple[dict[str, float | None], list[float | None]]:
    reviewed_scores: dict[str, float | None] = {}
    all_scores: list[float | None] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            pair = json.loads(line)
            score = qwen_score(pair)
            all_scores.append(score)
            if pair["pair_id"] in reviewed_ids:
                reviewed_scores[pair["pair_id"]] = score
    return reviewed_scores, all_scores


def recall_at_threshold(positive_scores: list[float | None], threshold: float) -> float:
    return sum(score is not None and score >= threshold for score in positive_scores) / len(
        positive_scores
    )


def best_threshold(positive_scores: list[float | None], target_recall: float) -> float | None:
    candidates = sorted({score for score in positive_scores if score is not None}, reverse=True)
    valid = [
        threshold
        for threshold in candidates
        if recall_at_threshold(positive_scores, threshold) >= target_recall
    ]
    return max(valid) if valid else None


def main() -> None:
    args = parse_args()
    reviews = load_latest_reviews(args.reviews)
    reviewed_scores, all_scores = load_scores(args.pairs, set(reviews))

    rows = []
    for pair_id, review in reviews.items():
        rows.append(
            {
                "pair_id": pair_id,
                "relation": canonical_relation(review["human_relation"]),
                "score": reviewed_scores.get(pair_id),
            }
        )

    relation_counts = Counter(row["relation"] for row in rows)
    supported_relations = {"interchangeable", "contains", "different_capability"}
    unexpected = sorted(set(relation_counts) - supported_relations)
    positives = [
        row["score"]
        for row in rows
        if row["relation"] in {"interchangeable", "contains"}
    ]
    negatives = [row["score"] for row in rows if row["relation"] == "different_capability"]

    print(f"Latest human reviews: {len(rows)}")
    print(f"Relations: {dict(sorted(relation_counts.items()))}")
    print(f"Reviewed pairs found in pool: {len(reviewed_scores)}/{len(rows)}")
    print(f"Positive relations: {len(positives)}; different_capability: {len(negatives)}")
    if unexpected:
        print(f"Ignored legacy relations: {unexpected}")
    if not positives:
        raise SystemExit("No interchangeable/contains reviews available for threshold selection")

    print("\nThreshold candidates (selected only from observed positive scores):")
    for target in (1.0, 0.95, 0.90):
        threshold = best_threshold(positives, target)
        if threshold is None:
            print(f"  target recall {target:.0%}: unavailable")
            continue
        positive_recall = recall_at_threshold(positives, threshold)
        negative_filtered = sum(score is None or score < threshold for score in negatives)
        negative_filter_rate = negative_filtered / len(negatives) if negatives else 0.0
        all_filtered = sum(score is None or score < threshold for score in all_scores)
        all_filter_rate = all_filtered / len(all_scores) if all_scores else 0.0
        print(
            f"  target recall {target:.0%}: threshold={threshold:.8f}; "
            f"observed positive recall={positive_recall:.1%}; "
            f"reviewed negatives filtered={negative_filter_rate:.1%}; "
            f"all pooled pairs filtered={all_filter_rate:.1%}"
        )

    print("\nFixed-threshold sensitivity:")
    for threshold in (0.80, 0.85, 0.90, 0.92, 0.95):
        positive_recall = recall_at_threshold(positives, threshold)
        negative_filtered = sum(score is None or score < threshold for score in negatives)
        negative_filter_rate = negative_filtered / len(negatives) if negatives else 0.0
        retained = sum(score is not None and score >= threshold for score in all_scores)
        print(
            f"  threshold={threshold:.2f}: positive recall={positive_recall:.1%}; "
            f"reviewed negatives filtered={negative_filter_rate:.1%}; "
            f"pooled pairs retained={retained}/{len(all_scores)} ({retained / len(all_scores):.1%})"
        )

    print("\nScores by relation:")
    for relation in ("interchangeable", "contains", "different_capability"):
        values = sorted(
            row["score"] for row in rows if row["relation"] == relation and row["score"] is not None
        )
        missing = sum(
            row["score"] is None for row in rows if row["relation"] == relation
        )
        if values:
            middle = values[len(values) // 2]
            print(
                f"  {relation}: n={len(values)}, missing_qwen={missing}, "
                f"min={values[0]:.8f}, median={middle:.8f}, max={values[-1]:.8f}"
            )
        else:
            print(f"  {relation}: n=0, missing_qwen={missing}")


if __name__ == "__main__":
    main()
