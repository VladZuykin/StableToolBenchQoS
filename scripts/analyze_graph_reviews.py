"""Analyze relation-graph decisions against the latest human reviews."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


VERSION = "graph-review-analysis-v1"
EMBEDDING_BINS = (0.0, 0.85, 0.90, 0.95, 1.0000001)
CUMULATIVE_CONFIDENCE = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0)
DIRECT_SOURCES = {"llm", "migrated_seed"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def embedding_score(row: dict[str, Any]) -> float | None:
    provenance = row.get("provenance") or {}
    source = provenance.get("source_provenance") or {}
    value = provenance.get("qwen_score", source.get("qwen_score"))
    return None if value is None else float(value)


def load_bm25_scores(path: Path, pair_ids: set[str]) -> dict[str, float]:
    """Return the strongest directional BM25 score for each requested pair."""
    scores = {}
    if not path.exists():
        return scores
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            pair_id = row.get("pair_id")
            if pair_id not in pair_ids:
                continue
            evidence = (row.get("retrieval") or {}).get("bm25") or []
            values = [float(item["score"]) for item in evidence if item.get("score") is not None]
            if values:
                scores[pair_id] = max(values)
    return scores


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute a quantile of an empty sequence")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def quartile_bins(values: list[float]) -> tuple[float, ...]:
    if not values:
        return ()
    boundaries = [
        min(values),
        quantile(values, 0.25),
        quantile(values, 0.50),
        quantile(values, 0.75),
        max(values),
    ]
    # Repeated scores can collapse adjacent quartiles.
    unique = []
    for value in boundaries:
        if not unique or value > unique[-1]:
            unique.append(value)
    if len(unique) == 1:
        unique.append(unique[0] + 1e-9)
    else:
        unique[-1] += 1e-9
    return tuple(unique)


def prompt_version(row: dict[str, Any]) -> str:
    provenance = row.get("provenance") or {}
    source = provenance.get("source_provenance") or {}
    return str(
        provenance.get("prompt_version")
        or source.get("prompt_version")
        or "unknown"
    )


def bin_label(lower: float, upper: float) -> str:
    visible_upper = min(upper, 1.0)
    return f"[{lower:.2f}, {visible_upper:.2f}{']' if upper > 1 else ')'}"


def relation_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = Counter(row[key] for row in rows)
    return {
        relation: counts.get(relation, 0)
        for relation in ("interchangeable", "contains", "different_capability")
    }


def bucket_embedding(rows: list[dict[str, Any]], relation_key: str) -> list[dict[str, Any]]:
    result = []
    for lower, upper in zip(EMBEDDING_BINS, EMBEDDING_BINS[1:]):
        selected = [
            row
            for row in rows
            if row["embedding_score"] is not None
            and lower <= row["embedding_score"] < upper
        ]
        if not selected:
            continue
        item = {
            "range": bin_label(lower, upper),
            "lower": lower,
            "upper": min(upper, 1.0),
            "count": len(selected),
            **relation_counts(selected, relation_key),
        }
        if all("is_false" in row for row in selected):
            false_count = sum(row["is_false"] for row in selected)
            item["false_count"] = false_count
            item["false_rate"] = false_count / len(selected)
        result.append(item)
    return result


def bucket_bm25(
    rows: list[dict[str, Any]], relation_key: str, bins: tuple[float, ...]
) -> list[dict[str, Any]]:
    result = []
    for index, (lower, upper) in enumerate(zip(bins, bins[1:])):
        selected = [
            row
            for row in rows
            if row.get("bm25_score") is not None
            and lower <= row["bm25_score"] < upper
        ]
        if not selected:
            continue
        visible_upper = upper if index < len(bins) - 2 else upper - 1e-9
        item = {
            "range": f"[{lower:.2f}, {visible_upper:.2f}{']' if index == len(bins) - 2 else ')'}",
            "lower": lower,
            "upper": visible_upper,
            "count": len(selected),
            **relation_counts(selected, relation_key),
        }
        if all("is_false" in row for row in selected):
            false_count = sum(row["is_false"] for row in selected)
            item["false_count"] = false_count
            item["false_rate"] = false_count / len(selected)
        result.append(item)
    return result


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def build_markdown(report: dict[str, Any]) -> str:
    direct_rows = []
    for row in report["all_direct_decisions"]["embedding_bins"]:
        direct_rows.append([
            row["range"],
            row["count"],
            row["interchangeable"],
            row["contains"],
            row["different_capability"],
        ])
    reviewed_embedding = []
    for row in report["human_review"]["embedding_bins"]:
        reviewed_embedding.append([
            row["range"],
            row["count"],
            row["interchangeable"],
            row["contains"],
            row["different_capability"],
            row["false_count"],
            f"{100 * row['false_rate']:.1f}%",
        ])
    direct_bm25 = []
    for row in report["all_direct_decisions"]["bm25_bins"]:
        direct_bm25.append([
            row["range"], row["count"], row["interchangeable"],
            row["contains"], row["different_capability"],
        ])
    reviewed_bm25 = []
    for row in report["human_review"]["bm25_bins"]:
        reviewed_bm25.append([
            row["range"], row["count"], row["interchangeable"],
            row["contains"], row["different_capability"], row["false_count"],
            f"{100 * row['false_rate']:.1f}%",
        ])
    confidence_rows = []
    for row in report["human_review"]["llm_confidence_exact"]:
        confidence_rows.append([
            f"{row['confidence']:.2f}",
            row["count"],
            row["false_count"],
            f"{100 * row['false_rate']:.1f}%",
        ])
    human = report["human_review"]
    return f"""# Анализ разметки графа API

Версия отчёта: `{VERSION}`.

## Все прямые решения

{markdown_table(
    ["Embedding score", "Всего", "interchangeable", "contains", "different_capability"],
    direct_rows,
)}

Всего прямых решений: **{report['all_direct_decisions']['count']}**.

## BM25 score

Для пары используется максимальный score среди двух направлений поиска.
Диапазоны являются квартилями прямых решений, у которых есть BM25 score.

{markdown_table(
    ["BM25 score", "Всего", "interchangeable", "contains", "different_capability"],
    direct_bm25,
)}

Прямых решений с BM25 score: **{report['all_direct_decisions']['bm25_count']}**.

## Человеческая проверка

Уникальных проверенных пар: **{human['unique_reviewed_pairs']}**.
Расхождений с LLM: **{human['false_count']}**
({100 * human['false_rate']:.1f}%).

{markdown_table(
    ["Embedding score", "Проверено", "interchangeable", "contains",
     "different_capability", "Ошибок", "Доля ошибок"],
    reviewed_embedding,
)}

### Проверенные пары по BM25 score

{markdown_table(
    ["BM25 score", "Проверено", "interchangeable", "contains",
     "different_capability", "Ошибок", "Доля ошибок"],
    reviewed_bm25,
)}

## Ошибки по confidence LLM

{markdown_table(
    ["Confidence", "Проверено", "Ошибок", "Доля ошибок"],
    confidence_rows,
)}

## Ограничение интерпретации

Проверенные пары не являются случайной выборкой: интерфейс позволяет выбирать
низкую уверенность LLM. Поэтому общую долю ошибок всех решений нельзя оценивать
непосредственно по этой выборке.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decisions",
        type=Path,
        default=Path("data/relation_graph/v16_migrated/pair_decisions.jsonl"),
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path("data/annotations/graph_review_queue.jsonl"),
    )
    parser.add_argument(
        "--retrieval-pairs",
        type=Path,
        default=Path("data/retrieval/pooled_pairs.jsonl"),
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=Path("data/annotations/human_graph_reviews.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/review_analysis/graph_reviews_v1"),
    )
    args = parser.parse_args()

    direct = []
    for row in load_jsonl(args.decisions):
        if row.get("status") != "ok" or row.get("decision_source") not in DIRECT_SOURCES:
            continue
        direct.append({
            "pair_id": row["pair_id"],
            "model_relation": row["annotation"]["relation"],
            "embedding_score": embedding_score(row),
            "llm_confidence": float(row["annotation"].get("confidence") or 0.0),
            "prompt_version": prompt_version(row),
        })

    pair_ids = {row["pair_id"] for row in direct}
    pair_ids.update(row["pair_id"] for row in load_jsonl(args.queue))
    bm25_by_pair = load_bm25_scores(args.retrieval_pairs, pair_ids)
    for row in direct:
        row["bm25_score"] = bm25_by_pair.get(row["pair_id"])

    queue = {row["pair_id"]: row for row in load_jsonl(args.queue)}
    latest_reviews = {}
    review_event_count = 0
    for row in load_jsonl(args.reviews):
        review_event_count += 1
        latest_reviews[row["pair_id"]] = row

    reviewed = []
    missing_from_queue = 0
    for pair_id, review in latest_reviews.items():
        source = queue.get(pair_id)
        if source is None:
            missing_from_queue += 1
            continue
        annotation = source["annotation"]
        human_relation = review["human_relation"]
        human_direction = review.get("human_direction", "none")
        model_relation = annotation["relation"]
        model_direction = annotation.get("direction", "none")
        is_false = (
            human_relation != model_relation
            or (
                human_relation == "contains"
                and human_direction != model_direction
            )
        )
        reviewed.append({
            "pair_id": pair_id,
            "model_relation": model_relation,
            "human_relation": human_relation,
            "embedding_score": embedding_score(source),
            "llm_confidence": float(annotation.get("confidence") or 0.0),
            "bm25_score": bm25_by_pair.get(pair_id),
            "prompt_version": prompt_version(source),
            "is_false": is_false,
        })

    exact_confidence = []
    for confidence in sorted({row["llm_confidence"] for row in reviewed}):
        selected = [row for row in reviewed if row["llm_confidence"] == confidence]
        false_count = sum(row["is_false"] for row in selected)
        exact_confidence.append({
            "confidence": confidence,
            "count": len(selected),
            "false_count": false_count,
            "false_rate": false_count / len(selected),
            **relation_counts(selected, "human_relation"),
        })

    cumulative_confidence = []
    for threshold in CUMULATIVE_CONFIDENCE:
        selected = [row for row in reviewed if row["llm_confidence"] <= threshold]
        if not selected:
            continue
        false_count = sum(row["is_false"] for row in selected)
        cumulative_confidence.append({
            "maximum_confidence": threshold,
            "count": len(selected),
            "false_count": false_count,
            "false_rate": false_count / len(selected),
        })

    false_count = sum(row["is_false"] for row in reviewed)
    bm25_values = [row["bm25_score"] for row in direct if row["bm25_score"] is not None]
    bm25_boundaries = quartile_bins(bm25_values)
    report = {
        "version": VERSION,
        "inputs": {
            "decisions": str(args.decisions),
            "queue": str(args.queue),
            "reviews": str(args.reviews),
            "retrieval_pairs": str(args.retrieval_pairs),
        },
        "all_direct_decisions": {
            "count": len(direct),
            "missing_embedding_score": sum(
                row["embedding_score"] is None for row in direct
            ),
            "relation_counts": relation_counts(direct, "model_relation"),
            "prompt_versions": dict(sorted(Counter(
                row["prompt_version"] for row in direct
            ).items())),
            "embedding_bins": bucket_embedding(direct, "model_relation"),
            "bm25_score_definition": "maximum directional BM25 score",
            "bm25_count": len(bm25_values),
            "missing_bm25_score": len(direct) - len(bm25_values),
            "bm25_quartile_boundaries": list(bm25_boundaries),
            "bm25_bins": bucket_bm25(direct, "model_relation", bm25_boundaries),
        },
        "human_review": {
            "review_event_count": review_event_count,
            "unique_reviewed_pairs": len(reviewed),
            "reviews_missing_from_queue": missing_from_queue,
            "human_relation_counts": relation_counts(reviewed, "human_relation"),
            "false_count": false_count,
            "false_rate": false_count / len(reviewed) if reviewed else None,
            "embedding_bins": bucket_embedding(reviewed, "human_relation"),
            "bm25_count": sum(row["bm25_score"] is not None for row in reviewed),
            "missing_bm25_score": sum(row["bm25_score"] is None for row in reviewed),
            "bm25_bins": bucket_bm25(reviewed, "human_relation", bm25_boundaries),
            "llm_confidence_exact": exact_confidence,
            "llm_confidence_cumulative": cumulative_confidence,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "analysis.json"
    markdown_path = args.output_dir / "report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = build_markdown(report)
    markdown_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
