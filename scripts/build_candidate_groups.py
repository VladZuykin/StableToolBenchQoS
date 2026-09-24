"""Build semantic nearest-neighbor groups from the audited API catalog."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors


QOS_FIELDS = (
    "avgServiceLevel",
    "avgLatency",
    "avgSuccessRate",
    "popularityScore",
)
EMBEDDING_TEXT_VERSION = "api_first_v2"


def standardize(value: str) -> str:
    result = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9_]", "_", value or "")
    return re.sub(r"_+", "_", result).strip("_").lower()


def numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parameter_signature(parameters: Iterable[dict[str, Any]]) -> tuple[tuple[str, str], ...]:
    signature = []
    for parameter in parameters:
        if not isinstance(parameter, dict):
            continue
        signature.append(
            (
                standardize(str(parameter.get("name") or "")),
                str(parameter.get("type") or "UNKNOWN").upper(),
            )
        )
    return tuple(sorted(signature))


def semantic_text(record: dict[str, Any]) -> str:
    parameter_parts = []
    for role, key in (("required", "required_parameters"), ("optional", "optional_parameters")):
        for parameter in record.get(key) or []:
            if not isinstance(parameter, dict):
                continue
            parameter_parts.append(
                f"{role} parameter {parameter.get('name', '')} "
                f"type {parameter.get('type', '')}: {parameter.get('description', '')}"
            )
    return "\n".join(
        part
        for part in (
            f"API: {record.get('api_name', '')}",
            f"API description: {record.get('api_description', '')}",
            *parameter_parts,
            f"Tool: {record.get('tool_name', '')}",
            f"Tool description: {record.get('tool_description', '')}",
            f"Category: {record.get('category', '')}",
        )
        if part.strip()
    )


def load_candidates(catalog_file: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records_by_id: dict[str, dict[str, Any]] = {}
    rejected: Counter[str] = Counter()
    with catalog_file.open(encoding="utf-8") as catalog:
        for line_number, line in enumerate(catalog, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
            if not str(record.get("api_description") or "").strip():
                rejected["missing_api_description"] += 1
                continue
            if int(record.get("cache_entries") or 0) <= 0:
                rejected["empty_cache"] += 1
                continue
            qos = record.get("qos") or {}
            if not all(numeric(qos.get(field)) for field in QOS_FIELDS):
                rejected["incomplete_qos"] += 1
                continue
            record["semantic_text"] = semantic_text(record)
            record["required_signature"] = parameter_signature(record.get("required_parameters") or [])
            record["optional_signature"] = parameter_signature(record.get("optional_parameters") or [])
            existing = records_by_id.get(record["api_id"])
            if existing is not None:
                rejected["duplicate_api_id"] += 1
                record_quality = (
                    len(record["api_description"]),
                    len(record["required_parameters"]),
                    len(record["optional_parameters"]),
                    record["semantic_text"],
                )
                existing_quality = (
                    len(existing["api_description"]),
                    len(existing["required_parameters"]),
                    len(existing["optional_parameters"]),
                    existing["semantic_text"],
                )
                if record_quality <= existing_quality:
                    continue
            records_by_id[record["api_id"]] = record
    records = sorted(records_by_id.values(), key=lambda item: item["api_id"])
    return records, dict(sorted(rejected.items()))


def catalog_fingerprint(records: list[dict[str, Any]], model_name: str) -> str:
    digest = hashlib.sha256()
    digest.update(model_name.encode("utf-8"))
    digest.update(EMBEDDING_TEXT_VERSION.encode("utf-8"))
    for record in records:
        digest.update(record["api_id"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(record["semantic_text"].encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def annotate_token_lengths(
    records: list[dict[str, Any]],
    model: SentenceTransformer,
    batch_size: int,
) -> dict[str, Any]:
    tokenizer = model.tokenizer
    max_length = int(model.max_seq_length)
    lengths: list[int] = []
    texts = [record["semantic_text"] for record in records]
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_length=True,
        )
        batch_lengths = [int(length) for length in encoded["length"]]
        lengths.extend(batch_lengths)

    for record, token_count in zip(records, lengths):
        record["token_count"] = token_count
        record["tokens_truncated"] = max(0, token_count - max_length)

    length_array = np.asarray(lengths, dtype=np.int32)
    truncated = length_array > max_length
    return {
        "max_seq_length": max_length,
        "records": len(lengths),
        "truncated_records": int(truncated.sum()),
        "truncated_ratio": float(truncated.mean()) if len(lengths) else 0.0,
        "min": int(length_array.min()) if len(lengths) else None,
        "max": int(length_array.max()) if len(lengths) else None,
        "mean": float(length_array.mean()) if len(lengths) else None,
        "p50": float(np.percentile(length_array, 50)) if len(lengths) else None,
        "p90": float(np.percentile(length_array, 90)) if len(lengths) else None,
        "p95": float(np.percentile(length_array, 95)) if len(lengths) else None,
        "p99": float(np.percentile(length_array, 99)) if len(lengths) else None,
        "total_tokens_truncated": int(sum(record["tokens_truncated"] for record in records)),
    }


def load_or_build_embeddings(
    records: list[dict[str, Any]],
    model: SentenceTransformer,
    model_name: str,
    embeddings_file: Path,
    batch_size: int,
) -> tuple[np.ndarray, str]:
    fingerprint = catalog_fingerprint(records, model_name)
    if embeddings_file.is_file():
        cached = np.load(embeddings_file, allow_pickle=False)
        cached_fingerprint = str(cached["fingerprint"].item())
        if cached_fingerprint == fingerprint:
            return cached["embeddings"], "cache"

    embeddings = model.encode(
        [record["semantic_text"] for record in records],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    embeddings_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(embeddings_file, fingerprint=np.array(fingerprint), embeddings=embeddings)
    return embeddings, "model"


def neighbor_record(
    source: dict[str, Any],
    candidate: dict[str, Any],
    similarity: float,
) -> dict[str, Any]:
    return {
        "api_id": candidate["api_id"],
        "tool_name": candidate["tool_name"],
        "api_name": candidate["api_name"],
        "category": candidate["category"],
        "similarity": round(float(similarity), 8),
        "same_tool": source["tool_id"] == candidate["tool_id"] and source["category"] == candidate["category"],
        "same_required_parameters": source["required_signature"] == candidate["required_signature"],
        "same_optional_parameters": source["optional_signature"] == candidate["optional_signature"],
        "cache_entries": candidate["cache_entries"],
        "pricing": candidate["pricing"],
        "qos": candidate["qos"],
        "token_count": candidate["token_count"],
        "tokens_truncated": candidate["tokens_truncated"],
    }


def compute_neighbors(
    records: list[dict[str, Any]],
    embeddings: np.ndarray,
    top_k: int,
    mode: str,
    allow_same_tool: bool,
) -> dict[int, list[dict[str, Any]]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        key = record["category"] if mode == "category" else "__all__"
        groups[key].append(index)

    result: dict[int, list[dict[str, Any]]] = {}
    for indices in groups.values():
        if len(indices) < 2:
            for index in indices:
                result[index] = []
            continue
        tool_counts = Counter((records[index]["category"], records[index]["tool_id"]) for index in indices)
        same_tool_allowance = max(tool_counts.values()) if not allow_same_tool else 0
        neighbor_count = min(top_k + 1 + same_tool_allowance, len(indices))
        group_embeddings = embeddings[indices]
        search = NearestNeighbors(n_neighbors=neighbor_count, metric="cosine", algorithm="brute", n_jobs=-1)
        search.fit(group_embeddings)
        distances, local_neighbors = search.kneighbors(group_embeddings)
        for source_position, source_index in enumerate(indices):
            neighbors = []
            for distance, candidate_position in zip(distances[source_position], local_neighbors[source_position]):
                candidate_index = indices[int(candidate_position)]
                if candidate_index == source_index:
                    continue
                same_tool = (
                    records[source_index]["category"] == records[candidate_index]["category"]
                    and records[source_index]["tool_id"] == records[candidate_index]["tool_id"]
                )
                if same_tool and not allow_same_tool:
                    continue
                neighbors.append(
                    neighbor_record(records[source_index], records[candidate_index], 1.0 - float(distance))
                )
                if len(neighbors) == top_k:
                    break
            result[source_index] = neighbors
    return result


def write_neighbors(
    records: list[dict[str, Any]],
    global_neighbors: dict[int, list[dict[str, Any]]],
    category_neighbors: dict[int, list[dict[str, Any]]],
    output_file: Path,
) -> None:
    with output_file.open("w", encoding="utf-8", newline="\n") as output:
        for index, record in enumerate(records):
            output_record = {
                "api_id": record["api_id"],
                "tool_name": record["tool_name"],
                "api_name": record["api_name"],
                "category": record["category"],
                "semantic_text": record["semantic_text"],
                "required_parameters": record["required_parameters"],
                "optional_parameters": record["optional_parameters"],
                "token_count": record["token_count"],
                "tokens_truncated": record["tokens_truncated"],
                "global_neighbors": global_neighbors[index],
                "category_neighbors": category_neighbors[index],
            }
            output.write(json.dumps(output_record, ensure_ascii=False, sort_keys=True) + "\n")


def write_review_csv(
    records: list[dict[str, Any]],
    global_neighbors: dict[int, list[dict[str, Any]]],
    category_neighbors: dict[int, list[dict[str, Any]]],
    output_file: Path,
    review_targets: int,
    review_neighbors: int,
    seed: int,
) -> None:
    random_generator = random.Random(seed)
    selected = random_generator.sample(range(len(records)), min(review_targets, len(records)))
    records_by_id = {record["api_id"]: record for record in records}
    fieldnames = [
        "target_api_id",
        "target_tool_name",
        "target_api_name",
        "target_tool_description",
        "target_api_description",
        "target_required_parameters",
        "target_optional_parameters",
        "target_token_count",
        "target_tokens_truncated",
        "mode",
        "rank",
        "neighbor_api_id",
        "neighbor_tool_name",
        "neighbor_api_name",
        "neighbor_tool_description",
        "neighbor_api_description",
        "neighbor_required_parameters",
        "neighbor_optional_parameters",
        "neighbor_token_count",
        "neighbor_tokens_truncated",
        "similarity",
        "same_tool",
        "same_required_parameters",
        "same_optional_parameters",
        "manual_relevance_0_1_2",
        "comment",
    ]
    with output_file.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for index in selected:
            target = records[index]
            for mode, neighbors in (
                ("global", global_neighbors[index]),
                ("category", category_neighbors[index]),
            ):
                for rank, neighbor in enumerate(neighbors[:review_neighbors], start=1):
                    writer.writerow(
                        {
                            "target_api_id": target["api_id"],
                            "target_tool_name": target["tool_name"],
                            "target_api_name": target["api_name"],
                            "target_tool_description": target["tool_description"],
                            "target_api_description": target["api_description"],
                            "target_required_parameters": json.dumps(
                                target["required_parameters"], ensure_ascii=False, sort_keys=True
                            ),
                            "target_optional_parameters": json.dumps(
                                target["optional_parameters"], ensure_ascii=False, sort_keys=True
                            ),
                            "target_token_count": target["token_count"],
                            "target_tokens_truncated": target["tokens_truncated"],
                            "mode": mode,
                            "rank": rank,
                            "neighbor_api_id": neighbor["api_id"],
                            "neighbor_tool_name": neighbor["tool_name"],
                            "neighbor_api_name": neighbor["api_name"],
                            "neighbor_tool_description": records_by_id[neighbor["api_id"]]["tool_description"],
                            "neighbor_api_description": records_by_id[neighbor["api_id"]]["api_description"],
                            "neighbor_required_parameters": json.dumps(
                                records_by_id[neighbor["api_id"]]["required_parameters"],
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                            "neighbor_optional_parameters": json.dumps(
                                records_by_id[neighbor["api_id"]]["optional_parameters"],
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                            "neighbor_token_count": neighbor["token_count"],
                            "neighbor_tokens_truncated": neighbor["tokens_truncated"],
                            "similarity": neighbor["similarity"],
                            "same_tool": neighbor["same_tool"],
                            "same_required_parameters": neighbor["same_required_parameters"],
                            "same_optional_parameters": neighbor["same_optional_parameters"],
                            "manual_relevance_0_1_2": "",
                            "comment": "",
                        }
                    )


def summarize_neighbors(neighbors: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    similarities = [neighbor["similarity"] for values in neighbors.values() for neighbor in values]
    return {
        "pairs": len(similarities),
        "mean_similarity": float(np.mean(similarities)) if similarities else None,
        "median_similarity": float(np.median(similarities)) if similarities else None,
        "min_similarity": min(similarities) if similarities else None,
        "max_similarity": max(similarities) if similarities else None,
        "same_tool_pairs": sum(neighbor["same_tool"] for values in neighbors.values() for neighbor in values),
        "same_required_parameter_pairs": sum(
            neighbor["same_required_parameters"] for values in neighbors.values() for neighbor in values
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog/tools.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/candidates"))
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--review-targets", type=int, default=50)
    parser.add_argument("--review-neighbors", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-same-tool",
        action="store_true",
        help="Allow APIs from the same tool to appear as candidates.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be positive")
    if not args.catalog.is_file():
        raise SystemExit(f"Catalog not found: {args.catalog}. Run scripts/build_tool_catalog.py first.")

    records, rejected = load_candidates(args.catalog)
    if len(records) < 2:
        raise SystemExit("At least two candidate APIs are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(args.model)
    token_statistics = annotate_token_lengths(records, model, args.batch_size)
    embeddings_file = args.output_dir / "embeddings.npz"
    embeddings, embedding_source = load_or_build_embeddings(
        records, model, args.model, embeddings_file, args.batch_size
    )
    global_neighbors = compute_neighbors(records, embeddings, args.top_k, "global", args.allow_same_tool)
    category_neighbors = compute_neighbors(records, embeddings, args.top_k, "category", args.allow_same_tool)

    neighbors_file = args.output_dir / "neighbors.jsonl"
    review_file = args.output_dir / "manual_review.csv"
    statistics_file = args.output_dir / "statistics.json"
    write_neighbors(records, global_neighbors, category_neighbors, neighbors_file)
    write_review_csv(
        records,
        global_neighbors,
        category_neighbors,
        review_file,
        args.review_targets,
        args.review_neighbors,
        args.seed,
    )
    statistics = {
        "model": args.model,
        "embedding_text_version": EMBEDDING_TEXT_VERSION,
        "embedding_dimension": int(embeddings.shape[1]),
        "embedding_source": embedding_source,
        "candidate_apis": len(records),
        "rejected": rejected,
        "top_k": args.top_k,
        "allow_same_tool": args.allow_same_tool,
        "seed": args.seed,
        "review_targets": min(args.review_targets, len(records)),
        "review_neighbors_per_mode": args.review_neighbors,
        "token_lengths": token_statistics,
        "global": summarize_neighbors(global_neighbors),
        "category": summarize_neighbors(category_neighbors),
    }
    statistics_file.write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Candidate APIs: {len(records)}")
    print(f"Embeddings: {embedding_source} ({embeddings.shape[1]} dimensions)")
    print(f"Neighbors: {neighbors_file}")
    print(f"Manual review: {review_file}")
    print(f"Statistics: {statistics_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
