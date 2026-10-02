"""Build ToolBench tasks augmented with interchangeable APIs from the relation graph."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


VERSION = "cluster-queries-v1"
RESERVED_NAMES = {"from", "class", "return", "false", "true", "id", "and"}


def standardize(value: str) -> str:
    result = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9_]", "_", value or "")
    result = re.sub(r"_+", "_", result).strip("_").lower()
    if result and result[0].isdigit():
        result = "get_" + result
    return result


def standardize_category(value: str) -> str:
    result = (value or "").replace(" ", "_").replace(",", "_").replace("/", "_")
    return re.sub(r"_+", "_", result)


def query_api_id(api: dict[str, Any]) -> str:
    category = standardize_category(str(api.get("category_name") or ""))
    tool = standardize(str(api.get("tool_name") or ""))
    name = standardize(str(api.get("api_name") or ""))
    if name in RESERVED_NAMES:
        name = "is_" + name
    return f"{category}/{tool}/{name}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def api_descriptor(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "category_name": row["category"],
        "tool_name": row["tool_name"],
        "api_name": row["api_name"],
        "api_description": row.get("api_description") or "",
        "required_parameters": row.get("required_parameters") or [],
        "optional_parameters": row.get("optional_parameters") or [],
        "method": row.get("method") or "GET",
    }


def fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-root",
        type=Path,
        default=Path("solvable_queries/test_instruction"),
    )
    parser.add_argument(
        "--clusters",
        type=Path,
        default=Path("data/relation_graph/v16_migrated/clusters.jsonl"),
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/catalog/tools.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/benchmark/cluster_queries_v1"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Write only the first N eligible queries for a pilot run.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    query_paths = sorted(args.query_root.glob("*.json"))
    if not query_paths:
        raise SystemExit(f"No JSON query files found in {args.query_root}")

    catalog = {row["api_id"]: row for row in load_jsonl(args.catalog)}
    api_to_cluster: dict[str, str] = {}
    cluster_members: dict[str, list[str]] = {}
    for cluster in load_jsonl(args.clusters):
        members = sorted(member["api_id"] for member in cluster.get("members", []))
        if len(members) < 2:
            continue
        cluster_id = cluster["cluster_id"]
        usable = [api_id for api_id in members if api_id in catalog]
        if len(usable) < 2:
            continue
        cluster_members[cluster_id] = usable
        for api_id in usable:
            api_to_cluster[api_id] = cluster_id

    output: list[dict[str, Any]] = []
    used_clusters: set[str] = set()
    missing_catalog: set[str] = set()
    source_queries = 0
    alternatives_added = 0

    for path in query_paths:
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            continue
        for query in records:
            source_queries += 1
            original_apis = query.get("api_list") or []
            original_ids = [query_api_id(api) for api in original_apis]
            matched_clusters = sorted(
                {api_to_cluster[api_id] for api_id in original_ids if api_id in api_to_cluster}
            )
            if not matched_clusters:
                continue

            expanded = list(original_apis)
            present = set(original_ids)
            added_ids: list[str] = []
            for cluster_id in matched_clusters:
                used_clusters.add(cluster_id)
                for api_id in cluster_members[cluster_id]:
                    if api_id in present:
                        continue
                    row = catalog.get(api_id)
                    if row is None:
                        missing_catalog.add(api_id)
                        continue
                    expanded.append(api_descriptor(row))
                    present.add(api_id)
                    added_ids.append(api_id)

            if not added_ids:
                continue
            result = dict(query)
            result["api_list"] = expanded
            result["cluster_benchmark"] = {
                "version": VERSION,
                "source_file": path.as_posix(),
                "original_api_ids": original_ids,
                "cluster_ids": matched_clusters,
                "added_api_ids": added_ids,
            }
            output.append(result)
            alternatives_added += len(added_ids)

    eligible_query_count = len(output)
    if args.limit is not None:
        output = output[: args.limit]
        used_clusters = {
            cluster_id
            for query in output
            for cluster_id in query["cluster_benchmark"]["cluster_ids"]
        }
        alternatives_added = sum(
            len(query["cluster_benchmark"]["added_api_ids"]) for query in output
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "queries.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "version": VERSION,
        "source_query_count": source_queries,
        "eligible_query_count": eligible_query_count,
        "output_query_count": len(output),
        "multi_api_cluster_count": len(cluster_members),
        "used_cluster_count": len(used_clusters),
        "alternatives_added": alternatives_added,
        "missing_catalog_api_count": len(missing_catalog),
        "query_root": args.query_root.as_posix(),
        "clusters": args.clusters.as_posix(),
        "catalog": args.catalog.as_posix(),
        "source_fingerprint_sha256": fingerprint(query_paths + [args.clusters, args.catalog]),
        "output": output_path.as_posix(),
        "limit": args.limit,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
