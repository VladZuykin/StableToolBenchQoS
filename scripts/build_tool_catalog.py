"""Build a normalized API catalog and audit statistics for StableToolBench."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from statistics import mean, median
from typing import Any


QOS_FIELDS = (
    "avgServiceLevel",
    "avgLatency",
    "avgSuccessRate",
    "popularityScore",
)


def standardize(value: str) -> str:
    """Match the identifier normalization used by ToolBench inference."""
    result = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9_]", "_", value or "")
    result = re.sub(r"_+", "_", result).strip("_").lower()
    if result and result[0].isdigit():
        result = "get_" + result
    return result


def numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "median": median(values),
    }


def load_cache_info(cache_file: Path) -> tuple[bool, int, str | None]:
    if not cache_file.is_file():
        return False, 0, None
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return True, 0, "cache root is not an object"
        return True, len(data), None
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return True, 0, str(error)


def build_catalog(tools_root: Path, cache_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    category_counts: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()
    pricing_counts: Counter[str] = Counter()
    qos_tool_coverage: Counter[str] = Counter()
    qos_values: dict[str, list[float]] = {field: [] for field in QOS_FIELDS}
    tools_with_score = 0
    tools_with_description = 0
    apis_with_description = 0
    apis_with_cache = 0
    apis_with_complete_qos = 0
    apis_with_cache_and_complete_qos = 0
    apis_ready_for_candidates = 0
    tools_with_cache: set[str] = set()
    total_cache_entries = 0
    cache_errors = 0
    valid_tool_files = 0

    tool_files = sorted(tools_root.glob("*/*.json"))
    for tool_file in tool_files:
        category = tool_file.parent.name
        try:
            tool = json.loads(tool_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            errors.append({"file": tool_file.as_posix(), "error": str(error)})
            continue
        if not isinstance(tool, dict):
            errors.append({"file": tool_file.as_posix(), "error": "tool root is not an object"})
            continue

        api_list = tool.get("api_list")
        if not isinstance(api_list, list):
            errors.append({"file": tool_file.as_posix(), "error": "api_list is not a list"})
            continue

        valid_tool_files += 1
        tool_name = str(tool.get("tool_name") or tool.get("name") or tool_file.stem)
        tool_id = standardize(tool_name)
        tool_description = str(tool.get("tool_description") or "").strip()
        pricing = str(tool.get("pricing") or "UNKNOWN").strip().upper()
        score = tool.get("score") if isinstance(tool.get("score"), dict) else {}
        has_complete_qos = all(numeric(score.get(field)) for field in QOS_FIELDS)

        category_counts[category] += 1
        pricing_counts[pricing] += 1
        if tool_description:
            tools_with_description += 1
        if score:
            tools_with_score += 1
        for field in QOS_FIELDS:
            value = score.get(field)
            if numeric(value):
                qos_tool_coverage[field] += 1
                qos_values[field].append(float(value))

        for api in api_list:
            if not isinstance(api, dict):
                errors.append({"file": tool_file.as_posix(), "error": "api_list contains a non-object item"})
                continue
            api_name = str(api.get("name") or "").strip()
            api_id = standardize(api_name)
            api_description = str(api.get("description") or "").strip()
            method = str(api.get("method") or "UNKNOWN").strip().upper()
            cache_file = cache_root / category / f"{tool_id}_for_{category}" / f"{api_id}.json"
            has_cache, cache_entries, cache_error = load_cache_info(cache_file)

            method_counts[method] += 1
            if api_description:
                apis_with_description += 1
            if has_complete_qos:
                apis_with_complete_qos += 1
            if has_cache:
                apis_with_cache += 1
                total_cache_entries += cache_entries
                tools_with_cache.add(f"{category}/{tool_id}")
                if has_complete_qos:
                    apis_with_cache_and_complete_qos += 1
                    if api_description and cache_entries > 0:
                        apis_ready_for_candidates += 1
            if cache_error:
                cache_errors += 1
                errors.append({"file": cache_file.as_posix(), "error": cache_error})

            records.append(
                {
                    "api_id": f"{category}/{tool_id}/{api_id}",
                    "category": category,
                    "tool_id": tool_id,
                    "tool_name": tool_name,
                    "tool_description": tool_description,
                    "api_name": api_name,
                    "api_description": api_description,
                    "method": method,
                    "required_parameters": api.get("required_parameters") or [],
                    "optional_parameters": api.get("optional_parameters") or [],
                    "pricing": pricing,
                    "qos": {field: score.get(field) for field in QOS_FIELDS},
                    "host": tool.get("host"),
                    "home_url": tool.get("home_url"),
                    "source_file": tool_file.relative_to(tools_root).as_posix(),
                    "cache_file": cache_file.relative_to(cache_root).as_posix() if has_cache else None,
                    "cache_entries": cache_entries,
                }
            )

    total_tools = valid_tool_files
    total_apis = len(records)
    statistics = {
        "paths": {
            "tools_root": tools_root.as_posix(),
            "cache_root": cache_root.as_posix(),
        },
        "totals": {
            "tool_files_discovered": len(tool_files),
            "valid_tools": total_tools,
            "apis": total_apis,
            "categories": len(category_counts),
            "errors": len(errors),
        },
        "coverage": {
            "tools_with_description": tools_with_description,
            "tools_with_score": tools_with_score,
            "qos_tools": dict(qos_tool_coverage),
            "apis_with_description": apis_with_description,
            "apis_with_cache": apis_with_cache,
            "tools_with_cache": len(tools_with_cache),
            "apis_with_complete_qos": apis_with_complete_qos,
            "apis_with_cache_and_complete_qos": apis_with_cache_and_complete_qos,
            "apis_ready_for_candidates": apis_ready_for_candidates,
            "cache_entries": total_cache_entries,
            "cache_errors": cache_errors,
        },
        "coverage_ratio": {
            "tools_with_description": tools_with_description / total_tools if total_tools else 0,
            "tools_with_score": tools_with_score / total_tools if total_tools else 0,
            "qos_tools": {
                field: qos_tool_coverage[field] / total_tools if total_tools else 0
                for field in QOS_FIELDS
            },
            "apis_with_description": apis_with_description / total_apis if total_apis else 0,
            "apis_with_cache": apis_with_cache / total_apis if total_apis else 0,
            "apis_with_complete_qos": apis_with_complete_qos / total_apis if total_apis else 0,
            "apis_with_cache_and_complete_qos": (
                apis_with_cache_and_complete_qos / total_apis if total_apis else 0
            ),
            "apis_ready_for_candidates": apis_ready_for_candidates / total_apis if total_apis else 0,
        },
        "qos_summary": {field: numeric_summary(values) for field, values in qos_values.items()},
        "pricing": dict(sorted(pricing_counts.items())),
        "methods": dict(sorted(method_counts.items())),
        "categories": dict(sorted(category_counts.items())),
        "errors": errors,
    }
    return records, statistics


def write_outputs(records: list[dict[str, Any]], statistics: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog_file = output_dir / "tools.jsonl"
    statistics_file = output_dir / "statistics.json"
    with catalog_file.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    statistics_file.write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools-root", type=Path, default=Path("server/tools"))
    parser.add_argument("--cache-root", type=Path, default=Path("server/tool_response_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/catalog"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.tools_root.is_dir():
        raise SystemExit(f"Tools directory not found: {args.tools_root}")
    if not args.cache_root.is_dir():
        raise SystemExit(f"Cache directory not found: {args.cache_root}")
    records, statistics = build_catalog(args.tools_root, args.cache_root)
    write_outputs(records, statistics, args.output_dir)
    totals = statistics["totals"]
    coverage = statistics["coverage"]
    print(f"Tools: {totals['valid_tools']}")
    print(f"APIs: {totals['apis']}")
    print(f"APIs with cache: {coverage['apis_with_cache']}")
    print(f"Catalog: {args.output_dir / 'tools.jsonl'}")
    print(f"Statistics: {args.output_dir / 'statistics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
