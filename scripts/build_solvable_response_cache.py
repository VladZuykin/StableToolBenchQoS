"""Build a resumable LLM-generated response-cache overlay for solvable APIs."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from openai import OpenAI


GENERATOR_VERSION = "solvable-response-cache-v3"
PROMPT_VERSION = "solvable-cache-examples-v2"
RESERVED_NAMES = {"from", "class", "return", "false", "true", "id", "and"}


def standardize(value: str) -> str:
    result = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9_]", "_", value or "")
    result = re.sub(r"_+", "_", result).strip("_").lower()
    if result and result[0].isdigit():
        result = "get_" + result
    return result


def change_name(value: str) -> str:
    return "is_" + value if value in RESERVED_NAMES else value


def standardize_category(value: str) -> str:
    result = (value or "").replace(" ", "_").replace(",", "_").replace("/", "_")
    return re.sub(r"_+", "_", result)


def canonical_input(value: Any) -> str:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            try:
                value = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Cache root is not an object: {path}")
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def load_attempt_counts(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not path.is_file():
        return counts
    with path.open(encoding="utf-8") as source:
        for line in source:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            api_id = record.get("api_id")
            if isinstance(api_id, str):
                counts[api_id] += 1
    return counts


def endpoint_id(api: dict[str, Any]) -> str:
    category = standardize_category(str(api.get("category_name") or ""))
    tool = standardize(str(api.get("tool_name") or ""))
    name = change_name(standardize(str(api.get("api_name") or "")))
    return f"{category}/{tool}/{name}"


def collect_endpoints(query_roots: list[Path]) -> tuple[dict[str, dict[str, Any]], int]:
    endpoints: dict[str, dict[str, Any]] = {}
    query_count = 0
    for root in query_roots:
        for path in sorted(root.rglob("*.json")):
            data = read_json(path)
            if not isinstance(data, list):
                continue
            for query in data:
                if not isinstance(query, dict):
                    continue
                api_list = query.get("api_list")
                if not isinstance(api_list, list):
                    continue
                query_count += 1
                for api in api_list:
                    if not isinstance(api, dict):
                        continue
                    api_id = endpoint_id(api)
                    candidate = {
                        "api_id": api_id,
                        "api": api,
                        "source_query_file": path.as_posix(),
                    }
                    current = endpoints.get(api_id)
                    if current is None or len(json.dumps(candidate, ensure_ascii=False)) > len(
                        json.dumps(current, ensure_ascii=False)
                    ):
                        endpoints[api_id] = candidate
    return endpoints, query_count


def cache_path(root: Path, api_id: str) -> Path:
    category, tool, api = api_id.split("/", 2)
    return root / category / f"{tool}_for_{category}" / f"{api}.json"


def target_example_count(record: dict[str, Any], configured_minimum: int) -> int:
    api = record["api"]
    parameters = (api.get("required_parameters") or []) + (
        api.get("optional_parameters") or []
    )
    return configured_minimum if parameters else 1


def request_example_count(needed: int, examples_per_call: int | None) -> int:
    if needed < 1:
        return 0
    if examples_per_call is None:
        return needed
    return min(needed, examples_per_call)


def load_api_document(tools_root: Path, record: dict[str, Any]) -> dict[str, Any]:
    category, tool, api_name = record["api_id"].split("/", 2)
    path = tools_root / category / f"{tool}.json"
    tool_doc = read_json(path) if path.is_file() else {}
    selected_api = None
    for candidate in tool_doc.get("api_list") or []:
        normalized = change_name(standardize(str(candidate.get("name") or "")))
        if normalized == api_name:
            selected_api = candidate
            break
    query_api = record["api"]
    return {
        "api_id": record["api_id"],
        "tool_description": str(tool_doc.get("tool_description") or "")[:4000],
        "api_name": query_api.get("api_name"),
        "api_description": str(query_api.get("api_description") or "")[:4000],
        "method": query_api.get("method"),
        "required_parameters": query_api.get("required_parameters") or [],
        "optional_parameters": query_api.get("optional_parameters") or [],
        "template_response": limited_value(query_api.get("template_response"), 8000),
        "tool_api_document": compact_tool_api(selected_api),
    }


def limited_value(value: Any, max_characters: int) -> Any:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(serialized) <= max_characters:
        return value
    return serialized[:max_characters] + "...<truncated>"


def compact_tool_api(api: Any) -> dict[str, Any] | None:
    if not isinstance(api, dict):
        return None
    return {
        "name": api.get("name"),
        "description": str(api.get("description") or "")[:4000],
        "method": api.get("method"),
        "required_parameters": api.get("required_parameters") or [],
        "optional_parameters": api.get("optional_parameters") or [],
        "schema": limited_value(api.get("schema"), 6000),
    }


def stable_seed(seed: int, api_id: str) -> int:
    digest = hashlib.sha256(f"{seed}\x1f{api_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def source_fingerprint(endpoints: dict[str, dict[str, Any]]) -> str:
    payload = json.dumps(endpoints, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_messages(
    document: dict[str, Any],
    needed: int,
    existing_inputs: list[str],
    attempt: int,
) -> list[dict[str, str]]:
    system = """You generate synthetic request/response examples for a virtual API server.
Return one JSON object only, with this shape:
{"examples":[{"input":{...},"response":<any JSON value>}]}

Rules:
- Generate exactly the requested number of distinct examples.
- Every input must include all required parameters using their documented names.
- Inputs must be plausible and different from the existing inputs.
- Follow documented parameter types and use defaults only as anchors, not for every example.
- The response must match the API purpose and template/schema when supplied.
- Keep every response compact: arrays contain at most 2 representative items,
  omit repetitive optional fields, never emit base64/binary data, and keep the
  complete JSON response below 1500 characters when possible.
- Do not return markdown, commentary, an error envelope, or invented authentication fields.
"""
    user = json.dumps(
        {
            "requested_example_count": needed,
            "generation_attempt": attempt,
            "existing_inputs_to_avoid": existing_inputs[:20],
            "api_document": document,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_model_output(
    output: str,
    required_parameters: list[dict[str, Any]],
    existing_keys: set[str],
    limit: int,
) -> tuple[dict[str, Any], list[str]]:
    cleaned = output.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    parsed = json.loads(cleaned)
    examples = parsed.get("examples") if isinstance(parsed, dict) else None
    if not isinstance(examples, list):
        raise ValueError("Model output does not contain an examples list")

    required = {
        change_name(standardize(str(parameter.get("name") or "")))
        for parameter in required_parameters
        if isinstance(parameter, dict) and parameter.get("name")
    }
    accepted: dict[str, Any] = {}
    rejected: list[str] = []
    for index, example in enumerate(examples):
        if len(accepted) >= limit:
            break
        if not isinstance(example, dict) or not isinstance(example.get("input"), dict):
            rejected.append(f"example {index}: input is not an object")
            continue
        normalized_input = {
            change_name(standardize(str(key))): value
            for key, value in example["input"].items()
        }
        missing = sorted(required - set(normalized_input))
        if missing:
            rejected.append(f"example {index}: missing required parameters {missing}")
            continue
        key = canonical_input(normalized_input)
        if key in existing_keys or key in accepted:
            rejected.append(f"example {index}: duplicate input")
            continue
        if "response" not in example:
            rejected.append(f"example {index}: response is missing")
            continue
        accepted[key] = {"error": "", "response": example["response"]}
    return accepted, rejected


def cache_keys(*caches: dict[str, Any]) -> set[str]:
    return {canonical_input(key) for cache in caches for key in cache}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-root", action="append", type=Path)
    parser.add_argument("--tools-root", type=Path, default=Path("server/tools"))
    parser.add_argument(
        "--official-cache-root", type=Path, default=Path("server/tool_response_cache")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/generated_cache/solvable_v1")
    )
    parser.add_argument("--min-examples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-llm-calls", type=int)
    parser.add_argument(
        "--examples-per-call",
        type=int,
        help="Maximum examples requested in one LLM call; omit to request all missing examples.",
    )
    parser.add_argument(
        "--max-attempts-per-endpoint",
        type=int,
        default=1,
        help="Maximum LLM calls for one endpoint during this run.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for one provider request.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--api-key", default=os.getenv("SIMULATOR_API_KEY"))
    parser.add_argument(
        "--api-base", default=os.getenv("SIMULATOR_API_BASE", "https://api.openai.com/v1")
    )
    parser.add_argument("--model", default=os.getenv("SIMULATOR_MODEL"))
    parser.add_argument(
        "--temperature", type=float, default=float(os.getenv("SIMULATOR_TEMPERATURE", "0"))
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    if args.min_examples < 1:
        parser.error("--min-examples must be positive")
    if args.max_llm_calls is not None and args.max_llm_calls < 0:
        parser.error("--max-llm-calls must be non-negative")
    if args.examples_per_call is not None and args.examples_per_call < 1:
        parser.error("--examples-per-call must be positive")
    if args.max_attempts_per_endpoint < 1:
        parser.error("--max-attempts-per-endpoint must be positive")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")

    query_roots = args.query_root or [
        Path("solvable_queries"),
        Path("solvable_queries_example"),
    ]
    endpoints, query_count = collect_endpoints(query_roots)
    records = [endpoints[api_id] for api_id in sorted(endpoints)]
    if args.limit is not None:
        records = records[: args.limit]

    responses_root = args.output_dir / "responses"
    plan = []
    existing_example_count = 0
    parameterless_endpoints = 0
    for record in records:
        official = read_cache(cache_path(args.official_cache_root, record["api_id"]))
        generated = read_cache(cache_path(responses_root, record["api_id"]))
        count = len(cache_keys(official, generated))
        existing_example_count += count
        target = target_example_count(record, args.min_examples)
        if target == 1:
            parameterless_endpoints += 1
        if count < target:
            plan.append((record, target - count))

    summary = {
        "generator_version": GENERATOR_VERSION,
        "prompt_version": PROMPT_VERSION,
        "query_roots": [path.as_posix() for path in query_roots],
        "query_records": query_count,
        "endpoint_count": len(records),
        "target_examples_per_endpoint": args.min_examples,
        "parameterless_endpoint_target": 1,
        "parameterless_endpoint_count": parameterless_endpoints,
        "endpoints_requiring_generation": len(plan),
        "endpoints_already_satisfied": len(records) - len(plan),
        "examples_required": sum(needed for _, needed in plan),
        "existing_unique_examples": existing_example_count,
        "seed": args.seed,
        "source_fingerprint_sha256": source_fingerprint(endpoints),
        "model": args.model,
        "api_base": args.api_base,
        "execute": args.execute,
        "examples_per_call": args.examples_per_call,
        "max_attempts_per_endpoint": args.max_attempts_per_endpoint,
        "request_timeout_seconds": args.request_timeout,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if not args.execute:
        print("Dry run only. Add --execute to call the LLM and write the overlay.")
        return
    if not args.api_key:
        parser.error("SIMULATOR_API_KEY or --api-key is required with --execute")
    if not args.model:
        parser.error("SIMULATOR_MODEL or --model is required with --execute")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(summary)
    metadata["execute"] = True
    atomic_write_json(args.output_dir / "metadata.json", metadata)
    errors_path = args.output_dir / "errors.jsonl"
    checkpoint_path = args.output_dir / "checkpoint.json"
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
        timeout=args.request_timeout,
        max_retries=0,
    )
    attempt_counts = load_attempt_counts(errors_path)

    llm_calls = 0
    completed = 0
    generated_examples = 0
    checkpoint = {
        "generator_version": GENERATOR_VERSION,
        "last_api_id": None,
        "planned_endpoints": len(plan),
        "processed_endpoints": 0,
        "completed_endpoints": 0,
        "remaining_endpoints": len(plan),
        "llm_calls": 0,
        "generated_examples": 0,
    }
    atomic_write_json(checkpoint_path, checkpoint)
    for index, (record, _) in enumerate(plan, start=1):
        if args.max_llm_calls is not None and llm_calls >= args.max_llm_calls:
            break
        api_id = record["api_id"]
        official_path = cache_path(args.official_cache_root, api_id)
        generated_path = cache_path(responses_root, api_id)
        official = read_cache(official_path)
        generated = read_cache(generated_path)
        existing = cache_keys(official, generated)
        target = target_example_count(record, args.min_examples)
        needed = max(0, target - len(existing))
        document = load_api_document(args.tools_root, record)
        historical_attempts = attempt_counts[api_id]
        endpoint_calls = 0

        while needed > 0 and endpoint_calls < args.max_attempts_per_endpoint:
            if args.max_llm_calls is not None and llm_calls >= args.max_llm_calls:
                break
            requested = request_example_count(needed, args.examples_per_call)
            attempt = historical_attempts + endpoint_calls + 1
            messages = build_messages(document, requested, sorted(existing), attempt)
            print(
                f"[{index}/{len(plan)}] {api_id}: request attempt={attempt}, "
                f"examples={requested}",
                flush=True,
            )
            llm_calls += 1
            endpoint_calls += 1
            accepted_count = 0
            rejected: list[str] = []
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    seed=stable_seed(args.seed, f"{api_id}\x1f{attempt}"),
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or ""
                new_examples, rejected = parse_model_output(
                    content,
                    document["required_parameters"],
                    existing,
                    requested,
                )
                generated.update(new_examples)
                atomic_write_json(generated_path, generated)
                accepted_count = len(new_examples)
                generated_examples += accepted_count
                existing.update(new_examples)
                needed = max(0, target - len(existing))
                if accepted_count < requested:
                    append_jsonl(
                        errors_path,
                        {
                            "api_id": api_id,
                            "generator_version": GENERATOR_VERSION,
                            "attempt": attempt,
                            "error": "model returned too few valid examples",
                            "needed": requested,
                            "accepted": accepted_count,
                            "rejected": rejected,
                        },
                    )
            except Exception as error:
                append_jsonl(
                    errors_path,
                    {
                        "api_id": api_id,
                        "generator_version": GENERATOR_VERSION,
                        "attempt": attempt,
                        "error": f"{type(error).__name__}: {error}",
                    },
                )
            print(
                f"[{index}/{len(plan)}] {api_id}: accepted={accepted_count}, "
                f"remaining={needed}, calls={llm_calls}",
                flush=True,
            )

        if needed == 0:
            completed += 1

        checkpoint = {
            "generator_version": GENERATOR_VERSION,
            "last_api_id": api_id,
            "planned_endpoints": len(plan),
            "processed_endpoints": index,
            "completed_endpoints": completed,
            "remaining_endpoints": len(plan) - completed,
            "llm_calls": llm_calls,
            "generated_examples": generated_examples,
        }
        atomic_write_json(checkpoint_path, checkpoint)

    print(json.dumps(read_json(checkpoint_path), ensure_ascii=False, indent=2))
    print(f"Generated cache overlay: {responses_root}")


if __name__ == "__main__":
    main()
