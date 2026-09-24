"""Annotate candidate API pairs with an OpenAI-compatible language model."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from tqdm.auto import tqdm


PROMPT_VERSION = "pair-relations-v12"
RELATIONS = {
    "interchangeable",
    "contains",
    "different_capability",
}
DIRECTIONS = {"none", "left_contains_right", "right_contains_left"}
ANNOTATION_FIELDS = {
    "relation", "direction", "confidence", "reasoning", "evidence", "needs_human_review",
}

SYSTEM_PROMPT = """You annotate relations between two software APIs using only their documentation.
Return one valid JSON object and no Markdown.

Choose exactly one relation:
- interchangeable: the APIs solve the same user need over equivalent entities and provide functionally equivalent results. This class also includes exact duplicates.
- contains: one API's documented result or supported scope includes the other's narrower capability. Use direction to say which one contains which. Set inclusion counts: for the same resource, "latest 10 drawings" contains "latest drawing" when the returned set includes the latest item.
- different_capability: no equivalence or containment edge should be created. Use this for any pair that exposes different capabilities, regardless of whether the APIs are semantically close, in the same domain, completely unrelated, or connected by a workflow.

Be conservative. Similar words, provider, category, operation verb, parameter shape, workflow, or both being search endpoints do not imply equivalence. A broader API only counts as contains when its documented result actually supplies the narrower capability. Parameter names need not match literally. Do not construct parameter mappings: the runtime agent will interpret the selected API documentation when making a call.

Decision examples:
1. city="Moscow" versus location="Moscow" can be interchangeable when both APIs otherwise solve the same task.
2. A date in YYYY-MM-DD versus a Unix timestamp can be interchangeable when the conversion is deterministic.
3. username versus user_id is not automatically interchangeable when converting one to the other requires another API call or unavailable external data.
4. Login endpoints for two different systems are different_capability because they authenticate against different user stores and create state in different systems, even when their names and schemas match.
5. Weather APIs from different providers can be interchangeable when they answer the same user need with functionally equivalent results.
6. Prices specifically from Binance versus Coinbase are not necessarily interchangeable because the requested source can be part of the result's meaning.

When documentation is insufficient or contradictory, choose the most plausible of the three relations, lower confidence, and set needs_human_review=true. Uncertainty is not a relation class.

Required JSON shape:
{
  "relation": "one relation above",
  "direction": "none | left_contains_right | right_contains_left",
  "confidence": 0.0,
  "reasoning": "short evidence-based explanation",
  "evidence": ["short fact from the supplied documentation"],
  "needs_human_review": false
}

For every relation except contains, direction must be none. Set needs_human_review=true for ambiguity, weak documentation, questionable equivalence, or confidence below 0.75."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {error}") from error
    return rows


def api_document(api: dict[str, Any]) -> dict[str, Any]:
    return {
        "api_id": api["api_id"],
        "category": api.get("category"),
        "tool_name": api.get("tool_name"),
        "tool_description": api.get("tool_description"),
        "api_name": api.get("api_name"),
        "api_description": api.get("api_description"),
        "required_parameters": api.get("required_parameters") or [],
        "optional_parameters": api.get("optional_parameters") or [],
    }


def build_user_prompt(pair: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> str:
    payload = {
        "task": "Classify the functional relationship between LEFT and RIGHT API.",
        "left": api_document(catalog[pair["left_api_id"]]),
        "right": api_document(catalog[pair["right_api_id"]]),
    }
    return "Analyze this pair and return JSON:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def normalize_annotation(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("The response root must be a JSON object")
    value = {field: field_value for field, field_value in value.items() if field in ANNOTATION_FIELDS}
    relation = value.get("relation")
    direction = value.get("direction")
    if relation not in RELATIONS:
        raise ValueError(f"Unknown relation: {relation!r}")
    if direction not in DIRECTIONS:
        raise ValueError(f"Unknown direction: {direction!r}")
    if relation == "contains" and direction == "none":
        raise ValueError("contains requires a direction")
    if relation != "contains" and direction != "none":
        raise ValueError(f"{relation} requires direction=none")
    confidence = value.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be a number")
    if not 0 <= float(confidence) <= 1:
        raise ValueError("confidence must be between 0 and 1")
    if not isinstance(value.get("reasoning"), str) or not value["reasoning"].strip():
        raise ValueError("reasoning must be a non-empty string")
    if not isinstance(value.get("evidence"), list):
        raise ValueError("evidence must be an array")
    if not isinstance(value.get("needs_human_review"), bool):
        raise ValueError("needs_human_review must be boolean")
    value["confidence"] = float(confidence)
    return value


def parse_response(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return normalize_annotation(json.loads(text))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def completed_pair_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["pair_id"] for row in load_jsonl(path) if row.get("status") == "ok"}


def interleave_strata(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Round-robin pairs so every small --limit covers different retrieval strata."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        stratum = str((pair.get("sampling") or {}).get("stratum") or "unspecified")
        groups[stratum].append(pair)
    ordered = []
    names = sorted(groups)
    max_size = max((len(group) for group in groups.values()), default=0)
    for position in range(max_size):
        for name in names:
            if position < len(groups[name]):
                ordered.append(groups[name][position])
    return ordered


def annotate(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("The model returned empty content")
    usage = response.usage.model_dump() if response.usage and hasattr(response.usage, "model_dump") else None
    return parse_response(content), usage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/retrieval_analysis/pilot_pairs.jsonl"))
    parser.add_argument("--catalog", type=Path, default=Path("data/retrieval/catalog.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/annotations/pilot_pair_annotations.jsonl"))
    parser.add_argument("--errors", type=Path, default=Path("data/annotations/pilot_pair_annotation_errors.jsonl"))
    parser.add_argument("--limit", type=int, default=20, help="Maximum pending pairs to process; 0 means all")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--dry-run", action="store_true", help="Print the first prompt without calling the API")
    args = parser.parse_args()

    if args.limit < 0 or args.retries < 0:
        parser.error("--limit and --retries must be non-negative")
    pairs = load_jsonl(args.input)
    catalog_rows = load_jsonl(args.catalog)
    catalog = {row["api_id"]: row for row in catalog_rows}
    for pair in pairs:
        for side in ("left_api_id", "right_api_id"):
            if pair[side] not in catalog:
                raise KeyError(f"{pair[side]!r} from pair {pair['pair_id']} is absent from the catalog")

    done = completed_pair_ids(args.output)
    pending = interleave_strata([pair for pair in pairs if pair["pair_id"] not in done])
    if args.limit:
        pending = pending[: args.limit]
    if not pending:
        print("No pending pairs.")
        return 0

    if args.dry_run:
        print(build_user_prompt(pending[0], catalog))
        strata = Counter((pair.get("sampling") or {}).get("stratum", "unspecified") for pair in pending)
        print("\nSelected retrieval strata: " + json.dumps(strata, ensure_ascii=False, sort_keys=True))
        print(f"\nPair: {pending[0]['pair_id']}; pending selected: {len(pending)}; API calls: 0")
        return 0

    api_key = os.getenv("ANNOTATOR_API_KEY")
    base_url = os.getenv("ANNOTATOR_API_BASE")
    model = os.getenv("ANNOTATOR_MODEL")
    missing = [name for name, value in (
        ("ANNOTATOR_API_KEY", api_key),
        ("ANNOTATOR_API_BASE", base_url),
        ("ANNOTATOR_MODEL", model),
    ) if not value]
    if missing:
        print(f"Missing environment variables: {', '.join(missing)}", file=sys.stderr)
        return 2

    try:
        from openai import OpenAI
    except ModuleNotFoundError:
        print(
            "Missing package 'openai'. Activate the main .venv used by StableToolBench "
            "or install it with: python -m pip install openai",
            file=sys.stderr,
        )
        return 2

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=args.timeout)
    strata = Counter((pair.get("sampling") or {}).get("stratum", "unspecified") for pair in pending)
    print("Selected retrieval strata: " + json.dumps(strata, ensure_ascii=False, sort_keys=True))
    successes = 0
    failures = 0
    for pair in tqdm(pending, desc="Annotating pairs", unit="pair"):
        prompt = build_user_prompt(pair, catalog)
        last_error: Exception | None = None
        for attempt in range(1, args.retries + 2):
            try:
                annotation, usage = annotate(
                    client, model, prompt, args.temperature, args.max_tokens
                )
                append_jsonl(args.output, {
                    "pair_id": pair["pair_id"],
                    "left_api_id": pair["left_api_id"],
                    "right_api_id": pair["right_api_id"],
                    "status": "ok",
                    "annotation": annotation,
                    "provenance": {
                        "model": model,
                        "api_base": base_url,
                        "prompt_version": PROMPT_VERSION,
                        "temperature": args.temperature,
                        "annotated_at": datetime.now(timezone.utc).isoformat(),
                        "usage": usage,
                    },
                })
                successes += 1
                last_error = None
                break
            except Exception as error:  # preserve progress and record provider/JSON failures
                last_error = error
                if attempt <= args.retries:
                    time.sleep(args.retry_delay * attempt)
        if last_error is not None:
            failures += 1
            append_jsonl(args.errors, {
                "pair_id": pair["pair_id"],
                "left_api_id": pair["left_api_id"],
                "right_api_id": pair["right_api_id"],
                "status": "error",
                "error_type": type(last_error).__name__,
                "error": str(last_error),
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "failed_at": datetime.now(timezone.utc).isoformat(),
            })

    print(f"Completed: {successes}; failed: {failures}; already present before run: {len(done)}")
    print(f"Annotations: {args.output}")
    if failures:
        print(f"Errors: {args.errors}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
