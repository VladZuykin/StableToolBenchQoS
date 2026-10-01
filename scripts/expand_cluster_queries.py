"""Generate resumable meaning-preserving paraphrases for cluster tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

from openai import OpenAI


VERSION = "cluster-query-paraphrases-v1"
PROMPT_VERSION = "meaning-preserving-paraphrases-v1"


def artifact_path(root: Path, query_id: object) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(query_id))
    suffix = hashlib.sha256(str(query_id).encode()).hexdigest()[:10]
    return root / f"{safe}_{suffix}.json"


def parse_paraphrases(text: str, count: int) -> list[str]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\\s*", "", text, flags=re.I)
        text = re.sub(r"\\s*```$", "", text)
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("paraphrases")
    if not isinstance(data, list):
        raise ValueError("Response is not a JSON array")
    result = []
    for value in data:
        if not isinstance(value, str):
            raise ValueError("A paraphrase is not a string")
        value = " ".join(value.split())
        if value and value not in result:
            result.append(value)
    if len(result) != count:
        raise ValueError(f"Expected {count} unique strings, received {len(result)}")
    return result


def make_prompt(query: str, count: int) -> str:
    return f"""Create exactly {count} diverse English paraphrases of this request.

Preserve every operation, entity, number, identifier, date, location, constraint,
and requested output detail. Do not add or remove requirements. Do not answer the
request. Do not name APIs or tools unless the source does. Vary wording and
sentence structure. Return only a JSON array of exactly {count} strings.

SOURCE REQUEST:
{query}
"""


def generate(client, model, query, count, temperature, max_tokens):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Return strict JSON with meaning-preserving paraphrases.",
            },
            {"role": "user", "content": make_prompt(query, count)},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    usage = response.usage
    return parse_paraphrases(response.choices[0].message.content or "", count), {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/benchmark/cluster_queries_v1/queries.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/benchmark/cluster_queries_paraphrased_v1"),
    )
    parser.add_argument("--paraphrases", type=int, default=20)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-llm-calls", type=int)
    parser.add_argument("--api-key")
    parser.add_argument("--api-base")
    parser.add_argument("--model")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=6000)
    args = parser.parse_args()

    if args.paraphrases < 1:
        parser.error("--paraphrases must be positive")
    if args.max_llm_calls is not None and args.max_llm_calls < 1:
        parser.error("--max-llm-calls must be positive")
    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    records = json.loads(args.input.read_text(encoding="utf-8"))
    ids = [str(row["query_id"]) for row in records]
    if len(ids) != len(set(ids)):
        raise SystemExit("Source query_id values must be unique")

    artifacts = args.output_dir / "generations"
    completed = {}
    for row in records:
        path = artifact_path(artifacts, row["query_id"])
        if not path.exists():
            continue
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            if (
                item["source_query"] == row["query"]
                and item["count"] == args.paraphrases
                and len(item["paraphrases"]) == args.paraphrases
            ):
                completed[str(row["query_id"])] = item
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    pending = [row for row in records if str(row["query_id"]) not in completed]
    api_base = (
        args.api_base
        or os.getenv("PARAPHRASE_API_BASE")
        or os.getenv("SIMULATOR_API_BASE")
        or "https://api.openai.com/v1"
    )
    model = (
        args.model
        or os.getenv("PARAPHRASE_MODEL")
        or os.getenv("SIMULATOR_MODEL")
    )
    plan = {
        "version": VERSION,
        "source_queries": len(records),
        "paraphrases_per_query": args.paraphrases,
        "expected_total_with_originals": len(records) * (args.paraphrases + 1),
        "completed_source_queries": len(completed),
        "pending_llm_calls": len(pending),
        "execute": args.execute,
        "api_base": api_base,
        "model": model,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    if not args.execute:
        print("Dry run only. Add --execute to call the LLM.")
        return

    api_key = (
        args.api_key
        or os.getenv("PARAPHRASE_API_KEY")
        or os.getenv("SIMULATOR_API_KEY")
    )
    if pending and not api_key:
        raise SystemExit("Set PARAPHRASE_API_KEY or SIMULATOR_API_KEY")
    if pending and not model:
        raise SystemExit("Set PARAPHRASE_MODEL or SIMULATOR_MODEL")

    limit = min(
        len(pending),
        args.max_llm_calls if args.max_llm_calls is not None else len(pending),
    )
    client = OpenAI(api_key=api_key, base_url=api_base) if pending else None
    artifacts.mkdir(parents=True, exist_ok=True)
    calls = failures = 0
    for position, row in enumerate(pending[:limit], start=1):
        try:
            variants, usage = generate(
                client, model, row["query"], args.paraphrases,
                args.temperature, args.max_tokens,
            )
            item = {
                "version": VERSION,
                "prompt_version": PROMPT_VERSION,
                "source_query_id": row["query_id"],
                "source_query": row["query"],
                "count": args.paraphrases,
                "model": model,
                "paraphrases": variants,
                "usage": usage,
            }
            artifact_path(artifacts, row["query_id"]).write_text(
                json.dumps(item, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            completed[str(row["query_id"])] = item
            calls += 1
            print(
                f"[{position}/{limit}] query_id={row['query_id']} "
                f"generated={len(variants)}"
            )
        except Exception as error:
            failures += 1
            print(f"[{position}/{limit}] query_id={row['query_id']} error={error}")

    output = []
    for row in records:
        original = dict(row)
        original["paraphrase"] = {
            "version": VERSION,
            "source_query_id": row["query_id"],
            "variant_index": 0,
            "is_original": True,
        }
        output.append(original)
        item = completed.get(str(row["query_id"]))
        if item is None:
            continue
        for index, text in enumerate(item["paraphrases"], start=1):
            variant = dict(row)
            variant["query"] = text
            variant["query_id"] = f"{row['query_id']}-p{index:02d}"
            variant["paraphrase"] = {
                "version": VERSION,
                "prompt_version": PROMPT_VERSION,
                "source_query_id": row["query_id"],
                "variant_index": index,
                "model": item["model"],
            }
            output.append(variant)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "queries.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = {
        **plan,
        "llm_calls_this_run": calls,
        "failures_this_run": failures,
        "completed_source_queries": len(completed),
        "remaining_source_queries": len(records) - len(completed),
        "output_query_count": len(output),
        "output": str(output_path),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
