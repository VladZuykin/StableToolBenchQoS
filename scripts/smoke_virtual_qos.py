"""Send repeated virtual API calls and summarize observed QoS behavior."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from statistics import mean, median
import time
from typing import Any

import requests


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    return {
        "count": len(values),
        "mean": round(mean(values), 4),
        "median": round(median(values), 4),
        "p95": round(float(percentile(values, 0.95)), 4),
    }


def make_payload(index: int) -> dict[str, Any]:
    return {
        "category": "Artificial Intelligence/Machine Learning",
        "tool_name": "ai_content_detector_v2",
        "api_name": "chat_gpt_detector_for_ai_content_detector_v2",
        "tool_input": {
            "text": (
                f"QoS smoke test request {index}. "
                "This sentence is used to verify virtual response generation."
            )
        },
        "strip": "",
        "toolbench_key": "",
    }


def call_endpoint(url: str, index: int, timeout: float) -> dict[str, Any]:
    started = time.monotonic()
    try:
        response = requests.post(url, json=make_payload(index), timeout=timeout)
        wall_time_ms = (time.monotonic() - started) * 1000.0
        try:
            body: Any = response.json()
        except ValueError:
            body = {"raw_body": response.text}
        return {
            "request_index": index,
            "http_status": response.status_code,
            "wall_time_ms": round(wall_time_ms, 4),
            "body": body,
        }
    except Exception as error:
        return {
            "request_index": index,
            "http_status": None,
            "wall_time_ms": round((time.monotonic() - started) * 1000.0, 4),
            "transport_error": f"{type(error).__name__}: {error}",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080/virtual")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/smoke/virtual_qos_100.jsonl"),
    )
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(call_endpoint, args.url, index, args.timeout): index
            for index in range(args.count)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            body = result.get("body")
            qos = body.get("qos") if isinstance(body, dict) else None
            state = (
                "success"
                if isinstance(qos, dict) and qos.get("succeeded") is True
                else "qos_failure"
                if isinstance(qos, dict) and qos.get("succeeded") is False
                else "http_or_transport_error"
            )
            print(
                f"[{completed}/{args.count}] request={result['request_index']} "
                f"state={state} wall_ms={result['wall_time_ms']}",
                flush=True,
            )

    results.sort(key=lambda item: item["request_index"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for result in results:
            output.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")

    qos_records = [
        result["body"]["qos"]
        for result in results
        if isinstance(result.get("body"), dict)
        and isinstance(result["body"].get("qos"), dict)
    ]
    successes = [record for record in qos_records if record.get("succeeded") is True]
    failures = [record for record in qos_records if record.get("succeeded") is False]
    http_errors = [result for result in results if result.get("http_status") != 200]
    application_errors = [
        result
        for result in results
        if isinstance(result.get("body"), dict)
        and result["body"].get("error")
        and not (
            isinstance(result["body"].get("qos"), dict)
            and result["body"]["qos"].get("succeeded") is False
        )
    ]
    summary = {
        "requests": len(results),
        "workers": args.workers,
        "qos_records": len(qos_records),
        "succeeded": len(successes),
        "qos_failed": len(failures),
        "observed_success_rate": (
            round(len(successes) / len(qos_records), 4) if qos_records else None
        ),
        "configured_success_rate": (
            qos_records[0].get("success_rate") if qos_records else None
        ),
        "http_or_transport_errors": len(http_errors),
        "unexpected_application_errors": len(application_errors),
        "sampled_latency_ms": numeric_summary(
            [float(record["latency_ms"]) for record in qos_records]
        ),
        "wall_time_ms": numeric_summary(
            [float(result["wall_time_ms"]) for result in results]
        ),
        "total_cost_units": round(
            sum(float(record.get("cost_units") or 0.0) for record in qos_records),
            10,
        ),
        "call_indices": sorted(
            record["call_index"]
            for record in qos_records
            if record.get("call_index") is not None
        ),
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if http_errors or application_errors or len(qos_records) != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
