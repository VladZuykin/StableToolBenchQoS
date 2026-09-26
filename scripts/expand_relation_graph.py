"""Expand an API relation graph in descending Qwen-score order with an LLM budget."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from tqdm.auto import tqdm

from annotate_candidate_pairs import (
    PROMPT_VERSION,
    annotate,
    append_jsonl,
    build_user_prompt,
    load_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]
QWEN_PREFIX = "dense:Qwen/"


def canonical_relation(value: str) -> str:
    return "interchangeable" if value == "exact_duplicate" else value


def stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


def qwen_score(pair: dict[str, Any]) -> float | None:
    scores = [
        float(item["score"])
        for method, items in pair.get("retrieval", {}).items()
        if method.startswith(QWEN_PREFIX)
        for item in items
    ]
    return max(scores) if scores else None


def qwen_ranks(pair: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "source_api_id": item["source_api_id"],
            "candidate_api_id": item["candidate_api_id"],
            "rank": item["rank"],
            "score": item["score"],
        }
        for method, items in pair.get("retrieval", {}).items()
        if method.startswith(QWEN_PREFIX)
        for item in items
    ]


class DisjointSet:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}
        self.size = {item: 1 for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> bool:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return False
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]
        return True


class RelationGraph:
    """Direct assertions plus relations derived after interchangeable contraction."""

    def __init__(self, api_ids: list[str]) -> None:
        self.dsu = DisjointSet(api_ids)
        self.equivalence_adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.assertions: list[dict[str, Any]] = []
        self.contains_adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self.different: dict[frozenset[str], list[str]] = defaultdict(list)
        self.diagnostics: list[dict[str, Any]] = []

    def add_direct(
        self,
        left_api_id: str,
        right_api_id: str,
        relation: str,
        direction: str,
        decision_id: str,
    ) -> None:
        if relation == "interchangeable":
            self.equivalence_adj[left_api_id].append((right_api_id, decision_id))
            self.equivalence_adj[right_api_id].append((left_api_id, decision_id))
            if self.dsu.union(left_api_id, right_api_id):
                self._rebuild_indexes()
            return
        self.assertions.append({
            "left_api_id": left_api_id,
            "right_api_id": right_api_id,
            "relation": relation,
            "direction": direction,
            "decision_id": decision_id,
        })
        self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        self.contains_adj = defaultdict(list)
        self.different = defaultdict(list)
        self.diagnostics = []
        for assertion in self.assertions:
            left_root = self.dsu.find(assertion["left_api_id"])
            right_root = self.dsu.find(assertion["right_api_id"])
            if left_root == right_root:
                self.diagnostics.append({
                    "issue_type": "relation_inside_interchangeable_cluster",
                    "decision_id": assertion["decision_id"],
                    "relation": assertion["relation"],
                    "cluster_root": left_root,
                })
                continue
            if assertion["relation"] == "different_capability":
                self.different[frozenset((left_root, right_root))].append(
                    assertion["decision_id"]
                )
                continue
            if assertion["direction"] == "left_contains_right":
                container, contained = left_root, right_root
            else:
                container, contained = right_root, left_root
            self.contains_adj[container].append((contained, assertion["decision_id"]))

    def _equivalence_path(self, start: str, target: str) -> list[str]:
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            node, path = queue.popleft()
            if node == target:
                return path
            for neighbor, decision_id in self.equivalence_adj.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [decision_id]))
        return []

    def _contains_path(self, start_root: str, target_root: str) -> list[str]:
        queue = deque([(start_root, [])])
        visited = {start_root}
        while queue:
            node, path = queue.popleft()
            if node == target_root:
                return path
            for neighbor, decision_id in self.contains_adj.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [decision_id]))
        return []

    def infer(self, left_api_id: str, right_api_id: str) -> dict[str, Any] | None:
        left_root = self.dsu.find(left_api_id)
        right_root = self.dsu.find(right_api_id)
        if left_root == right_root:
            return {
                "relation": "interchangeable",
                "direction": "none",
                "inference_rule": "same_interchangeable_cluster",
                "supporting_decision_ids": self._equivalence_path(left_api_id, right_api_id),
            }

        different_support = self.different.get(frozenset((left_root, right_root)))
        if different_support:
            return {
                "relation": "different_capability",
                "direction": "none",
                "inference_rule": "cluster_different_propagation",
                "supporting_decision_ids": different_support,
            }

        forward = self._contains_path(left_root, right_root)
        if forward:
            return {
                "relation": "contains",
                "direction": "left_contains_right",
                "inference_rule": (
                    "cluster_contains_propagation" if len(forward) == 1
                    else "transitive_cluster_contains"
                ),
                "supporting_decision_ids": forward,
            }
        backward = self._contains_path(right_root, left_root)
        if backward:
            return {
                "relation": "contains",
                "direction": "right_contains_left",
                "inference_rule": (
                    "cluster_contains_propagation" if len(backward) == 1
                    else "transitive_cluster_contains"
                ),
                "supporting_decision_ids": backward,
            }
        return None

    def cluster_statistics(self) -> dict[str, Any]:
        sizes = Counter(self.dsu.find(api_id) for api_id in self.dsu.parent)
        return {
            "cluster_count": len(sizes),
            "non_singleton_clusters": sum(size > 1 for size in sizes.values()),
            "largest_cluster_size": max(sizes.values(), default=0),
            "contains_direct_assertions": sum(
                assertion["relation"] == "contains" for assertion in self.assertions
            ),
            "different_direct_assertions": sum(
                assertion["relation"] == "different_capability" for assertion in self.assertions
            ),
            "diagnostic_issue_count": len(self.diagnostics),
        }

    def candidate_coverage(
        self,
        pairs: list[dict[str, Any]],
        direct_pair_ids: set[str],
        completed_pair_ids: set[str],
    ) -> dict[str, Any]:
        """Count candidate pairs whose relation follows from the current graph."""
        roots = {self.dsu.find(api_id) for api_id in self.dsu.parent}
        reachable: dict[str, set[str]] = {}
        for root in roots:
            visited: set[str] = set()
            queue = deque([root])
            while queue:
                node = queue.popleft()
                for neighbor, _decision_id in self.contains_adj.get(node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            reachable[root] = visited

        counts: Counter[str] = Counter()
        eligible = 0
        already_processed_inferred = 0
        remaining_inferable = 0
        for pair in pairs:
            if pair["pair_id"] in direct_pair_ids:
                continue
            eligible += 1
            left_root = self.dsu.find(pair["left_api_id"])
            right_root = self.dsu.find(pair["right_api_id"])
            if left_root == right_root:
                counts["interchangeable"] += 1
                inferred = True
            elif frozenset((left_root, right_root)) in self.different:
                counts["different_capability"] += 1
                inferred = True
            elif (
                right_root in reachable.get(left_root, set())
                or left_root in reachable.get(right_root, set())
            ):
                counts["contains"] += 1
                inferred = True
            else:
                inferred = False
            if inferred:
                if pair["pair_id"] in completed_pair_ids:
                    already_processed_inferred += 1
                else:
                    remaining_inferable += 1
        covered = sum(counts.values())
        return {
            "eligible_candidate_pairs": eligible,
            "covered_without_llm": covered,
            "already_processed_inferred": already_processed_inferred,
            "remaining_inferable": remaining_inferable,
            "coverage_rate": covered / eligible if eligible else 0.0,
            "by_relation": {
                relation: counts.get(relation, 0)
                for relation in ("interchangeable", "contains", "different_capability")
            },
        }


def latest_human_reviews(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in load_jsonl(path):
        latest[record["pair_id"]] = record
    return latest


def direct_decisions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        row for row in load_jsonl(path)
        if row.get("status") == "ok"
        and row.get("decision_source") in {"llm", "migrated_seed"}
    ]


def write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=Path("data/retrieval/pooled_pairs.jsonl"))
    parser.add_argument("--catalog", type=Path, default=Path("data/retrieval/catalog.jsonl"))
    parser.add_argument("--human-reviews", type=Path, default=Path("data/annotations/human_pair_reviews.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/relation_graph/pair_decisions.jsonl"))
    parser.add_argument("--errors", type=Path, default=Path("data/relation_graph/errors.jsonl"))
    parser.add_argument("--checkpoint", type=Path, default=Path("data/relation_graph/checkpoint.json"))
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Store pair_decisions.jsonl, errors.jsonl, and checkpoint.json in this directory",
    )
    parser.add_argument(
        "--no-human-seeds",
        action="store_true",
        help="Build the graph without loading human_pair_reviews.jsonl as initial decisions",
    )
    parser.add_argument("--max-llm-calls", type=int, default=1000)
    parser.add_argument("--max-pairs", type=int, default=0, help="Debug limit; 0 means no pair limit")
    parser.add_argument(
        "--min-qwen-score",
        type=float,
        default=None,
        help="Only consider Qwen candidate pairs with cosine similarity at least this value",
    )
    parser.add_argument(
        "--cross-tool-only",
        action="store_true",
        help="Only consider pairs whose category/tool_id prefixes differ",
    )
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_llm_calls < 0 or args.max_pairs < 0 or args.retries < 0:
        parser.error("limits and retries must be non-negative")
    if args.run_dir is not None:
        args.output = args.run_dir / "pair_decisions.jsonl"
        args.errors = args.run_dir / "errors.jsonl"
        args.checkpoint = args.run_dir / "checkpoint.json"

    catalog_rows = load_jsonl(args.catalog)
    catalog = {row["api_id"]: row for row in catalog_rows}
    pair_by_id: dict[str, dict[str, Any]] = {}
    qwen_pairs: list[dict[str, Any]] = []
    with args.pairs.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            pair = json.loads(line)
            pair_by_id[pair["pair_id"]] = pair
            score = qwen_score(pair)
            if score is not None:
                if args.min_qwen_score is not None and score < args.min_qwen_score:
                    continue
                if args.cross_tool_only:
                    left_tool = pair["left_api_id"].split("/", 2)[:2]
                    right_tool = pair["right_api_id"].split("/", 2)[:2]
                    if left_tool == right_tool:
                        continue
                pair["_qwen_score"] = score
                qwen_pairs.append(pair)
    qwen_pairs.sort(key=lambda pair: (-pair["_qwen_score"], pair["pair_id"]))

    graph = RelationGraph(list(catalog))
    reviews = {} if args.no_human_seeds else latest_human_reviews(args.human_reviews)
    seed_counts: Counter[str] = Counter()
    direct_pair_ids: set[str] = set()
    # Union all human interchangeable decisions before projecting other relations to clusters.
    for pair_id, review in reviews.items():
        relation = canonical_relation(review["human_relation"])
        if relation != "interchangeable" or pair_id not in pair_by_id:
            continue
        pair = pair_by_id[pair_id]
        decision_id = stable_id("human", pair_id, review.get("reviewed_at", ""))
        graph.add_direct(pair["left_api_id"], pair["right_api_id"], relation, "none", decision_id)
        direct_pair_ids.add(pair_id)
        seed_counts[relation] += 1
    for pair_id, review in reviews.items():
        relation = canonical_relation(review["human_relation"])
        if relation == "interchangeable" or relation not in {"contains", "different_capability"}:
            continue
        if pair_id not in pair_by_id:
            continue
        pair = pair_by_id[pair_id]
        direction = review.get("human_direction", "none")
        decision_id = stable_id("human", pair_id, review.get("reviewed_at", ""))
        graph.add_direct(pair["left_api_id"], pair["right_api_id"], relation, direction, decision_id)
        direct_pair_ids.add(pair_id)
        seed_counts[relation] += 1

    existing_direct = direct_decisions(args.output)
    llm_decision_counts_total: Counter[str] = Counter()
    migrated_seed_counts: Counter[str] = Counter()
    for row in existing_direct:
        annotation = row["annotation"]
        if row.get("decision_source") == "llm":
            llm_decision_counts_total[annotation["relation"]] += 1
        else:
            migrated_seed_counts[annotation["relation"]] += 1
        graph.add_direct(
            row["left_api_id"], row["right_api_id"], annotation["relation"],
            annotation["direction"], row["decision_id"],
        )
        direct_pair_ids.add(row["pair_id"])
    completed = {
        row["pair_id"] for row in load_jsonl(args.output)
    } if args.output.exists() else set()
    existing_requests = sum(
        int(row.get("request_count", 0)) for row in load_jsonl(args.output)
    ) if args.output.exists() else 0
    existing_requests += sum(
        int(row.get("request_count", 0)) for row in load_jsonl(args.errors)
    ) if args.errors.exists() else 0
    remaining_requests = max(0, args.max_llm_calls - existing_requests)
    graph_coverage = graph.candidate_coverage(qwen_pairs, direct_pair_ids, completed)
    graph_coverage_before_run = deepcopy(graph_coverage)

    preview_counts: Counter[str] = Counter()
    preview_unknown: dict[str, Any] | None = None
    examined = 0
    for pair in qwen_pairs:
        if pair["pair_id"] in completed or pair["pair_id"] in direct_pair_ids:
            continue
        if args.max_pairs and examined >= args.max_pairs:
            break
        examined += 1
        inferred = graph.infer(pair["left_api_id"], pair["right_api_id"])
        if inferred:
            preview_counts[f"inferred:{inferred['relation']}"] += 1
        else:
            preview_counts["requires_llm"] += 1
            if preview_unknown is None:
                preview_unknown = pair
        if args.dry_run and examined >= 10000:
            break

    if args.dry_run:
        print(f"Catalog APIs: {len(catalog)}")
        print(f"Qwen candidate pairs: {len(qwen_pairs)}")
        print(f"Human seeds: {dict(sorted(seed_counts.items()))}")
        print(f"Human seeds enabled: {not args.no_human_seeds}")
        print(f"Run directory: {args.output.parent}")
        print(f"Existing direct LLM decisions: {len(existing_direct)}")
        print(f"Migrated seeds: {dict(sorted(migrated_seed_counts.items()))}")
        print(f"Existing HTTP requests: {existing_requests}/{args.max_llm_calls}")
        print(f"Previewed pending pairs: {examined}")
        print("Preview decisions: " + json.dumps(preview_counts, sort_keys=True))
        print("Graph: " + json.dumps(graph.cluster_statistics(), sort_keys=True))
        print("Graph candidate coverage: " + json.dumps(graph_coverage, sort_keys=True))
        if preview_unknown:
            print(
                "First pair requiring LLM: "
                f"{preview_unknown['pair_id']} score={preview_unknown['_qwen_score']:.8f} "
                f"{preview_unknown['left_api_id']} <> {preview_unknown['right_api_id']}"
            )
        print("API calls: 0")
        return 0

    if remaining_requests == 0:
        print(f"LLM request budget already exhausted: {existing_requests}/{args.max_llm_calls}")
        return 0

    api_key = os.getenv("ANNOTATOR_API_KEY")
    base_url = os.getenv("ANNOTATOR_API_BASE")
    model = os.getenv("ANNOTATOR_MODEL")
    missing = [name for name, value in (
        ("ANNOTATOR_API_KEY", api_key), ("ANNOTATOR_API_BASE", base_url),
        ("ANNOTATOR_MODEL", model),
    ) if not value]
    if missing:
        print(f"Missing environment variables: {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        from openai import OpenAI
    except ModuleNotFoundError:
        print("Missing package 'openai'. Activate the main .venv.", file=sys.stderr)
        return 2
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=args.timeout)

    run_counts: Counter[str] = Counter()
    requests_this_run = 0
    examined_this_run = 0
    inferred_this_run = 0
    last_coverage_delta = 0
    request_progress = tqdm(
        total=args.max_llm_calls,
        initial=existing_requests,
        desc="DeepSeek requests",
        unit="request",
    )
    request_progress.set_postfix(
        pairs=0,
        inferred=0,
        covered=graph_coverage["covered_without_llm"],
        saved=graph_coverage["already_processed_inferred"],
        refresh=False,
    )
    for pair in qwen_pairs:
        if pair["pair_id"] in completed or pair["pair_id"] in direct_pair_ids:
            continue
        if args.max_pairs and examined_this_run >= args.max_pairs:
            break
        examined_this_run += 1
        inferred = graph.infer(pair["left_api_id"], pair["right_api_id"])
        now = datetime.now(timezone.utc).isoformat()
        if inferred:
            decision_id = stable_id("inferred", pair["pair_id"], inferred["inference_rule"])
            append_jsonl(args.output, {
                "decision_id": decision_id,
                "pair_id": pair["pair_id"],
                "left_api_id": pair["left_api_id"],
                "right_api_id": pair["right_api_id"],
                "status": "ok",
                "decision_source": "inferred",
                "annotation": {
                    "relation": inferred["relation"],
                    "direction": inferred["direction"],
                },
                "provenance": {
                    "inference_rule": inferred["inference_rule"],
                    "supporting_decision_ids": inferred["supporting_decision_ids"],
                    "qwen_score": pair["_qwen_score"],
                    "qwen_retrieval": qwen_ranks(pair),
                    "decided_at": now,
                },
                "request_count": 0,
            })
            completed.add(pair["pair_id"])
            run_counts[f"inferred:{inferred['relation']}"] += 1
            inferred_this_run += 1
            graph_coverage["already_processed_inferred"] += 1
            graph_coverage["remaining_inferable"] -= 1
            request_progress.set_postfix(
                pairs=examined_this_run,
                inferred=inferred_this_run,
                covered=graph_coverage["covered_without_llm"],
                saved=graph_coverage["already_processed_inferred"],
                delta=last_coverage_delta,
                refresh=False,
            )
            continue

        if existing_requests + requests_this_run >= args.max_llm_calls:
            break
        prompt = build_user_prompt(pair, catalog)
        last_error: Exception | None = None
        requests_for_pair = 0
        annotation: dict[str, Any] | None = None
        usage: dict[str, Any] | None = None
        for attempt in range(1, args.retries + 2):
            if existing_requests + requests_this_run >= args.max_llm_calls:
                break
            requests_this_run += 1
            requests_for_pair += 1
            request_progress.update(1)
            request_progress.set_postfix(
                pairs=examined_this_run,
                inferred=inferred_this_run,
                covered=graph_coverage["covered_without_llm"],
                saved=graph_coverage["already_processed_inferred"],
                delta=last_coverage_delta,
                refresh=False,
            )
            try:
                annotation, usage = annotate(
                    client, model, prompt, args.temperature, args.max_tokens
                )
                last_error = None
                break
            except Exception as error:
                last_error = error
                if attempt <= args.retries and existing_requests + requests_this_run < args.max_llm_calls:
                    time.sleep(args.retry_delay * attempt)
        if annotation is None:
            if last_error is not None:
                append_jsonl(args.errors, {
                    "pair_id": pair["pair_id"],
                    "left_api_id": pair["left_api_id"],
                    "right_api_id": pair["right_api_id"],
                    "error_type": type(last_error).__name__,
                    "error": str(last_error),
                    "request_count": requests_for_pair,
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                })
                run_counts["llm:error"] += 1
            if existing_requests + requests_this_run >= args.max_llm_calls:
                break
            continue
        decision_id = stable_id("llm", pair["pair_id"], model, PROMPT_VERSION)
        coverage_before = graph_coverage
        graph.add_direct(
            pair["left_api_id"], pair["right_api_id"], annotation["relation"],
            annotation["direction"], decision_id,
        )
        direct_pair_ids.add(pair["pair_id"])
        graph_coverage = graph.candidate_coverage(qwen_pairs, direct_pair_ids, completed)
        coverage_delta = (
            graph_coverage["covered_without_llm"]
            - coverage_before["covered_without_llm"]
        )
        last_coverage_delta = coverage_delta
        append_jsonl(args.output, {
            "decision_id": decision_id,
            "pair_id": pair["pair_id"],
            "left_api_id": pair["left_api_id"],
            "right_api_id": pair["right_api_id"],
            "status": "ok",
            "decision_source": "llm",
            "annotation": annotation,
            "provenance": {
                "model": model,
                "api_base": base_url,
                "prompt_version": PROMPT_VERSION,
                "temperature": args.temperature,
                "usage": usage,
                "qwen_score": pair["_qwen_score"],
                "qwen_retrieval": qwen_ranks(pair),
                "decided_at": datetime.now(timezone.utc).isoformat(),
                "candidate_coverage_impact": {
                    "before": coverage_before,
                    "after": graph_coverage,
                    "delta_covered_without_llm": coverage_delta,
                },
            },
            "request_count": requests_for_pair,
        })
        completed.add(pair["pair_id"])
        run_counts[f"llm:{annotation['relation']}"] += 1
        llm_decision_counts_total[annotation["relation"]] += 1
        request_progress.set_postfix(
            pairs=examined_this_run,
            inferred=inferred_this_run,
            covered=graph_coverage["covered_without_llm"],
            saved=graph_coverage["already_processed_inferred"],
            delta=coverage_delta,
            refresh=False,
        )

    request_progress.close()

    checkpoint = {
        "checkpoint_version": "relation-graph-v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "ordering": "qwen_score_desc",
        "min_qwen_score": args.min_qwen_score,
        "cross_tool_only": args.cross_tool_only,
        "human_seeds_enabled": not args.no_human_seeds,
        "run_directory": str(args.output.parent),
        "max_llm_calls": args.max_llm_calls,
        "llm_requests_before_run": existing_requests,
        "llm_requests_this_run": requests_this_run,
        "llm_requests_total": existing_requests + requests_this_run,
        "completed_pair_count": len(completed),
        "run_counts": dict(sorted(run_counts.items())),
        "graph_statistics": graph.cluster_statistics(),
        "graph_candidate_coverage": graph_coverage,
        "graph_candidate_coverage_before_run": graph_coverage_before_run,
        "graph_candidate_coverage_delta": (
            graph_coverage["covered_without_llm"]
            - graph_coverage_before_run["covered_without_llm"]
        ),
        "llm_decision_counts_total": dict(sorted(llm_decision_counts_total.items())),
        "migrated_seed_counts": dict(sorted(migrated_seed_counts.items())),
        "llm_retry_or_failed_request_count_total": (
            existing_requests + requests_this_run - sum(llm_decision_counts_total.values())
        ),
        "diagnostics": graph.diagnostics,
    }
    write_checkpoint(args.checkpoint, checkpoint)
    print(json.dumps(checkpoint, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
