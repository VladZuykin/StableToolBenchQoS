"""Export the relation decision log into explicit API clusters and cluster edges."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from expand_relation_graph import RelationGraph


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def cluster_id(member_ids: list[str]) -> str:
    payload = "\x1f".join(sorted(member_ids)).encode("utf-8")
    return "cluster_" + hashlib.sha256(payload).hexdigest()[:20]


def direct_decisions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("decision_source") in {"llm", "migrated_seed"}
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("data/relation_graph/v16_migrated"),
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/retrieval/catalog.jsonl"),
    )
    args = parser.parse_args()

    catalog_rows = load_jsonl(args.catalog)
    catalog = {row["api_id"]: row for row in catalog_rows}
    decisions_path = args.run_dir / "pair_decisions.jsonl"
    decisions = direct_decisions(load_jsonl(decisions_path))

    graph = RelationGraph(list(catalog))
    # Build equivalence components first so every non-equivalence assertion is
    # projected directly onto the final component identifiers.
    for row in decisions:
        annotation = row["annotation"]
        if annotation["relation"] == "interchangeable":
            graph.add_direct(
                row["left_api_id"],
                row["right_api_id"],
                "interchangeable",
                "none",
                row["decision_id"],
            )
    for row in decisions:
        annotation = row["annotation"]
        if annotation["relation"] != "interchangeable":
            graph.add_direct(
                row["left_api_id"],
                row["right_api_id"],
                annotation["relation"],
                annotation["direction"],
                row["decision_id"],
            )

    members_by_root: dict[str, list[str]] = defaultdict(list)
    for api_id in sorted(catalog):
        members_by_root[graph.dsu.find(api_id)].append(api_id)
    id_by_root = {
        root: cluster_id(members) for root, members in members_by_root.items()
    }
    id_by_api = {
        api_id: id_by_root[graph.dsu.find(api_id)] for api_id in catalog
    }

    equivalence_support: dict[str, list[str]] = defaultdict(list)
    for row in decisions:
        if row["annotation"]["relation"] == "interchangeable":
            equivalence_support[id_by_api[row["left_api_id"]]].append(row["decision_id"])

    clusters = []
    api_map = []
    for root, members in sorted(
        members_by_root.items(), key=lambda item: id_by_root[item[0]]
    ):
        cid = id_by_root[root]
        clusters.append({
            "cluster_id": cid,
            "member_count": len(members),
            "members": [
                {
                    "api_id": api_id,
                    "api_name": catalog[api_id].get("api_name"),
                    "tool_name": catalog[api_id].get("tool_name"),
                    "category": catalog[api_id].get("category"),
                }
                for api_id in members
            ],
            "supporting_interchangeable_decision_ids": sorted(
                set(equivalence_support.get(cid, []))
            ),
        })
        api_map.extend(
            {"api_id": api_id, "cluster_id": cid} for api_id in members
        )

    edge_support: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        annotation = row["annotation"]
        relation = annotation["relation"]
        if relation == "interchangeable":
            continue
        left_cluster = id_by_api[row["left_api_id"]]
        right_cluster = id_by_api[row["right_api_id"]]
        if left_cluster == right_cluster:
            continue
        if relation == "different_capability":
            source, target = sorted((left_cluster, right_cluster))
        elif annotation["direction"] == "left_contains_right":
            source, target = left_cluster, right_cluster
        else:
            source, target = right_cluster, left_cluster
        edge_support[(relation, source, target)].append({
            "decision_id": row["decision_id"],
            "pair_id": row["pair_id"],
            "decision_source": row["decision_source"],
            "left_api_id": row["left_api_id"],
            "right_api_id": row["right_api_id"],
            "confidence": row["annotation"].get("confidence"),
            "prompt_version": (
                row.get("provenance", {}).get("prompt_version")
                or row.get("provenance", {}).get("source_provenance", {}).get("prompt_version")
            ),
        })

    edges = [
        {
            "edge_id": "edge_" + hashlib.sha256(
                "\x1f".join(key).encode("utf-8")
            ).hexdigest()[:20],
            "relation": key[0],
            "source_cluster_id": key[1],
            "target_cluster_id": key[2],
            "support_count": len(support),
            "support": support,
        }
        for key, support in sorted(edge_support.items())
    ]

    diagnostics = list(graph.diagnostics)
    relation_by_pair: dict[frozenset[str], set[str]] = defaultdict(set)
    for edge in edges:
        key = frozenset((edge["source_cluster_id"], edge["target_cluster_id"]))
        relation_by_pair[key].add(edge["relation"])
    for cluster_pair, relations in relation_by_pair.items():
        if len(relations) > 1:
            diagnostics.append({
                "issue_type": "multiple_relations_between_clusters",
                "cluster_ids": sorted(cluster_pair),
                "relations": sorted(relations),
            })

    contains_adj: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        if edge["relation"] == "contains":
            contains_adj[edge["source_cluster_id"]].add(edge["target_cluster_id"])
    cycle_nodes = set()
    for start in contains_adj:
        pending = list(contains_adj[start])
        visited = set()
        while pending:
            node = pending.pop()
            if node == start:
                cycle_nodes.add(start)
                break
            if node not in visited:
                visited.add(node)
                pending.extend(contains_adj.get(node, set()))
    if cycle_nodes:
        diagnostics.append({
            "issue_type": "contains_cycle",
            "cluster_ids": sorted(cycle_nodes),
        })

    summary = {
        "source_decisions": str(decisions_path),
        "api_count": len(catalog),
        "direct_decision_count": len(decisions),
        "cluster_count": len(clusters),
        "singleton_cluster_count": sum(row["member_count"] == 1 for row in clusters),
        "non_singleton_cluster_count": sum(row["member_count"] > 1 for row in clusters),
        "largest_cluster_size": max((row["member_count"] for row in clusters), default=0),
        "cluster_size_distribution": dict(sorted(Counter(
            row["member_count"] for row in clusters
        ).items())),
        "edge_count": len(edges),
        "edge_relation_distribution": dict(sorted(Counter(
            row["relation"] for row in edges
        ).items())),
        "diagnostic_count": len(diagnostics),
    }

    write_jsonl(args.run_dir / "clusters.jsonl", clusters)
    write_jsonl(args.run_dir / "api_to_cluster.jsonl", sorted(api_map, key=lambda row: row["api_id"]))
    write_jsonl(args.run_dir / "cluster_edges.jsonl", edges)
    write_jsonl(args.run_dir / "graph_diagnostics.jsonl", diagnostics)
    (args.run_dir / "graph_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
