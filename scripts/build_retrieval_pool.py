"""Build a deduplicated tool-pair pool from dense retrieval and BM25.

The output is intended for subsequent LLM annotation. Retrieval signals are
kept separate from tool metadata so an annotator can be blinded later.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import torch
from tqdm.auto import tqdm

from build_candidate_groups import EMBEDDING_TEXT_VERSION, load_candidates


DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_MODE = "api_to_api_instruction_v1"
DEFAULT_QUERY_INSTRUCTION = (
    "Find API tools that provide the same capability, a substitutable capability, "
    "or a closely related capability. Pay attention to inputs, outputs, temporal "
    "semantics, side effects, and provider-specific identifiers."
)

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
LEXICAL_STOPWORDS = {
    "a", "an", "and", "api", "are", "as", "at", "be", "by", "category",
    "description", "for", "from", "in", "is", "it", "of", "on", "optional",
    "or", "parameter", "parameters", "required", "string", "the", "this",
    "to", "tool", "type", "use", "with",
}


def model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_").lower()


def fingerprint(
    records: list[dict[str, Any]],
    model_name: str,
    instruction: str,
    dtype: str,
    max_seq_length: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(EMBEDDING_TEXT_VERSION.encode("utf-8"))
    digest.update(EMBEDDING_MODE.encode("utf-8"))
    digest.update(model_name.encode("utf-8"))
    digest.update(instruction.encode("utf-8"))
    digest.update(dtype.encode("utf-8"))
    digest.update(str(max_seq_length).encode("ascii"))
    for record in records:
        digest.update(str(record["api_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(record["semantic_text"].encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def encode_dense(
    records: list[dict[str, Any]],
    model_name: str,
    instruction: str,
    device: str,
    dtype: str,
    max_seq_length: int,
    batch_size: int,
    cache_file: Path,
    show_progress: bool,
) -> tuple[np.ndarray, str]:
    expected_fingerprint = fingerprint(records, model_name, instruction, dtype, max_seq_length)
    if cache_file.is_file():
        cached = np.load(cache_file, allow_pickle=False)
        if str(cached["fingerprint"].item()) == expected_fingerprint and "embeddings" in cached:
            return cached["embeddings"], "cache"

    model_kwargs: dict[str, Any] = {}
    if dtype == "float16":
        model_kwargs["torch_dtype"] = torch.float16
    elif dtype == "float32":
        model_kwargs["torch_dtype"] = torch.float32
    model = SentenceTransformer(model_name, device=device, model_kwargs=model_kwargs)
    model.max_seq_length = max_seq_length
    texts = [record["semantic_text"] for record in records]
    encode_kwargs: dict[str, Any] = {}
    if instruction:
        encode_kwargs["prompt"] = f"Instruct: {instruction}\nQuery: "
    print("Encoding instruction-aware API embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
        **encode_kwargs,
    ).astype(np.float32)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_file,
        fingerprint=np.array(expected_fingerprint),
        embeddings=embeddings,
    )
    return embeddings, "model"


def dense_neighbors(
    embeddings: np.ndarray,
    top_k: int,
    n_jobs: int,
) -> dict[int, list[tuple[int, float]]]:
    neighbor_count = min(top_k + 1, len(embeddings))
    index = NearestNeighbors(
        n_neighbors=neighbor_count,
        metric="cosine",
        algorithm="brute",
        n_jobs=n_jobs,
    )
    index.fit(embeddings)
    distances, indices = index.kneighbors(embeddings)
    result: dict[int, list[tuple[int, float]]] = {}
    for source_index, (source_distances, source_neighbors) in enumerate(zip(distances, indices)):
        values = []
        for distance, candidate_index in zip(source_distances, source_neighbors):
            candidate_index = int(candidate_index)
            if candidate_index == source_index:
                continue
            values.append((candidate_index, 1.0 - float(distance)))
            if len(values) == top_k:
                break
        result[source_index] = values
    return result


def lexical_tokens(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if len(token) > 1 and token not in LEXICAL_STOPWORDS
    ]


def bm25_neighbors(
    records: list[dict[str, Any]],
    top_k: int,
    k1: float,
    b: float,
    show_progress: bool,
) -> dict[int, list[tuple[int, float]]]:
    documents = [lexical_tokens(record["semantic_text"]) for record in records]
    term_frequencies = [Counter(document) for document in documents]
    document_lengths = [len(document) for document in documents]
    average_length = sum(document_lengths) / len(document_lengths) if document_lengths else 0.0
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for document_index, frequencies in tqdm(
        enumerate(term_frequencies),
        total=len(term_frequencies),
        desc="BM25 index",
        unit="api",
        disable=not show_progress,
    ):
        for term, frequency in frequencies.items():
            postings[term].append((document_index, frequency))

    document_count = len(documents)
    inverse_document_frequency = {
        term: math.log(1.0 + (document_count - len(values) + 0.5) / (len(values) + 0.5))
        for term, values in postings.items()
    }
    result: dict[int, list[tuple[int, float]]] = {}
    for query_index, query_terms in tqdm(
        enumerate(term_frequencies),
        total=len(term_frequencies),
        desc="BM25 search",
        unit="api",
        disable=not show_progress,
    ):
        scores: dict[int, float] = defaultdict(float)
        for term in query_terms:
            idf = inverse_document_frequency[term]
            for document_index, frequency in postings[term]:
                if document_index == query_index:
                    continue
                length_normalization = 1.0 - b
                if average_length:
                    length_normalization += b * document_lengths[document_index] / average_length
                denominator = frequency + k1 * length_normalization
                scores[document_index] += idf * frequency * (k1 + 1.0) / denominator
        result[query_index] = heapq.nlargest(top_k, scores.items(), key=lambda item: (item[1], -item[0]))
    return result


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "api_id": record["api_id"],
        "tool_id": record["tool_id"],
        "tool_name": record["tool_name"],
        "api_name": record["api_name"],
        "category": record["category"],
        "tool_description": record["tool_description"],
        "api_description": record["api_description"],
        "required_parameters": record["required_parameters"],
        "optional_parameters": record["optional_parameters"],
    }


def add_method_pairs(
    pair_pool: dict[tuple[int, int], dict[str, Any]],
    method: str,
    neighbors: dict[int, list[tuple[int, float]]],
    show_progress: bool,
) -> None:
    for source_index, candidates in tqdm(
        neighbors.items(),
        total=len(neighbors),
        desc=f"Pooling {method}",
        unit="api",
        disable=not show_progress,
    ):
        for rank, (candidate_index, score) in enumerate(candidates, start=1):
            left_index, right_index = sorted((source_index, candidate_index))
            pair = pair_pool.setdefault(
                (left_index, right_index),
                {"left_index": left_index, "right_index": right_index, "retrieval": {}},
            )
            method_evidence = pair["retrieval"].setdefault(method, [])
            method_evidence.append(
                {
                    "source_index": source_index,
                    "candidate_index": candidate_index,
                    "rank": rank,
                    "score": round(float(score), 8),
                }
            )


def write_neighbors(
    records: list[dict[str, Any]],
    methods: dict[str, dict[int, list[tuple[int, float]]]],
    output_file: Path,
) -> None:
    with output_file.open("w", encoding="utf-8", newline="\n") as output:
        for source_index, source in enumerate(records):
            row = {"api_id": source["api_id"], "neighbors": {}}
            for method, neighbors in methods.items():
                row["neighbors"][method] = [
                    {
                        "api_id": records[candidate_index]["api_id"],
                        "rank": rank,
                        "score": round(float(score), 8),
                    }
                    for rank, (candidate_index, score) in enumerate(neighbors[source_index], start=1)
                ]
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_pairs(
    records: list[dict[str, Any]],
    pair_pool: dict[tuple[int, int], dict[str, Any]],
    output_file: Path,
    show_progress: bool,
) -> None:
    with output_file.open("w", encoding="utf-8", newline="\n") as output:
        sorted_pairs = sorted(pair_pool, key=lambda pair: (records[pair[0]]["api_id"], records[pair[1]]["api_id"]))
        for key in tqdm(
            sorted_pairs,
            desc="Writing pooled pairs",
            unit="pair",
            disable=not show_progress,
        ):
            pair = pair_pool[key]
            retrieval = {
                method: [
                    {
                        "source_api_id": records[evidence["source_index"]]["api_id"],
                        "candidate_api_id": records[evidence["candidate_index"]]["api_id"],
                        "rank": evidence["rank"],
                        "score": evidence["score"],
                    }
                    for evidence in evidence_list
                ]
                for method, evidence_list in pair["retrieval"].items()
            }
            row = {
                "pair_id": hashlib.sha256(
                    f'{records[key[0]]["api_id"]}\0{records[key[1]]["api_id"]}'.encode("utf-8")
                ).hexdigest()[:20],
                "left_api_id": records[key[0]]["api_id"],
                "right_api_id": records[key[1]]["api_id"],
                "retrieval": retrieval,
                "retrieved_by": sorted(retrieval),
                "llm_annotation": None,
                "human_annotation": None,
            }
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_annotation_catalog(records: list[dict[str, Any]], output_file: Path) -> None:
    with output_file.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(compact_record(record), ensure_ascii=False, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog/tools.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/retrieval"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--query-instruction", default=DEFAULT_QUERY_INSTRUCTION)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float16", "float32"), default="auto")
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--skip-bm25", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--limit", type=int, help="Use only the first N APIs for a smoke test.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be positive")
    if args.max_seq_length < 1:
        raise SystemExit("--max-seq-length must be positive")
    if args.skip_dense and args.skip_bm25:
        raise SystemExit("At least one retrieval method must be enabled")
    if not args.catalog.is_file():
        raise SystemExit(f"Catalog not found: {args.catalog}. Run scripts/build_tool_catalog.py first.")

    records, rejected = load_candidates(args.catalog)
    if args.limit is not None:
        if args.limit < 2:
            raise SystemExit("--limit must be at least 2")
        records = records[: args.limit]
    if len(records) < 2:
        raise SystemExit("At least two candidate APIs are required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    show_progress = not args.no_progress
    methods: dict[str, dict[int, list[tuple[int, float]]]] = {}
    dense_source = None
    dense_device = None
    dense_dtype = None
    dense_method = f"dense:{args.model}"
    if not args.skip_dense:
        dense_device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if dense_device == "auto":
            dense_device = "cpu"
        if dense_device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("--device cuda requested, but torch.cuda.is_available() is false")
        dense_dtype = args.dtype
        if dense_dtype == "auto":
            dense_dtype = "float16" if dense_device == "cuda" else "float32"
        if dense_device == "cpu" and dense_dtype == "float16":
            raise SystemExit("float16 is not supported for this CPU retrieval path; use --dtype float32")
        cache_file = args.output_dir / "model_indexes" / f"{model_slug(args.model)}.npz"
        embeddings, dense_source = encode_dense(
            records,
            args.model,
            args.query_instruction,
            dense_device,
            dense_dtype,
            args.max_seq_length,
            args.batch_size,
            cache_file,
            show_progress,
        )
        methods[dense_method] = dense_neighbors(
            embeddings,
            args.top_k,
            args.n_jobs,
        )
    if not args.skip_bm25:
        methods["bm25"] = bm25_neighbors(
            records,
            args.top_k,
            args.bm25_k1,
            args.bm25_b,
            show_progress,
        )

    pair_pool: dict[tuple[int, int], dict[str, Any]] = {}
    for method, neighbors in methods.items():
        add_method_pairs(pair_pool, method, neighbors, show_progress)

    neighbors_file = args.output_dir / "neighbors.jsonl"
    pairs_file = args.output_dir / "pooled_pairs.jsonl"
    annotation_catalog_file = args.output_dir / "catalog.jsonl"
    statistics_file = args.output_dir / "statistics.json"
    write_neighbors(records, methods, neighbors_file)
    write_pairs(records, pair_pool, pairs_file, show_progress)
    write_annotation_catalog(records, annotation_catalog_file)

    method_pair_counts = {
        method: sum(method in pair["retrieval"] for pair in pair_pool.values())
        for method in methods
    }
    overlap_count = sum(len(pair["retrieval"]) > 1 for pair in pair_pool.values())
    statistics = {
        "catalog": str(args.catalog),
        "embedding_text_version": EMBEDDING_TEXT_VERSION,
        "candidate_apis": len(records),
        "rejected": rejected,
        "top_k": args.top_k,
        "methods": sorted(methods),
        "dense_model": None if args.skip_dense else args.model,
        "dense_embedding_mode": None if args.skip_dense else EMBEDDING_MODE,
        "dense_embedding_source": dense_source,
        "dense_device": dense_device,
        "dense_dtype": dense_dtype,
        "dense_max_seq_length": None if args.skip_dense else args.max_seq_length,
        "query_instruction": None if args.skip_dense else args.query_instruction,
        "bm25": None if args.skip_bm25 else {"k1": args.bm25_k1, "b": args.bm25_b},
        "unique_pairs": len(pair_pool),
        "pairs_by_method": method_pair_counts,
        "pairs_retrieved_by_multiple_methods": overlap_count,
        "outputs": {
            "neighbors": str(neighbors_file),
            "pooled_pairs": str(pairs_file),
            "catalog": str(annotation_catalog_file),
        },
    }
    statistics_file.write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Candidate APIs: {len(records)}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Unique pairs: {len(pair_pool)}")
    print(f"Pairs from multiple methods: {overlap_count}")
    print(f"Pooled pairs: {pairs_file}")
    print(f"Statistics: {statistics_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
