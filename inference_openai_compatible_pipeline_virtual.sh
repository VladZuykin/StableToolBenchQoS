#!/usr/bin/env bash
set -euo pipefail

: "${AGENT_API_KEY:?Set AGENT_API_KEY before running this script}"
: "${AGENT_API_BASE:?Set AGENT_API_BASE before running this script}"
: "${AGENT_MODEL:?Set AGENT_MODEL before running this script}"

export NO_PROXY="localhost,127.0.0.1"
export no_proxy="$NO_PROXY"
export PYTHONPATH="${PYTHONPATH:-.}"
export SERVICE_URL="${SERVICE_URL:-http://localhost:8080/virtual}"

TOOL_ROOT_DIR="${TOOL_ROOT_DIR:-server/tools}"
INPUT_QUERY_FILE="${INPUT_QUERY_FILE:-solvable_queries_example/smoke/cache_hit.json}"
OUTPUT_DIR="${OUTPUT_DIR:-data/answer/agent_smoke/llm_virtual}"
TOOLBENCH_KEY="${TOOLBENCH_KEY:-dummy}"
SINGLE_CHAIN_MAX_STEP="${SINGLE_CHAIN_MAX_STEP:-20}"
OVERWRITE="${OVERWRITE:-false}"

if [[ ! -d "$TOOL_ROOT_DIR" ]]; then
    echo "Tool directory not found: $TOOL_ROOT_DIR" >&2
    echo "Download and unpack the StableToolBench tools before inference." >&2
    exit 2
fi

if [[ ! -f "$INPUT_QUERY_FILE" ]]; then
    echo "Query file not found: $INPUT_QUERY_FILE" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"

EXTRA_ARGS=()
if [[ "$OVERWRITE" == "true" ]]; then
    EXTRA_ARGS+=(--overwrite)
fi

python toolbench/inference/qa_pipeline_multithread.py \
    --tool_root_dir "$TOOL_ROOT_DIR" \
    --backbone_model chatgpt_function \
    --chatgpt_model "$AGENT_MODEL" \
    --openai_key "$AGENT_API_KEY" \
    --base_url "$AGENT_API_BASE" \
    --max_observation_length 1024 \
    --method CoT@1 \
    --input_query_file "$INPUT_QUERY_FILE" \
    --output_answer_file "$OUTPUT_DIR" \
    --toolbench_key "$TOOLBENCH_KEY" \
    --single_chain_max_step "$SINGLE_CHAIN_MAX_STEP" \
    --num_thread 1 \
    "${EXTRA_ARGS[@]}"
