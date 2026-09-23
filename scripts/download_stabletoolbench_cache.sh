#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESTINATION="${1:-$REPOSITORY_ROOT/server}"
CACHE_REVISION="${STABLETOOLBENCH_CACHE_REVISION:-c6aca0c}"
CACHE_URL="https://huggingface.co/datasets/stabletoolbench/Cache/resolve/${CACHE_REVISION}/server_cache.zip"

if [[ -d "$DESTINATION/tools" && -d "$DESTINATION/tool_response_cache" ]]; then
    echo "StableToolBench cache is already present in: $DESTINATION"
    exit 0
fi

command -v curl >/dev/null 2>&1 || {
    echo "curl is required but was not found." >&2
    exit 1
}
command -v python >/dev/null 2>&1 || {
    echo "python is required but was not found." >&2
    exit 1
}

TEMP_DIR="$(mktemp -d)"
trap 'rm -rf -- "$TEMP_DIR"' EXIT
ARCHIVE="$TEMP_DIR/server_cache.zip"
EXTRACT_DIR="$TEMP_DIR/extracted"

echo "Downloading StableToolBench Cache (${CACHE_REVISION})..."
curl --fail --location --retry 3 --output "$ARCHIVE" "$CACHE_URL"

echo "Validating and extracting the archive..."
python - "$ARCHIVE" "$EXTRACT_DIR" <<'PY'
from pathlib import Path
import sys
import zipfile

archive = Path(sys.argv[1])
destination = Path(sys.argv[2]).resolve()
destination.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(archive) as zip_file:
    for member in zip_file.infolist():
        target = (destination / member.filename).resolve()
        if target != destination and destination not in target.parents:
            raise RuntimeError(f"Unsafe path in archive: {member.filename}")
    zip_file.extractall(destination)
PY

SOURCE_ROOT="$EXTRACT_DIR"
if [[ -d "$EXTRACT_DIR/server/tools" && -d "$EXTRACT_DIR/server/tool_response_cache" ]]; then
    SOURCE_ROOT="$EXTRACT_DIR/server"
fi

if [[ ! -d "$SOURCE_ROOT/tools" || ! -d "$SOURCE_ROOT/tool_response_cache" ]]; then
    echo "Unexpected archive layout: tools/ or tool_response_cache/ is missing." >&2
    exit 1
fi

mkdir -p "$DESTINATION"
cp -R "$SOURCE_ROOT/tools" "$DESTINATION/"
cp -R "$SOURCE_ROOT/tool_response_cache" "$DESTINATION/"

echo "StableToolBench cache installed:"
echo "  $DESTINATION/tools"
echo "  $DESTINATION/tool_response_cache"
