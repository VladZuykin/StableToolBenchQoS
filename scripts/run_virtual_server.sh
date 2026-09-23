#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SERVER_MODE="${SERVER_MODE:-cache_only}"

cd "$REPOSITORY_ROOT/server"
exec python main.py
