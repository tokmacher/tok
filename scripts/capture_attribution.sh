#!/usr/bin/env bash
set -euo pipefail

SESSION="${1:-}"
if [[ -z "$SESSION" ]]; then
    echo "Usage: scripts/capture_attribution.sh <session-name>" >&2
    exit 2
fi

mkdir -p tmp

echo "=== ${SESSION} pre-session ==="
uv run tok --version
uv run tok bridge status || true
git status --short

echo "=== Starting session with trace ==="
TOK_TRACE=1 TOK_TRACE_CAPTURE_ARTIFACTS=1 uv run tok claude

echo "=== ${SESSION} post-session ==="
uv run tok bridge status > "tmp/attribution_${SESSION}_status.txt" 2>&1 || true
uv run tok doctor > "tmp/attribution_${SESSION}_doctor.txt" 2>&1 || true
uv run tok stats > "tmp/attribution_${SESSION}_stats.txt" 2>&1 || true
uv run tok audit --latest > "tmp/attribution_${SESSION}_audit.txt" 2>&1 || true
uv run tok bridge logs 120 > "tmp/attribution_${SESSION}_logs.txt" 2>&1 || true
uv run tok bridge stop || true

echo "=== Captures saved to tmp/attribution_${SESSION}_*.txt ==="
