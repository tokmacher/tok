#!/usr/bin/env bash
set -euo pipefail
exec uv run python scripts/run_agent_smoke.py "$@"
