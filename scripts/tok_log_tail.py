#!/usr/bin/env python3
"""Rolling tail view of the Tok proxy log with live savings stats."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tok.stats import SavingsTracker

LOG_PATH = Path(os.getenv("TOK_PROXY_LOG", "/tmp/tok_proxy.log"))
TAIL_LINES = int(os.getenv("TOK_TAIL_LINES", "50"))
REFRESH_SECONDS = float(os.getenv("TOK_TAIL_REFRESH", "1.5"))


def _clear_screen() -> None:
    subprocess.run(["clear"], check=False) if os.name == "posix" else subprocess.run(["cls"], check=False)  # nosec: B605


def _tail_lines(path: Path, num_lines: int) -> list[str]:
    if not path.exists():
        return [f"❌ Log file not found: {path}"]
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    return lines[-num_lines:] if len(lines) > num_lines else lines


def _savings_summary() -> tuple[int, int, float]:
    tracker = SavingsTracker()
    stats = tracker.load_stats()
    models = stats.get("models", {})
    actual_tokens = 0
    saved_tokens = 0
    for model_stats in models.values():
        actual_tokens += int(model_stats.get("actual_input_tokens", 0))
        actual_tokens += int(model_stats.get("actual_output_tokens", 0))
        actual_tokens += int(model_stats.get("cache_read_tokens", 0))
        actual_tokens += int(model_stats.get("cache_write_tokens", 0))
        saved_tokens += int(model_stats.get("input_saved_tokens", 0))
        saved_tokens += int(model_stats.get("output_saved_tokens", 0))
    baseline_tokens = actual_tokens + saved_tokens
    saved_pct = (saved_tokens / baseline_tokens * 100) if baseline_tokens else 0.0
    return (saved_tokens, baseline_tokens, saved_pct)


def render() -> None:
    _saved, _baseline, _pct = _savings_summary()
    _clear_screen()
    for _line in _tail_lines(LOG_PATH, TAIL_LINES):
        pass


def main() -> None:
    try:
        while True:
            render()
            time.sleep(REFRESH_SECONDS)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
