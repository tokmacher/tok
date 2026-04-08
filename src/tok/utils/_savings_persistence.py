"""Persistence helpers for savings tracker."""

from __future__ import annotations

import contextlib
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from ._savings_quality import GLOBAL_LEDGER_FILENAME, SESSION_STATS_FILENAME

__all__ = [
    "GLOBAL_LEDGER_FILENAME",
    "SESSION_STATS_FILENAME",
    "STATS_KEY_MAP",
    "STATS_KEY_MAP_INV",
    "default_ledger_path",
    "default_savings_file",
    "empty_stats",
    "legacy_ledger_path",
    "parse_kv_string",
    "parse_model_line",
]

STATS_KEY_MAP: dict[str, str] = {
    "c": "calls",
    "in": "actual_input_tokens",
    "out": "actual_output_tokens",
    "cr": "cache_read_tokens",
    "cw": "cache_write_tokens",
    "ins": "input_saved_tokens",
    "outs": "output_saved_tokens",
    "act": "actual_cost_usd",
    "base": "baseline_cost_usd",
    "bpt": "baseline_prompt_tokens",
    "ppt": "prepared_prompt_tokens",
    "spt": "saved_prompt_tokens",
    "hht": "hot_hint_tokens_added",
    "rat": "reacquisition_tokens_avoided_estimate",
}

STATS_KEY_MAP_INV = {
    "calls": "c",
    "actual_input_tokens": "in",
    "actual_output_tokens": "out",
    "cache_read_tokens": "cr",
    "cache_write_tokens": "cw",
    "input_saved_tokens": "ins",
    "output_saved_tokens": "outs",
    "actual_cost_usd": "act",
    "baseline_cost_usd": "base",
    "baseline_prompt_tokens": "bpt",
    "prepared_prompt_tokens": "ppt",
    "saved_prompt_tokens": "spt",
    "hot_hint_tokens_added": "hht",
    "reacquisition_tokens_avoided_estimate": "rat",
}


def default_savings_file() -> str:
    """Return the default path for savings statistics file."""
    default_path = os.path.join(tempfile.gettempdir(), SESSION_STATS_FILENAME)
    return os.getenv("TOK_SAVINGS_FILE", default_path)


def default_ledger_path() -> Path:
    """Return the default path for the global savings ledger."""
    tok_dir = os.getenv("TOK_PROJECT_DIR", "")
    if tok_dir:
        return Path(tok_dir) / GLOBAL_LEDGER_FILENAME
    return Path.home() / ".tok" / GLOBAL_LEDGER_FILENAME


def legacy_ledger_path() -> Path:
    """Return the legacy path for the savings ledger."""
    tok_dir = os.getenv("TOK_PROJECT_DIR", "")
    if tok_dir:
        return Path(tok_dir) / "savings.tok"
    return Path.home() / ".tok" / "savings.tok"


def parse_kv_string(s: str) -> dict[str, int]:
    """Parse a key=value comma-separated string into a dictionary."""
    pairs: dict[str, int] = {}
    for pair in s.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            with contextlib.suppress(ValueError):
                pairs[k] = int(v)
    return pairs


def parse_model_line(line: str) -> tuple[str, dict[str, Any]]:
    """Parse a model statistics line from the savings file."""
    parts = line.split(" m:", 1)[1].strip().split("|")
    model_name = parts[0]
    model_stats: dict[str, Any] = {
        "type_breakdown": {},
        "behavior_signals": {},
    }
    for part in parts[1:]:
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        if key == "breakdown":
            model_stats["type_breakdown"] = parse_kv_string(value)
        elif key == "signals":
            model_stats["behavior_signals"] = parse_kv_string(value)
        elif key in STATS_KEY_MAP:
            model_stats[STATS_KEY_MAP[key]] = float(value) if "." in value or key in ("act", "base") else int(value)
    return model_name, model_stats


def empty_stats() -> dict[str, Any]:
    """Return an empty statistics dictionary with session timestamp."""
    return {
        "session_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": {},
    }
