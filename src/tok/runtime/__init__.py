"""Lazy runtime package exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_SYMBOLS = {
    "UniversalTokRuntime": (".core", "UniversalTokRuntime"),
    "PreparedRuntimeRequest": (".types", "PreparedRuntimeRequest"),
    "ProcessedRuntimeResponse": (".types", "ProcessedRuntimeResponse"),
    "RuntimeRequest": (".types", "RuntimeRequest"),
    "RuntimeSession": (".core", "RuntimeSession"),
    "DEFAULT_KEEP_TURNS": (".config", "DEFAULT_KEEP_TURNS"),
    "DEFAULT_MODE": (".config", "DEFAULT_MODE"),
    "DEFAULT_MODEL_FAMILY": (".config", "DEFAULT_MODEL_FAMILY"),
    "DEFAULT_MAX_WORKING_MEMORY": (".config", "DEFAULT_MAX_WORKING_MEMORY"),
    "DEFAULT_MIN_WORKING_MEMORY": (".config", "DEFAULT_MIN_WORKING_MEMORY"),
    "DEFAULT_HARD_MEMORY_LIMIT": (".config", "DEFAULT_HARD_MEMORY_LIMIT"),
    "DEFAULT_SOFT_MEMORY_LIMIT": (".config", "DEFAULT_SOFT_MEMORY_LIMIT"),
    "DEFAULT_TOKEN_LIMIT": (".config", "DEFAULT_TOKEN_LIMIT"),
    "DEFAULT_INJECTION_THRESHOLD": (".config", "DEFAULT_INJECTION_THRESHOLD"),
    "DEFAULT_KEEP_RECENT_WINDOW": (".config", "DEFAULT_KEEP_RECENT_WINDOW"),
    "DEFAULT_COMPRESSION_WINDOW": (".config", "DEFAULT_COMPRESSION_WINDOW"),
    "DEFAULT_FAMILY_MODE": (".config", "DEFAULT_FAMILY_MODE"),
    "DEFAULT_FALLBACK_MODE": (".config", "DEFAULT_FALLBACK_MODE"),
    "DEFAULT_REPLAY_GATE_MODE": (".config", "DEFAULT_REPLAY_GATE_MODE"),
    "DEFAULT_REPLAY_THRESHOLD": (".config", "DEFAULT_REPLAY_THRESHOLD"),
    "DEFAULT_MUTATION_THRESHOLD": (".config", "DEFAULT_MUTATION_THRESHOLD"),
    "DEFAULT_TELEMETRY_MODE": (".config", "DEFAULT_TELEMETRY_MODE"),
    "DEFAULT_MEMORY_PROFILE": (".config", "DEFAULT_MEMORY_PROFILE"),
    "DEFAULT_PROJECT_MARKER_PATTERNS": (
        ".config",
        "DEFAULT_PROJECT_MARKER_PATTERNS",
    ),
    "RuntimeMetrics": (".metrics", "RuntimeMetrics"),
    "calculate_tokens_saved": (".metrics", "calculate_tokens_saved"),
    "calculate_compression_ratio": (".metrics", "calculate_compression_ratio"),
    "calculate_invisible_pressure": (
        ".metrics",
        "calculate_invisible_pressure",
    ),
    "TOOL_DENSITY_THRESHOLD": (".config", "TOOL_DENSITY_THRESHOLD"),
    "calculate_reasoning_depth_per_token": (
        ".metrics",
        "calculate_reasoning_depth_per_token",
    ),
    "RuntimeToolExecutor": (".tools", "RuntimeToolExecutor"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _SYMBOLS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = list(_SYMBOLS)
