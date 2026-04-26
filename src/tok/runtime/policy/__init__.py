"""Lazy exports for runtime policy helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_SYMBOLS = {
    "MemoryProjectionProfile": (".smart_policy", "MemoryProjectionProfile"),
    "FamilyAdaptiveState": (".smart_policy", "FamilyAdaptiveState"),
    "advance_state": (".smart_policy", "advance_state"),
    "initial_state": (".smart_policy", "initial_state"),
    "policy_for_model": (".smart_policy", "policy_for_model"),
    "pressure_score": (".smart_policy", "pressure_score"),
    "detect_task_type": (".smart_policy", "detect_task_type"),
    "select_optimal_mode": (".smart_policy", "select_optimal_mode"),
    "CANONICAL_WIRE_FIELD_ORDER": (
        ".smart_policy",
        "CANONICAL_WIRE_FIELD_ORDER",
    ),
    "calculate_invisible_pressure": (
        ".semantic_validation",
        "calculate_invisible_pressure",
    ),
    "IS_TOK": (".translator", "IS_TOK"),
    "postprocess_response": (".translator", "postprocess_response"),
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


__all__ = tuple(_SYMBOLS)
