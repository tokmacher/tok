"""Lazy exports for runtime memory helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_SYMBOLS = {
    "BridgeMemoryState": (".bridge_memory", "BridgeMemoryState"),
    "clean_system_context": (".bridge_memory", "clean_system_context"),
    "TokState": (".tok_state", "TokState"),
    "build_state_line": (".tok_state", "build_state_line"),
    "parse_state_line": (".tok_state", "parse_state_line"),
    "apply_state_update": (".tok_state", "apply_state_update"),
    "AnswerMemory": (".answer_memory", "AnswerMemory"),
    "extract_answers": (".answer_memory", "extract_answers"),
    "ground_answers_in_memory": (".answer_memory", "ground_answers_in_memory"),
    "SessionHelpers": (".session_helpers", "SessionHelpers"),
    "initialize_session": (".session_helpers", "initialize_session"),
    "update_session_state": (".session_helpers", "update_session_state"),
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
