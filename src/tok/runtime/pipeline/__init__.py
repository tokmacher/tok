from __future__ import annotations

"""Lazy exports for runtime pipeline helpers."""

from importlib import import_module
from typing import Any

_SYMBOLS = {
    "prepare_request": (".request_preparation", "prepare_request"),
    "build_system_context": (".request_preparation", "build_system_context"),
    "inject_tok_directive": (".request_preparation", "inject_tok_directive"),
    "validate_request": (".request_validation", "validate_request"),
    "validate_tool_compatibility": (
        ".request_validation",
        "validate_tool_compatibility",
    ),
    "validate_compression_eligible": (
        ".request_validation",
        "validate_compression_eligible",
    ),
    "handle_response": (".response_handling", "handle_response"),
    "validate_response": (".response_handling", "validate_response"),
    "classify_response_mode": (".response_handling", "classify_response_mode"),
    "process_response": (".response_processing", "process_response"),
    "extract_visible_content": (
        ".response_processing",
        "extract_visible_content",
    ),
    "parse_tok_response": (".response_processing", "parse_tok_response"),
    "parse_markdown_fallback": (
        ".response_processing",
        "parse_markdown_fallback",
    ),
    "process_tool_calls": (".tool_processing", "process_tool_calls"),
    "build_tool_use_id_to_context": (
        ".tool_processing",
        "build_tool_use_id_to_context",
    ),
    "extract_tool_events": (".tool_processing", "extract_tool_events"),
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
