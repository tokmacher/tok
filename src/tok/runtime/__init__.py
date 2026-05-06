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
    "TOOL_DENSITY_THRESHOLD": (".config", "TOOL_DENSITY_THRESHOLD"),
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
