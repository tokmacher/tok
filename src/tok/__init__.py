"""
Tok — bridge-first CLI and runtime for Claude Code.

The only supported root export for 0.1.x is `Bridge`. All other symbols are
accessible via their submodules (e.g. `tok.runtime.core.RuntimeSession`) but
are not part of the defended public release surface.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .protocol.format_bridge import Bridge

_SUPPORTED_SYMBOLS: dict[str, tuple[str, str]] = {
    "Bridge": (".protocol.format_bridge", "Bridge"),
}


def __getattr__(name: str) -> object:
    if name in _SUPPORTED_SYMBOLS:
        module_name, attr_name = _SUPPORTED_SYMBOLS[name]
        module = import_module(module_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(
        f"module 'tok' has no attribute {name!r}; see tok.protocol, tok.runtime, tok.compression for submodule access"
    )


__all__: list[str] = ["Bridge"]

__version__ = "0.2.0"
