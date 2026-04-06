from __future__ import annotations

"""Top-level Tok package with lazy exports to avoid eager import cascades.

Convenience re-exports live here, but the canonical 0.1.0 protocol IDL lives in
`tok.protocol.schema` and `tok.protocol.models`.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .protocol.encoder import TokEncoder
    from .protocol.format_bridge import Bridge
    from .protocol.models import TokNode
    from .protocol.parser import TokParser
    from .protocol.schema import BlockSchema, TokSchema
    from .runtime.core import UniversalTokRuntime
    from .runtime.core import RuntimeSession
    from .runtime.types import (
        PreparedRuntimeRequest,
        ProcessedRuntimeResponse,
        RuntimeRequest,
    )
    from .utils.sifter import Sifter
    from .utils.tok_registry import TokRegistry
    from .utils.transformer import DocumentTransformer

_RUNTIME = None

_SYMBOLS = {
    "Bridge": (".protocol.format_bridge", "Bridge"),
    "BlockSchema": (".protocol.schema", "BlockSchema"),
    "ClaudeBridgeAdapter": (".adapters.adapters", "ClaudeBridgeAdapter"),
    "DEFAULT_SCHEMA": (".protocol.schema", "DEFAULT_SCHEMA"),
    "DocumentTransformer": (".utils.transformer", "DocumentTransformer"),
    "explore_file": (".explorer", "explore_file"),
    "explore_module": (".explorer", "explore_module"),
    "get_file_overview": (".explorer", "get_file_overview"),
    "list_large_files": (".explorer", "list_large_files"),
    "OpenAIChatAdapter": (".adapters.adapters", "OpenAIChatAdapter"),
    "OrchestratorAdapter": (".adapters.adapters", "OrchestratorAdapter"),
    "PreparedRuntimeRequest": (".runtime.types", "PreparedRuntimeRequest"),
    "ProcessedRuntimeResponse": (".runtime.types", "ProcessedRuntimeResponse"),
    "RuntimeRequest": (".runtime.types", "RuntimeRequest"),
    "RuntimeSession": (".runtime.core", "RuntimeSession"),
    "Sifter": (".utils.sifter", "Sifter"),
    "TextLoopAdapter": (".adapters.adapters", "TextLoopAdapter"),
    "TokEncoder": (".protocol.encoder", "TokEncoder"),
    "TokNode": (".protocol.models", "TokNode"),
    "TokParser": (".protocol.parser", "TokParser"),
    "TokRegistry": (".utils.tok_registry", "TokRegistry"),
    "TokSchema": (".protocol.schema", "TokSchema"),
    "UniversalTokRuntime": (".runtime.core", "UniversalTokRuntime"),
    "serialize": (".protocol.parser", "serialize"),
    "tok_to_dict": (".protocol.parser", "tok_to_dict"),
    "tok_to_tok": (".protocol.parser", "tok_to_tok"),
    "TokError": (".exceptions", "TokError"),
    "CompressionError": (".exceptions", "CompressionError"),
    "SessionError": (".exceptions", "SessionError"),
    "BridgeUnavailableError": (".exceptions", "BridgeUnavailableError"),
    "ReplayGateError": (".exceptions", "ReplayGateError"),
    "InvalidSessionStateError": (".exceptions", "InvalidSessionStateError"),
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


def _runtime() -> UniversalTokRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        runtime_cls = __getattr__("UniversalTokRuntime")
        _RUNTIME = cast("UniversalTokRuntime", runtime_cls())
    return _RUNTIME


def wrap(
    messages: list[dict[str, Any]],
    model: str,
    session: RuntimeSession,
    *,
    system: str | list[dict[str, Any]] | None = None,
    tool_compatible: bool = True,
) -> PreparedRuntimeRequest:
    """Experimental request preparation helper."""
    request_cls = __getattr__("RuntimeRequest")
    request = request_cls(
        model=model,
        messages=messages,
        system=system,
        adapter_kind="wrap",
        tool_compatible=tool_compatible,
    )
    return _runtime().prepare_request(request, session)


def process(
    response_text: str,
    model: str,
    session: RuntimeSession,
    *,
    tool_compatible: bool = True,
) -> ProcessedRuntimeResponse:
    """Experimental response-processing helper."""
    return _runtime().process_response(
        response_text,
        model=model,
        session=session,
        tool_compatible=tool_compatible,
    )


__all__ = [
    "Bridge",
    "BlockSchema",
    "DEFAULT_SCHEMA",
    "DocumentTransformer",
    "explore_file",
    "explore_module",
    "get_file_overview",
    "list_large_files",
    "PreparedRuntimeRequest",
    "ProcessedRuntimeResponse",
    "RuntimeRequest",
    "RuntimeSession",
    "Sifter",
    "TokEncoder",
    "TokNode",
    "TokParser",
    "TokRegistry",
    "TokSchema",
    "UniversalTokRuntime",
    "serialize",
    "tok_to_dict",
    "tok_to_tok",
    "TokError",
    "CompressionError",
    "SessionError",
    "BridgeUnavailableError",
    "ReplayGateError",
    "InvalidSessionStateError",
]

__version__ = "0.1.0"
