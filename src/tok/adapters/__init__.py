"""Tok adapter module for various LLM runtime integrations."""

from .adapters import (
    ClaudeBridgeAdapter,
    OpenAIChatAdapter,
    OrchestratorAdapter,
    RuntimeAdapter,
    TextLoopAdapter,
)


def __getattr__(name: str) -> object:
    """Lazy-load TokOrchestrator to avoid circular imports."""
    if name == "TokOrchestrator":
        from .orchestrator import TokOrchestrator

        return TokOrchestrator
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "ClaudeBridgeAdapter",
    "OpenAIChatAdapter",
    "OrchestratorAdapter",
    "RuntimeAdapter",
    "TextLoopAdapter",
    "TokOrchestrator",
]
