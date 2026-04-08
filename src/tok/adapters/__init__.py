"""Tok adapter module for various LLM runtime integrations."""

from .adapters import (
    ClaudeBridgeAdapter,
    OpenAIChatAdapter,
    OrchestratorAdapter,
    RuntimeAdapter,
    TextLoopAdapter,
)


class OrchestratorConfig:
    """Placeholder for orchestrator configuration (not yet implemented)."""


def __getattr__(name: str) -> object:
    """Lazy-load Agent and TokOrchestrator to avoid circular imports."""
    if name == "Agent":
        from .agent import Agent

        return Agent
    if name == "TokOrchestrator":
        from .orchestrator import TokOrchestrator

        return TokOrchestrator
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    # Agent (lazy — requires OPENROUTER_API_KEY at runtime)
    "Agent",
    "ClaudeBridgeAdapter",
    "OpenAIChatAdapter",
    "OrchestratorAdapter",
    "OrchestratorConfig",
    # Adapters
    "RuntimeAdapter",
    "TextLoopAdapter",
    # Orchestrator
    "TokOrchestrator",
]
