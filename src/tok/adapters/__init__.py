from typing import Any

from .adapters import (
    RuntimeAdapter,
    ClaudeBridgeAdapter,
    OpenAIChatAdapter,
    OrchestratorAdapter,
    TextLoopAdapter,
)


class OrchestratorConfig:
    """Placeholder for orchestrator configuration (not yet implemented)."""


def __getattr__(name: str) -> Any:
    if name == "Agent":
        from .agent import Agent

        return Agent
    if name == "TokOrchestrator":
        from .orchestrator import TokOrchestrator

        return TokOrchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Adapters
    "RuntimeAdapter",
    "ClaudeBridgeAdapter",
    "OpenAIChatAdapter",
    "OrchestratorAdapter",
    "TextLoopAdapter",
    # Agent (lazy — requires OPENROUTER_API_KEY at runtime)
    "Agent",
    # Orchestrator
    "TokOrchestrator",
    "OrchestratorConfig",
]
