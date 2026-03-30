from typing import Any

from .adapters import (
    RuntimeAdapter,
    ClaudeBridgeAdapter,
    OpenAIChatAdapter,
    OrchestratorAdapter,
    TextLoopAdapter,
)


def __getattr__(name: str) -> Any:
    if name == "Agent":
        from .agent import Agent

        return Agent
    if name in ("TokOrchestrator", "OrchestratorConfig"):
        from .orchestrator import TokOrchestrator

        if name == "TokOrchestrator":
            return TokOrchestrator
        # OrchestratorConfig was never defined; return a stub dataclass
        import dataclasses

        OrchestratorConfig: Any = dataclasses.dataclass(
            type("OrchestratorConfig", (), {})
        )
        return OrchestratorConfig
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
