"""Thin adapter shells over the universal Tok runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from tok.runtime.core import (
    PreparedRuntimeRequest,
    ProcessedRuntimeResponse,
    RuntimeRequest,
    RuntimeSession,
    UniversalTokRuntime,
)
from tok.runtime.types import SignalPacket, SurfaceMetadata


def _system_to_messages(
    system: str | list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Convert system prompt to list of message dicts."""
    if not system:
        return []
    if isinstance(system, str):
        return [{"role": "system", "content": system}]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": block.get("text", "")} for block in system if isinstance(block, dict)
    ]
    return messages


def _render_text(content_blocks: list[dict[str, Any]]) -> str:
    """Extract and join text from content blocks."""
    return "\n".join(
        str(block.get("text", "")).strip()
        for block in content_blocks
        if block.get("type") == "text" and str(block.get("text", "")).strip()
    )


@dataclass
class RuntimeAdapter:
    """
    Transport-thin shell over UniversalTokRuntime.

    Runtime semantics such as compression, fallback behavior,
    memory extraction, and response classification belong in the runtime.
    Adapters should only map transport-specific inputs into RuntimeRequest fields.
    """

    adapter_kind: str
    runtime: UniversalTokRuntime = field(default_factory=UniversalTokRuntime)
    session: RuntimeSession = field(default_factory=RuntimeSession)

    @property
    def surface(self) -> SurfaceMetadata:
        return SurfaceMetadata.from_adapter_kind(self.adapter_kind)

    def prepare(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
        tool_compatible: bool = False,
        grammar: str | None = None,
        todo: str | None = None,
        deltas: str | None = None,
    ) -> PreparedRuntimeRequest:
        """Prepare a runtime request with the given parameters."""
        request = RuntimeRequest(
            model=model,
            messages=messages,
            system=system,
            adapter_kind=self.adapter_kind,
            surface=self.surface,
            tool_compatible=tool_compatible,
            grammar=grammar,
            todo=todo,
            deltas=deltas,
        )
        return self.runtime.prepare_signal_packet(
            SignalPacket.from_request(request),
            self.session,
        )

    def finalize(
        self,
        *,
        text: str,
        model: str,
        behavior_signals: dict[str, int] | None = None,
    ) -> ProcessedRuntimeResponse:
        """Process and finalize the response text."""
        return self.runtime.process_response(
            text,
            model=model,
            session=self.session,
            behavior_signals=behavior_signals,
        )


@dataclass
class ClaudeBridgeAdapter(RuntimeAdapter):
    """Adapter for Claude Bridge integration."""

    adapter_kind: str = "claude-bridge"

    @property
    def surface(self) -> SurfaceMetadata:
        return SurfaceMetadata.claude_bridge()


@dataclass
class OpenAIChatAdapter(RuntimeAdapter):
    """Adapter for OpenAI Chat API integration."""

    adapter_kind: str = "openai-chat"

    def build_chat_messages(
        self,
        *,
        model: str,
        user_text: str,
        system_prompt: str | None = None,
    ) -> tuple[list[dict[str, Any]], PreparedRuntimeRequest]:
        prepared = self.prepare(
            model=model,
            messages=[{"role": "user", "content": user_text}],
            system=system_prompt,
        )
        chat_messages = _system_to_messages(cast("Any", prepared.body.get("system"))) + cast(
            "list[dict[str, Any]]", prepared.body.get("messages", [])
        )
        return chat_messages, prepared

    def visible_text(self, processed: ProcessedRuntimeResponse) -> str:
        """Extract visible text from processed response."""
        return _render_text(processed.content_blocks)


@dataclass
class TextLoopAdapter(RuntimeAdapter):
    """Adapter for text loop integration."""

    adapter_kind: str = "text-loop"

    def prepare_messages(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> tuple[list[dict[str, Any]], PreparedRuntimeRequest]:
        prepared = self.prepare(model=model, messages=messages, system=system_prompt)
        chat_messages = _system_to_messages(cast("Any", prepared.body.get("system"))) + cast(
            "list[dict[str, Any]]", prepared.body.get("messages", [])
        )
        return chat_messages, prepared


@dataclass
class OrchestratorAdapter(RuntimeAdapter):
    """Adapter for Tok orchestrator integration."""

    adapter_kind: str = "orchestrator"

    def prepare_turn(
        self,
        *,
        model: str,
        system_prompt: str,
        dynamic_messages: list[dict[str, Any]],
        grammar: str | None = None,
        todo: str | None = None,
        deltas: str | None = None,
    ) -> tuple[list[dict[str, Any]], PreparedRuntimeRequest]:
        prepared = self.prepare(
            model=model,
            messages=dynamic_messages,
            system=system_prompt,
            tool_compatible=False,
            grammar=grammar,
            todo=todo,
            deltas=deltas,
        )
        chat_messages = _system_to_messages(cast("Any", prepared.body.get("system"))) + cast(
            "list[dict[str, Any]]", prepared.body.get("messages", [])
        )
        return chat_messages, prepared
