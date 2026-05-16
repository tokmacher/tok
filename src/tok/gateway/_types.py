"""Shared gateway types used across pipeline, app factory, and streaming."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Any

from tok.runtime._request_lifecycle import RequestLifecycle
from tok.spec.trace import TRACE_VERSION

_PACKAGE_VERSION = version("tok-protocol")


@dataclass(frozen=True)
class BridgeCapabilityManifest:
    trace_version: str
    supported_evidence_forms: tuple[str, ...]
    supported_actions: tuple[str, ...]
    supported_delta_algorithms: tuple[str, ...]
    fallback_modes: tuple[str, ...]
    fixture_pack_version: str
    max_conformance_level: str
    bridge_mode: str


def build_capability_manifest(bridge_mode: str = "tool-compatible") -> BridgeCapabilityManifest:
    return BridgeCapabilityManifest(
        trace_version=TRACE_VERSION,
        supported_evidence_forms=("exact", "summary", "skeleton", "reference"),
        supported_actions=(
            "pass_through",
            "store",
            "reference",
            "delta",
            "fallback",
            "skeleton_reference",
            "summary_reference",
        ),
        supported_delta_algorithms=("unified_diff",),
        fallback_modes=("fail_open", "baseline"),
        fixture_pack_version=_PACKAGE_VERSION,
        max_conformance_level="L2",
        bridge_mode=bridge_mode,
    )


@dataclass(frozen=True)
class PromptMetrics:
    baseline_prompt_tokens: int = 0
    prepared_prompt_tokens: int = 0
    saved_prompt_tokens: int = 0
    hot_hint_tokens_added: int = 0
    reacquisition_tokens_avoided_estimate: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "baseline_prompt_tokens": int(self.baseline_prompt_tokens),
            "prepared_prompt_tokens": int(self.prepared_prompt_tokens),
            "saved_prompt_tokens": int(self.saved_prompt_tokens),
            "hot_hint_tokens_added": int(self.hot_hint_tokens_added),
            "reacquisition_tokens_avoided_estimate": int(self.reacquisition_tokens_avoided_estimate),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> PromptMetrics:
        if not data:
            return cls()
        return cls(
            baseline_prompt_tokens=int(data.get("baseline_prompt_tokens", 0) or 0),
            prepared_prompt_tokens=int(data.get("prepared_prompt_tokens", 0) or 0),
            saved_prompt_tokens=int(data.get("saved_prompt_tokens", 0) or 0),
            hot_hint_tokens_added=int(data.get("hot_hint_tokens_added", 0) or 0),
            reacquisition_tokens_avoided_estimate=int(data.get("reacquisition_tokens_avoided_estimate", 0) or 0),
        )


@dataclass(frozen=True)
class PipelineEarlyExit:
    status_code: int
    content: bytes
    media_type: str = "application/json"


@dataclass
class BridgePreparedPayload:
    body: dict[str, Any]
    behavior_signals: dict[str, int]
    request_policy: str
    request_tool_compatible: bool
    compressed: bool
    saved_toks: int
    tool_breakdown: dict[str, int]
    prompt_metrics: PromptMetrics
    retry_forbidden: bool
    provider_safe_original_body: dict[str, Any] = field(default_factory=dict)
    request_model: str = ""
    request_messages: list[dict[str, Any]] = field(default_factory=list)
    lifecycle: RequestLifecycle | None = None
    surface_runtime: str = ""
    surface_adapter: str = ""


__all__ = [
    "BridgeCapabilityManifest",
    "BridgePreparedPayload",
    "PipelineEarlyExit",
    "PromptMetrics",
    "build_capability_manifest",
]
