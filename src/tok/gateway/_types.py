"""Shared gateway types used across pipeline, app factory, and streaming."""

from __future__ import annotations

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


@dataclass
class BridgePreparedPayload:
    body: dict[str, Any]
    behavior_signals: dict[str, int]
    request_policy: str
    request_tool_compatible: bool
    compressed: bool
    saved_toks: int
    tool_breakdown: dict[str, int]
    prompt_metrics: dict[str, int]
    retry_forbidden: bool
    provider_safe_original_body: dict[str, Any] = field(default_factory=dict)
    request_model: str = ""
    request_messages: list[dict[str, Any]] = field(default_factory=list)
    lifecycle: RequestLifecycle | None = None


__all__ = ["BridgeCapabilityManifest", "BridgePreparedPayload", "build_capability_manifest"]
