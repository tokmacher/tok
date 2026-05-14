from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from tok.runtime.core import RuntimeSession, UniversalTokRuntime
from tok.runtime.pipeline._answer_expectation import (
    record_structured_answer_expectation,
    restore_latest_assistant_thinking,
)
from tok.runtime.pipeline.request_preparation import mutation_signals
from tok.runtime.pipeline.request_validation import canonicalize_anthropic_bridge_body
from tok.runtime.types import PreparedRuntimeRequest, RuntimeRequest


@dataclass
class Step9Result:
    prepared_request: PreparedRuntimeRequest | None = None


def run_step_9(
    runtime_self: UniversalTokRuntime | None = None,
    request: RuntimeRequest | None = None,
    session: RuntimeSession | None = None,
    body: dict[str, Any] | None = None,
    original_body: dict[str, Any] | None = None,
    thinking_snapshot: str | None = None,
    compressed: bool = False,
    saved_tokens: int = 0,
    type_breakdown: dict[str, int] | None = None,
    behavior_signals: dict[str, int] | None = None,
    mode: Any = None,
    request_policy: Literal["legacy_tool_compatible", "natural_first", "forced_baseline"] = "legacy_tool_compatible",
    effective_tool_compatible: bool = False,
    request_policy_escalated: bool = False,
    normalized_tool_events: list[Any] | None = None,
    baseline_prompt_tokens: int = 0,
    prepared_prompt_tokens: int = 0,
    hot_hint_metrics: dict[str, int] | None = None,
    seen_mutation_pairs: set[tuple[str, str]] | None = None,
    _pre_existing_session_signals: dict[str, int] | None = None,
) -> Step9Result:
    if behavior_signals is None:
        behavior_signals = {}
    if type_breakdown is None:
        type_breakdown = {}
    if hot_hint_metrics is None:
        hot_hint_metrics = {}
    if seen_mutation_pairs is None:
        seen_mutation_pairs = set()
    if _pre_existing_session_signals is None:
        _pre_existing_session_signals = {}

    saved_prompt_tokens = 0

    if body is None:
        body = {}
    if original_body is None:
        original_body = {}

    _mut_signals = mutation_signals(original_body, body)
    for key, value in _mut_signals.items():
        behavior_signals[key] = behavior_signals.get(key, 0) + value

    if _mut_signals.get("tok_preflight_rejected"):
        body = original_body
        if session and session._pending_exact_evidence_keys:
            session._first_exact_evidence_seen.update(session._pending_exact_evidence_keys)
            session._pending_exact_evidence_keys.clear()
        if session:
            session._bump_signals(_mut_signals)
            session._save_bridge_memory()
            record_structured_answer_expectation(session, body)
        return Step9Result(
            prepared_request=PreparedRuntimeRequest(
                body=body,
                compressed=False,
                input_saved_tokens=0,
                type_breakdown={},
                behavior_signals=behavior_signals,
                mode=mode or "",
                request_policy=request_policy,
                effective_tool_compatible=effective_tool_compatible,
                request_policy_escalated=request_policy_escalated,
                normalized_tool_events=normalized_tool_events or [],
            )
        )

    if session:
        prepared_prompt_tokens = session.prepared_prompt_tokens(body)
        baseline_prompt_tokens = session.prepared_prompt_tokens(original_body)
        saved_prompt_tokens = max(0, baseline_prompt_tokens - prepared_prompt_tokens)
        if saved_prompt_tokens > 0:
            compressed = True
            saved_tokens += saved_prompt_tokens

        for key, value in session.pending_behavior_signals.items():
            if value and value > _pre_existing_session_signals.get(key, 0):
                behavior_signals[key] = behavior_signals.get(key, 0) + value - _pre_existing_session_signals.get(key, 0)

        for key, value in hot_hint_metrics.items():
            if value:
                behavior_signals[key] = behavior_signals.get(key, 0) + value

        session._bump_signals(behavior_signals)
        session._bump_signals(hot_hint_metrics)

        if session._pending_exact_evidence_keys:
            session._first_exact_evidence_seen.update(session._pending_exact_evidence_keys)
            session._pending_exact_evidence_keys.clear()
        session._save_bridge_memory()

    if request and request.requires_provider_canonicalization:
        canonical_body, canonicalized, canonical_signals = canonicalize_anthropic_bridge_body(
            body, seen_mutation_pairs=seen_mutation_pairs
        )
        if canonicalized:
            body = canonical_body
            for key, value in canonical_signals.items():
                behavior_signals[key] = behavior_signals.get(key, 0) + value
    else:
        canonical_body = body

    if thinking_snapshot is not None:
        restore_latest_assistant_thinking(body.get("messages", []), thinking_snapshot)

    if session:
        record_structured_answer_expectation(session, body)

    return Step9Result(
        prepared_request=PreparedRuntimeRequest(
            body=body,
            compressed=compressed,
            input_saved_tokens=saved_tokens,
            type_breakdown=type_breakdown,
            behavior_signals=behavior_signals,
            mode=mode or "",
            request_policy=request_policy,
            effective_tool_compatible=effective_tool_compatible,
            request_policy_escalated=request_policy_escalated,
            normalized_tool_events=normalized_tool_events or [],
            baseline_prompt_tokens=baseline_prompt_tokens,
            prepared_prompt_tokens=prepared_prompt_tokens,
            saved_prompt_tokens=saved_prompt_tokens,
            hot_hint_tokens_added=hot_hint_metrics.get("hot_hint_tokens_added", 0),
            reacquisition_tokens_avoided_estimate=hot_hint_metrics.get("reacquisition_tokens_avoided_estimate", 0),
        )
    )
