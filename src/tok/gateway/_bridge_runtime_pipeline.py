"""Shared bridge preparation pipeline used by gateway and benchmark harnesses."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, replace
from typing import Any, Literal, cast

from fastapi import Response

from tok.runtime._request_lifecycle import RequestLifecycle
from tok.universal_runtime import RuntimeRequest

from . import _RUNTIME, BridgeSession, logger
from ._bridge_preflight import _run_bridge_preflight


def _plan_finalization_min_saved_tokens() -> int:
    raw = os.getenv("TOK_PLAN_FINALIZATION_MIN_SAVED_TOKENS", "500")
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(
            "Invalid integer config TOK_PLAN_FINALIZATION_MIN_SAVED_TOKENS=%r; using fallback 500",
            raw,
        )
        return 500


_PLAN_FINALIZATION_MIN_SAVED_TOKENS = _plan_finalization_min_saved_tokens()


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


def _empty_prompt_metrics() -> dict[str, int]:
    return {
        "baseline_prompt_tokens": 0,
        "prepared_prompt_tokens": 0,
        "saved_prompt_tokens": 0,
        "hot_hint_tokens_added": 0,
        "reacquisition_tokens_avoided_estimate": 0,
    }


def _extract_allowed_tools_from_body(body: dict[str, Any]) -> tuple[str, ...]:
    tools = body.get("tools")
    if not isinstance(tools, list):
        return ()
    names: list[str] = []
    seen: set[str] = set()
    for item in tools:
        if not isinstance(item, dict):
            continue
        raw_name = item.get("name")
        if raw_name is None and isinstance(item.get("function"), dict):
            raw_name = item["function"].get("name")
        name = str(raw_name or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return tuple(names)


def _apply_plan_finalization_spend_guard(
    *,
    session: BridgeSession,
    prepared_body: dict[str, Any],
    provider_safe_original_body: dict[str, Any],
    behavior_signals: dict[str, int],
    compressed: bool,
    saved_toks: int,
    prompt_metrics: dict[str, int],
) -> tuple[dict[str, Any], bool, int, dict[str, int], dict[str, int], bool]:
    """Force final-answer/plan turns to provider-safe passthrough unless Tok clearly saves input tokens."""
    if not behavior_signals.get("plan_finalization_turn", 0):
        return prepared_body, compressed, saved_toks, prompt_metrics, behavior_signals, False

    original_prompt_tokens = int(prompt_metrics.get("baseline_prompt_tokens", 0))
    prepared_prompt_tokens = int(prompt_metrics.get("prepared_prompt_tokens", 0))
    if original_prompt_tokens <= 0 and prepared_prompt_tokens <= 0:
        original_prompt_tokens = session.runtime_session.prepared_prompt_tokens(provider_safe_original_body)
        prepared_prompt_tokens = original_prompt_tokens
    elif original_prompt_tokens <= 0:
        original_prompt_tokens = session.runtime_session.prepared_prompt_tokens(provider_safe_original_body)
    elif prepared_prompt_tokens <= 0:
        prepared_prompt_tokens = session.runtime_session.prepared_prompt_tokens(prepared_body)
    observed_saved_tokens = max(0, original_prompt_tokens - prepared_prompt_tokens)
    minimum_saved_tokens = max(0, _PLAN_FINALIZATION_MIN_SAVED_TOKENS)

    behavior_signals["plan_finalization_original_prompt_tokens"] = original_prompt_tokens
    behavior_signals["plan_finalization_prepared_prompt_tokens"] = prepared_prompt_tokens
    behavior_signals["plan_finalization_saved_prompt_tokens"] = observed_saved_tokens
    behavior_signals["plan_finalization_min_saved_tokens"] = minimum_saved_tokens

    tok_added_tokens = prepared_prompt_tokens > original_prompt_tokens
    insufficient_savings = observed_saved_tokens < minimum_saved_tokens
    if tok_added_tokens:
        behavior_signals["plan_finalization_tok_overhead_blocked"] = 1
    if insufficient_savings:
        behavior_signals["plan_finalization_passthrough"] = 1

    if not tok_added_tokens and not insufficient_savings:
        return prepared_body, compressed, saved_toks, prompt_metrics, behavior_signals, False

    passthrough_metrics = dict(prompt_metrics)
    passthrough_metrics.update(
        {
            "baseline_prompt_tokens": original_prompt_tokens,
            "prepared_prompt_tokens": original_prompt_tokens,
            "saved_prompt_tokens": 0,
            "hot_hint_tokens_added": 0,
            "reacquisition_tokens_avoided_estimate": 0,
        }
    )
    logger.info(
        "plan_finalization_passthrough: original_prompt_tokens=%d prepared_prompt_tokens=%d saved=%d min_saved=%d",
        original_prompt_tokens,
        prepared_prompt_tokens,
        observed_saved_tokens,
        minimum_saved_tokens,
    )
    return (
        copy.deepcopy(provider_safe_original_body),
        False,
        0,
        passthrough_metrics,
        behavior_signals,
        True,
    )


def prepare_bridge_payload(
    *,
    session: BridgeSession,
    body: dict[str, Any],
    headers: dict[str, str],
    path: str,
    tok_tool_header: str = "",
    allowed_tools: tuple[str, ...] | None = None,
    request_state: dict[str, bool] | None = None,
) -> tuple[BridgePreparedPayload, Response | None]:
    active_request_state = request_state if request_state is not None else {"fallback_recorded": False}

    lifecycle = RequestLifecycle()
    compressed = False
    saved_toks = 0
    tool_breakdown: dict[str, int] = {}
    behavior_signals: dict[str, int] = {}
    prompt_metrics = _empty_prompt_metrics()
    request_tool_compatible = False
    request_policy = "forced_baseline"
    retry_forbidden = False

    original_body = copy.deepcopy(body)
    (
        provider_safe_original_body,
        behavior_signals,
        source_retry_forbidden,
        preflight_response,
    ) = _run_bridge_preflight(
        session,
        body=copy.deepcopy(body),
        original_body=original_body,
        headers=headers,
        behavior_signals=behavior_signals,
        compressed=False,
        request_state=active_request_state,
        path=path,
        emit_ready_log=(path == "v1/messages/count_tokens"),
        emit_repair_logs=(path == "v1/messages/count_tokens"),
        reset_recovery_state=(path == "v1/messages/count_tokens"),
    )
    retry_forbidden = source_retry_forbidden

    request_model = str(provider_safe_original_body.get("model", ""))
    request_messages = provider_safe_original_body.get("messages", [])
    if not isinstance(request_messages, list):
        request_messages = []
    lifecycle = replace(lifecycle, initial_preflight=True, model_extraction=True)

    payload = BridgePreparedPayload(
        body=copy.deepcopy(provider_safe_original_body),
        behavior_signals=dict(behavior_signals),
        request_policy=request_policy,
        request_tool_compatible=request_tool_compatible,
        compressed=compressed,
        saved_toks=saved_toks,
        tool_breakdown=tool_breakdown,
        prompt_metrics=prompt_metrics,
        retry_forbidden=retry_forbidden,
        provider_safe_original_body=copy.deepcopy(provider_safe_original_body),
        request_model=request_model,
        request_messages=copy.deepcopy(request_messages),
        lifecycle=lifecycle,
    )
    if preflight_response is not None:
        return payload, preflight_response

    cooldown_remaining = getattr(session.runtime_session, "_stream_recovery_cooldown_remaining", 0)
    if cooldown_remaining > 0:
        session.runtime_session._stream_recovery_cooldown_remaining = max(0, cooldown_remaining - 1)

    if path != "v1/messages":
        return payload, None

    if tok_tool_header.lower() in {"0", "false", "off", "no"}:
        request_tool_compatible = False
        request_policy = "forced_baseline" if session.request_policy_default == "forced_baseline" else "natural_first"
    elif session.runtime_session._baseline_only:
        request_tool_compatible = False
        request_policy = "forced_baseline"
        behavior_signals["baseline_only_session"] = 1
        behavior_signals["tok_fallback_activated"] = 1
        logger.warning("tok_fallback_activated: session is in baseline-only mode, serving without compression")
    else:
        request_tool_compatible = True
        request_policy = session.request_policy_default
    source_behavior_signals = dict(behavior_signals)
    lifecycle = replace(lifecycle, tool_compatibility_check=True)

    logger.info(
        "Request mode: model=%s, request_policy=%s, tool_compatible_allowed=%s (tools present: %s, header=%s)",
        request_model,
        request_policy,
        request_tool_compatible,
        bool(provider_safe_original_body.get("tools")),
        tok_tool_header or "<unset>",
    )
    requested_request_policy = request_policy
    requested_tool_compatible = request_tool_compatible

    prepared = _RUNTIME.prepare_request(
        RuntimeRequest(
            model=request_model,
            messages=request_messages,
            system=provider_safe_original_body.get("system", ""),
            adapter_kind="claude-bridge",
            tool_compatible=request_tool_compatible,
            request_policy=cast(
                Literal["legacy_tool_compatible", "natural_first", "forced_baseline"],
                request_policy,
            ),
            request_has_tools=bool(provider_safe_original_body.get("tools")),
            allowed_tools=allowed_tools
            if allowed_tools
            else _extract_allowed_tools_from_body(provider_safe_original_body),
        ),
        session.runtime_session,
        result_cache=session.result_cache,
    )
    lifecycle = replace(lifecycle, runtime_preparation=True)
    request_policy = prepared.request_policy
    request_tool_compatible = prepared.effective_tool_compatible
    compressed = prepared.compressed
    saved_toks = prepared.input_saved_tokens
    tool_breakdown = dict(prepared.type_breakdown)
    behavior_signals = dict(prepared.behavior_signals)
    for key, value in source_behavior_signals.items():
        behavior_signals[key] = behavior_signals.get(key, 0) + value
    behavior_signals[f"request_policy_requested_{requested_request_policy}"] = (
        behavior_signals.get(f"request_policy_requested_{requested_request_policy}", 0) + 1
    )
    if requested_tool_compatible:
        behavior_signals["request_policy_requested_tool_compatible"] = (
            behavior_signals.get("request_policy_requested_tool_compatible", 0) + 1
        )
    else:
        behavior_signals["request_policy_requested_non_tool_compatible"] = (
            behavior_signals.get("request_policy_requested_non_tool_compatible", 0) + 1
        )
    if request_tool_compatible:
        behavior_signals["request_policy_effective_tool_compatible"] = (
            behavior_signals.get("request_policy_effective_tool_compatible", 0) + 1
        )
    else:
        behavior_signals["request_policy_effective_natural_first"] = (
            behavior_signals.get("request_policy_effective_natural_first", 0) + 1
        )
    prompt_metrics = {
        "baseline_prompt_tokens": prepared.baseline_prompt_tokens,
        "prepared_prompt_tokens": prepared.prepared_prompt_tokens,
        "saved_prompt_tokens": prepared.saved_prompt_tokens,
        "hot_hint_tokens_added": prepared.hot_hint_tokens_added,
        "reacquisition_tokens_avoided_estimate": prepared.reacquisition_tokens_avoided_estimate,
    }
    policy_reasons = sorted(
        key.removeprefix("request_policy_reason_")
        for key, value in behavior_signals.items()
        if key.startswith("request_policy_reason_") and value
    )
    logger.info(
        "Prepared request policy: requested_policy=%s, requested_tool_compatible=%s, effective_policy=%s, effective_tool_compatible=%s, reasons=%s, escalated=%s",
        requested_request_policy,
        requested_tool_compatible,
        request_policy,
        request_tool_compatible,
        ",".join(policy_reasons) if policy_reasons else "<none>",
        prepared.request_policy_escalated,
    )
    lifecycle = replace(lifecycle, signals_and_metrics=True)

    prepared_body = copy.deepcopy(provider_safe_original_body)
    prepared_body["messages"] = prepared.body.get("messages", [])
    prepared_body["system"] = prepared.body.get("system", prepared_body.get("system", ""))

    (
        prepared_body,
        behavior_signals,
        prepared_retry_forbidden,
        preflight_response,
    ) = _run_bridge_preflight(
        session,
        body=prepared_body,
        original_body=provider_safe_original_body,
        headers=headers,
        behavior_signals=behavior_signals,
        compressed=compressed,
        request_state=active_request_state,
        path=path,
    )
    retry_forbidden = retry_forbidden or prepared_retry_forbidden
    lifecycle = replace(lifecycle, prepared_preflight=True)
    if behavior_signals.get("tok_bridge_pairing_degraded_to_provider_safe", 0):
        if saved_toks > 0:
            behavior_signals["tok_compression_worked_before_pairing_degraded"] = 1
        compressed = False
        saved_toks = 0
        tool_breakdown = {}
        prompt_metrics = _empty_prompt_metrics()

    (
        prepared_body,
        compressed,
        saved_toks,
        prompt_metrics,
        behavior_signals,
        plan_finalization_passthrough,
    ) = _apply_plan_finalization_spend_guard(
        session=session,
        prepared_body=prepared_body,
        provider_safe_original_body=provider_safe_original_body,
        behavior_signals=behavior_signals,
        compressed=compressed,
        saved_toks=saved_toks,
        prompt_metrics=prompt_metrics,
    )
    if plan_finalization_passthrough:
        request_tool_compatible = False
        tool_breakdown = {}
    lifecycle = replace(lifecycle, plan_finalization_guard=True)

    lifecycle = replace(lifecycle, final_payload_construction=True)
    payload = BridgePreparedPayload(
        body=prepared_body,
        behavior_signals=dict(behavior_signals),
        request_policy=request_policy,
        request_tool_compatible=request_tool_compatible,
        compressed=compressed,
        saved_toks=saved_toks,
        tool_breakdown=tool_breakdown,
        prompt_metrics=prompt_metrics,
        retry_forbidden=retry_forbidden,
        provider_safe_original_body=copy.deepcopy(provider_safe_original_body),
        request_model=request_model,
        request_messages=copy.deepcopy(request_messages),
        lifecycle=lifecycle,
    )
    return payload, preflight_response
