"""Per-request operation artifacts shared by streaming and non-streaming paths."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import BridgeSession

logger = logging.getLogger("tok.gateway")


def _fallback_zeroed(
    *,
    fallback: bool,
    compressed: bool,
    input_saved: int,
    output_saved: int,
    tool_breakdown: dict[str, int] | None,
    prompt_metrics: dict[str, int] | None,
) -> tuple[bool, int, int, dict[str, int], dict[str, int]]:
    if fallback:
        return False, 0, 0, {}, {}
    return (
        bool(compressed),
        max(0, int(input_saved)),
        max(0, int(output_saved)),
        dict(tool_breakdown or {}),
        dict(prompt_metrics or {}),
    )


def emit_operation_receipt(
    active_session: BridgeSession,
    *,
    request_policy: str,
    compressed: bool,
    fallback: bool,
    input_saved: int,
    output_saved: int,
    prompt_metrics: dict[str, int] | None,
) -> None:
    if bool(getattr(active_session, "_operation_receipt_emitted", False)):
        return
    effective_compressed, effective_input_saved, effective_output_saved, _, effective_prompt_metrics = _fallback_zeroed(
        fallback=fallback,
        compressed=compressed,
        input_saved=input_saved,
        output_saved=output_saved,
        tool_breakdown=None,
        prompt_metrics=prompt_metrics,
    )
    try:
        from tok.receipt import emit_bridge_receipt_for_session

        runtime_session = active_session.runtime_session
        evidence_summary: dict[str, Any] = {}
        if hasattr(runtime_session, "evidence_safety_audit_summary"):
            evidence_summary = dict(runtime_session.evidence_safety_audit_summary())
        lifecycle = getattr(active_session, "_last_request_lifecycle", None)
        if lifecycle is not None and hasattr(lifecycle, "incomplete_gateway_stages"):
            incomplete = lifecycle.incomplete_gateway_stages()
            evidence_summary["request_lifecycle"] = {
                "gateway_stages_complete": not incomplete,
                "incomplete_gateway_stages": incomplete,
            }
        emit_bridge_receipt_for_session(
            active_session,
            request_policy=request_policy,
            compression_applied=effective_compressed,
            evidence_summary=evidence_summary,
            savings={
                "input_saved_tokens": effective_input_saved,
                "output_saved_tokens": effective_output_saved,
                "tokens_saved": effective_input_saved + effective_output_saved,
                **{str(k): int(v) for k, v in effective_prompt_metrics.items()},
            },
            fallback=fallback,
        )
        active_session._operation_receipt_emitted = True
    except Exception:
        logger.debug("tok_receipt_emit_failed", exc_info=True)


def emit_savings_event(
    active_session: BridgeSession,
    *,
    model: str,
    usage: dict[str, Any],
    request_policy: str,
    compressed: bool,
    fallback: bool,
    input_saved: int,
    output_saved: int,
    tool_breakdown: dict[str, int] | None,
    prompt_metrics: dict[str, int] | None,
) -> None:
    if bool(getattr(active_session, "_savings_event_emitted", False)):
        return
    (
        effective_compressed,
        effective_input_saved,
        effective_output_saved,
        effective_tool_breakdown,
        effective_prompt_metrics,
    ) = _fallback_zeroed(
        fallback=fallback,
        compressed=compressed,
        input_saved=input_saved,
        output_saved=output_saved,
        tool_breakdown=tool_breakdown,
        prompt_metrics=prompt_metrics,
    )
    try:
        from tok.receipt import _session_id_from_session as receipt_session_id
        from tok.receipt import bridge_receipt_path
        from tok.utils.pricing import get_pricing as _get_pricing
        from tok.utils.savings_event import SavingsEvent, append_savings_event

        memory_dir = active_session.memory_dir or Path.home() / ".tok"
        session_id = receipt_session_id(active_session)
        events_path = bridge_receipt_path(memory_dir=memory_dir, session_id=session_id).with_name(
            "savings_events.jsonl"
        )
        actual_input = int(usage.get("input_tokens", 0) or 0)
        actual_output = int(usage.get("output_tokens", 0) or 0)
        inp_rate, out_rate, cache_read_rate, cache_write_rate = _get_pricing(model)
        per_million = 1_000_000
        cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
        cache_write = int(usage.get("cache_creation_input_tokens", 0) or 0)
        actual_cost = (
            actual_input * inp_rate / per_million
            + actual_output * out_rate / per_million
            + cache_read * cache_read_rate / per_million
            + cache_write * cache_write_rate / per_million
        )
        baseline_input = actual_input + effective_input_saved
        baseline_output = actual_output + effective_output_saved
        baseline_cost = (
            baseline_input * inp_rate / per_million
            + baseline_output * out_rate / per_million
            + cache_read * cache_read_rate / per_million
            + cache_write * cache_write_rate / per_million
        )
        event = SavingsEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now(UTC).isoformat(),
            model=model,
            mode="tok" if effective_compressed else "baseline",
            request_policy=request_policy,
            baseline_input_tokens=actual_input + effective_input_saved,
            actual_input_tokens=actual_input,
            input_tokens_saved=effective_input_saved,
            baseline_output_tokens=actual_output + effective_output_saved,
            actual_output_tokens=actual_output,
            output_tokens_saved=effective_output_saved,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            baseline_cost_usd=0.0 if fallback else baseline_cost,
            actual_cost_usd=0.0 if fallback else actual_cost,
            cost_saved_usd=0.0 if fallback else baseline_cost - actual_cost,
            fallback=bool(fallback),
            degraded_to_baseline=bool(fallback or not compressed),
            compression_paths={str(k): int(v) for k, v in effective_tool_breakdown.items()},
            non_headline_estimates={str(k): int(v) for k, v in effective_prompt_metrics.items()},
        )
        append_savings_event(event, events_path)
        active_session._savings_event_emitted = True
    except Exception:
        logger.debug("tok_savings_event_emit_failed", exc_info=True)


__all__ = ["emit_operation_receipt", "emit_savings_event"]
