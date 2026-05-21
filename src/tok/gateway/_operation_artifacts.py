"""Per-request operation artifacts shared by streaming and non-streaming paths."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import BridgeSession

logger = logging.getLogger("tok.gateway")


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
    effective_compressed = bool(compressed and not fallback)
    effective_input_saved = 0 if fallback else int(input_saved)
    effective_output_saved = 0 if fallback else int(output_saved)
    effective_prompt_metrics = {} if fallback else dict(prompt_metrics or {})
    try:
        from tok.receipt import emit_bridge_receipt_for_session

        runtime_session = active_session.runtime_session
        evidence_summary: dict[str, Any] = {}
        if hasattr(runtime_session, "evidence_safety_audit_summary"):
            evidence_summary = dict(runtime_session.evidence_safety_audit_summary())
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
    effective_compressed = bool(compressed and not fallback)
    effective_input_saved = 0 if fallback else max(0, int(input_saved))
    effective_output_saved = 0 if fallback else max(0, int(output_saved))
    effective_tool_breakdown = {} if fallback else dict(tool_breakdown or {})
    effective_prompt_metrics = {} if fallback else dict(prompt_metrics or {})
    try:
        from tok.receipt import _session_id_from_session as receipt_session_id
        from tok.receipt import bridge_receipt_path
        from tok.utils.savings_event import SavingsEvent, append_savings_event

        memory_dir = active_session.memory_dir or Path.home() / ".tok"
        session_id = receipt_session_id(active_session)
        events_path = bridge_receipt_path(memory_dir=memory_dir, session_id=session_id).with_name(
            "savings_events.jsonl"
        )
        actual_input = int(usage.get("input_tokens", 0) or 0)
        actual_output = int(usage.get("output_tokens", 0) or 0)
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
            cache_read_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
            cache_write_tokens=int(usage.get("cache_creation_input_tokens", 0) or 0),
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
