"""Runtime orchestration helpers for request/response processing."""

from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING, Any, cast

from .config import (
    ANSWER_READY_REPAIR_HINT,
    ANSWER_READY_RUNTIME_HINT,
    LATE_ANSWER_ASSEMBLY_ANSWER_ONLY_REPAIR_HINT,
    LATE_ANSWER_ASSEMBLY_TOOL_ONLY_REPAIR_HINT,
    LATE_ANSWER_FOLLOWTHROUGH_HINT,
    RUNTIME_HINTS_MAX_PER_TURN,
    TOK_RUNTIME_HINT_COOLDOWN_TURNS,
)
from .memory.answer_memory import (
    _process_answer_memory,
    _should_persist_to_durable,
)
from .memory.tok_state import (
    _prepare_tool_compatible_state,
    _select_resend_reason,
)
from .metrics import report_protocol_drift
from .pipeline.request_preparation import (
    _annotate_reacquisition_diagnostics,
    _apply_tool_compatible_resend_diagnostics,
    _inject_system,
    _is_answer_ready_turn,
    _is_read_only_audit_turn,
    _runtime_hints_for_turn,
)
from .pipeline.response_handling import handle_answer_repair
from .pipeline.response_processing import (
    _expected_structured_labels,
    _is_answer_like_visible_text,
    _is_strict_structured_answer_response,
    _visible_text_from_content_blocks,
    heal_drift,
    is_safe_visible_contract_output,
    response_behavior_signals,
    response_contract_for_mode,
)
from .pipeline.tool_processing import count_tokens
from .policy.answer_repair import _mark_late_answer_assembly_mode_signal
from .policy.macro_handling import _attribute_macro_savings
from .policy.semantic_validation import (
    semantic_pressure_score as _semantic_pressure_score,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from .core import RuntimeSession, UniversalTokRuntime
    from .types import ProcessedRuntimeResponse, RuntimeRequest


_HINT_COOLDOWN_EXEMPT = frozenset(
    {
        ANSWER_READY_RUNTIME_HINT,
        ANSWER_READY_REPAIR_HINT,
        LATE_ANSWER_FOLLOWTHROUGH_HINT,
        LATE_ANSWER_ASSEMBLY_TOOL_ONLY_REPAIR_HINT,
        LATE_ANSWER_ASSEMBLY_ANSWER_ONLY_REPAIR_HINT,
    }
)


def _apply_runtime_hint_cooldown(
    session: RuntimeSession,
    runtime_hints: list[str],
    *,
    cooldown_turns: int,
) -> tuple[list[str], int]:
    if cooldown_turns <= 0 or not runtime_hints:
        return runtime_hints, 0
    current_turn = max(1, int(session.bridge_memory.turn))
    filtered: list[str] = []
    suppressed = 0
    for hint in runtime_hints:
        if hint in _HINT_COOLDOWN_EXEMPT:
            filtered.append(hint)
            continue
        hint_key = hashlib.sha256(hint.encode("utf-8")).hexdigest()[:16]
        last_turn = int(session._runtime_hint_last_turn.get(hint_key, 0))
        if last_turn and (current_turn - last_turn) <= cooldown_turns:
            suppressed += 1
            continue
        session._runtime_hint_last_turn[hint_key] = current_turn
        filtered.append(hint)
    if len(session._runtime_hint_last_turn) > 512:
        cutoff = current_turn - max(cooldown_turns + 8, 12)
        session._runtime_hint_last_turn = {
            key: turn for key, turn in session._runtime_hint_last_turn.items() if int(turn) >= cutoff
        }
    return filtered, suppressed


def build_tool_compatible_resend(
    runtime: UniversalTokRuntime,
    request: RuntimeRequest,
    session: RuntimeSession,
    memory: str,
    skip_reason: str | None,
    behavior_signals: dict[str, Any],
    runtime_hints: list[str],
    current_pressure: int,
    hot_hint_metrics: dict[str, int],
    translated_messages: list[dict[str, Any]] | None = None,
    should_skip_history: bool = False,
    has_answer_anchor_param: bool | None = None,
) -> tuple[
    str,
    list[str],
    dict[str, Any],
    dict[str, int],
    dict[str, Any],
    dict[str, Any],
    bool,
]:
    """Build a tool-compatible resend payload with state compression and hints."""
    del runtime
    if request.tool_compatible:
        pre_resend_memory = memory
        previous_comparable = dict(session._last_tool_compatible_state_fields)
        (
            _,
            comparable_state,
            has_answer_anchor,
        ) = _prepare_tool_compatible_state(pre_resend_memory, previous_comparable)
        if has_answer_anchor_param is not None:
            has_answer_anchor = has_answer_anchor_param

        if translated_messages is None:
            answer_ready = False
        else:
            answer_ready = _is_answer_ready_turn(
                translated_messages,
                tool_compatible=request.tool_compatible,
                has_answer_anchor=has_answer_anchor,
                baseline_only=session._baseline_only,
            )

        resend_reason = _select_resend_reason(comparable_state, previous_comparable, has_answer_anchor)
        (
            processed_memory,
            resend_signals,
        ) = session.maybe_suppress_tool_compatible_state(
            memory,
            force_resend_on_answer_ready=bool(answer_ready and has_answer_anchor),
        )
        behavior_signals.update({key: behavior_signals.get(key, 0) + value for key, value in resend_signals.items()})
        _apply_tool_compatible_resend_diagnostics(
            behavior_signals,
            processed_memory,
            resend_signals,
            has_answer_anchor=has_answer_anchor,
            resend_reason=resend_reason,
            skip_reason_hint=skip_reason if should_skip_history else None,
            tok_history_compression_skipped=bool(behavior_signals.get("tok_history_compression_skipped", 0)),
            tool_compatible_compression=bool(behavior_signals.get("tool_compatible_compression", 0)),
        )
        if answer_ready:
            behavior_signals["answer_ready_turn"] = 1
        if session._late_answer_followthrough_active:
            behavior_signals["late_answer_followthrough_active"] = 1
        if session._answer_ready_repair_active:
            behavior_signals["answer_ready_repair_active"] = 1
        if session._late_answer_assembly_repair_active:
            behavior_signals["late_answer_assembly_repair_active"] = 1
        _mark_late_answer_assembly_mode_signal(
            behavior_signals,
            session._late_answer_assembly_repair_mode_active,
        )
        runtime_hints.extend(
            _runtime_hints_for_turn(
                answer_ready=answer_ready,
                answer_ready_repair_active=session._answer_ready_repair_active,
                late_answer_followthrough_active=session._late_answer_followthrough_active,
                late_answer_assembly_repair_mode=session._late_answer_assembly_repair_mode_active,
            )
        )
        if not runtime_hints:
            pass
        hot_recent_hints, hot_metrics = session.hot_recent_runtime_hints(max_hints=None)
        if hot_recent_hints:
            runtime_hints.extend(hot_recent_hints)
            for key, value in hot_metrics.items():
                hot_hint_metrics[key] = hot_hint_metrics.get(key, 0) + value
        runtime_hints, suppressed_hint_count = _apply_runtime_hint_cooldown(
            session,
            runtime_hints,
            cooldown_turns=TOK_RUNTIME_HINT_COOLDOWN_TURNS,
        )
        if suppressed_hint_count > 0:
            behavior_signals["runtime_hint_cooldown_suppressed"] = suppressed_hint_count
        if len(runtime_hints) > RUNTIME_HINTS_MAX_PER_TURN:
            runtime_hints = runtime_hints[:RUNTIME_HINTS_MAX_PER_TURN]
        exploration_mode = bool(translated_messages is not None and _is_read_only_audit_turn(translated_messages))
        _annotate_reacquisition_diagnostics(
            behavior_signals,
            answer_ready=answer_ready,
            answer_ready_repair_active=session._answer_ready_repair_active,
            exploration_mode=exploration_mode,
        )
        from .core import logger

        logger.debug(
            "tool-compatible resend: mode=%s payload_chars=%d anchor=%d",
            next(
                (k for k in resend_signals if k.startswith("state_resend_")),
                "none",
            ),
            len(processed_memory),
            behavior_signals.get("answer_anchor_present", 0),
        )
        processed_body = _inject_system(
            {},
            processed_memory,
            runtime_hints,
            tool_compatible=request.tool_compatible,
            grammar=bool(request.grammar),
            todo=request.todo or "",
            deltas=bool(request.deltas),
            pressure=current_pressure,
            behavior_signals=behavior_signals,
            current_turn=session.bridge_memory.turn,
        )

        return (
            processed_memory,
            runtime_hints,
            behavior_signals,
            hot_hint_metrics,
            processed_body,
            resend_signals,
            answer_ready,
        )

    return (
        memory,
        runtime_hints,
        behavior_signals,
        hot_hint_metrics,
        {},
        {},
        False,
    )


def process_response_impl(
    runtime: UniversalTokRuntime,
    text: str,
    *,
    model: str,
    session: RuntimeSession,
    behavior_signals: dict[str, int] | None = None,
    tool_compatible: bool = False,
    jit_executor: Callable[[RuntimeSession, str, str], str] | None = None,
) -> ProcessedRuntimeResponse:
    """Process a raw LLM response into structured content with drift handling."""
    del jit_executor
    from .types import ProcessedRuntimeResponse

    contract = response_contract_for_mode(text, tool_compatible=tool_compatible, session=session)
    expected_labels = _expected_structured_labels(session)
    strict_structured_answer = bool(
        expected_labels
        and _is_strict_structured_answer_response(
            _visible_text_from_content_blocks(contract.content_blocks),
            expected_labels=expected_labels,
        )
        and not any(block.get("type") == "tool_use" for block in contract.content_blocks)
    )
    response_side_signals = (
        {}
        if strict_structured_answer
        else response_behavior_signals(
            text,
            tool_compatible=tool_compatible,
            session=session,
        )
    )
    drift_signals = (
        runtime.semantic_validator.validate_drift(text, contract.behavior_signals) if not tool_compatible else {}
    )
    merged_signals: dict[str, int] = {
        **session.consume_behavior_signals(),
        **(behavior_signals or {}),
        **response_side_signals,
        **contract.behavior_signals,
        **drift_signals,
    }
    if strict_structured_answer:
        merged_signals.pop("non_tok_response", None)
        merged_signals.pop("fail_open_compat_response", None)
        merged_signals.pop("tok_drift_healed", None)

    healed_text = (
        text if strict_structured_answer else heal_drift(text, merged_signals, tool_compatible=tool_compatible)
    )
    if not strict_structured_answer and healed_text != text:
        merged_signals["tok_drift_healed"] = 1
        contract = response_contract_for_mode(healed_text, tool_compatible=tool_compatible, session=session)

    visible_text = "\n".join(
        cast("str", block.get("text", ""))
        for block in contract.content_blocks
        if block.get("type") == "text" and str(block.get("text", "")).strip()
    ).strip()
    has_tool = any(block.get("type") == "tool_use" for block in contract.content_blocks)
    has_answer_text = _is_answer_like_visible_text(visible_text)

    natural_response_acceptable = bool(getattr(session, "_natural_response_acceptable_this_turn", False))
    recovered_valid = is_safe_visible_contract_output(
        visible_text,
        content_blocks=contract.content_blocks,
        expected_labels=expected_labels,
        session=session,
    )
    malformed_present = any(
        merged_signals.get(key, 0) > 0
        for key in (
            "malformed_tok_response",
            "malformed_tok_hybrid_tool",
            "malformed_tok_non_inverted_msg",
            "malformed_tok_markdown_fallback",
            "malformed_tok_bad_header",
        )
    )
    suppress_contract_friction = natural_response_acceptable or (
        strict_structured_answer and not malformed_present and not merged_signals.get("tok_drift_healed")
    )
    if recovered_valid and (
        natural_response_acceptable
        or strict_structured_answer
        or merged_signals.get("tok_drift_healed")
        or malformed_present
    ):
        if suppress_contract_friction:
            for key in (
                "non_tok_response",
                "fail_open_compat_response",
                "malformed_tok_response",
                "malformed_tok_hybrid_tool",
                "malformed_tok_non_inverted_msg",
                "malformed_tok_markdown_fallback",
                "malformed_tok_bad_header",
                "tok_drift_healed",
            ):
                merged_signals.pop(key, None)
        merged_signals["response_contract_recovered_valid"] = 1
        if natural_response_acceptable:
            merged_signals["natural_response_contract_accepted"] = 1

    handle_answer_repair(
        session,
        merged_signals=merged_signals,
        has_tool=has_tool,
        has_answer_text=has_answer_text,
        tool_compatible=tool_compatible,
    )
    structured_fields = _process_answer_memory(session, visible_text)
    if structured_fields:
        for field, values in structured_fields.items():
            for value in values:
                session.bridge_memory._upsert(
                    session.bridge_memory.hot,
                    field,
                    value,
                    score_delta=3,
                )
                if _should_persist_to_durable(field, value):
                    session.bridge_memory._upsert(
                        session.bridge_memory.durable,
                        field,
                        value,
                        score_delta=2,
                    )
        session._save_bridge_memory()

    should_write_healed_memory = not (tool_compatible and merged_signals.get("tok_drift_healed"))
    updated_memory = session.write_memory(healed_text) if should_write_healed_memory else ""
    if updated_memory:
        _attribute_macro_savings(session, updated_memory)
    family_mode = session.update_family_mode(model, merged_signals)

    report_protocol_drift(
        model=model,
        merged_signals=merged_signals,
        mode=contract.mode,
        session=session,
        content_blocks=contract.content_blocks,
    )

    session._step_count += 1
    session._token_count += count_tokens(text)
    for block in contract.content_blocks:
        if block.get("type") == "tool_use" and block.get("name"):
            session._tool_names_seen.add(cast("str", block["name"]))
            tool_input = block.get("input", {})
            if isinstance(tool_input, dict):
                input_key = next(iter(tool_input.values()), "") if tool_input else ""
            else:
                input_key = ""
            loop_detected = session.observe_tool_action(cast("str", block["name"]), str(input_key)[:120])
            if loop_detected:
                merged_signals["loop_detected"] = 1
            tool_name_lower = cast("str", block["name"]).lower()
            if tool_name_lower in ("edit_file", "edit", "write_file", "write", "replace", "create_file"):
                edited_path = (
                    cast("dict", tool_input).get("path")
                    or cast("dict", tool_input).get("file_path")
                    or cast("dict", tool_input).get("filename")
                    or ""
                )
                if edited_path:
                    from tok.runtime.repeat_targets import normalize_path_target

                    session.mark_file_edited(normalize_path_target(edited_path))

    if os.getenv("TOK_NEURO_REACTOR", "0") == "1" and "EXECUTE_JIT(@" in text:
        merged_signals["jit_detected_not_executed"] = 1

    session._drift_detected_previous_turn = bool(
        merged_signals.get("semantic_drift_detected") or merged_signals.get("non_tok_response")
    )

    return ProcessedRuntimeResponse(
        content_blocks=contract.content_blocks,
        output_saved_tokens=contract.output_saved_tokens,
        behavior_signals=merged_signals,
        mode=contract.mode,
        family_mode=family_mode,
        updated_memory=updated_memory,
    )


def pressure_score(signals: dict[str, int]) -> int:
    """Calculate semantic pressure score from behavior signals."""
    return _semantic_pressure_score(signals)
