"""Breakpoint classification and coverage helpers for stress harness."""

from __future__ import annotations

from typing import Any

from tok.runtime.core import RuntimeSession
from tok.runtime.pipeline.response_processing import response_contract_for_mode
from .models import (
    StressTask,
    StressBreakpoint,
    StressObservation,
    StressHarnessConfig,
    LATE_STAGED_RETRY_PHASES,
    LATE_FOLLOWTHROUGH_MIN_EVIDENCE_CHARS,
    PRE_PRESSURE_MIN_EVIDENCE_CHARS,
    EXCLUDED_GROUNDED_PATH_FRAGMENTS,
)
from .utils import (
    _extract_labeled_fields,
    _render_visible_text,
)


def inferred_cause_for_class(breakpoint_class: str) -> str:
    causes = {
        "protocol_drift": (
            "Grounded task pressure still allowed confident structured answers to drift away from supported repo facts."
        ),
        "retention_loss": (
            "Previously grounded answer anchors were not carried forward reliably across later turns."
        ),
        "reacquisition_loop": (
            "The language re-opened validated facts with new tools instead of reusing bridged memory."
        ),
        "compaction_loss": (
            "State projection compacted answer-bearing facts too aggressively once evidence volume rose."
        ),
        "baseline_fallback": (
            "Repeated grounded-discipline failures accumulated until the session degraded to baseline-only behavior."
        ),
        "tool_contract_failure": (
            "The model emitted unsupported tool usage or answered without the required fresh evidence."
        ),
        "macro_redundancy_loss": (
            "The model failed to use available macros for repetitive tasks, choosing raw tools instead."
        ),
    }
    return causes[breakpoint_class]


def refactor_target_for_class(breakpoint_class: str) -> str:
    targets = {
        "protocol_drift": "response contract pressure",
        "retention_loss": "src/tok/bridge.py",
        "reacquisition_loop": "directive density",
        "compaction_loss": "state projection",
        "baseline_fallback": "fallback policy clarity",
        "tool_contract_failure": "tool contract constraints",
        "macro_redundancy_loss": "src/tok/universal_runtime.py",
    }
    return targets[breakpoint_class]


def _late_tool_contract_grace_kind(
    *,
    task: StressTask,
    retry_index: int,
    payload_pressure_ready: bool,
    input_signals: dict[str, int],
    output_signals: dict[str, int],
    protocol_failure: bool,
    tool_contract_failure: bool,
) -> str | None:
    if not (protocol_failure or tool_contract_failure):
        return None
    if task.phase_name != "tool-contract":
        return None
    if not payload_pressure_ready:
        return None
    if input_signals.get(
        "validated_target_exact_reacquired"
    ) or input_signals.get("validated_target_reacquired"):
        return None
    if output_signals.get("unsupported_tool_event"):
        return None
    if output_signals.get("bad_tool_args_event"):
        return None
    if output_signals.get("mixed_answer_tool_event"):
        return "mixed"
    if output_signals.get("toolless_fresh_answer_event"):
        return "toolless"
    if not input_signals.get("validated_target_reconfirmation_attempt"):
        return None
    if retry_index < 0:
        return None
    return "reconfirmation"


def _fallback_pressure_cause(
    *,
    input_signals: dict[str, int],
    output_signals: dict[str, int],
) -> str | None:
    if input_signals.get(
        "validated_target_exact_reacquired"
    ) or input_signals.get("validated_target_reacquired"):
        return "exact_reacquisition"
    if output_signals.get("unsupported_tool_event"):
        return "unsupported_tool"
    if output_signals.get("bad_tool_args_event"):
        return "bad_args"
    if output_signals.get("mixed_answer_tool_event"):
        return "mixed_turn"
    if output_signals.get("toolless_fresh_answer_event"):
        return "toolless_fresh"
    return None


def _retry_prompt_shape(
    *,
    task: StressTask,
    retry_index: int,
    target_already_validated: bool,
    payload_pressure_ready: bool,
    validated_target_exact_reacquired: bool,
    validated_target_reconfirmation_attempt: bool,
    mixed_answer_tool_event: bool,
    toolless_fresh_answer_event: bool,
    unsupported_tool_event: bool,
    bad_tool_args_event: bool,
) -> str:
    if (
        task.phase_name == "tool-contract"
        and target_already_validated
        and payload_pressure_ready
    ):
        if validated_target_exact_reacquired:
            return "exact_target_reread"
        if mixed_answer_tool_event:
            return "mixed_turn"
        if toolless_fresh_answer_event:
            return "toolless_fresh"
        if unsupported_tool_event:
            return "unsupported_tool"
        if bad_tool_args_event:
            return "bad_args"
        if validated_target_reconfirmation_attempt and retry_index >= 0:
            return "generic_retry"
    return "generic_retry"


def _late_retry_contract_stage(
    *,
    task: StressTask,
    prompt_shape: str,
    payload_pressure_ready: bool,
    validated_target_exact_reacquired: bool,
    unsupported_tool_event: bool,
    bad_tool_args_event: bool,
    prior_turn_has_valid_supporting_tool_backing: bool,
    current_turn_was_tool_only_retry: bool,
    current_turn_satisfied_tool_only_stage: bool,
) -> str | None:
    if prompt_shape != "generic_retry":
        return None
    if (
        not payload_pressure_ready
        or task.phase_name not in LATE_STAGED_RETRY_PHASES
    ):
        return None
    if (
        validated_target_exact_reacquired
        or unsupported_tool_event
        or bad_tool_args_event
    ):
        return None
    if (
        current_turn_was_tool_only_retry
        and current_turn_satisfied_tool_only_stage
    ):
        return "answer_only"
    if (
        task.require_fresh_evidence
        and not prior_turn_has_valid_supporting_tool_backing
    ):
        return "tool_only"
    return "answer_only"


def _early_retry_contract_stage(
    *,
    task: StressTask,
    payload_pressure_ready: bool,
    validated_target_exact_reacquired: bool,
    unsupported_tool_event: bool,
    bad_tool_args_event: bool,
    mixed_answer_tool_event: bool,
    prior_turn_has_valid_supporting_tool_backing: bool,
    current_turn_was_tool_only_retry: bool,
    current_turn_satisfied_tool_only_stage: bool,
) -> str | None:
    if task.phase_name != "tool-contract" or payload_pressure_ready:
        return None
    if validated_target_exact_reacquired or unsupported_tool_event:
        return None
    if (
        current_turn_was_tool_only_retry
        and current_turn_satisfied_tool_only_stage
    ):
        return "answer_only"
    if bad_tool_args_event:
        return "tool_only_bad_args"
    if mixed_answer_tool_event:
        if (
            task.require_fresh_evidence
            and not prior_turn_has_valid_supporting_tool_backing
        ):
            return "tool_only_mixed"
        return "answer_only_mixed"
    return None


def _runtime_turn_context_signals(
    *, payload_pressure_ready: bool, compaction_eligible_ready: bool = False
) -> dict[str, int]:
    signals: dict[str, int] = {}
    if payload_pressure_ready:
        signals["payload_pressure_ready"] = 1
    if compaction_eligible_ready:
        signals["compaction_eligible_ready"] = 1
    return signals


def _runtime_retry_context_signals(
    retry_prompt_signals: dict[str, int],
) -> dict[str, int]:
    signals: dict[str, int] = {}
    if retry_prompt_signals.get("late_retry_contract_stage_tool_only"):
        signals["late_retry_contract_stage_tool_only"] = 1
        signals["late_staged_retry_context"] = 1
    if retry_prompt_signals.get("late_retry_contract_stage_answer_only"):
        signals["late_retry_contract_stage_answer_only"] = 1
        signals["late_staged_retry_context"] = 1
    return signals


def _followthrough_evidence_sufficient(
    *,
    evidence_chars: int,
    payload_pressure_ready: bool,
    tool_results: list[dict[str, Any]],
) -> bool:
    if payload_pressure_ready:
        threshold = LATE_FOLLOWTHROUGH_MIN_EVIDENCE_CHARS
        min_tools = 2
    else:
        threshold = PRE_PRESSURE_MIN_EVIDENCE_CHARS
        min_tools = 1

    if evidence_chars < threshold:
        return False

    successful_tools = [
        r
        for r in tool_results
        if not r.get("is_error")
        and not str(r.get("content", "")).startswith("ERROR:")
    ]
    return len(successful_tools) >= min_tools


def _preprocess_runtime_contract_signals(
    *,
    task: StressTask,
    raw_response: str,
    attempt_tool_count_before_turn: int,
    payload_pressure_ready: bool,
    request_behavior_signals: dict[str, int],
    session: RuntimeSession | None = None,
) -> dict[str, int]:
    if task.phase_name == "checkpoint":
        return {}
    contract = response_contract_for_mode(raw_response, tool_compatible=True)
    tool_uses = [
        block
        for block in contract.content_blocks
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]
    fields = _extract_labeled_fields(
        _render_visible_text(contract.content_blocks), session=session
    )
    signals: dict[str, int] = {}
    if request_behavior_signals.get("late_staged_retry_context"):
        signals["late_staged_retry_context"] = 1
    if request_behavior_signals.get("late_retry_contract_stage_tool_only"):
        signals["late_retry_contract_stage_tool_only"] = 1
    if request_behavior_signals.get("late_retry_contract_stage_answer_only"):
        signals["late_retry_contract_stage_answer_only"] = 1
    if payload_pressure_ready:
        signals["payload_pressure_ready"] = 1
    late_context = payload_pressure_ready or signals.get(
        "late_staged_retry_context"
    )
    if tool_uses and fields and late_context:
        signals["mixed_answer_tool_event"] = 1
        signals["late_mixed_signal_promoted"] = 1
    if (
        task.require_fresh_evidence
        and fields
        and attempt_tool_count_before_turn + len(tool_uses)
        < max(task.require_tool_count, 1)
    ):
        signals["toolless_fresh_answer_event"] = 1
        signals["late_freshness_signal_promoted"] = 1
    return signals


def classify_breakpoints(
    observation: StressObservation,
    seen_classes: set[str],
    task: StressTask | None = None,
) -> list[StressBreakpoint]:
    candidates: list[str] = []
    output_signals = observation.output_behavior_signals
    input_signals = observation.input_behavior_signals

    if observation.tool_contract_failure:
        candidates.append("tool_contract_failure")

    if (
        task
        and task.required_tool_names
        and any(name.startswith("@") for name in task.required_tool_names)
        and not (set(task.required_tool_names) & set(observation.active_tools))
    ):
        candidates.append("macro_redundancy_loss")

    if (
        _is_protocol_failure(
            observation.prompt,
            observation.visible_response,
            observation.observed_fields,
            output_signals,
        )
        or observation.repeated_oracle_miss
    ):
        candidates.append("protocol_drift")

    if observation.target_already_validated and (
        observation.validated_target_exact_reacquired
        or observation.validated_target_reacquired
        or input_signals.get("validated_target_exact_reacquired")
        or input_signals.get("validated_target_reacquired")
        or (
            observation.active_tools
            and (
                input_signals.get("repeat_search")
                or input_signals.get("repeat_file_read")
                or input_signals.get("reacquisition_cost_tokens")
            )
        )
    ):
        candidates.append("reacquisition_loop")

    if observation.expected_fields and observation.phase in {
        "checkpoint",
        "late-recovery",
        "retention-probe",
    }:
        expected_file = observation.expected_fields.get("file", "").lower()
        expected_verification = observation.expected_fields.get(
            "verification", ""
        ).lower()
        observed_file = observation.observed_fields.get("file", "").lower()
        observed_verification = observation.observed_fields.get(
            "verification", ""
        ).lower()
        if (
            expected_file
            and expected_file not in observed_file
            or expected_verification
            and expected_verification not in observed_verification
        ):
            candidates.append("retention_loss")
    if observation.retention_latest_substitution:
        candidates.append("retention_loss")

    if (
        observation.payload_pressure_ready
        and observation.expected_fields
        and observation.phase in {"checkpoint", "late-recovery"}
        and (
            observation.input_behavior_signals.get("answer_anchor_present", 1)
            == 0
            or (
                observation.state_payload_chars
                and observation.state_payload_chars < 100
                and observation.resend_mode in {"suppressed", "delta", "none"}
            )
        )
    ):
        candidates.append("compaction_loss")

    if observation.baseline_only:
        candidates.append("baseline_fallback")

    breakpoints: list[StressBreakpoint] = []
    for breakpoint_class in candidates:
        if breakpoint_class in seen_classes:
            continue
        seen_classes.add(breakpoint_class)
        breakpoints.append(
            StressBreakpoint(
                breakpoint_class=breakpoint_class,
                task_id=observation.task_id,
                turn_index=observation.turn_index,
                prompt=observation.prompt,
                visible_response=observation.visible_response,
                active_tools=observation.active_tools,
                input_behavior_signals=dict(
                    observation.input_behavior_signals
                ),
                output_behavior_signals=dict(
                    observation.output_behavior_signals
                ),
                state_payload_chars=observation.state_payload_chars,
                resend_mode=observation.resend_mode,
                transcript_slice=list(observation.transcript_slice),
                inferred_cause=inferred_cause_for_class(breakpoint_class),
                refactor_target=refactor_target_for_class(breakpoint_class),
            )
        )
    return breakpoints


def _is_protocol_failure(
    prompt: str,
    visible_response: str,
    observed_fields: dict[str, str],
    output_signals: dict[str, int],
) -> bool:
    return bool(
        output_signals.get("semantic_drift_detected")
        or output_signals.get("non_tok_response")
        or output_signals.get("malformed_tok_response")
        or output_signals.get("fail_open_compat_response")
        or output_signals.get("malformed_tok_hybrid_tool")
        or output_signals.get("malformed_tok_markdown_fallback")
        or output_signals.get("grounded_oracle_miss_streak")
        or (
            output_signals.get("tool_compatible_response")
            and ("File=<" in prompt or "File=" in prompt)
            and not observed_fields.get("file")
            and len(visible_response.split()) > 15
        )
    )


def should_stop_run(
    *,
    breakpoint_count: int,
    baseline_only: bool,
    tasks_completed: int,
    seen_classes: set[str],
    config: StressHarnessConfig,
) -> bool:
    del breakpoint_count
    return (
        required_class_coverage(seen_classes, config.required_classes)[
            "complete"
        ]
        or baseline_only
        or tasks_completed >= config.max_tasks
    )


def required_class_coverage(
    seen_classes: set[str] | list[str], required_classes: tuple[str, ...]
) -> dict[str, Any]:
    seen = set(seen_classes)
    covered: list[str] = []
    missing: list[str] = []
    for item in required_classes:
        options = [
            option.strip() for option in item.split("|") if option.strip()
        ]
        if any(option in seen for option in options):
            covered.append(item)
        else:
            missing.append(item)
    return {"covered": covered, "missing": missing, "complete": not missing}


def _is_excluded_grounded_path(path: str) -> bool:
    normalized = path.strip()
    return any(
        fragment in normalized for fragment in EXCLUDED_GROUNDED_PATH_FRAGMENTS
    )
