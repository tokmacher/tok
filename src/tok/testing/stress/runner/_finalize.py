from __future__ import annotations

from typing import Any

from ..classification import required_class_coverage
from ..models import StressRunResult
from ..utils import _extract_labeled_fields, _is_supported_read_only_tool_name, _iso_now
from ._turn_validation import payload_pressure_reached, turn_has_valid_supporting_tool_backing


def turn_satisfies_tool_only_retry_stage(turn: Any, session: Any) -> bool:
    if len(turn.tool_uses) != 1:
        return False
    if not _is_supported_read_only_tool_name(turn.tool_uses[0].get("name", "")):
        return False
    if turn.validated_target_exact_reacquired:
        return False
    if any(
        turn.output_behavior_signals.get(key)
        for key in (
            "mixed_answer_tool_event",
            "unsupported_tool_event",
            "bad_tool_args_event",
        )
    ):
        return False
    return not _extract_labeled_fields(turn.visible_response, session=session)


def turn_satisfies_answer_only_retry_stage(turn: Any, session: Any) -> bool:
    if turn.tool_uses:
        return False
    if any(turn.output_behavior_signals.get(key) for key in ("unsupported_tool_event", "bad_tool_args_event")):
        return False
    fields = _extract_labeled_fields(turn.visible_response, session=session)
    return bool(fields.get("file") and fields.get("verification"))


def failed_task_retry_family(task_turns: list) -> str:
    if any(
        turn.output_behavior_signals.get(key)
        for turn in task_turns
        for key in (
            "retry_prompt_shape_exact_target_reread",
            "retry_prompt_shape_mixed_turn",
            "retry_prompt_shape_toolless_fresh",
            "retry_prompt_shape_unsupported_tool",
            "retry_prompt_shape_bad_args",
        )
    ):
        return "validated_target"
    if any(
        turn.output_behavior_signals.get("late_retry_contract_stage_tool_only")
        or turn.output_behavior_signals.get("late_retry_contract_stage_answer_only")
        for turn in task_turns
    ):
        return "late_staged"
    if any(
        turn.output_behavior_signals.get("early_retry_contract_stage_tool_only")
        or turn.output_behavior_signals.get("early_retry_contract_stage_answer_only")
        for turn in task_turns
    ):
        return "early_staged"
    if any(turn.output_behavior_signals.get("retry_prompt_shape_generic_retry") for turn in task_turns):
        return "generic"
    return "none"


def first_irreversible_miss_kind(task_turns: list, turn_has_valid_supporting_fn) -> str:
    prior_valid_grounding = False
    for turn in task_turns:
        output_signals = turn.output_behavior_signals
        fields = _extract_labeled_fields(turn.visible_response, session=None)
        if output_signals.get("bad_tool_args_event"):
            return "first_miss_bad_args"
        if output_signals.get("unsupported_tool_event"):
            return "first_miss_unsupported_tool"
        if output_signals.get("mixed_answer_tool_event"):
            return "first_miss_mixed_answer_tool"
        if output_signals.get("toolless_fresh_answer_event"):
            return "first_miss_toolless_fresh"
        if turn.tool_uses:
            if turn_has_valid_supporting_fn(turn):
                prior_valid_grounding = True
            elif not fields:
                return "first_miss_tool_only_insufficient"
        if prior_valid_grounding and fields and not turn.task_completed_validated:
            return "first_miss_answer_after_grounding"
        if not turn.tool_uses and turn.visible_response.strip() and not fields:
            return "first_miss_prose_no_tool"
    return "first_miss_unknown"


def failed_task_summaries(
    turns: list,
    failed_task_retry_family_fn,
    first_irreversible_miss_kind_fn,
) -> list:
    task_order: list[str] = []
    task_turns_map: dict[str, list] = {}
    for turn in turns:
        if turn.task_id not in task_turns_map:
            task_order.append(turn.task_id)
            task_turns_map[turn.task_id] = []
        task_turns_map[turn.task_id].append(turn)
    failed: list[dict[str, str]] = []
    for task_id in task_order:
        task_turns = task_turns_map[task_id]
        if any(turn.task_completed_validated for turn in task_turns):
            continue
        failed.append(
            {
                "task_id": task_id,
                "phase_name": (task_turns[0].phase_name if task_turns else ""),
                "retry_family": failed_task_retry_family_fn(task_turns),
                "first_irreversible_miss_kind": first_irreversible_miss_kind_fn(task_turns),
            }
        )
    return failed


def dominant_failure_locus(
    *,
    failed_task_summaries_list: list,
    answer_anchor_reacquisition_events_seen: int,
    answer_ready_reacquisition_events_seen: int,
    repair_phase_reacquisition_events_seen: int,
    answer_ready_repair_failed_count: int,
    fallback_after_compaction_eligible: bool,
) -> str:
    if not failed_task_summaries_list:
        return "mixed"
    total_failed = len(failed_task_summaries_list)
    before_any = sum(1 for item in failed_task_summaries_list if item["retry_family"] == "none")
    harness_shaped = sum(
        1
        for item in failed_task_summaries_list
        if item["retry_family"] in {"generic", "early_staged", "late_staged", "validated_target"}
    )
    tok_evidence = (
        answer_anchor_reacquisition_events_seen
        + answer_ready_reacquisition_events_seen
        + repair_phase_reacquisition_events_seen
        + answer_ready_repair_failed_count
        + int(fallback_after_compaction_eligible)
    )
    if tok_evidence > max(before_any, harness_shaped) and tok_evidence >= total_failed:
        return "tok"
    if before_any * 2 >= total_failed:
        return "agent"
    if harness_shaped * 2 >= total_failed and before_any == 0:
        return "harness"
    return "mixed"


def first_anchor_failure_mode(
    *,
    seed_direct_reads: int,
    seed_evidence_sufficient: bool,
    seed_wrong_field_attempts: int,
    seed_unstructured_answer_attempts: int,
) -> str:
    if seed_evidence_sufficient:
        if seed_unstructured_answer_attempts > 0:
            return "answer_assembly"
        if seed_wrong_field_attempts > 0:
            return "extraction"
        return "answer_assembly"
    if seed_direct_reads > 0:
        return "extraction"
    return "navigation"


def finalize_result(
    *,
    config: Any,
    session: Any,
    tasks: Any,
    started_at: str,
    task_count: int,
    total_prompt_tokens: int,
    total_completion_tokens: int,
    anchor_history: list,
    tool_backed_turns: int,
    resend_modes_seen: set[str],
    total_evidence_chars: int,
    breakpoints: list,
    turns: list,
    notes: list[str],
    seen_classes: set[str],
    reuse_checks_run: int,
    checkpoint_checks_run: int,
    reuse_probe_attempts: int,
    reuse_probe_successes: int,
    retention_probe_attempts: int,
    retention_probe_successes: int,
    late_retention_probe_attempts: int,
    late_retention_probe_successes: int,
    tool_contract_probe_attempts: int,
    tool_contract_failure_events_seen: int,
    mixed_answer_tool_events_seen: int,
    unsupported_tool_events_seen: int,
    bad_tool_args_events_seen: int,
    toolless_fresh_answer_events_seen: int,
    reacquisition_events_seen: int,
    validated_target_reacquisition_events_seen: int,
    validated_target_exact_reacquisition_events_seen: int,
    validated_target_reconfirmation_events_seen: int,
    answer_anchor_reacquisition_events_seen: int,
    answer_ready_reacquisition_events_seen: int,
    repair_phase_reacquisition_events_seen: int,
    benign_reverification_events_seen: int,
    retention_substitution_events_seen: int,
    compaction_eligible_turns: int,
    anchors_before_baseline: int | None,
    seed_searches: int,
    seed_direct_reads: int,
    seed_answer_attempts: int,
    seed_evidence_sufficient: bool,
    seed_wrong_field_attempts: int,
    seed_unstructured_answer_attempts: int,
) -> StressRunResult:
    coverage = required_class_coverage(seen_classes, config.required_classes)
    _payload_pressure = payload_pressure_reached(
        total_evidence_chars, len(anchor_history), config.min_payload_pressure_bytes
    )
    compaction_eligible = compaction_eligible_turns > 0
    first_payload_pressure_turn_record = next(
        (turn for turn in turns if getattr(turn, "payload_pressure_ready", False)),
        None,
    )
    first_compaction_eligible_turn_record = next(
        (turn for turn in turns if getattr(turn, "compaction_eligible_ready", False)),
        None,
    )
    first_baseline_fallback_turn_record = next(
        (turn for turn in turns if getattr(turn, "baseline_only", False)),
        None,
    )
    answer_ready_repair_requested_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("answer_ready_repair_requested")
    )
    answer_ready_repair_active_count = sum(
        1 for turn in turns if turn.input_behavior_signals.get("answer_ready_repair_active")
    )
    answer_ready_repair_resolved_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("answer_ready_repair_resolved")
    )
    answer_ready_repair_failed_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("answer_ready_repair_failed")
    )
    late_freshness_signal_promoted_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("late_freshness_signal_promoted")
    )
    late_freshness_signal_consumed_by_tok_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("late_freshness_signal_consumed_by_tok")
    )
    late_mixed_signal_promoted_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("late_mixed_signal_promoted")
    )
    late_mixed_signal_consumed_by_tok_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("late_mixed_signal_consumed_by_tok")
    )
    late_answer_assembly_repair_answer_only_requested_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("late_answer_assembly_repair_answer_only_requested")
    )
    late_answer_assembly_repair_answer_only_resolved_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("late_answer_assembly_repair_answer_only_resolved")
    )
    late_answer_assembly_repair_answer_only_failed_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("late_answer_assembly_repair_answer_only_failed")
    )
    late_answer_followthrough_requested_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("late_answer_followthrough_requested")
    )
    late_answer_followthrough_active_count = sum(
        1 for turn in turns if turn.input_behavior_signals.get("late_answer_followthrough_active")
    )
    late_answer_followthrough_resolved_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("late_answer_followthrough_resolved")
    )
    late_answer_followthrough_failed_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("late_answer_followthrough_failed")
    )
    late_answer_followthrough_after_tool_only_repair_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("late_answer_followthrough_after_tool_only_repair")
    )
    late_answer_followthrough_blocked_insufficient_evidence_count = sum(
        1
        for turn in turns
        if turn.output_behavior_signals.get("late_answer_followthrough_blocked_insufficient_evidence")
    )
    late_tool_contract_reconfirmation_grace_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("late_tool_contract_reconfirmation_grace")
    )
    late_tool_contract_mixed_grace_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("late_tool_contract_mixed_grace")
    )
    late_tool_contract_toolless_grace_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("late_tool_contract_toolless_grace")
    )
    late_tool_contract_reconfirmation_retry_failure_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("late_tool_contract_reconfirmation_retry_failure")
    )
    late_tool_contract_mixed_retry_failure_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("late_tool_contract_mixed_retry_failure")
    )
    late_tool_contract_toolless_retry_failure_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("late_tool_contract_toolless_retry_failure")
    )
    fallback_pressure_incremented_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("fallback_pressure_incremented")
    )
    fallback_pressure_suppressed_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("fallback_pressure_suppressed")
    )
    fallback_pressure_cause_exact_reacquisition_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("fallback_pressure_cause_exact_reacquisition")
    )
    fallback_pressure_cause_mixed_turn_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("fallback_pressure_cause_mixed_turn")
    )
    fallback_pressure_cause_toolless_fresh_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("fallback_pressure_cause_toolless_fresh")
    )
    fallback_pressure_cause_bad_args_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("fallback_pressure_cause_bad_args")
    )
    fallback_pressure_cause_unsupported_tool_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("fallback_pressure_cause_unsupported_tool")
    )
    retry_prompt_shape_exact_target_reread_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("retry_prompt_shape_exact_target_reread")
    )
    retry_prompt_shape_mixed_turn_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("retry_prompt_shape_mixed_turn")
    )
    retry_prompt_shape_toolless_fresh_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("retry_prompt_shape_toolless_fresh")
    )
    retry_prompt_shape_unsupported_tool_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("retry_prompt_shape_unsupported_tool")
    )
    retry_prompt_shape_bad_args_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("retry_prompt_shape_bad_args")
    )
    retry_prompt_shape_generic_retry_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("retry_prompt_shape_generic_retry")
    )
    retry_prompt_no_exact_reread_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("retry_prompt_no_exact_reread")
    )
    retry_prompt_requires_supporting_tool_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("retry_prompt_requires_supporting_tool")
    )
    early_retry_contract_stage_tool_only_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("early_retry_contract_stage_tool_only")
    )
    early_retry_contract_stage_answer_only_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("early_retry_contract_stage_answer_only")
    )
    early_retry_bad_args_tool_only_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("early_retry_bad_args_tool_only")
    )
    early_retry_tool_only_satisfied_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("early_retry_tool_only_satisfied")
    )
    early_retry_tool_only_failed_mixed_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("early_retry_tool_only_failed_mixed")
    )
    early_retry_tool_only_failed_toolless_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("early_retry_tool_only_failed_toolless")
    )
    early_retry_answer_only_satisfied_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("early_retry_answer_only_satisfied")
    )
    early_retry_answer_only_failed_tool_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("early_retry_answer_only_failed_tool")
    )
    late_retry_contract_stage_tool_only_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("late_retry_contract_stage_tool_only")
    )
    late_retry_contract_stage_answer_only_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("late_retry_contract_stage_answer_only")
    )
    late_retry_tool_only_satisfied_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("late_retry_tool_only_satisfied")
    )
    late_retry_tool_only_failed_mixed_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("late_retry_tool_only_failed_mixed")
    )
    late_retry_tool_only_failed_toolless_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("late_retry_tool_only_failed_toolless")
    )
    late_retry_answer_only_satisfied_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("late_retry_answer_only_satisfied")
    )
    late_retry_answer_only_failed_tool_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("late_retry_answer_only_failed_tool")
    )
    late_retry_no_exact_target_count = sum(
        1 for turn in turns if turn.output_behavior_signals.get("late_retry_no_exact_target")
    )
    exact_target_reread_after_no_exact_retry_count = 0
    exact_target_reread_after_late_retry_no_exact_target_count = 0
    retry_prompt_supporting_tool_satisfied_count = 0
    retry_prompt_supporting_tool_missed_count = 0
    retry_prompt_supporting_tool_missed_mixed_count = 0
    retry_prompt_supporting_tool_missed_toolless_count = 0
    early_retry_tool_only_satisfied_count = 0
    early_retry_tool_only_failed_mixed_count = 0
    early_retry_tool_only_failed_toolless_count = 0
    early_retry_answer_only_satisfied_count = 0
    early_retry_answer_only_failed_tool_count = 0
    late_retry_tool_only_satisfied_count = 0
    late_retry_tool_only_failed_mixed_count = 0
    late_retry_tool_only_failed_toolless_count = 0
    late_retry_answer_only_satisfied_count = 0
    late_retry_answer_only_failed_tool_count = 0
    previous_no_exact_retry = False
    previous_late_retry_no_exact_target = False
    previous_supporting_tool_shape: str | None = None
    previous_early_retry_stage: str | None = None
    previous_late_retry_stage: str | None = None
    for turn in turns:
        if previous_no_exact_retry and turn.validated_target_exact_reacquired:
            exact_target_reread_after_no_exact_retry_count += 1
        if previous_late_retry_no_exact_target and turn.validated_target_exact_reacquired:
            exact_target_reread_after_late_retry_no_exact_target_count += 1
        previous_no_exact_retry = bool(turn.output_behavior_signals.get("retry_prompt_no_exact_reread"))
        previous_late_retry_no_exact_target = bool(turn.output_behavior_signals.get("late_retry_no_exact_target"))
        if previous_supporting_tool_shape and turn.retry_index > 0:
            invalid_tool_use = any(
                turn.output_behavior_signals.get(key)
                for key in (
                    "unsupported_tool_event",
                    "bad_tool_args_event",
                )
            )
            if turn.tool_uses and not turn.validated_target_exact_reacquired and not invalid_tool_use:
                retry_prompt_supporting_tool_satisfied_count += 1
                turn.output_behavior_signals["retry_prompt_supporting_tool_satisfied"] = 1
            else:
                retry_prompt_supporting_tool_missed_count += 1
                turn.output_behavior_signals["retry_prompt_supporting_tool_missed"] = 1
                if previous_supporting_tool_shape == "mixed":
                    retry_prompt_supporting_tool_missed_mixed_count += 1
                elif previous_supporting_tool_shape == "toolless":
                    retry_prompt_supporting_tool_missed_toolless_count += 1
        if previous_early_retry_stage == "tool_only" and turn.retry_index > 0:
            if turn_satisfies_tool_only_retry_stage(turn, session):
                early_retry_tool_only_satisfied_count += 1
                turn.output_behavior_signals["early_retry_tool_only_satisfied"] = 1
            elif turn.output_behavior_signals.get("mixed_answer_tool_event"):
                early_retry_tool_only_failed_mixed_count += 1
                turn.output_behavior_signals["early_retry_tool_only_failed_mixed"] = 1
            else:
                early_retry_tool_only_failed_toolless_count += 1
                turn.output_behavior_signals["early_retry_tool_only_failed_toolless"] = 1
        elif previous_early_retry_stage == "answer_only" and turn.retry_index > 0:
            if turn_satisfies_answer_only_retry_stage(turn, session):
                early_retry_answer_only_satisfied_count += 1
                turn.output_behavior_signals["early_retry_answer_only_satisfied"] = 1
            elif turn.tool_uses or any(
                turn.output_behavior_signals.get(key)
                for key in (
                    "unsupported_tool_event",
                    "bad_tool_args_event",
                )
            ):
                early_retry_answer_only_failed_tool_count += 1
                turn.output_behavior_signals["early_retry_answer_only_failed_tool"] = 1
        if previous_late_retry_stage == "tool_only" and turn.retry_index > 0:
            if turn_satisfies_tool_only_retry_stage(turn, session):
                late_retry_tool_only_satisfied_count += 1
                turn.output_behavior_signals["late_retry_tool_only_satisfied"] = 1
            elif turn.output_behavior_signals.get("mixed_answer_tool_event"):
                late_retry_tool_only_failed_mixed_count += 1
                turn.output_behavior_signals["late_retry_tool_only_failed_mixed"] = 1
            else:
                late_retry_tool_only_failed_toolless_count += 1
                turn.output_behavior_signals["late_retry_tool_only_failed_toolless"] = 1
        elif previous_late_retry_stage == "answer_only" and turn.retry_index > 0:
            if turn_satisfies_answer_only_retry_stage(turn, session):
                late_retry_answer_only_satisfied_count += 1
                turn.output_behavior_signals["late_retry_answer_only_satisfied"] = 1
            elif turn.tool_uses or any(
                turn.output_behavior_signals.get(key)
                for key in (
                    "unsupported_tool_event",
                    "bad_tool_args_event",
                )
            ):
                late_retry_answer_only_failed_tool_count += 1
                turn.output_behavior_signals["late_retry_answer_only_failed_tool"] = 1
        previous_supporting_tool_shape = None
        previous_early_retry_stage = None
        previous_late_retry_stage = None
        if turn.output_behavior_signals.get("retry_prompt_requires_supporting_tool"):
            if turn.output_behavior_signals.get("retry_prompt_shape_mixed_turn"):
                previous_supporting_tool_shape = "mixed"
            elif turn.output_behavior_signals.get("retry_prompt_shape_toolless_fresh"):
                previous_supporting_tool_shape = "toolless"
        if turn.output_behavior_signals.get("early_retry_contract_stage_tool_only"):
            previous_early_retry_stage = "tool_only"
        elif turn.output_behavior_signals.get("early_retry_contract_stage_answer_only"):
            previous_early_retry_stage = "answer_only"
        if turn.output_behavior_signals.get("late_retry_contract_stage_tool_only"):
            previous_late_retry_stage = "tool_only"
        elif turn.output_behavior_signals.get("late_retry_contract_stage_answer_only"):
            previous_late_retry_stage = "answer_only"
    baseline_fallback_turns_after_payload_pressure = (
        sum(
            1
            for turn in turns
            if turn.baseline_only
            and first_payload_pressure_turn_record is not None
            and turn.turn_index >= first_payload_pressure_turn_record.turn_index
        )
        if first_payload_pressure_turn_record is not None
        else 0
    )
    baseline_fallback_turns_after_compaction_eligible = (
        sum(
            1
            for turn in turns
            if turn.baseline_only
            and first_compaction_eligible_turn_record is not None
            and turn.turn_index >= first_compaction_eligible_turn_record.turn_index
        )
        if first_compaction_eligible_turn_record is not None
        else 0
    )
    _failed_task_summaries = failed_task_summaries(
        turns,
        failed_task_retry_family,
        lambda task_turns: first_irreversible_miss_kind(task_turns, turn_has_valid_supporting_tool_backing),
    )
    failed_tasks_before_any_retry_contract_count = sum(
        1 for item in _failed_task_summaries if item["retry_family"] == "none"
    )
    failed_tasks_after_generic_retry_only_count = sum(
        1 for item in _failed_task_summaries if item["retry_family"] == "generic"
    )
    failed_tasks_after_early_staged_retry_count = sum(
        1 for item in _failed_task_summaries if item["retry_family"] == "early_staged"
    )
    failed_tasks_after_late_staged_retry_count = sum(
        1 for item in _failed_task_summaries if item["retry_family"] == "late_staged"
    )
    failed_tasks_after_validated_target_retry_count = sum(
        1 for item in _failed_task_summaries if item["retry_family"] == "validated_target"
    )
    first_failed_phase = _failed_task_summaries[0]["phase_name"] if _failed_task_summaries else ""
    first_failed_task = _failed_task_summaries[0]["task_id"] if _failed_task_summaries else ""
    _first_irreversible_miss_kind = (
        _failed_task_summaries[0]["first_irreversible_miss_kind"] if _failed_task_summaries else "first_miss_unknown"
    )
    fallback_after_compaction_eligible = baseline_fallback_turns_after_compaction_eligible > 0
    _dominant_failure_locus = dominant_failure_locus(
        failed_task_summaries_list=_failed_task_summaries,
        answer_anchor_reacquisition_events_seen=answer_anchor_reacquisition_events_seen,
        answer_ready_reacquisition_events_seen=answer_ready_reacquisition_events_seen,
        repair_phase_reacquisition_events_seen=repair_phase_reacquisition_events_seen,
        answer_ready_repair_failed_count=answer_ready_repair_failed_count,
        fallback_after_compaction_eligible=fallback_after_compaction_eligible,
    )
    weak_run_reasons: list[str] = []
    if not _payload_pressure:
        weak_run_reasons.append("payload_pressure_not_reached")
    if turns and tool_backed_turns / len(turns) < 0.35:
        weak_run_reasons.append("low_tool_backed_turn_ratio")
    if _payload_pressure and not compaction_eligible:
        weak_run_reasons.append("payload_not_compaction_eligible")
    if compaction_eligible and not {"delta", "suppressed"} & resend_modes_seen:
        weak_run_reasons.append("never_left_full_resend")
    _first_anchor_failure_mode = first_anchor_failure_mode(
        seed_direct_reads=seed_direct_reads,
        seed_evidence_sufficient=seed_evidence_sufficient,
        seed_wrong_field_attempts=seed_wrong_field_attempts,
        seed_unstructured_answer_attempts=seed_unstructured_answer_attempts,
    )
    missing = set(coverage["missing"])
    has_tool_contract_probe_phase = any(getattr(task, "phase_name", "") == "tool-contract" for task in tasks)
    if session._baseline_only and (reuse_checks_run < 1 or checkpoint_checks_run < 1):
        run_diagnosis = f"early_contract_collapse:{_first_anchor_failure_mode}"
    elif "tool_contract_failure" in missing and has_tool_contract_probe_phase and tool_contract_probe_attempts == 0:
        run_diagnosis = "tool_contract_surface_unexercised"
    elif (
        "tool_contract_failure" in missing
        and has_tool_contract_probe_phase
        and tool_contract_probe_attempts > 0
        and tool_contract_failure_events_seen == 0
    ):
        run_diagnosis = "tool_contract_surface_held"
    elif "reacquisition_loop" in missing and reuse_probe_attempts == 0:
        run_diagnosis = "reuse_surface_unexercised"
    elif "retention_loss" in missing and retention_probe_attempts == 0:
        run_diagnosis = "retention_surface_unexercised"
    elif (
        "retention_loss" in missing
        and retention_probe_attempts > 0
        and retention_probe_successes > 0
        and late_retention_probe_attempts == 0
        and retention_substitution_events_seen == 0
    ):
        run_diagnosis = "retention_surface_held_early_only"
    elif (
        "retention_loss" in missing
        and late_retention_probe_attempts > 0
        and late_retention_probe_successes > 0
        and retention_substitution_events_seen == 0
    ):
        run_diagnosis = "retention_surface_held"
    elif reuse_checks_run >= 1 and checkpoint_checks_run >= 1 and not _payload_pressure:
        run_diagnosis = "payload_surface_not_reached"
    elif (
        reuse_checks_run >= 1
        and checkpoint_checks_run >= 1
        and coverage["missing"]
        and reuse_probe_attempts > 0
        and late_retention_probe_attempts > 0
    ):
        run_diagnosis = "memory_surface_reached_but_not_broken"
    elif reuse_checks_run >= 1 and checkpoint_checks_run >= 1 and coverage["missing"]:
        run_diagnosis = "memory_surface_reached"
    elif coverage["missing"] and weak_run_reasons:
        run_diagnosis = "weak_harness_pressure"
    elif coverage["missing"]:
        run_diagnosis = "tok_resisted_under_pressure"
    else:
        run_diagnosis = "required_coverage_reached"
    completed_at = _iso_now()
    return StressRunResult(
        model=config.model,
        provider=config.provider,
        started_at=started_at,
        completed_at=completed_at,
        target_breakpoints=config.target_breakpoints,
        required_classes=config.required_classes,
        max_tasks=config.max_tasks,
        max_tool_rounds=config.max_tool_rounds,
        tasks_completed=task_count,
        baseline_only=session._baseline_only,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=total_prompt_tokens + total_completion_tokens,
        validated_anchor_count=len(anchor_history),
        tool_backed_turns=tool_backed_turns,
        resend_modes_seen=sorted(resend_modes_seen),
        payload_pressure_reached=_payload_pressure,
        compaction_eligible=compaction_eligible,
        reuse_checks_run=reuse_checks_run,
        checkpoint_checks_run=checkpoint_checks_run,
        reuse_probe_attempts=reuse_probe_attempts,
        reuse_probe_successes=reuse_probe_successes,
        retention_probe_attempts=retention_probe_attempts,
        retention_probe_successes=retention_probe_successes,
        late_retention_probe_attempts=late_retention_probe_attempts,
        late_retention_probe_successes=late_retention_probe_successes,
        tool_contract_probe_attempts=tool_contract_probe_attempts,
        tool_contract_failure_events_seen=tool_contract_failure_events_seen,
        mixed_answer_tool_events_seen=mixed_answer_tool_events_seen,
        unsupported_tool_events_seen=unsupported_tool_events_seen,
        bad_tool_args_events_seen=bad_tool_args_events_seen,
        toolless_fresh_answer_events_seen=toolless_fresh_answer_events_seen,
        reacquisition_events_seen=reacquisition_events_seen,
        validated_target_reacquisition_events_seen=validated_target_reacquisition_events_seen,
        validated_target_exact_reacquisition_events_seen=validated_target_exact_reacquisition_events_seen,
        validated_target_reconfirmation_events_seen=validated_target_reconfirmation_events_seen,
        answer_anchor_reacquisition_events_seen=answer_anchor_reacquisition_events_seen,
        answer_ready_reacquisition_events_seen=answer_ready_reacquisition_events_seen,
        repair_phase_reacquisition_events_seen=repair_phase_reacquisition_events_seen,
        benign_reverification_events_seen=benign_reverification_events_seen,
        answer_ready_repair_requested_count=answer_ready_repair_requested_count,
        answer_ready_repair_active_count=answer_ready_repair_active_count,
        answer_ready_repair_resolved_count=answer_ready_repair_resolved_count,
        answer_ready_repair_failed_count=answer_ready_repair_failed_count,
        late_freshness_signal_promoted_count=late_freshness_signal_promoted_count,
        late_freshness_signal_consumed_by_tok_count=late_freshness_signal_consumed_by_tok_count,
        late_mixed_signal_promoted_count=late_mixed_signal_promoted_count,
        late_mixed_signal_consumed_by_tok_count=late_mixed_signal_consumed_by_tok_count,
        late_answer_assembly_repair_answer_only_requested_count=late_answer_assembly_repair_answer_only_requested_count,
        late_answer_assembly_repair_answer_only_resolved_count=late_answer_assembly_repair_answer_only_resolved_count,
        late_answer_assembly_repair_answer_only_failed_count=late_answer_assembly_repair_answer_only_failed_count,
        late_answer_followthrough_requested_count=late_answer_followthrough_requested_count,
        late_answer_followthrough_active_count=late_answer_followthrough_active_count,
        late_answer_followthrough_resolved_count=late_answer_followthrough_resolved_count,
        late_answer_followthrough_failed_count=late_answer_followthrough_failed_count,
        late_answer_followthrough_after_tool_only_repair_count=late_answer_followthrough_after_tool_only_repair_count,
        late_answer_followthrough_blocked_insufficient_evidence_count=late_answer_followthrough_blocked_insufficient_evidence_count,
        late_tool_contract_reconfirmation_grace_count=late_tool_contract_reconfirmation_grace_count,
        late_tool_contract_mixed_grace_count=late_tool_contract_mixed_grace_count,
        late_tool_contract_toolless_grace_count=late_tool_contract_toolless_grace_count,
        late_tool_contract_reconfirmation_retry_failure_count=late_tool_contract_reconfirmation_retry_failure_count,
        late_tool_contract_mixed_retry_failure_count=late_tool_contract_mixed_retry_failure_count,
        late_tool_contract_toolless_retry_failure_count=late_tool_contract_toolless_retry_failure_count,
        fallback_pressure_incremented_count=fallback_pressure_incremented_count,
        fallback_pressure_suppressed_count=fallback_pressure_suppressed_count,
        fallback_pressure_cause_exact_reacquisition_count=fallback_pressure_cause_exact_reacquisition_count,
        fallback_pressure_cause_mixed_turn_count=fallback_pressure_cause_mixed_turn_count,
        fallback_pressure_cause_toolless_fresh_count=fallback_pressure_cause_toolless_fresh_count,
        fallback_pressure_cause_bad_args_count=fallback_pressure_cause_bad_args_count,
        fallback_pressure_cause_unsupported_tool_count=fallback_pressure_cause_unsupported_tool_count,
        retry_prompt_shape_exact_target_reread_count=retry_prompt_shape_exact_target_reread_count,
        retry_prompt_shape_mixed_turn_count=retry_prompt_shape_mixed_turn_count,
        retry_prompt_shape_toolless_fresh_count=retry_prompt_shape_toolless_fresh_count,
        retry_prompt_shape_unsupported_tool_count=retry_prompt_shape_unsupported_tool_count,
        retry_prompt_shape_bad_args_count=retry_prompt_shape_bad_args_count,
        retry_prompt_shape_generic_retry_count=retry_prompt_shape_generic_retry_count,
        retry_prompt_no_exact_reread_count=retry_prompt_no_exact_reread_count,
        retry_prompt_requires_supporting_tool_count=retry_prompt_requires_supporting_tool_count,
        retry_prompt_supporting_tool_satisfied_count=retry_prompt_supporting_tool_satisfied_count,
        retry_prompt_supporting_tool_missed_count=retry_prompt_supporting_tool_missed_count,
        retry_prompt_supporting_tool_missed_mixed_count=retry_prompt_supporting_tool_missed_mixed_count,
        retry_prompt_supporting_tool_missed_toolless_count=retry_prompt_supporting_tool_missed_toolless_count,
        exact_target_reread_after_no_exact_retry_count=exact_target_reread_after_no_exact_retry_count,
        early_retry_contract_stage_tool_only_count=early_retry_contract_stage_tool_only_count,
        early_retry_contract_stage_answer_only_count=early_retry_contract_stage_answer_only_count,
        early_retry_bad_args_tool_only_count=early_retry_bad_args_tool_only_count,
        early_retry_tool_only_satisfied_count=early_retry_tool_only_satisfied_count,
        early_retry_tool_only_failed_mixed_count=early_retry_tool_only_failed_mixed_count,
        early_retry_tool_only_failed_toolless_count=early_retry_tool_only_failed_toolless_count,
        early_retry_answer_only_satisfied_count=early_retry_answer_only_satisfied_count,
        early_retry_answer_only_failed_tool_count=early_retry_answer_only_failed_tool_count,
        late_retry_contract_stage_tool_only_count=late_retry_contract_stage_tool_only_count,
        late_retry_contract_stage_answer_only_count=late_retry_contract_stage_answer_only_count,
        late_retry_tool_only_satisfied_count=late_retry_tool_only_satisfied_count,
        late_retry_tool_only_failed_mixed_count=late_retry_tool_only_failed_mixed_count,
        late_retry_tool_only_failed_toolless_count=late_retry_tool_only_failed_toolless_count,
        late_retry_answer_only_satisfied_count=late_retry_answer_only_satisfied_count,
        late_retry_answer_only_failed_tool_count=late_retry_answer_only_failed_tool_count,
        late_retry_no_exact_target_count=late_retry_no_exact_target_count,
        exact_target_reread_after_late_retry_no_exact_target_count=exact_target_reread_after_late_retry_no_exact_target_count,
        failed_tasks_before_any_retry_contract_count=failed_tasks_before_any_retry_contract_count,
        failed_tasks_after_generic_retry_only_count=failed_tasks_after_generic_retry_only_count,
        failed_tasks_after_early_staged_retry_count=failed_tasks_after_early_staged_retry_count,
        failed_tasks_after_late_staged_retry_count=failed_tasks_after_late_staged_retry_count,
        failed_tasks_after_validated_target_retry_count=failed_tasks_after_validated_target_retry_count,
        first_failed_phase=first_failed_phase,
        first_failed_task=first_failed_task,
        first_irreversible_miss_kind=_first_irreversible_miss_kind,
        dominant_failure_locus=_dominant_failure_locus,
        first_payload_pressure_turn=getattr(first_payload_pressure_turn_record, "turn_index", None),
        first_payload_pressure_task=getattr(first_payload_pressure_turn_record, "task_id", ""),
        first_compaction_eligible_turn=getattr(first_compaction_eligible_turn_record, "turn_index", None),
        first_compaction_eligible_task=getattr(first_compaction_eligible_turn_record, "task_id", ""),
        first_baseline_fallback_turn=getattr(first_baseline_fallback_turn_record, "turn_index", None),
        first_baseline_fallback_task=getattr(first_baseline_fallback_turn_record, "task_id", ""),
        baseline_fallback_turns_after_payload_pressure=baseline_fallback_turns_after_payload_pressure,
        baseline_fallback_turns_after_compaction_eligible=baseline_fallback_turns_after_compaction_eligible,
        fallback_after_payload_pressure=baseline_fallback_turns_after_payload_pressure > 0,
        fallback_after_compaction_eligible=fallback_after_compaction_eligible,
        retention_substitution_events_seen=retention_substitution_events_seen,
        compaction_eligible_turns=compaction_eligible_turns,
        anchors_before_baseline=(
            anchors_before_baseline if anchors_before_baseline is not None else len(anchor_history)
        ),
        seed_searches=seed_searches,
        seed_direct_reads=seed_direct_reads,
        seed_answer_attempts=seed_answer_attempts,
        seed_evidence_sufficient=seed_evidence_sufficient,
        first_anchor_failure_mode=_first_anchor_failure_mode,
        run_diagnosis=run_diagnosis,
        weak_run_reasons=weak_run_reasons,
        breakpoints=breakpoints,
        turns=turns,
        notes=[
            *notes,
            f"required_coverage_missing:{','.join(coverage['missing']) or 'none'}",
        ],
    )
