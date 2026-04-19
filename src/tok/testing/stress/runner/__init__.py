from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from openai import OpenAI

from tok.runtime import (
    TOOL_DENSITY_THRESHOLD,
    RuntimeRequest,
    RuntimeSession,
    UniversalTokRuntime,
)
from tok.runtime.pipeline.response_processing import response_contract_for_mode

from ..catalog import TASK_CATALOG
from ..classification import (
    _fallback_pressure_cause,
    _followthrough_evidence_sufficient,
    _is_protocol_failure,
    _late_tool_contract_grace_kind,
    _preprocess_runtime_contract_signals,
    _runtime_retry_context_signals,
    _runtime_turn_context_signals,
    classify_breakpoints,
    should_stop_run,
)
from ..executor import ReadOnlyToolExecutor
from ..models import (
    StressBreakpoint,
    StressHarnessConfig,
    StressObservation,
    StressRunResult,
    StressTask,
    StressTurnRecord,
    ValidatedAnchor,
)
from ..utils import (
    _classify_validated_target_tool_use,
    _compact_message,
    _extract_labeled_fields,
    _fields_key,
    _iso_now,
    _normalize_chat_messages,
    _render_visible_text,
    _sanitize_tool_use_block,
    _system_to_messages,
)
from ._finalize import (
    dominant_failure_locus,
    failed_task_retry_family,
    failed_task_summaries,
    finalize_result,
    first_anchor_failure_mode,
    first_irreversible_miss_kind,
    turn_satisfies_answer_only_retry_stage,
    turn_satisfies_tool_only_retry_stage,
)
from ._retry_prompts import (
    checkpoint_prompt,
    retry_prompt,
    seed_synthesis_prompt,
)
from ._scheduling import (
    expected_fields_for_task,
    task_prompt,
    task_ready,
)
from ._signals import (
    repetition_signals,
    resend_decision_reason,
    resend_mode,
    tool_signature,
    update_failure_counter,
)
from ._turn_validation import (
    fields_match_expected,
    payload_pressure_reached,
    seed_tool_summary,
    task_answer_validated,
    turn_has_valid_supporting_tool_backing,
)


class StressHarness:
    def __init__(
        self,
        config: StressHarnessConfig,
        *,
        client: Any | None = None,
        runtime: UniversalTokRuntime | None = None,
        session: RuntimeSession | None = None,
        workspace_root: Path | None = None,
    ) -> None:
        load_dotenv()
        self.config = config
        self.runtime = runtime or UniversalTokRuntime()
        self.session = session or RuntimeSession()
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
        self.tool_executor = ReadOnlyToolExecutor(self.workspace_root)
        self.tasks = config.task_catalog or TASK_CATALOG
        api_key = config.api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if client is None and not api_key:
            msg = (
                "Missing API key. Set OPENROUTER_API_KEY (preferred) or OPENAI_API_KEY, "
                "or pass api_key in StressHarnessConfig."
            )
            raise ValueError(msg)
        self.client = client or OpenAI(
            base_url=config.api_base,
            api_key=api_key,
            timeout=120.0,
            max_retries=0,
        )
        self._consecutive_failures = 0
        self._progress = config.progress

    def run(self) -> StressRunResult:
        started_at = _iso_now()
        system_prompt = self._system_prompt()
        conversation: list[dict[str, Any]] = []
        breakpoints: list[StressBreakpoint] = []
        turns: list[StressTurnRecord] = []
        notes: list[str] = []
        seen_classes: set[str] = set()
        anchor_history: list[ValidatedAnchor] = []
        validated_targets: set[str] = set()
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_evidence_chars = 0
        tool_backed_turns = 0
        resend_modes_seen: set[str] = set()
        reuse_checks_run = 0
        checkpoint_checks_run = 0
        anchors_before_baseline: int | None = None
        seed_searches = 0
        seed_direct_reads = 0
        seed_answer_attempts = 0
        seed_evidence_sufficient = False
        seed_wrong_field_attempts = 0
        seed_unstructured_answer_attempts = 0
        reuse_probe_attempts = 0
        reuse_probe_successes = 0
        retention_probe_attempts = 0
        retention_probe_successes = 0
        late_retention_probe_attempts = 0
        late_retention_probe_successes = 0
        tool_contract_probe_attempts = 0
        tool_contract_failure_events_seen = 0
        mixed_answer_tool_events_seen = 0
        unsupported_tool_events_seen = 0
        bad_tool_args_events_seen = 0
        toolless_fresh_answer_events_seen = 0
        reacquisition_events_seen = 0
        validated_target_reacquisition_events_seen = 0
        validated_target_exact_reacquisition_events_seen = 0
        validated_target_reconfirmation_events_seen = 0
        answer_anchor_reacquisition_events_seen = 0
        answer_ready_reacquisition_events_seen = 0
        repair_phase_reacquisition_events_seen = 0
        benign_reverification_events_seen = 0
        retention_substitution_events_seen = 0
        compaction_eligible_turns = 0
        task_count = 0
        global_turn = 0
        task_cursor = 0

        while task_count < self.config.max_tasks:
            task = self.tasks[task_cursor % len(self.tasks)]
            task_cursor += 1
            expected_fields = self._expected_fields_for_task(task, anchor_history)
            if not self._task_ready(
                task,
                anchor_history,
                total_evidence_chars,
                expected_fields,
                reuse_checks_run=reuse_checks_run,
                reuse_probe_attempts=reuse_probe_attempts,
                checkpoint_checks_run=checkpoint_checks_run,
            ):
                notes.append(f"task_skipped:{task.id}:not_ready")
                if task_cursor > len(self.tasks) * 3:
                    break
                continue

            task_prompt = self._task_prompt(task, expected_fields)
            self._log(
                f"[stress] task {task_count + 1}/{self.config.max_tasks} start: {task.id} phase={task.phase_name}"
            )
            conversation.append({"role": "user", "content": task_prompt})
            task_count += 1
            if task.phase_name == "reuse-probe":
                reuse_probe_attempts += 1
            if task.phase_name == "retention-probe":
                retention_probe_attempts += 1
                if task.id != "retention_probe_early":
                    late_retention_probe_attempts += 1
            if task.phase_name == "tool-contract":
                tool_contract_probe_attempts += 1
            retry_index = 0
            attempt_tool_count = 0
            attempt_tool_names: set[str] = set()
            attempt_evidence_chars = 0
            attempt_validated_reacquisition = False
            grounded_miss_count = 0
            attempt_seed_evidence_sufficient = False
            task_finished = False
            checkpoint_expected: dict[str, str] = {}

            total_task_steps = self.config.max_tool_rounds + self.config.max_retries_per_task
            for _ in range(total_task_steps):
                global_turn += 1
                turn, new_breakpoints, has_tool_use, fields = self._run_turn(
                    conversation=conversation,
                    system_prompt=system_prompt,
                    prompt=conversation[-1]["content"],
                    task=task,
                    expected_fields=(expected_fields if task.phase_name != "checkpoint" else checkpoint_expected),
                    turn_index=global_turn,
                    retry_index=retry_index,
                    attempt_tool_count_before_turn=attempt_tool_count,
                    target_already_validated=_fields_key(expected_fields) in validated_targets,
                    payload_pressure_ready=self._payload_pressure_reached(
                        total_evidence_chars + attempt_evidence_chars,
                        len(anchor_history),
                    ),
                    grounded_miss_count=grounded_miss_count,
                    attempt_seed_evidence_sufficient_before_turn=attempt_seed_evidence_sufficient,
                    latest_anchor_fields=(anchor_history[-1].to_fields() if anchor_history else {}),
                    validated_target_keys=set(validated_targets),
                    seen_classes=seen_classes,
                )
                turns.append(turn)
                breakpoints.extend(new_breakpoints)
                resend_modes_seen.add(turn.resend_mode)
                total_prompt_tokens += turn.usage.get("prompt_tokens", 0)
                total_completion_tokens += turn.usage.get("completion_tokens", 0)
                if turn.tool_uses:
                    tool_backed_turns += 1
                    attempt_tool_count += len(turn.tool_uses)
                    attempt_tool_names.update(str(block.get("name", "")).strip().lower() for block in turn.tool_uses)
                    attempt_evidence_chars += turn.evidence_chars
                    total_evidence_chars += turn.evidence_chars
                    if (
                        task.phase_name == "payload-pressure"
                        and attempt_tool_count >= 1
                        and attempt_evidence_chars >= 500
                        and len(anchor_history) >= 2
                    ):
                        compaction_eligible_turns = max(compaction_eligible_turns, 1)
                        turns[-1] = replace(turns[-1], compaction_eligible_ready=True)
                if task.phase_name == "anchor-seed":
                    seed_searches += turn.input_behavior_signals.get("seed_search_tools_used", 0)
                    seed_direct_reads += turn.input_behavior_signals.get("seed_direct_read_tools_used", 0)
                    if not turn.tool_uses:
                        seed_answer_attempts += 1
                        if fields:
                            if not self._fields_match_expected(expected_fields, fields):
                                seed_wrong_field_attempts += 1
                        else:
                            seed_unstructured_answer_attempts += 1
                    if turn.input_behavior_signals.get("seed_evidence_sufficient"):
                        attempt_seed_evidence_sufficient = True
                        seed_evidence_sufficient = True
                if turn.validated_target_exact_reacquired:
                    attempt_validated_reacquisition = True
                    reacquisition_events_seen += 1
                    validated_target_reacquisition_events_seen += 1
                    validated_target_exact_reacquisition_events_seen += 1
                elif task.forbid_reacquisition and (turn.validated_target_reconfirmation_attempt):
                    attempt_validated_reacquisition = True
                if turn.validated_target_reconfirmation_attempt:
                    validated_target_reconfirmation_events_seen += 1
                if turn.answer_anchor_reacquisition_attempt:
                    answer_anchor_reacquisition_events_seen += 1
                if turn.answer_ready_reacquisition_attempt:
                    answer_ready_reacquisition_events_seen += 1
                if turn.repair_phase_reacquisition_attempt:
                    repair_phase_reacquisition_events_seen += 1
                if turn.benign_reverification_attempt:
                    benign_reverification_events_seen += 1
                if turn.output_behavior_signals.get("mixed_answer_tool_event"):
                    mixed_answer_tool_events_seen += 1
                if turn.output_behavior_signals.get("unsupported_tool_event"):
                    unsupported_tool_events_seen += 1
                if turn.output_behavior_signals.get("bad_tool_args_event"):
                    bad_tool_args_events_seen += 1
                if turn.output_behavior_signals.get("toolless_fresh_answer_event"):
                    toolless_fresh_answer_events_seen += 1
                if turn.tool_contract_failure and task.phase_name == "tool-contract":
                    tool_contract_failure_events_seen += 1
                if turn.output_behavior_signals.get("retention_latest_substitution"):
                    retention_substitution_events_seen += 1
                if anchors_before_baseline is None and self.session._baseline_only:
                    anchors_before_baseline = len(anchor_history)

                self._log(
                    f"[stress] turn {turn.turn_index} phase={turn.phase_name}/{turn.phase} "
                    f"retry={turn.retry_index} tools={len(turn.tool_uses)} "
                    f"turn_validated={turn.validated} breakpoints={len(breakpoints)} "
                    f"baseline_only={self.session._baseline_only}"
                )
                for breakpoint in new_breakpoints:
                    self._log(
                        f"[stress] breakpoint: {breakpoint.breakpoint_class} "
                        f"task={breakpoint.task_id} turn={breakpoint.turn_index} "
                        f"target={breakpoint.refactor_target}"
                    )

                response_blocks: list[dict[str, Any]] = []
                if turn.visible_response:
                    response_blocks.append({"type": "text", "text": turn.visible_response})
                for tool in turn.tool_uses:
                    response_blocks.append(tool)
                if response_blocks:
                    conversation.append({"role": "assistant", "content": response_blocks})
                else:
                    conversation.append({"role": "assistant", "content": turn.raw_response})
                if turn.tool_results:
                    conversation.extend(turn.tool_results)

                if should_stop_run(
                    breakpoint_count=len(seen_classes),
                    baseline_only=self.session._baseline_only,
                    tasks_completed=max(0, task_count - 1),
                    seen_classes=seen_classes,
                    config=self.config,
                ):
                    return self._finalize_result(
                        started_at=started_at,
                        task_count=task_count,
                        total_prompt_tokens=total_prompt_tokens,
                        total_completion_tokens=total_completion_tokens,
                        anchor_history=anchor_history,
                        tool_backed_turns=tool_backed_turns,
                        resend_modes_seen=resend_modes_seen,
                        total_evidence_chars=total_evidence_chars,
                        breakpoints=breakpoints,
                        turns=turns,
                        notes=notes,
                        seen_classes=seen_classes,
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
                        retention_substitution_events_seen=retention_substitution_events_seen,
                        compaction_eligible_turns=compaction_eligible_turns,
                        anchors_before_baseline=anchors_before_baseline,
                        seed_searches=seed_searches,
                        seed_direct_reads=seed_direct_reads,
                        seed_answer_attempts=seed_answer_attempts,
                        seed_evidence_sufficient=seed_evidence_sufficient,
                        seed_wrong_field_attempts=seed_wrong_field_attempts,
                        seed_unstructured_answer_attempts=seed_unstructured_answer_attempts,
                    )

                if has_tool_use:
                    if task.phase_name == "anchor-seed":
                        conversation.append(
                            {
                                "role": "user",
                                "content": self._seed_synthesis_prompt(
                                    expected_fields,
                                    evidence_sufficient=attempt_seed_evidence_sufficient,
                                ),
                            }
                        )
                    continue

                if task.phase_name == "late-recovery":
                    checkpoint_expected = expected_fields
                    if self._fields_match_expected(checkpoint_expected, fields):
                        task_finished = True
                        break
                    notes.append(f"late_recovery_miss:{task.id}")
                    task_finished = True
                    break

                validated = self._task_answer_validated(
                    task=task,
                    expected_fields=expected_fields,
                    observed_fields=fields,
                    attempt_tool_count=attempt_tool_count,
                    attempt_tool_names=attempt_tool_names,
                    validated_reacquisition=attempt_validated_reacquisition,
                )
                if validated:
                    turns[-1] = replace(turns[-1], task_completed_validated=True)
                    if expected_fields:
                        anchor = ValidatedAnchor(
                            task_id=task.id,
                            phase_name=task.phase_name,
                            file=fields.get("file", expected_fields.get("file", "")),
                            verification=fields.get(
                                "verification",
                                expected_fields.get("verification", ""),
                            ),
                            turn_index=turn.turn_index,
                            evidence_chars=attempt_evidence_chars,
                        )
                        anchor_history.append(anchor)
                        validated_targets.add(_fields_key(anchor.to_fields()))
                    if task.phase_name == "reuse-vs-reacquire":
                        reuse_checks_run += 1
                    if task.phase_name == "reuse-probe":
                        reuse_probe_successes += 1
                        notes.append(f"reuse_probe_memory_success:{task.id}")
                    if task.phase_name == "retention-probe":
                        retention_probe_successes += 1
                        if task.id != "retention_probe_early":
                            late_retention_probe_successes += 1
                    self._log(
                        f"[stress] task validated: {task.id} turn={turn.turn_index} task_completed_validated=True"
                    )
                    checkpoint_expected = {}
                    if len(anchor_history) >= 2 and checkpoint_checks_run == 0:
                        checkpoint_expected = anchor_history[0].to_fields()
                        conversation.append(
                            {
                                "role": "user",
                                "content": self._checkpoint_prompt(anchor_history),
                            }
                        )
                    task_finished = True
                    break

                grounded_miss_count += 1
                if retry_index >= self.config.max_retries_per_task:
                    notes.append(f"task_incomplete:{task.id}:retry_budget")
                    break

                current_turn_was_tool_only_retry = bool(
                    retry_index > 0
                    and len(turns) >= 2
                    and turns[-2].output_behavior_signals.get("late_retry_contract_stage_tool_only")
                )
                current_turn_satisfied_tool_only_stage = (
                    current_turn_was_tool_only_retry and self._turn_satisfies_tool_only_retry_stage(turn)
                )
                retry_prompt, retry_prompt_signals = self._retry_prompt(
                    task=task,
                    expected_fields=expected_fields,
                    observed_fields=fields,
                    attempt_tool_count=attempt_tool_count,
                    attempt_tool_names=attempt_tool_names,
                    validated_reacquisition=attempt_validated_reacquisition,
                    target_already_validated=_fields_key(expected_fields) in validated_targets,
                    payload_pressure_ready=self._payload_pressure_reached(total_evidence_chars, len(anchor_history)),
                    validated_target_exact_reacquired=turn.validated_target_exact_reacquired,
                    validated_target_reconfirmation_attempt=turn.validated_target_reconfirmation_attempt,
                    mixed_answer_tool_event=bool(turn.output_behavior_signals.get("mixed_answer_tool_event")),
                    toolless_fresh_answer_event=bool(turn.output_behavior_signals.get("toolless_fresh_answer_event")),
                    unsupported_tool_event=bool(turn.output_behavior_signals.get("unsupported_tool_event")),
                    bad_tool_args_event=bool(turn.output_behavior_signals.get("bad_tool_args_event")),
                    prior_turn_has_valid_supporting_tool_backing=self._turn_has_valid_supporting_tool_backing(turn),
                    current_turn_was_tool_only_retry=current_turn_was_tool_only_retry,
                    current_turn_satisfied_tool_only_stage=current_turn_satisfied_tool_only_stage,
                    retry_index=retry_index + 1,
                )
                turns[-1] = replace(
                    turns[-1],
                    output_behavior_signals={
                        **turns[-1].output_behavior_signals,
                        **retry_prompt_signals,
                    },
                )
                self.session._bump_signals(_runtime_retry_context_signals(retry_prompt_signals))
                conversation.append({"role": "user", "content": retry_prompt})
                retry_index += 1
                attempt_tool_count = 0
                attempt_tool_names = set()
                attempt_evidence_chars = 0
                attempt_validated_reacquisition = False
                attempt_seed_evidence_sufficient = False

            if not task_finished:
                notes.append(f"task_incomplete:{task.id}")
                self._log(f"[stress] task incomplete: {task.id}")
                continue

            if checkpoint_expected:
                global_turn += 1
                checkpoint_turn, new_breakpoints, _, _ = self._run_turn(
                    conversation=conversation,
                    system_prompt=system_prompt,
                    prompt=conversation[-1]["content"],
                    task=StressTask(
                        id=f"{task.id}_checkpoint",
                        phase_name="checkpoint",
                        prompt=conversation[-1]["content"],
                    ),
                    expected_fields=checkpoint_expected,
                    turn_index=global_turn,
                    retry_index=0,
                    attempt_tool_count_before_turn=0,
                    target_already_validated=True,
                    payload_pressure_ready=self._payload_pressure_reached(total_evidence_chars, len(anchor_history)),
                    grounded_miss_count=0,
                    attempt_seed_evidence_sufficient_before_turn=False,
                    latest_anchor_fields=(anchor_history[-1].to_fields() if anchor_history else {}),
                    validated_target_keys=set(validated_targets),
                    seen_classes=seen_classes,
                )
                turns.append(checkpoint_turn)
                breakpoints.extend(new_breakpoints)
                resend_modes_seen.add(checkpoint_turn.resend_mode)
                total_prompt_tokens += checkpoint_turn.usage.get("prompt_tokens", 0)
                total_completion_tokens += checkpoint_turn.usage.get("completion_tokens", 0)
                if checkpoint_turn.validated_target_exact_reacquired:
                    reacquisition_events_seen += 1
                    validated_target_reacquisition_events_seen += 1
                    validated_target_exact_reacquisition_events_seen += 1
                if checkpoint_turn.validated_target_reconfirmation_attempt:
                    validated_target_reconfirmation_events_seen += 1
                if checkpoint_turn.answer_anchor_reacquisition_attempt:
                    answer_anchor_reacquisition_events_seen += 1
                if checkpoint_turn.answer_ready_reacquisition_attempt:
                    answer_ready_reacquisition_events_seen += 1
                if checkpoint_turn.repair_phase_reacquisition_attempt:
                    repair_phase_reacquisition_events_seen += 1
                if checkpoint_turn.benign_reverification_attempt:
                    benign_reverification_events_seen += 1
                checkpoint_checks_run += 1
                if checkpoint_turn.validated:
                    turns[-1] = replace(turns[-1], task_completed_validated=True)
                if anchors_before_baseline is None and self.session._baseline_only:
                    anchors_before_baseline = len(anchor_history)
                self._log(
                    f"[stress] turn {checkpoint_turn.turn_index} phase={checkpoint_turn.phase_name}/{checkpoint_turn.phase} "
                    f"retry=0 tools={len(checkpoint_turn.tool_uses)} turn_validated={checkpoint_turn.validated} "
                    f"breakpoints={len(breakpoints)} baseline_only={self.session._baseline_only}"
                )
                response_blocks = []
                if checkpoint_turn.visible_response:
                    response_blocks.append(
                        {
                            "type": "text",
                            "text": checkpoint_turn.visible_response,
                        }
                    )
                for tool in checkpoint_turn.tool_uses:
                    response_blocks.append(tool)
                if response_blocks:
                    conversation.append({"role": "assistant", "content": response_blocks})
                else:
                    conversation.append(
                        {
                            "role": "assistant",
                            "content": checkpoint_turn.raw_response,
                        }
                    )
                if checkpoint_turn.tool_results:
                    conversation.extend(checkpoint_turn.tool_results)

                if should_stop_run(
                    breakpoint_count=len(seen_classes),
                    baseline_only=self.session._baseline_only,
                    tasks_completed=max(0, task_count - 1),
                    seen_classes=seen_classes,
                    config=self.config,
                ):
                    return self._finalize_result(
                        started_at=started_at,
                        task_count=task_count,
                        total_prompt_tokens=total_prompt_tokens,
                        total_completion_tokens=total_completion_tokens,
                        anchor_history=anchor_history,
                        tool_backed_turns=tool_backed_turns,
                        resend_modes_seen=resend_modes_seen,
                        total_evidence_chars=total_evidence_chars,
                        breakpoints=breakpoints,
                        turns=turns,
                        notes=notes,
                        seen_classes=seen_classes,
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
                        retention_substitution_events_seen=retention_substitution_events_seen,
                        compaction_eligible_turns=compaction_eligible_turns,
                        anchors_before_baseline=anchors_before_baseline,
                        seed_searches=seed_searches,
                        seed_direct_reads=seed_direct_reads,
                        seed_answer_attempts=seed_answer_attempts,
                        seed_evidence_sufficient=seed_evidence_sufficient,
                        seed_wrong_field_attempts=seed_wrong_field_attempts,
                        seed_unstructured_answer_attempts=seed_unstructured_answer_attempts,
                    )

        return self._finalize_result(
            started_at=started_at,
            task_count=task_count,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            anchor_history=anchor_history,
            tool_backed_turns=tool_backed_turns,
            resend_modes_seen=resend_modes_seen,
            total_evidence_chars=total_evidence_chars,
            breakpoints=breakpoints,
            turns=turns,
            notes=notes,
            seen_classes=seen_classes,
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
            retention_substitution_events_seen=retention_substitution_events_seen,
            compaction_eligible_turns=compaction_eligible_turns,
            anchors_before_baseline=anchors_before_baseline,
            seed_searches=seed_searches,
            seed_direct_reads=seed_direct_reads,
            seed_answer_attempts=seed_answer_attempts,
            seed_evidence_sufficient=seed_evidence_sufficient,
            seed_wrong_field_attempts=seed_wrong_field_attempts,
            seed_unstructured_answer_attempts=seed_unstructured_answer_attempts,
        )

    def _run_turn(
        self,
        *,
        conversation: list[dict[str, Any]],
        system_prompt: str,
        prompt: str,
        task: StressTask,
        expected_fields: dict[str, str],
        turn_index: int,
        retry_index: int,
        attempt_tool_count_before_turn: int,
        target_already_validated: bool,
        payload_pressure_ready: bool,
        grounded_miss_count: int,
        attempt_seed_evidence_sufficient_before_turn: bool,
        latest_anchor_fields: dict[str, str],
        validated_target_keys: set[str],
        seen_classes: set[str],
    ) -> tuple[StressTurnRecord, list[StressBreakpoint], bool, dict[str, str]]:
        self.session._bump_signals(
            _runtime_turn_context_signals(
                payload_pressure_ready=payload_pressure_ready,
            )
        )
        prepared = self.runtime.prepare_request(
            RuntimeRequest(
                model=self.config.model,
                messages=conversation,
                system=system_prompt,
                adapter_kind="stress-harness",
                tool_compatible=True,
            ),
            self.session,
        )
        outbound_messages = _system_to_messages(prepared.body.get("system")) + _normalize_chat_messages(
            prepared.body.get("messages", [])
        )
        started = time.time()
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=cast("Any", outbound_messages),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            extra_body=self.config.provider_options,
        )
        latency_ms = round((time.time() - started) * 1000, 2)
        raw_response = response.choices[0].message.content or ""
        runtime_contract_signals = _preprocess_runtime_contract_signals(
            task=task,
            raw_response=raw_response,
            attempt_tool_count_before_turn=attempt_tool_count_before_turn,
            payload_pressure_ready=payload_pressure_ready,
            request_behavior_signals=prepared.behavior_signals,
            session=self.session,
        )
        processed = self.runtime.process_response(
            raw_response,
            model=self.config.model,
            session=self.session,
            behavior_signals={
                **prepared.behavior_signals,
                **runtime_contract_signals,
            },
            tool_compatible=True,
        )
        content_blocks = list(processed.content_blocks)
        processed_behavior_signals = dict(processed.behavior_signals)
        direct_contract = response_contract_for_mode(raw_response, tool_compatible=True)
        processed_tool_uses = [
            block for block in content_blocks if isinstance(block, dict) and block.get("type") == "tool_use"
        ]
        direct_tool_uses = [
            block
            for block in direct_contract.content_blocks
            if isinstance(block, dict) and block.get("type") == "tool_use"
        ]
        if direct_tool_uses and not processed_tool_uses:
            content_blocks = list(direct_contract.content_blocks)
            processed_behavior_signals = dict(direct_contract.behavior_signals)
        visible_response = _render_visible_text(content_blocks) or raw_response
        tool_uses = [
            _sanitize_tool_use_block(block)
            for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "tool_use"
        ]
        repetition_signals = self._repetition_signals(
            conversation,
            tool_uses,
            target_already_validated=target_already_validated,
        )
        merged_input_signals = dict(prepared.behavior_signals)
        for key, value in repetition_signals.items():
            merged_input_signals[key] = merged_input_signals.get(key, 0) + value
        if target_already_validated and tool_uses:
            target_tool_use_signals = _classify_validated_target_tool_use(tool_uses, expected_fields)
            for key, value in target_tool_use_signals.items():
                merged_input_signals[key] = merged_input_signals.get(key, 0) + value
            if target_tool_use_signals.get("validated_target_reacquired"):
                merged_input_signals["reacquisition_cost_tokens"] = (
                    merged_input_signals.get("reacquisition_cost_tokens", 0) + 100
                )

        tool_results: list[dict[str, Any]] = []
        tool_contract_failure = False
        unsupported_tool_event = False
        bad_tool_args_event = False
        evidence_chars = 0
        seed_tool_summary = self._seed_tool_summary(
            task=task,
            expected_fields=expected_fields,
            tool_uses=tool_uses,
            attempt_tool_count_before_turn=attempt_tool_count_before_turn,
            attempt_seed_evidence_sufficient_before_turn=attempt_seed_evidence_sufficient_before_turn,
        )
        for block in tool_uses:
            result, blocked = self.tool_executor.execute(block)
            if blocked:
                tool_contract_failure = True
            content = str(result.get("content", ""))
            contract_signal = str(result.get("contract_signal", "")).strip()
            if contract_signal == "unsupported_tool":
                unsupported_tool_event = True
            if contract_signal == "bad_tool_args":
                bad_tool_args_event = True
            if result.get("is_error") or content.startswith("ERROR:"):
                tool_contract_failure = True
            if any(str(key).startswith("Tool use (") for key in block.get("input", {})):
                tool_contract_failure = True
                bad_tool_args_event = True
            evidence_chars += len(content)
            tool_results.append(result)

        fields = _extract_labeled_fields(visible_response, session=self.session)
        observed_match = self._fields_match_expected(expected_fields, fields)
        mixed_answer_tool_event = bool(tool_uses and fields)
        if processed_behavior_signals.get(
            "late_answer_followthrough_requested"
        ) and not _followthrough_evidence_sufficient(
            evidence_chars=evidence_chars,
            payload_pressure_ready=payload_pressure_ready,
            tool_results=tool_results,
        ):
            processed_behavior_signals["late_answer_followthrough_blocked_insufficient_evidence"] = 1
            processed_behavior_signals["late_answer_followthrough_requested"] = 0
            self.session._late_answer_followthrough_pending = False
            self.session._late_answer_assembly_repair_pending = True
            self.session._late_answer_assembly_repair_mode_pending = "tool_only"
        if task.phase_name == "reuse-probe" and not tool_uses and observed_match:
            merged_input_signals["reuse_probe_memory_success"] = 1
        repeated_oracle_miss = bool(
            task.phase_name != "checkpoint"
            and expected_fields
            and tool_uses
            and not observed_match
            and grounded_miss_count > 0
        )
        output_behavior_signals = dict(processed_behavior_signals)
        merged_input_signals.update(seed_tool_summary)
        if repeated_oracle_miss:
            output_behavior_signals["grounded_oracle_miss_streak"] = grounded_miss_count + 1
        if (
            task.phase_name == "retention-probe"
            and latest_anchor_fields
            and not observed_match
            and _fields_key(fields) in validated_target_keys
        ):
            output_behavior_signals["retention_latest_substitution"] = 1
        if seed_tool_summary["repeated_seed_search_without_read"]:
            tool_contract_failure = True
            output_behavior_signals["seed_navigation_pressure"] = 1
        if seed_tool_summary["repeated_seed_tool_after_evidence"]:
            output_behavior_signals["seed_answer_assembly_pressure"] = 1
        if mixed_answer_tool_event:
            output_behavior_signals["mixed_answer_tool_event"] = 1
            if task.phase_name == "tool-contract":
                tool_contract_failure = True
        if unsupported_tool_event:
            output_behavior_signals["unsupported_tool_event"] = 1
        if bad_tool_args_event:
            output_behavior_signals["bad_tool_args_event"] = 1
        resend_mode = self._resend_mode(prepared.behavior_signals)
        resend_decision_reason = self._resend_decision_reason(prepared.behavior_signals)
        transcript_slice = [_compact_message(message) for message in conversation[-6:]]

        if task.phase_name != "checkpoint":
            if (
                task.require_fresh_evidence
                and fields
                and attempt_tool_count_before_turn + len(tool_uses) < max(task.require_tool_count, 1)
            ):
                output_behavior_signals["toolless_fresh_answer_event"] = 1
            if (
                task.require_fresh_evidence
                and fields
                and attempt_tool_count_before_turn + len(tool_uses) < max(task.require_tool_count, 1)
            ):
                tool_contract_failure = True
            if target_already_validated and task.forbid_reacquisition and tool_uses:
                tool_contract_failure = True
            if task.phase_name == "anchor-seed" and seed_tool_summary["repeated_seed_tool_after_evidence"]:
                tool_contract_failure = True

        protocol_failure = (
            _is_protocol_failure(
                str(prompt),
                visible_response,
                fields,
                output_behavior_signals,
            )
            or repeated_oracle_miss
        )
        late_grace_kind = _late_tool_contract_grace_kind(
            task=task,
            retry_index=retry_index,
            payload_pressure_ready=payload_pressure_ready,
            input_signals=merged_input_signals,
            output_signals=output_behavior_signals,
            protocol_failure=protocol_failure,
            tool_contract_failure=tool_contract_failure,
        )
        suppress_fallback_increment = late_grace_kind is not None and retry_index == 0
        if suppress_fallback_increment:
            output_behavior_signals[f"late_tool_contract_{late_grace_kind}_grace"] = 1
        elif late_grace_kind:
            output_behavior_signals[f"late_tool_contract_{late_grace_kind}_retry_failure"] = 1
        fallback_incremented = self._update_failure_counter(
            protocol_failure=protocol_failure,
            tool_contract_failure=tool_contract_failure,
            suppress_failure_increment=suppress_fallback_increment,
        )
        if protocol_failure or tool_contract_failure:
            if suppress_fallback_increment:
                output_behavior_signals["fallback_pressure_suppressed"] = 1
            elif fallback_incremented:
                output_behavior_signals["fallback_pressure_incremented"] = 1
                fallback_cause = _fallback_pressure_cause(
                    input_signals=merged_input_signals,
                    output_signals=output_behavior_signals,
                )
                if fallback_cause:
                    output_behavior_signals[f"fallback_pressure_cause_{fallback_cause}"] = 1

        validated = (task.phase_name == "checkpoint" and observed_match) or (
            task.phase_name == "late-recovery" and observed_match
        )

        observation = StressObservation(
            task_id=task.id,
            turn_index=turn_index,
            prompt=str(prompt),
            phase=task.phase_name,
            visible_response=visible_response,
            active_tools=[str(block.get("name", "")) for block in tool_uses],
            input_behavior_signals=dict(merged_input_signals),
            output_behavior_signals=dict(output_behavior_signals),
            state_payload_chars=int(prepared.behavior_signals.get("state_payload_chars", 0)),
            resend_mode=resend_mode,
            transcript_slice=transcript_slice,
            expected_fields=expected_fields,
            observed_fields=fields,
            baseline_only=self.session._baseline_only,
            tool_contract_failure=tool_contract_failure,
            repeated_oracle_miss=repeated_oracle_miss,
            validated_target_reacquired=bool(merged_input_signals.get("validated_target_reacquired")),
            validated_target_exact_reacquired=bool(merged_input_signals.get("validated_target_exact_reacquired")),
            validated_target_reconfirmation_attempt=bool(
                merged_input_signals.get("validated_target_reconfirmation_attempt")
            ),
            target_already_validated=target_already_validated,
            payload_pressure_ready=payload_pressure_ready,
            seed_evidence_sufficient=bool(merged_input_signals.get("seed_evidence_sufficient")),
            repeated_seed_search_without_read=bool(merged_input_signals.get("repeated_seed_search_without_read")),
            repeated_seed_tool_after_evidence=bool(merged_input_signals.get("repeated_seed_tool_after_evidence")),
            retention_latest_substitution=bool(output_behavior_signals.get("retention_latest_substitution")),
        )
        breakpoints = classify_breakpoints(observation, seen_classes, task=task)
        usage = {
            "prompt_tokens": int(getattr(response.usage, "prompt_tokens", 0)),
            "completion_tokens": int(getattr(response.usage, "completion_tokens", 0)),
            "total_tokens": int(getattr(response.usage, "total_tokens", 0)),
        }
        turn = StressTurnRecord(
            task_id=task.id,
            phase_name=task.phase_name,
            turn_index=turn_index,
            phase=task.phase_name,
            prompt=str(prompt),
            raw_response=raw_response,
            visible_response=visible_response,
            tool_uses=[dict(block) for block in tool_uses],
            tool_results=[dict(result) for result in tool_results],
            evidence_chars=evidence_chars,
            retry_index=retry_index,
            validated=validated,
            input_behavior_signals=dict(merged_input_signals),
            output_behavior_signals=dict(output_behavior_signals),
            input_saved_tokens=prepared.input_saved_tokens,
            output_saved_tokens=processed.output_saved_tokens,
            tool_contract_failure=tool_contract_failure,
            state_payload_chars=int(prepared.behavior_signals.get("state_payload_chars", 0)),
            resend_mode=resend_mode,
            resend_decision_reason=resend_decision_reason,
            memory_loaded_chars=int(prepared.behavior_signals.get("state_payload_chars", 0)),
            tool_result_volume_chars=evidence_chars,
            tool_dense_session=bool(
                prepared.behavior_signals.get("tool_dense_session", 0)
                or getattr(self.session, "_current_tool_density", 0.0) >= TOOL_DENSITY_THRESHOLD
            ),
            answer_fact_projection_present=bool(prepared.behavior_signals.get("answer_anchor_present", 0)),
            payload_pressure_ready=payload_pressure_ready,
            compaction_eligible_ready=False,
            validated_target_reacquired=bool(merged_input_signals.get("validated_target_reacquired")),
            validated_target_exact_reacquired=bool(merged_input_signals.get("validated_target_exact_reacquired")),
            validated_target_reconfirmation_attempt=bool(
                merged_input_signals.get("validated_target_reconfirmation_attempt")
            ),
            answer_anchor_reacquisition_attempt=bool(
                prepared.behavior_signals.get("answer_anchor_reacquisition_attempt", 0)
            ),
            answer_ready_reacquisition_attempt=bool(
                prepared.behavior_signals.get("answer_ready_reacquisition_attempt", 0)
            ),
            repair_phase_reacquisition_attempt=bool(
                prepared.behavior_signals.get("repair_phase_reacquisition_attempt", 0)
            ),
            benign_reverification_attempt=bool(prepared.behavior_signals.get("benign_reverification_attempt", 0)),
            request_messages=len(outbound_messages),
            latency_ms=latency_ms,
            usage=usage,
            baseline_only=self.session._baseline_only,
        )
        return turn, breakpoints, bool(tool_uses), fields

    def _system_prompt(self) -> str:
        return (
            "You are a Tok language stress agent running on a long-lived compressed session.\n"
            "Use Tok-compatible plain text. Prefer `@Tool view_file`, `@Tool grep_search`, and `@Tool list_dir` only.\n"
            "CRITICAL: Adhere to a strict turn contract:\n"
            "1. TOOL TURN: Emit ONLY `@Tool` blocks (or `@msg` wrapped tools). Do NOT include any thoughts, greetings, or explanations.\n"
            "2. ANSWER TURN: Emit the `File=` / `Verification=` answer block. You MAY wrap this in `@msg` or use Tok v7 macros (`*A`) for filenames.\n"
            "Never use write, edit, run, or any mutating tool.\n"
            "If fresh evidence is requested, gather it before answering.\n"
            "When you answer, be concrete and use exactly the requested `File=` / `Verification=` shape.\n"
            "This is an adversarial durability loop: preserve anchors across time and repeated prompts."
        )

    def _log(self, _message: str) -> None:
        if self._progress:
            pass

    def _expected_fields_for_task(self, task: StressTask, anchor_history: list[ValidatedAnchor]) -> dict[str, str]:
        return expected_fields_for_task(task, anchor_history)

    def _task_prompt(self, task: StressTask, expected_fields: dict[str, str]) -> str:
        return task_prompt(task, expected_fields)

    def _task_ready(
        self,
        task: StressTask,
        anchor_history: list[ValidatedAnchor],
        total_evidence_chars: int,
        expected_fields: dict[str, str],
        *,
        reuse_checks_run: int,
        reuse_probe_attempts: int,
        checkpoint_checks_run: int,
    ) -> bool:
        return task_ready(
            task,
            anchor_history,
            total_evidence_chars,
            expected_fields,
            reuse_checks_run=reuse_checks_run,
            reuse_probe_attempts=reuse_probe_attempts,
            checkpoint_checks_run=checkpoint_checks_run,
            min_payload_pressure_bytes=self.config.min_payload_pressure_bytes,
        )

    def _task_answer_validated(
        self,
        *,
        task: StressTask,
        expected_fields: dict[str, str],
        observed_fields: dict[str, str],
        attempt_tool_count: int,
        attempt_tool_names: set[str],
        validated_reacquisition: bool,
    ) -> bool:
        return task_answer_validated(
            task=task,
            expected_fields=expected_fields,
            observed_fields=observed_fields,
            attempt_tool_count=attempt_tool_count,
            attempt_tool_names=attempt_tool_names,
            validated_reacquisition=validated_reacquisition,
        )

    def _fields_match_expected(self, expected_fields: dict[str, str], observed_fields: dict[str, str]) -> bool:
        return fields_match_expected(expected_fields, observed_fields)

    def _payload_pressure_reached(self, total_evidence_chars: int, validated_anchor_count: int) -> bool:
        return payload_pressure_reached(
            total_evidence_chars, validated_anchor_count, self.config.min_payload_pressure_bytes
        )

    def _seed_tool_summary(
        self,
        *,
        task: StressTask,
        expected_fields: dict[str, str],
        tool_uses: list[dict[str, Any]],
        attempt_tool_count_before_turn: int,
        attempt_seed_evidence_sufficient_before_turn: bool,
    ) -> dict[str, int]:
        return seed_tool_summary(
            task=task,
            expected_fields=expected_fields,
            tool_uses=tool_uses,
            attempt_tool_count_before_turn=attempt_tool_count_before_turn,
            attempt_seed_evidence_sufficient_before_turn=attempt_seed_evidence_sufficient_before_turn,
        )

    def _turn_has_valid_supporting_tool_backing(self, turn: StressTurnRecord) -> bool:
        return turn_has_valid_supporting_tool_backing(turn)

    def _retry_prompt(
        self,
        *,
        task: StressTask,
        expected_fields: dict[str, str],
        observed_fields: dict[str, str],
        attempt_tool_count: int,
        attempt_tool_names: set[str],
        validated_reacquisition: bool,
        target_already_validated: bool,
        payload_pressure_ready: bool,
        validated_target_exact_reacquired: bool,
        validated_target_reconfirmation_attempt: bool,
        mixed_answer_tool_event: bool,
        toolless_fresh_answer_event: bool,
        unsupported_tool_event: bool,
        bad_tool_args_event: bool,
        prior_turn_has_valid_supporting_tool_backing: bool,
        current_turn_was_tool_only_retry: bool,
        current_turn_satisfied_tool_only_stage: bool,
        retry_index: int,
    ) -> tuple[str, dict[str, int]]:
        return retry_prompt(
            task=task,
            expected_fields=expected_fields,
            observed_fields=observed_fields,
            attempt_tool_count=attempt_tool_count,
            attempt_tool_names=attempt_tool_names,
            validated_reacquisition=validated_reacquisition,
            target_already_validated=target_already_validated,
            payload_pressure_ready=payload_pressure_ready,
            validated_target_exact_reacquired=validated_target_exact_reacquired,
            validated_target_reconfirmation_attempt=validated_target_reconfirmation_attempt,
            mixed_answer_tool_event=mixed_answer_tool_event,
            toolless_fresh_answer_event=toolless_fresh_answer_event,
            unsupported_tool_event=unsupported_tool_event,
            bad_tool_args_event=bad_tool_args_event,
            prior_turn_has_valid_supporting_tool_backing=prior_turn_has_valid_supporting_tool_backing,
            current_turn_was_tool_only_retry=current_turn_was_tool_only_retry,
            current_turn_satisfied_tool_only_stage=current_turn_satisfied_tool_only_stage,
            retry_index=retry_index,
        )

    def _seed_synthesis_prompt(self, expected_fields: dict[str, str], *, evidence_sufficient: bool) -> str:
        return seed_synthesis_prompt(expected_fields, evidence_sufficient=evidence_sufficient)

    def _checkpoint_prompt(self, anchor_history: list[ValidatedAnchor]) -> str:
        return checkpoint_prompt(anchor_history)

    def _repetition_signals(
        self,
        conversation: list[dict[str, Any]],
        tool_uses: list[dict[str, Any]],
        *,
        target_already_validated: bool,
    ) -> dict[str, int]:
        return repetition_signals(
            conversation,
            tool_uses,
            target_already_validated=target_already_validated,
        )

    def _resend_mode(self, behavior_signals: dict[str, int]) -> str:
        return resend_mode(behavior_signals)

    def _resend_decision_reason(self, behavior_signals: dict[str, int]) -> str:
        return resend_decision_reason(behavior_signals)

    def _update_failure_counter(
        self,
        *,
        protocol_failure: bool,
        tool_contract_failure: bool,
        suppress_failure_increment: bool = False,
    ) -> bool:
        self._consecutive_failures, incremented, trigger_baseline = update_failure_counter(
            self._consecutive_failures,
            protocol_failure=protocol_failure,
            tool_contract_failure=tool_contract_failure,
            suppress_failure_increment=suppress_failure_increment,
            fallback_threshold=self.config.fallback_threshold,
        )
        if trigger_baseline:
            self.session._baseline_only = True
        return incremented

    def _tool_signature(self, block: dict[str, Any]) -> tuple[str, str]:
        return tool_signature(block)

    def _finalize_result(
        self,
        *,
        started_at: str,
        task_count: int,
        total_prompt_tokens: int,
        total_completion_tokens: int,
        anchor_history: list[ValidatedAnchor],
        tool_backed_turns: int,
        resend_modes_seen: set[str],
        total_evidence_chars: int,
        breakpoints: list[StressBreakpoint],
        turns: list[StressTurnRecord],
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
        return finalize_result(
            config=self.config,
            session=self.session,
            tasks=self.tasks,
            started_at=started_at,
            task_count=task_count,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            anchor_history=anchor_history,
            tool_backed_turns=tool_backed_turns,
            resend_modes_seen=resend_modes_seen,
            total_evidence_chars=total_evidence_chars,
            breakpoints=breakpoints,
            turns=turns,
            notes=notes,
            seen_classes=seen_classes,
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
            retention_substitution_events_seen=retention_substitution_events_seen,
            compaction_eligible_turns=compaction_eligible_turns,
            anchors_before_baseline=anchors_before_baseline,
            seed_searches=seed_searches,
            seed_direct_reads=seed_direct_reads,
            seed_answer_attempts=seed_answer_attempts,
            seed_evidence_sufficient=seed_evidence_sufficient,
            seed_wrong_field_attempts=seed_wrong_field_attempts,
            seed_unstructured_answer_attempts=seed_unstructured_answer_attempts,
        )

    def _failed_task_summaries(self, turns: list[StressTurnRecord]) -> list[dict[str, str]]:
        return failed_task_summaries(
            turns,
            failed_task_retry_family,
            lambda task_turns: first_irreversible_miss_kind(task_turns, turn_has_valid_supporting_tool_backing),
        )

    def _failed_task_retry_family(self, task_turns: list[StressTurnRecord]) -> str:
        return failed_task_retry_family(task_turns)

    def _dominant_failure_locus(
        self,
        *,
        failed_task_summaries: list[dict[str, str]],
        answer_anchor_reacquisition_events_seen: int,
        answer_ready_reacquisition_events_seen: int,
        repair_phase_reacquisition_events_seen: int,
        answer_ready_repair_failed_count: int,
        fallback_after_compaction_eligible: bool,
    ) -> str:
        return dominant_failure_locus(
            failed_task_summaries_list=failed_task_summaries,
            answer_anchor_reacquisition_events_seen=answer_anchor_reacquisition_events_seen,
            answer_ready_reacquisition_events_seen=answer_ready_reacquisition_events_seen,
            repair_phase_reacquisition_events_seen=repair_phase_reacquisition_events_seen,
            answer_ready_repair_failed_count=answer_ready_repair_failed_count,
            fallback_after_compaction_eligible=fallback_after_compaction_eligible,
        )

    def _first_anchor_failure_mode(
        self,
        *,
        seed_direct_reads: int,
        seed_evidence_sufficient: bool,
        seed_wrong_field_attempts: int,
        seed_unstructured_answer_attempts: int,
    ) -> str:
        return first_anchor_failure_mode(
            seed_direct_reads=seed_direct_reads,
            seed_evidence_sufficient=seed_evidence_sufficient,
            seed_wrong_field_attempts=seed_wrong_field_attempts,
            seed_unstructured_answer_attempts=seed_unstructured_answer_attempts,
        )

    def _first_irreversible_miss_kind(self, task_turns: list[StressTurnRecord]) -> str:
        return first_irreversible_miss_kind(task_turns, turn_has_valid_supporting_tool_backing)

    def _turn_satisfies_tool_only_retry_stage(self, turn: StressTurnRecord) -> bool:
        return turn_satisfies_tool_only_retry_stage(turn, self.session)

    def _turn_satisfies_answer_only_retry_stage(self, turn: StressTurnRecord) -> bool:
        return turn_satisfies_answer_only_retry_stage(turn, self.session)
