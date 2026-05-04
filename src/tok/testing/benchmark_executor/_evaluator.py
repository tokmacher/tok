from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from tok.testing.benchmark_suite import BenchmarkComparisonRun, BenchmarkTaskManifest

from ._models import (
    DEFAULT_EVALUATOR_BUNDLE_DIR,
    BenchmarkTaskRunResult,
    ShellCommandResult,
    TaskEvaluationResult,
    ToolExecutionRecord,
)
from ._utils import (
    _LOOP_ASYMMETRY_ABSOLUTE_THRESHOLD,
    _LOOP_ASYMMETRY_DELTA_THRESHOLD,
    _clean_answer_text,
    _count_note_prefix,
    _evidence_lines,
    _file_mentioned,
    _git,
    _normalize_pytest_output,
    _pytest_execution_command,
    _run_zsh_command,
    _sentence_count,
    _truncate,
)


class FamilyEvaluator:
    """Deterministic evaluators for each benchmark family."""

    def __init__(self, *, catalog_root: Path | None = None) -> None:
        base = (
            catalog_root.resolve() if catalog_root is not None else Path(__file__).resolve().parents[3] / "benchmarks"
        )
        self.catalog_root = base.resolve()
        self.evaluator_bundle_root = (self.catalog_root / DEFAULT_EVALUATOR_BUNDLE_DIR).resolve()

    def validate_execution_evaluators(self, tasks: tuple[BenchmarkTaskManifest, ...]) -> None:
        for task in tasks:
            if task.family != "execution_patch":
                continue
            self._load_hidden_evaluator_spec(task)

    def evaluate(
        self,
        materialized: Any,
        *,
        answer_text: str,
        clean_exit: bool,
        invalid_tool_calls: int,
        tool_calls: int,
        tool_records: list[ToolExecutionRecord],
        workspace_root: Path,
    ) -> TaskEvaluationResult:
        task = materialized.task
        if task.family == "execution_patch":
            return self._evaluate_execution_patch(
                task,
                workspace_root=workspace_root,
                clean_exit=clean_exit,
                tool_records=tool_records,
            )
        if task.family == "repo_grounding":
            return self._evaluate_repo_grounding(
                task,
                answer_text=answer_text,
                clean_exit=clean_exit,
                invalid_tool_calls=invalid_tool_calls,
                tool_calls=tool_calls,
                workspace_root=workspace_root,
            )
        return self._evaluate_real_session(
            task,
            answer_text=answer_text,
            clean_exit=clean_exit,
            invalid_tool_calls=invalid_tool_calls,
            tool_calls=tool_calls,
            workspace_root=workspace_root,
            tool_records=tool_records,
        )

    def compare_pair(
        self,
        *,
        task: BenchmarkTaskManifest,
        lane_id: str,
        repeat_index: int,
        baseline: BenchmarkTaskRunResult,
        candidate: BenchmarkTaskRunResult,
    ) -> BenchmarkComparisonRun:
        paired_result_stable = not any(note == "step_budget_exhausted" for note in (*baseline.notes, *candidate.notes))
        format_contract_violations = tuple(
            note
            for note in candidate.evaluation.notes
            if note in {"answer_contract_sentence_limit", "evidence_block_count", "invalid_citations"}
        )
        min_grounded_steps = int(task.success_evaluator.get("min_grounded_retrieval_steps", 0) or 0)
        baseline_details = baseline.evaluation.details if isinstance(baseline.evaluation.details, dict) else {}
        candidate_details = candidate.evaluation.details if isinstance(candidate.evaluation.details, dict) else {}
        baseline_command_invoked = bool(baseline_details.get("command_invoked", True))
        candidate_command_invoked = bool(candidate_details.get("command_invoked", True))
        baseline_suspicious_noop_pass = bool(baseline_details.get("suspicious_noop_pass", False))
        candidate_suspicious_noop_pass = bool(candidate_details.get("suspicious_noop_pass", False))
        baseline_execution_contract_met = bool(
            baseline_details.get("execution_contract_met", task.family != "execution_patch")
        )
        candidate_execution_contract_met = bool(
            candidate_details.get("execution_contract_met", task.family != "execution_patch")
        )
        baseline_loop_recovery_triggers = _count_note_prefix(baseline.notes, "read_only_loop_recovery_step_")
        candidate_loop_recovery_triggers = _count_note_prefix(candidate.notes, "read_only_loop_recovery_step_")
        loop_recovery_asymmetry_delta = abs(baseline_loop_recovery_triggers - candidate_loop_recovery_triggers)
        loop_recovery_asymmetry_material = candidate_loop_recovery_triggers > baseline_loop_recovery_triggers and (
            loop_recovery_asymmetry_delta >= _LOOP_ASYMMETRY_DELTA_THRESHOLD
            or candidate_loop_recovery_triggers >= _LOOP_ASYMMETRY_ABSOLUTE_THRESHOLD
        )
        baseline_adapter_contract_failure = baseline.local_failure == "adapter_payload_contract_error"
        candidate_adapter_contract_failure = candidate.local_failure == "adapter_payload_contract_error"
        baseline_premature_final_count = _count_note_prefix(baseline.notes, "premature_final_step_")
        candidate_premature_final_count = _count_note_prefix(candidate.notes, "premature_final_step_")
        baseline_tool_required_latch_active_count = self._count_turn_response_signal(
            baseline, "tool_required_latch_active"
        )
        candidate_tool_required_latch_active_count = self._count_turn_response_signal(
            candidate, "tool_required_latch_active"
        )
        integrity_artifact_flags: list[str] = []
        integrity_asymmetry_flags: list[str] = []

        def _append_once(target: list[str], value: str) -> None:
            if value not in target:
                target.append(value)

        if not baseline_command_invoked or not candidate_command_invoked:
            _append_once(integrity_artifact_flags, "command_not_executed_artifact")
        if baseline_command_invoked != candidate_command_invoked:
            _append_once(integrity_asymmetry_flags, "command_not_executed_asymmetry")
        if baseline_adapter_contract_failure or candidate_adapter_contract_failure:
            _append_once(integrity_artifact_flags, "adapter_payload_contract_error_artifact")
        if baseline_adapter_contract_failure != candidate_adapter_contract_failure:
            _append_once(integrity_asymmetry_flags, "adapter_payload_contract_error_asymmetry")
        if baseline_suspicious_noop_pass or candidate_suspicious_noop_pass:
            _append_once(integrity_artifact_flags, "suspicious_noop_pass_artifact")
        if baseline_suspicious_noop_pass != candidate_suspicious_noop_pass:
            _append_once(integrity_asymmetry_flags, "suspicious_noop_pass_asymmetry")
        if task.family == "execution_patch" and (
            not baseline_execution_contract_met or not candidate_execution_contract_met
        ):
            _append_once(integrity_artifact_flags, "execution_contract_not_met_artifact")
        if task.family == "execution_patch" and baseline_execution_contract_met != candidate_execution_contract_met:
            _append_once(integrity_asymmetry_flags, "execution_contract_asymmetry")
        if baseline_loop_recovery_triggers != candidate_loop_recovery_triggers:
            _append_once(integrity_asymmetry_flags, "read_only_loop_recovery_asymmetry")
            if loop_recovery_asymmetry_material:
                _append_once(integrity_asymmetry_flags, "read_only_loop_recovery_asymmetry_material")
            else:
                _append_once(integrity_asymmetry_flags, "read_only_loop_recovery_asymmetry_non_material")

        decision_grade_blockers: list[str] = []
        decision_grade_advisories: list[str] = []
        if (
            baseline.success
            and candidate.success
            and "read_only_loop_recovery_asymmetry" in integrity_asymmetry_flags
            and loop_recovery_asymmetry_material
        ):
            decision_grade_blockers.append("read_only_loop_recovery_asymmetry_matched_pair")
        elif (
            baseline.success and candidate.success and "read_only_loop_recovery_asymmetry" in integrity_asymmetry_flags
        ):
            decision_grade_advisories.append("read_only_loop_recovery_asymmetry_below_materiality")
        decision_grade = not integrity_artifact_flags and not decision_grade_blockers

        tool_engagement_stats = {
            "baseline_tool_calls": baseline.tool_calls,
            "tok_tool_calls": candidate.tool_calls,
            "grounded_retrieval_target": max(0, min_grounded_steps),
            "tok_missing_grounded_retrieval_target": bool(
                task.family == "repo_grounding" and min_grounded_steps > 0 and candidate.tool_calls < min_grounded_steps
            ),
            "baseline_command_invoked": baseline_command_invoked,
            "tok_command_invoked": candidate_command_invoked,
            "baseline_suspicious_noop_pass": baseline_suspicious_noop_pass,
            "tok_suspicious_noop_pass": candidate_suspicious_noop_pass,
            "baseline_execution_contract_met": baseline_execution_contract_met,
            "tok_execution_contract_met": candidate_execution_contract_met,
            "baseline_loop_recovery_triggers": baseline_loop_recovery_triggers,
            "tok_loop_recovery_triggers": candidate_loop_recovery_triggers,
            "loop_recovery_asymmetry_delta": loop_recovery_asymmetry_delta,
            "loop_recovery_asymmetry_material": loop_recovery_asymmetry_material,
            "loop_recovery_asymmetry_delta_threshold": _LOOP_ASYMMETRY_DELTA_THRESHOLD,
            "loop_recovery_asymmetry_absolute_threshold": _LOOP_ASYMMETRY_ABSOLUTE_THRESHOLD,
            "baseline_premature_final_count": baseline_premature_final_count,
            "tok_premature_final_count": candidate_premature_final_count,
            "baseline_tool_required_latch_active_count": baseline_tool_required_latch_active_count,
            "tok_tool_required_latch_active_count": candidate_tool_required_latch_active_count,
            "baseline_adapter_payload_contract_failure": baseline_adapter_contract_failure,
            "tok_adapter_payload_contract_failure": candidate_adapter_contract_failure,
            "integrity_artifact_flags": integrity_artifact_flags,
            "integrity_asymmetry_flags": integrity_asymmetry_flags,
            "decision_grade": decision_grade,
            "decision_grade_blockers": decision_grade_blockers,
            "decision_grade_advisories": decision_grade_advisories,
        }
        return BenchmarkComparisonRun(
            lane_id=lane_id,
            task_id=task.id,
            family=task.family,
            repeat_index=repeat_index,
            public_release=task.public_release,
            baseline_success=baseline.success,
            tok_success=candidate.success,
            quality_gate_passed=candidate.success and not integrity_artifact_flags,
            total_token_delta=int(candidate.provider_usage["total_tokens"])
            - int(baseline.provider_usage["total_tokens"]),
            latency_delta_ms=float(candidate.provider_usage["latency_ms"])
            - float(baseline.provider_usage["latency_ms"]),
            reacquisition_events=candidate.reacquisition_events,
            invalid_tool_calls=candidate.invalid_tool_calls,
            paired_result_stable=paired_result_stable,
            baseline_grounding_success=baseline.grounding_success,
            tok_grounding_success=candidate.grounding_success,
            baseline_tool_calls=baseline.tool_calls,
            tok_tool_calls=candidate.tool_calls,
            format_contract_violations=format_contract_violations,
            tool_engagement_stats=tool_engagement_stats,
            matched_completion_pair=baseline.success and candidate.success,
        )

    def _count_turn_response_signal(self, run: BenchmarkTaskRunResult, signal_name: str) -> int:
        count = 0
        for turn in run.turns:
            response_metrics = turn.get("response_metrics") if isinstance(turn, dict) else None
            response_signals = (
                response_metrics.get("response_behavior_signals") if isinstance(response_metrics, dict) else None
            )
            if not isinstance(response_signals, dict):
                continue
            raw_value = response_signals.get(signal_name, 0)
            try:
                count += int(raw_value)
            except (TypeError, ValueError):
                continue
        return count

    def _evaluate_execution_patch(
        self,
        task: BenchmarkTaskManifest,
        *,
        workspace_root: Path,
        clean_exit: bool,
        tool_records: list[ToolExecutionRecord],
    ) -> TaskEvaluationResult:
        hidden_spec = self._load_hidden_evaluator_spec(task)
        timeout_seconds = int(
            hidden_spec.get("timeout_seconds") or task.success_evaluator.get("hidden_tests_timeout_seconds", 60) or 60
        )
        hidden_command = str(hidden_spec.get("command") or "").strip()
        hidden_tests = tuple(hidden_spec.get("selectors") or hidden_spec.get("hidden_tests") or task.hidden_tests)
        if not hidden_command:
            hidden_command = self._pytest_command(hidden_tests)
        hidden_execution_command = _pytest_execution_command(hidden_command)
        hidden_result = self._run_shell(hidden_execution_command, cwd=workspace_root, timeout_seconds=timeout_seconds)
        modified_files = self._modified_files(workspace_root)
        allowed_paths_ok = all(path in task.allowed_paths for path in modified_files)
        command_invoked = hidden_result.command_invoked
        expect_initial_hidden_failure = bool(task.success_evaluator.get("expect_initial_hidden_failure", False))
        hidden_ok = command_invoked and hidden_result.returncode == 0
        suspicious_noop_pass = bool(hidden_ok and not modified_files)
        edit_file_calls = sum(1 for record in tool_records if record.canonical_tool_name == "edit_file")
        run_tests_calls = sum(1 for record in tool_records if record.canonical_tool_name == "run_tests")
        execution_contract_met = edit_file_calls > 0 and run_tests_calls > 0
        initial_hidden_failure_gate_ok = True
        if expect_initial_hidden_failure and suspicious_noop_pass:
            initial_hidden_failure_gate_ok = False
        success = (
            hidden_ok and allowed_paths_ok and initial_hidden_failure_gate_ok and execution_contract_met and clean_exit
        )
        notes: list[str] = []
        if not command_invoked:
            notes.append("hidden_tests_not_executed")
        if not hidden_ok:
            notes.append("hidden_tests_failed")
        if not allowed_paths_ok:
            notes.append("allowed_path_check_failed")
        if not initial_hidden_failure_gate_ok:
            notes.append("initial_hidden_failure_not_observed")
        if not execution_contract_met:
            notes.append("execution_contract_not_met")
        if not clean_exit:
            notes.append("clean_exit_required_for_success")
        return TaskEvaluationResult(
            success=success,
            grounding_success=success,
            details={
                "hidden_tests_command": hidden_command,
                "command_invoked": command_invoked,
                "command_argv": list(hidden_result.argv),
                "command_returncode": hidden_result.returncode,
                "command_execution_error": hidden_result.execution_error,
                "hidden_tests_returncode": hidden_result.returncode,
                "hidden_tests_output": _truncate(
                    _normalize_pytest_output((hidden_result.stdout or "") + (hidden_result.stderr or ""))
                ),
                "evaluator_spec": task.evaluator_spec_ref(),
                "modified_files": list(modified_files),
                "allowed_paths_ok": allowed_paths_ok,
                "expect_initial_hidden_failure": expect_initial_hidden_failure,
                "initial_hidden_failure_gate_ok": initial_hidden_failure_gate_ok,
                "suspicious_noop_pass": suspicious_noop_pass,
                "execution_contract_met": execution_contract_met,
                "execution_contract_edit_file_calls": edit_file_calls,
                "execution_contract_run_tests_calls": run_tests_calls,
                "clean_exit": clean_exit,
            },
            notes=tuple(notes),
        )

    def _evaluate_repo_grounding(
        self,
        task: BenchmarkTaskManifest,
        *,
        answer_text: str,
        clean_exit: bool,
        invalid_tool_calls: int,
        tool_calls: int,
        workspace_root: Path,
    ) -> TaskEvaluationResult:
        cleaned_answer = _clean_answer_text(answer_text)
        payload = self._load_gold_answer_payload(task, workspace_root=workspace_root)
        required_files = tuple(payload.get("required_files") or task.required_files)
        required_symbols = tuple(payload.get("required_symbols") or task.required_symbols)
        supporting_spans = tuple(payload.get("supporting_spans") or [dict(item) for item in task.supporting_spans])
        sentence_limit = 6
        sentence_ok = _sentence_count(cleaned_answer) <= sentence_limit
        evidence = _evidence_lines(cleaned_answer)
        evidence_count_ok = 2 <= len(evidence) <= 4
        citations_valid = self._citations_valid(evidence, supporting_spans)
        file_hit = any(_file_mentioned(path, cleaned_answer) for path in required_files)
        symbol_hit = any(symbol in cleaned_answer for symbol in required_symbols)
        min_steps = int(task.success_evaluator.get("min_grounded_retrieval_steps", 2) or 2)
        grounded_steps_ok = tool_calls >= min_steps
        completion_signals_ok = clean_exit and file_hit and symbol_hit and invalid_tool_calls == 0
        success = completion_signals_ok
        grounding_success = completion_signals_ok
        notes: list[str] = []
        if not sentence_ok:
            notes.append("answer_contract_sentence_limit")
        if not evidence_count_ok:
            notes.append("evidence_block_count")
        if not citations_valid:
            notes.append("invalid_citations")
        if not file_hit:
            notes.append("required_file_missing")
        if not symbol_hit:
            notes.append("required_symbol_missing")
        if not grounded_steps_ok:
            notes.append("grounded_retrieval_steps_missing")
        if invalid_tool_calls:
            notes.append("invalid_tool_calls_present")
        return TaskEvaluationResult(
            success=success,
            grounding_success=grounding_success,
            details={
                "completion_signals_ok": completion_signals_ok,
                "sentence_count": _sentence_count(cleaned_answer),
                "evidence_lines": evidence,
                "citations_valid": citations_valid,
                "required_file_hit": file_hit,
                "required_symbol_hit": symbol_hit,
                "tool_calls": tool_calls,
                "invalid_tool_calls": invalid_tool_calls,
                "clean_exit": clean_exit,
                "format_contract_violations": [
                    note
                    for note, ok in (
                        ("answer_contract_sentence_limit", sentence_ok),
                        ("evidence_block_count", evidence_count_ok),
                        ("invalid_citations", citations_valid),
                    )
                    if not ok
                ],
                "tool_engagement_stats": {
                    "tool_calls": tool_calls,
                    "min_grounded_retrieval_steps": min_steps,
                    "grounded_retrieval_steps_met": grounded_steps_ok,
                },
            },
            notes=tuple(notes),
        )

    def _evaluate_real_session(
        self,
        task: BenchmarkTaskManifest,
        *,
        answer_text: str,
        clean_exit: bool,
        invalid_tool_calls: int,
        tool_calls: int,
        workspace_root: Path,
        tool_records: list[ToolExecutionRecord],
    ) -> TaskEvaluationResult:
        milestone_type = str(task.success_evaluator.get("milestone_type") or task.episode_type or "").strip()
        invalid_limit = int(task.success_evaluator.get("invalid_tool_calls_limit", 0) or 0)
        if milestone_type in {"grounded_answer", "bug_diagnosis"}:
            grounding = self._evaluate_repo_grounding(
                task,
                answer_text=answer_text,
                clean_exit=clean_exit,
                invalid_tool_calls=invalid_tool_calls,
                tool_calls=tool_calls,
                workspace_root=workspace_root,
            )
            success = grounding.success and invalid_tool_calls <= invalid_limit
            details = dict(grounding.details)
            details["milestone_type"] = milestone_type
            notes = list(grounding.notes)
            if invalid_tool_calls > invalid_limit:
                notes.append("invalid_tool_call_limit_exceeded")
            return TaskEvaluationResult(
                success=success, grounding_success=grounding.grounding_success, details=details, notes=tuple(notes)
            )

        execution = self._evaluate_execution_patch(
            task,
            workspace_root=workspace_root,
            clean_exit=clean_exit,
            tool_records=tool_records,
        )
        success = execution.success and invalid_tool_calls <= invalid_limit
        details = dict(execution.details)
        details["milestone_type"] = milestone_type or "patch_completion"
        details["tool_records"] = [record.to_dict() for record in tool_records]
        notes = list(execution.notes)
        if invalid_tool_calls > invalid_limit:
            notes.append("invalid_tool_call_limit_exceeded")
        return TaskEvaluationResult(success=success, grounding_success=success, details=details, notes=tuple(notes))

    def _pytest_command(self, tests: tuple[str, ...]) -> str:
        if not tests:
            msg = "execution_patch evaluator requires selectors or an explicit command"
            raise RuntimeError(msg)
        return "python -m pytest -q " + " ".join(tests)

    def _load_hidden_evaluator_spec(self, task: BenchmarkTaskManifest) -> dict[str, Any]:
        evaluator_spec = task.evaluator_spec_ref()
        if evaluator_spec:
            raw = Path(evaluator_spec)
            candidate = raw if raw.is_absolute() else (self.catalog_root / raw)
            candidate = candidate.resolve()
            if not candidate.is_file():
                msg = f"task {task.id} references evaluator spec '{evaluator_spec}', but no file exists at {candidate}"
                raise RuntimeError(msg)
            payload: dict[str, Any] = json.loads(candidate.read_text())
            payload["source"] = str(candidate)
            return payload
        if task.hidden_tests:
            return {"selectors": list(task.hidden_tests)}
        return {}

    def _load_gold_answer_payload(
        self,
        task: BenchmarkTaskManifest,
        *,
        workspace_root: Path | None,
    ) -> dict[str, Any]:
        gold_answer_path = str(task.family_payload.get("gold_answer_path") or "").strip()
        if not gold_answer_path:
            return dict(task.effective_family_payload())
        candidate_bases: list[Path] = []
        if workspace_root is not None:
            candidate_bases.append(workspace_root)
            candidate_bases.append(workspace_root.parent)
        candidate_bases.extend(
            [
                Path(task.asset_dir),
                Path.cwd() / task.asset_dir,
                Path(__file__).resolve().parents[3] / "benchmarks" / task.asset_dir,
            ]
        )
        for base in candidate_bases:
            candidate = base / gold_answer_path
            if candidate.exists():
                result = json.loads(candidate.read_text())
                if isinstance(result, dict):
                    return result
                msg = f"gold answer at {candidate} is not a dict"
                raise RuntimeError(msg)
        return dict(task.effective_family_payload())

    def _run_shell(self, command: str, *, cwd: Path, timeout_seconds: int) -> ShellCommandResult:
        return _run_zsh_command(command, cwd=cwd, timeout_seconds=timeout_seconds)

    def _modified_files(self, workspace_root: Path) -> tuple[str, ...]:
        completed = _git(["status", "--porcelain", "--untracked-files=no"], cwd=workspace_root)
        if completed.returncode != 0:
            return ()
        paths: list[str] = []
        for raw_line in completed.stdout.splitlines():
            if not raw_line.strip():
                continue
            path_field = raw_line[3:].strip() if len(raw_line) >= 4 else ""
            if " -> " in path_field:
                path_field = path_field.split(" -> ", 1)[1].strip()
            if path_field:
                paths.append(path_field)
        return tuple(sorted(paths))

    def _citations_valid(self, citations: list[str], supporting_spans: tuple[dict[str, Any], ...]) -> bool:
        if not citations:
            return False
        line_ref_pattern = re.compile(r"(?:\bline\s+\d+\b|:\d+\b|#L\d+\b)", re.IGNORECASE)
        for citation in citations:
            if not any(
                (str(span.get("file", "")) in citation or Path(str(span.get("file", ""))).name in citation)
                and (
                    str(span.get("anchor", "")) in citation
                    or any(
                        token and token in citation
                        for token in re.findall(
                            r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\b",
                            str(span.get("anchor", "")),
                        )
                    )
                    or bool(line_ref_pattern.search(citation))
                )
                for span in supporting_spans
            ):
                return False
        return True
