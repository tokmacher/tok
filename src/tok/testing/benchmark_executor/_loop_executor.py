from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tok.gateway import BridgeSession
from tok.runtime.core import RuntimeSession
from tok.testing.benchmark_suite import BenchmarkTaskManifest
from tok.testing.live_benchmark import LiveBenchmarkRunner

from ._evaluator import FamilyEvaluator
from ._models import (
    BenchmarkTaskRunResult,
    MaterializedBenchmarkTask,
    ToolExecutionRecord,
)
from ._tool_executor import BenchmarkToolExecutor
from ._utils import (
    _EXECUTION_PATCH_REQUIRED_TOOLS,
    _READ_ONLY_LOOP_REPEAT_THRESHOLD,
    _READ_ONLY_LOOP_TOOLS,
    _REDUNDANT_RUN_TESTS_SUPPRESSION_THRESHOLD,
    _TOK_CONTROLLED_CONDITION,
    _TOK_RUNTIME_CONDITIONS,
    _assistant_message_content,
    _count_reacquisition_events,
    _extract_text_tool_calls,
    _normalize_pytest_command,
    _truncate,
)


class ToolLoopExecutor:
    """Run a live benchmark task with the shared conversation transport."""

    def __init__(self, runner: LiveBenchmarkRunner, *, catalog_root: Path) -> None:
        self.runner = runner
        self.evaluator = FamilyEvaluator(catalog_root=catalog_root)

    def run_task(
        self,
        materialized: MaterializedBenchmarkTask,
        *,
        output_root: Path,
    ) -> BenchmarkTaskRunResult:
        task = materialized.task
        workspace_root = Path(materialized.workspace_root)
        timeout_seconds = int(
            task.success_evaluator.get("hidden_tests_timeout_seconds", task.time_budget_minutes * 60) or 120
        )
        tool_executor = BenchmarkToolExecutor(
            workspace_root,
            allowed_tools=task.allowed_tools,
            timeout_seconds=max(30, timeout_seconds),
        )
        condition = materialized.condition
        bridge_session: BridgeSession | None = None
        if condition in _TOK_RUNTIME_CONDITIONS:
            bridge_kwargs: dict[str, Any] = {"memory_dir": workspace_root / ".tok_benchmark_memory"}
            if condition == _TOK_CONTROLLED_CONDITION:
                bridge_kwargs["request_policy_default"] = "forced_baseline"
            bridge_session = BridgeSession(**bridge_kwargs)
        session = bridge_session.runtime_session if bridge_session is not None else RuntimeSession()
        session.model = self.runner.model
        runtime_mode = "tok-universal" if condition in _TOK_RUNTIME_CONDITIONS else condition
        conversation: list[dict[str, Any]] = [{"role": "user", "content": task.prompt}]
        turns: list[dict[str, Any]] = []
        tool_records: list[ToolExecutionRecord] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_latency_ms = 0.0
        invalid_tool_calls = 0
        tool_calls = 0
        reacquisition_events = 0
        saw_edit_file = False
        saw_run_tests = False
        read_only_loop_signature: str | None = None
        read_only_loop_repeats = 0
        loop_recovery_trigger_count = 0
        search_miss_recovery_count = 0
        post_test_finalize_nudge_sent = False
        redundant_run_tests_suppressed_count = 0
        saw_successful_edit = False
        saw_successful_run_tests_after_edit = False
        successful_run_tests_since_last_edit: dict[str, int] = {}
        answer_text = ""
        raw_response = ""
        clean_exit = False
        notes: list[str] = []

        for step_index in range(1, task.step_budget + 1):
            step = self._run_conversation_step_with_retry(
                conversation=conversation,
                system_prompt=self._system_prompt(task, condition),
                mode=runtime_mode,
                session=session,
                bridge_session=bridge_session,
                allowed_tools=task.allowed_tools,
                notes=notes,
                step_index=step_index,
            )
            total_prompt_tokens += int(step.provider_usage.prompt_tokens)
            total_completion_tokens += int(step.provider_usage.completion_tokens)
            total_latency_ms += float(step.provider_usage.latency_ms)
            reacquisition_events += _count_reacquisition_events(step.response_metrics["response_behavior_signals"])
            tool_blocks = [dict(block) for block in step.content_blocks if block.get("type") == "tool_use"]
            extracted_from_text = False

            if not tool_blocks and step.raw_response:
                text_tool_blocks = _extract_text_tool_calls(step.raw_response, task.allowed_tools)
                if text_tool_blocks:
                    tool_blocks.extend(text_tool_blocks)
                    extracted_from_text = True
                    notes.append(f"text_tool_extraction_step_{step_index}")

            if extracted_from_text:
                assistant_content: str | list[dict[str, Any]] = copy.deepcopy(tool_blocks)
            else:
                assistant_content = _assistant_message_content(
                    step.content_blocks, step.visible_response or step.raw_response
                )
            conversation.append({"role": "assistant", "content": assistant_content})

            turns.append(
                {
                    "step": step_index,
                    "provider_usage": asdict(step.provider_usage),
                    "visible_response": step.visible_response,
                    "raw_response": step.raw_response,
                    "response_metrics": dict(step.response_metrics),
                    "compression_metrics": dict(step.compression_metrics),
                    "tool_use_count": len(tool_blocks),
                }
            )
            raw_response = step.raw_response
            if not tool_blocks:
                proposed_answer = step.visible_response or step.raw_response
                if task.family == "execution_patch" and saw_successful_edit and not saw_successful_run_tests_after_edit:
                    notes.append(f"premature_final_step_{step_index}")
                    notes.append(f"premature_final_after_failed_tests_step_{step_index}")
                    conversation.append(
                        {
                            "role": "user",
                            "content": self._tooling_recovery_prompt(
                                task=task,
                                tool_calls=tool_calls,
                                missing_requirements=self._missing_execution_patch_requirements(
                                    saw_edit_file=saw_edit_file,
                                    saw_run_tests=saw_run_tests,
                                ),
                                reason="failed_tests_after_edit",
                            ),
                        }
                    )
                    continue
                missing_requirements = self._missing_execution_patch_requirements(
                    saw_edit_file=saw_edit_file,
                    saw_run_tests=saw_run_tests,
                )
                if self._requires_more_tooling(
                    task,
                    tool_calls=tool_calls,
                    missing_requirements=missing_requirements,
                ):
                    notes.append(f"premature_final_step_{step_index}")
                    conversation.append(
                        {
                            "role": "user",
                            "content": self._tooling_recovery_prompt(
                                task=task,
                                tool_calls=tool_calls,
                                missing_requirements=missing_requirements,
                                reason="premature_final",
                            ),
                        }
                    )
                    continue
                answer_text = proposed_answer
                clean_exit = True
                break

            step_records: list[ToolExecutionRecord] = []
            step_has_successful_run_tests = False
            step_edit_file_search_miss_paths: list[str] = []
            for block in tool_blocks:
                canonical_name = tool_executor._canonical_tool_name(str(block.get("name", "")).strip())
                tool_input = block.get("input", {})
                if not isinstance(tool_input, dict):
                    tool_input = {}

                if (
                    task.family == "execution_patch"
                    and canonical_name == "run_tests"
                    and saw_successful_edit
                    and saw_successful_run_tests_after_edit
                ):
                    raw_command = self._run_tests_command_from_input(tool_input)
                    normalized_command = _normalize_pytest_command(raw_command)
                    if normalized_command is not None:
                        run_count = successful_run_tests_since_last_edit.get(normalized_command, 0)
                        if run_count >= _REDUNDANT_RUN_TESTS_SUPPRESSION_THRESHOLD:
                            suppressed_content = self._redundant_run_tests_suppressed_prompt(normalized_command)
                            tool_message = tool_executor._tool_result_message(
                                block,
                                suppressed_content,
                                signal="redundant_run_tests_suppressed",
                            )
                            record = ToolExecutionRecord(
                                step_index=step_index,
                                tool_name=str(block.get("name", "")).strip(),
                                canonical_tool_name="run_tests",
                                tool_input=dict(tool_input),
                                invalid=False,
                                is_error=False,
                                content_preview=_truncate(suppressed_content),
                            )
                            tool_calls += 1
                            tool_records.append(record)
                            step_records.append(record)
                            conversation.append(tool_message)
                            redundant_run_tests_suppressed_count += 1
                            notes.append(f"redundant_run_tests_suppressed_step_{step_index}")
                            continue

                tool_message, invalid, record = tool_executor.execute_tool(block, step_index=step_index)
                tool_calls += 1
                invalid_tool_calls += int(invalid)
                tool_records.append(record)
                step_records.append(record)
                conversation.append(tool_message)
                if record.canonical_tool_name == "edit_file":
                    saw_edit_file = True
                    if not record.is_error:
                        saw_successful_edit = True
                        saw_successful_run_tests_after_edit = False
                        successful_run_tests_since_last_edit.clear()
                    if self._is_edit_file_search_miss(record):
                        raw_path = (
                            record.tool_input.get("path")
                            or record.tool_input.get("file_path")
                            or record.tool_input.get("target")
                        )
                        if isinstance(raw_path, str) and raw_path.strip():
                            step_edit_file_search_miss_paths.append(raw_path.strip())
                elif record.canonical_tool_name == "run_tests":
                    saw_run_tests = True
                    if not record.is_error:
                        step_has_successful_run_tests = True
                        if saw_successful_edit:
                            saw_successful_run_tests_after_edit = True
                        normalized_command = _normalize_pytest_command(
                            self._run_tests_command_from_input(record.tool_input)
                        )
                        if normalized_command is not None:
                            successful_run_tests_since_last_edit[normalized_command] = (
                                successful_run_tests_since_last_edit.get(normalized_command, 0) + 1
                            )

            if task.family == "execution_patch" and step_edit_file_search_miss_paths:
                search_miss_recovery_count += len(step_edit_file_search_miss_paths)
                notes.append(f"edit_file_search_miss_recovery_step_{step_index}")
                conversation.append(
                    {
                        "role": "user",
                        "content": self._edit_search_miss_recovery_prompt(
                            file_paths=tuple(step_edit_file_search_miss_paths),
                            miss_count=search_miss_recovery_count,
                        ),
                    }
                )

            if (
                task.family == "execution_patch"
                and saw_edit_file
                and saw_successful_run_tests_after_edit
                and step_has_successful_run_tests
                and not post_test_finalize_nudge_sent
            ):
                notes.append(f"post_test_finalize_nudge_step_{step_index}")
                conversation.append(
                    {
                        "role": "user",
                        "content": self._post_test_finalize_prompt(),
                    }
                )
                post_test_finalize_nudge_sent = True

            execution_contract_met = saw_edit_file and saw_run_tests
            if task.family != "execution_patch" or execution_contract_met:
                read_only_loop_signature = None
                read_only_loop_repeats = 0
                continue

            step_signature = self._read_only_step_signature(step_records)
            if not step_signature:
                read_only_loop_signature = None
                read_only_loop_repeats = 0
                continue

            if step_signature == read_only_loop_signature:
                read_only_loop_repeats += 1
            else:
                read_only_loop_signature = step_signature
                read_only_loop_repeats = 1

            if read_only_loop_repeats >= _READ_ONLY_LOOP_REPEAT_THRESHOLD:
                loop_recovery_trigger_count += 1
                notes.append(f"read_only_loop_recovery_step_{step_index}")
                conversation.append(
                    {
                        "role": "user",
                        "content": self._tooling_recovery_prompt(
                            task=task,
                            tool_calls=tool_calls,
                            missing_requirements=self._missing_execution_patch_requirements(
                                saw_edit_file=saw_edit_file,
                                saw_run_tests=saw_run_tests,
                            ),
                            reason="read_only_loop",
                        ),
                    }
                )
                read_only_loop_signature = None
                read_only_loop_repeats = 0
        else:
            notes.append("step_budget_exhausted")

        if task.family == "execution_patch":
            if saw_edit_file:
                notes.append("execution_contract_edit_file_seen")
            else:
                notes.append("execution_contract_missing_edit_file")
            if saw_run_tests:
                notes.append("execution_contract_run_tests_seen")
            else:
                notes.append("execution_contract_missing_run_tests")
            notes.append("execution_contract_met" if (saw_edit_file and saw_run_tests) else "execution_contract_unmet")
            notes.append(f"read_only_loop_recovery_count_{loop_recovery_trigger_count}")
            notes.append(f"edit_file_search_miss_recovery_count_{search_miss_recovery_count}")
            notes.append(f"redundant_run_tests_suppressed_count_{redundant_run_tests_suppressed_count}")

        evaluation = self.evaluator.evaluate(
            materialized,
            answer_text=answer_text,
            clean_exit=clean_exit,
            invalid_tool_calls=invalid_tool_calls,
            tool_calls=tool_calls,
            tool_records=tool_records,
            workspace_root=workspace_root,
        )
        provider_usage = {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "latency_ms": round(total_latency_ms, 2),
        }
        result = BenchmarkTaskRunResult(
            lane_id=materialized.lane.id,
            condition=condition,
            task_id=task.id,
            family=task.family,
            repeat_index=materialized.repeat_index,
            workspace_root=str(workspace_root),
            answer_text=answer_text,
            raw_response=raw_response,
            provider_usage=provider_usage,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
            reacquisition_events=reacquisition_events,
            clean_exit=clean_exit,
            modified_files=tool_executor.modified_files(),
            tool_records=tuple(tool_records),
            turns=tuple(turns),
            evaluation=evaluation,
            notes=tuple(notes),
        )
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "run.json").write_text(json.dumps(result.to_dict(), indent=2))
        return result

    def _is_transient_connection_error(self, error: Exception) -> bool:
        message = str(error).lower()
        return (
            "connection error" in message
            or "timed out" in message
            or "timeout" in message
            or "temporarily unavailable" in message
            or "rate limit" in message
            or "429" in message
            or "502" in message
            or "503" in message
            or "504" in message
        )

    def _run_conversation_step_with_retry(
        self,
        *,
        conversation: list[dict[str, Any]],
        system_prompt: str,
        mode: str,
        session: RuntimeSession,
        bridge_session: BridgeSession | None,
        allowed_tools: tuple[str, ...],
        notes: list[str],
        step_index: int,
    ) -> Any:
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                return self.runner.run_conversation_step(
                    conversation=conversation,
                    system_prompt=system_prompt,
                    mode=mode,
                    session=session,
                    bridge_session=bridge_session,
                    allowed_tools=allowed_tools,
                )
            except Exception as exc:
                if not self._is_transient_connection_error(exc) or attempt >= max_attempts:
                    raise
                notes.append(f"transient_connection_retry_step_{step_index}_attempt_{attempt}")
                time.sleep(0.5 * attempt)
        msg = "retry loop exhausted without raising final exception"
        raise RuntimeError(msg)

    def _requires_more_tooling(
        self,
        task: BenchmarkTaskManifest,
        *,
        tool_calls: int,
        missing_requirements: tuple[str, ...],
    ) -> bool:
        if task.family == "repo_grounding":
            return False
        if task.family == "execution_patch":
            return bool(missing_requirements) or tool_calls < 1
        return False

    def _missing_execution_patch_requirements(
        self,
        *,
        saw_edit_file: bool,
        saw_run_tests: bool,
    ) -> tuple[str, ...]:
        missing: list[str] = []
        for tool_name in _EXECUTION_PATCH_REQUIRED_TOOLS:
            if tool_name == "edit_file" and not saw_edit_file:
                missing.append(tool_name)
            if tool_name == "run_tests" and not saw_run_tests:
                missing.append(tool_name)
        return tuple(missing)

    def _read_only_step_signature(self, records: list[ToolExecutionRecord]) -> str:
        if not records:
            return ""
        parts: list[str] = []
        for record in records:
            if record.canonical_tool_name not in _READ_ONLY_LOOP_TOOLS:
                return ""
            normalized_input = json.dumps(record.tool_input, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            parts.append(f"{record.canonical_tool_name}:{normalized_input}")
        return "|".join(parts)

    def _is_edit_file_search_miss(self, record: ToolExecutionRecord) -> bool:
        if record.canonical_tool_name != "edit_file":
            return False
        if not record.invalid:
            return False
        preview = record.content_preview.lower()
        return "search string not found" in preview

    def _run_tests_command_from_input(self, tool_input: dict[str, Any]) -> str:
        return str(tool_input.get("command") or tool_input.get("cmd") or tool_input.get("text") or "").strip()

    def _edit_search_miss_recovery_prompt(self, *, file_paths: tuple[str, ...], miss_count: int) -> str:
        from ._utils import _EDIT_SEARCH_MISS_ESCALATION_THRESHOLD

        escalation = (
            " You have hit this mismatch multiple times; copy the exact current lines from view_file before editing again."
            if miss_count >= _EDIT_SEARCH_MISS_ESCALATION_THRESHOLD
            else ""
        )
        if file_paths:
            files = ", ".join(f"`{path}`" for path in sorted(set(file_paths)))
            return (
                "Your edit_file search text did not match the current file content. "
                f"Inspect {files} with view_file, then retry edit_file with an exact current snippet "
                "or provide full replacement content." + escalation
            )
        return (
            "Your edit_file search text did not match the current file content. "
            "Use view_file to inspect the latest file, then retry edit_file with an exact snippet "
            "or full replacement content." + escalation
        )

    def _redundant_run_tests_suppressed_prompt(self, command: str) -> str:
        return (
            "run_tests suppressed: this command already succeeded without a new edit. "
            f"Command: `{command}`. Finalize now, or make another edit before re-running tests."
        )

    def _post_test_finalize_prompt(self) -> str:
        return (
            "You have applied an edit and completed run_tests successfully. "
            "If the fix is complete, provide the final answer now instead of continuing extra tool calls."
        )

    def _tooling_recovery_prompt(
        self,
        *,
        task: BenchmarkTaskManifest,
        tool_calls: int,
        missing_requirements: tuple[str, ...] = (),
        reason: str = "premature_final",
    ) -> str:
        if task.family == "repo_grounding":
            return (
                "Continue investigating with the allowed tools as needed before finalizing. "
                f"You currently have {tool_calls} tool calls. Prefer grounded citations in your final answer when possible."
            )
        if task.family == "execution_patch":
            requirement_hint = ""
            if missing_requirements:
                requirement_hint = (
                    " Required before finalizing: " + ", ".join(f"`{item}`" for item in missing_requirements) + "."
                )
            if reason == "failed_tests_after_edit":
                return (
                    "Your latest patch has not produced a passing run_tests result yet. "
                    "Use view_file/edit_file to fix the failure, then run_tests again before finalizing."
                    + requirement_hint
                )
            if reason == "read_only_loop":
                return (
                    "You are repeating read-only inspection without patch progress. "
                    "Pick one concrete allowed file, apply the fix with edit_file, then run_tests to verify."
                    + requirement_hint
                )
            return (
                "Continue using the allowed tools before finalizing. "
                "Run the necessary read/edit/test steps to verify the fix, then provide the final answer."
                + requirement_hint
            )
        return "Continue using the allowed tools and then provide the final answer."

    def _system_prompt(self, task: BenchmarkTaskManifest, condition: str = "baseline") -> str:
        tool_list = ", ".join(f"`{tool}`" for tool in task.allowed_tools)
        tool_params = (
            "Tool parameters: "
            "view_file(path, start?, end?) — start/end are 1-based line numbers for reading a range. "
            "grep_search(pattern, path?) — search file contents. "
            "list_dir(path) — list directory contents. "
            "edit_file(path, old_string, new_string) — replace text in a file. "
            "run_tests(command) — run a pytest command. "
            "bash(command) — run a shell command. "
        )
        del condition
        tool_instruction = "When you need a tool, call it using the function calling interface. " + tool_params
        shared = (
            "You are running a controlled benchmark task in a local repository workspace. "
            f"Use only these tools: {tool_list}. "
            + tool_instruction
            + "Do not invent file contents, tests, or evidence. "
            "If a tool errors, recover using the allowed tools. "
        )
        if task.family == "execution_patch":
            return (
                shared
                + "Fix the bug, keep changes inside the allowed files, and only finish after you have run the most relevant tests."
            )
        if task.family == "repo_grounding":
            return (
                shared
                + "When you have gathered enough information, stop calling tools and write your final answer. "
                + "Prefer ending with an Evidence section in this format when possible:\n"
                + "Evidence:\n"
                + "- <file_path> line <number>: <what this line does>\n"
                + "- <file_path> line <number>: <what this line does>\n"
                + "Keep the answer concise and grounded in required files and symbols from the codebase."
            )
        if task.family == "real_session":
            return shared + f"Continue only to the declared milestone: {task.next_milestone}"
        return shared
