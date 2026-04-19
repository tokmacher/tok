from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from typing import Any

ASSET_LOCK_FILENAME = "asset.lock.json"
DEFAULT_EVALUATOR_BUNDLE_DIR = "evaluators"


@dataclass(frozen=True)
class ShellCommandResult:
    command: str
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    execution_error: str = ""

    @property
    def command_invoked(self) -> bool:
        return not bool(self.execution_error)

    def as_completed_process(self) -> subprocess.CompletedProcess[str]:
        stderr = self.stderr
        if self.execution_error:
            stderr = f"{stderr}\n{self.execution_error}".strip() if stderr else self.execution_error
        return subprocess.CompletedProcess(list(self.argv), self.returncode, self.stdout, stderr)


@dataclass(frozen=True)
class MaterializedBenchmarkTask:
    task: Any
    lane: Any
    repeat_index: int
    condition: str
    asset_root: str
    workspace_root: str
    resolved_ref: str
    reportable: bool
    setup_ran: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolExecutionRecord:
    step_index: int
    tool_name: str
    canonical_tool_name: str
    tool_input: dict[str, Any]
    invalid: bool
    is_error: bool
    content_preview: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskEvaluationResult:
    success: bool
    grounding_success: bool
    details: dict[str, Any]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "grounding_success": self.grounding_success,
            "details": dict(self.details),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class BenchmarkTaskRunResult:
    lane_id: str
    condition: str
    task_id: str
    family: str
    repeat_index: int
    workspace_root: str
    answer_text: str
    raw_response: str
    provider_usage: dict[str, Any]
    tool_calls: int
    invalid_tool_calls: int
    reacquisition_events: int
    clean_exit: bool
    modified_files: tuple[str, ...]
    tool_records: tuple[ToolExecutionRecord, ...]
    turns: tuple[dict[str, Any], ...]
    evaluation: TaskEvaluationResult
    local_failure: str = ""
    notes: tuple[str, ...] = ()

    @property
    def success(self) -> bool:
        return self.evaluation.success

    @property
    def grounding_success(self) -> bool:
        return self.evaluation.grounding_success

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "condition": self.condition,
            "task_id": self.task_id,
            "family": self.family,
            "repeat_index": self.repeat_index,
            "workspace_root": self.workspace_root,
            "answer_text": self.answer_text,
            "raw_response": self.raw_response,
            "provider_usage": dict(self.provider_usage),
            "tool_calls": self.tool_calls,
            "invalid_tool_calls": self.invalid_tool_calls,
            "reacquisition_events": self.reacquisition_events,
            "clean_exit": self.clean_exit,
            "modified_files": list(self.modified_files),
            "tool_records": [record.to_dict() for record in self.tool_records],
            "turns": [dict(turn) for turn in self.turns],
            "evaluation": self.evaluation.to_dict(),
            "local_failure": self.local_failure,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class CatalogBenchmarkRun:
    lane_id: str
    selected_task_ids: tuple[str, ...]
    runs: tuple[Any, ...]
    report: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "selected_task_ids": list(self.selected_task_ids),
            "runs": [run.to_dict() for run in self.runs],
            "report": self.report.to_dict(),
        }
