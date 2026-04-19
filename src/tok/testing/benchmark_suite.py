"""Structured production benchmark catalogs, lane definitions, and reports."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VALID_BENCHMARK_FAMILIES = {"execution_patch", "repo_grounding", "real_session"}
VALID_CLAIM_SCOPES = {"headline", "secondary"}
VALID_BENCHMARK_CONDITIONS = {"baseline", "tok-universal"}
VALID_WORKSPACE_SOURCE_KINDS = {"asset_snapshot", "local_checkout"}
VALID_SUMMARY_SCOPES = {"public_production", "supplemental"}
MIN_ABSOLUTE_SUCCESS_FLOOR = 0.5
MIN_ABSOLUTE_GROUNDING_FLOOR = 0.5
MIN_QUALITY_GATE_RATE = 0.5
BENCHMARK_REPORT_STATEMENT = (
    "Comparison is baseline vs production Tok (`tok-universal`) only. No other Tok modes are benchmark candidates."
)
DEFAULT_BENCHMARK_ROOT = Path(__file__).resolve().parents[3] / "benchmarks"


def _clean_str(value: Any) -> str:
    return str(value or "").strip()


def _str_tuple(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        cleaned = _clean_str(values)
        return (cleaned,) if cleaned else ()
    if not isinstance(values, list | tuple):
        cleaned = _clean_str(values)
        return (cleaned,) if cleaned else ()
    result: list[str] = []
    for value in values:
        cleaned = _clean_str(value)
        if cleaned:
            result.append(cleaned)
    return tuple(result)


def _dict_tuple(values: Any) -> tuple[dict[str, Any], ...]:
    if values is None:
        return ()
    if isinstance(values, dict):
        return (dict(values),)
    if not isinstance(values, list | tuple):
        return ()
    result: list[dict[str, Any]] = []
    for value in values:
        if isinstance(value, dict):
            result.append(dict(value))
    return tuple(result)


def _load_json_records(path: Path, *, key: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and key in payload:
        records = payload[key]
    else:
        records = payload
    if isinstance(records, dict):
        return [records]
    if isinstance(records, list):
        return [dict(item) for item in records if isinstance(item, dict)]
    msg = f"{path} does not contain a valid '{key}' payload"
    raise ValueError(msg)


@dataclass(frozen=True)
class BenchmarkLane:
    id: str
    runtime_path: str
    transport_shape: str
    model_family: str
    provider: str
    adapter_name: str
    adapter_notes: str
    claim_scope: str
    normalized_differences: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkLane:
        lane = cls(
            id=_clean_str(data.get("id")),
            runtime_path=_clean_str(data.get("runtime_path")),
            transport_shape=_clean_str(data.get("transport_shape")),
            model_family=_clean_str(data.get("model_family")),
            provider=_clean_str(data.get("provider")),
            adapter_name=_clean_str(data.get("adapter_name")),
            adapter_notes=_clean_str(data.get("adapter_notes")),
            claim_scope=_clean_str(data.get("claim_scope")),
            normalized_differences=_str_tuple(data.get("normalized_differences")),
        )
        errors = lane.validate()
        if errors:
            msg = f"invalid benchmark lane '{lane.id or '<missing>'}': {'; '.join(errors)}"
            raise ValueError(msg)
        return lane

    @property
    def is_adapter(self) -> bool:
        return self.claim_scope == "secondary"

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.id:
            errors.append("missing id")
        if not self.runtime_path:
            errors.append("missing runtime_path")
        if not self.transport_shape:
            errors.append("missing transport_shape")
        if not self.model_family:
            errors.append("missing model_family")
        if not self.provider:
            errors.append("missing provider")
        if self.claim_scope not in VALID_CLAIM_SCOPES:
            errors.append(f"claim_scope must be one of {sorted(VALID_CLAIM_SCOPES)}")
        if self.id == "production_claude_lane":
            if self.claim_scope != "headline":
                errors.append("production_claude_lane must use claim_scope=headline")
            if self.adapter_name:
                errors.append("production_claude_lane must not declare an adapter_name")
            if "UniversalTokRuntime" not in self.runtime_path:
                errors.append("production_claude_lane runtime_path must mention UniversalTokRuntime")
            if "Claude Code" not in self.runtime_path:
                errors.append("production_claude_lane runtime_path must mention Claude Code")
        elif self.claim_scope == "headline":
            errors.append("only production_claude_lane may use claim_scope=headline")
        if self.claim_scope == "secondary":
            if not self.id.startswith("adapter_"):
                errors.append("secondary lanes must use an adapter_* id")
            if not self.adapter_name:
                errors.append("secondary lanes must declare adapter_name")
            if not self.adapter_notes:
                errors.append("secondary lanes must declare adapter_notes")
            if not self.normalized_differences:
                errors.append("secondary lanes must list normalized_differences")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "runtime_path": self.runtime_path,
            "transport_shape": self.transport_shape,
            "model_family": self.model_family,
            "provider": self.provider,
            "adapter_name": self.adapter_name,
            "adapter_notes": self.adapter_notes,
            "claim_scope": self.claim_scope,
            "normalized_differences": list(self.normalized_differences),
        }


@dataclass(frozen=True)
class BenchmarkTaskManifest:
    id: str
    family: str
    title: str
    summary: str
    repo: str
    ref: str
    setup_script: str
    prompt: str
    allowed_tools: tuple[str, ...]
    time_budget_minutes: int
    step_budget: int
    success_evaluator: dict[str, Any]
    artifact_policy: dict[str, Any]
    public_release: bool
    asset_dir: str = ""
    workspace_source: dict[str, Any] = field(default_factory=dict)
    family_payload: dict[str, Any] = field(default_factory=dict)
    seed_patch: str = ""
    prompt_forbidden_terms: tuple[str, ...] = ()
    allowed_paths: tuple[str, ...] = ()
    visible_tests: tuple[str, ...] = ()
    hidden_tests: tuple[str, ...] = ()
    required_files: tuple[str, ...] = ()
    required_symbols: tuple[str, ...] = ()
    supporting_spans: tuple[dict[str, Any], ...] = ()
    answer_contract: str = ""
    capture_source: str = ""
    redaction_version: str = ""
    episode_type: str = ""
    next_milestone: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkTaskManifest:
        manifest = cls(
            id=_clean_str(data.get("id")),
            family=_clean_str(data.get("family")),
            title=_clean_str(data.get("title")),
            summary=_clean_str(data.get("summary")),
            repo=_clean_str(data.get("repo")),
            ref=_clean_str(data.get("ref")),
            setup_script=_clean_str(data.get("setup_script")),
            prompt=_clean_str(data.get("prompt")),
            allowed_tools=_str_tuple(data.get("allowed_tools")),
            time_budget_minutes=int(data.get("time_budget_minutes", 0) or 0),
            step_budget=int(data.get("step_budget", 0) or 0),
            success_evaluator=dict(data.get("success_evaluator") or {}),
            artifact_policy=dict(data.get("artifact_policy") or {}),
            public_release=bool(data.get("public_release")),
            asset_dir=_clean_str(data.get("asset_dir")),
            workspace_source=dict(data.get("workspace_source") or {}),
            family_payload=dict(data.get("family_payload") or {}),
            seed_patch=_clean_str(data.get("seed_patch")),
            prompt_forbidden_terms=_str_tuple(data.get("prompt_forbidden_terms")),
            allowed_paths=_str_tuple(data.get("allowed_paths")),
            visible_tests=_str_tuple(data.get("visible_tests")),
            hidden_tests=_str_tuple(data.get("hidden_tests")),
            required_files=_str_tuple(data.get("required_files")),
            required_symbols=_str_tuple(data.get("required_symbols")),
            supporting_spans=_dict_tuple(data.get("supporting_spans")),
            answer_contract=_clean_str(data.get("answer_contract")),
            capture_source=_clean_str(data.get("capture_source")),
            redaction_version=_clean_str(data.get("redaction_version")),
            episode_type=_clean_str(data.get("episode_type")),
            next_milestone=_clean_str(data.get("next_milestone")),
        )
        errors = manifest.validate()
        if errors:
            msg = f"invalid benchmark task '{manifest.id or '<missing>'}': {'; '.join(errors)}"
            raise ValueError(msg)
        return manifest

    def prompt_leak_violations(self) -> tuple[str, ...]:
        prompt_lower = self.prompt.lower()
        return tuple(term for term in self.prompt_forbidden_terms if term.lower() in prompt_lower)

    def effective_asset_dir(self) -> str:
        return self.asset_dir

    def effective_workspace_source(self) -> dict[str, Any]:
        return dict(self.workspace_source)

    def effective_family_payload(self) -> dict[str, Any]:
        return dict(self.family_payload)

    def evaluator_spec_ref(self) -> str:
        return _clean_str(self.success_evaluator.get("evaluator_spec"))

    def workspace_source_kind(self) -> str:
        return _clean_str(self.workspace_source.get("kind"))

    def workspace_source_path(self) -> str:
        return _clean_str(self.workspace_source.get("path"))

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.id:
            errors.append("missing id")
        if self.family not in VALID_BENCHMARK_FAMILIES:
            errors.append(f"family must be one of {sorted(VALID_BENCHMARK_FAMILIES)}")
        if not self.title:
            errors.append("missing title")
        if not self.summary:
            errors.append("missing summary")
        if not self.repo:
            errors.append("missing repo")
        if not self.ref:
            errors.append("missing ref")
        if not self.setup_script:
            errors.append("missing setup_script")
        if not self.prompt:
            errors.append("missing prompt")
        if not self.allowed_tools:
            errors.append("missing allowed_tools")
        if self.time_budget_minutes <= 0:
            errors.append("time_budget_minutes must be > 0")
        if self.step_budget <= 0:
            errors.append("step_budget must be > 0")
        if not self.success_evaluator:
            errors.append("missing success_evaluator")
        if not self.artifact_policy:
            errors.append("missing artifact_policy")
        if not self.asset_dir:
            errors.append("missing asset_dir")
        if not self.workspace_source:
            errors.append("missing workspace_source")
        if not self.family_payload:
            errors.append("missing family_payload")
        leak_violations = self.prompt_leak_violations()
        if leak_violations:
            errors.append(f"prompt leaks forbidden terms: {', '.join(leak_violations)}")
        workspace_source_kind = self.workspace_source_kind()
        if workspace_source_kind not in VALID_WORKSPACE_SOURCE_KINDS:
            errors.append(f"workspace_source.kind must be one of {sorted(VALID_WORKSPACE_SOURCE_KINDS)}")
        if not self.workspace_source_path():
            errors.append("workspace_source.path is required")
        if self.public_release and workspace_source_kind != "asset_snapshot":
            errors.append("public_release tasks must use workspace_source.kind=asset_snapshot")
        if self.public_release and self.family == "real_session":
            errors.append("real_session tasks must not be public_release")

        if self.family == "execution_patch":
            if not self.allowed_paths:
                errors.append("execution_patch tasks must declare allowed_paths")
            if len(self.allowed_paths) > 3:
                errors.append("execution_patch tasks may touch at most 3 allowed_paths")
            payload_allowed_paths = _str_tuple(self.family_payload.get("allowed_paths"))
            if payload_allowed_paths != self.allowed_paths:
                errors.append("execution_patch family_payload.allowed_paths must match allowed_paths")
            payload_visible_tests = _str_tuple(self.family_payload.get("visible_tests"))
            if payload_visible_tests != self.visible_tests:
                errors.append("execution_patch family_payload.visible_tests must match visible_tests")
            if self.workspace_source_kind() == "asset_snapshot":
                if not self.seed_patch:
                    errors.append("asset-backed execution_patch tasks must declare seed_patch")
                if not _clean_str(self.family_payload.get("seed_patch_path")):
                    errors.append("execution_patch tasks must declare family_payload.seed_patch_path")
                if _clean_str(self.family_payload.get("seed_patch_path")) != self.seed_patch:
                    errors.append("execution_patch family_payload.seed_patch_path must match seed_patch")
            if self.public_release:
                if not self.hidden_tests and not self.evaluator_spec_ref():
                    errors.append(
                        "public execution_patch tasks must declare hidden_tests or success_evaluator.evaluator_spec"
                    )
                if self.hidden_tests:
                    errors.append("public execution_patch tasks must not declare hidden_tests")
            elif not self.hidden_tests and not self.evaluator_spec_ref():
                errors.append("execution_patch tasks must declare hidden_tests or success_evaluator.evaluator_spec")

        if self.family == "repo_grounding":
            if not self.required_files:
                errors.append("repo_grounding tasks must declare required_files")
            if not self.required_symbols:
                errors.append("repo_grounding tasks must declare required_symbols")
            if not self.supporting_spans:
                errors.append("repo_grounding tasks must declare supporting_spans")
            if not self.answer_contract:
                errors.append("repo_grounding tasks must declare answer_contract")
            if int(self.success_evaluator.get("min_grounded_retrieval_steps", 0) or 0) < 0:
                errors.append("repo_grounding min_grounded_retrieval_steps must be >= 0")
            for span in self.supporting_spans:
                if not _clean_str(span.get("file")) or not _clean_str(span.get("anchor")):
                    errors.append("repo_grounding supporting_spans require file and anchor")
                    break
            if not _clean_str(self.family_payload.get("gold_answer_path")):
                errors.append("repo_grounding tasks must declare family_payload.gold_answer_path")

        if self.family == "real_session":
            if not self.capture_source:
                errors.append("real_session tasks must declare capture_source")
            if not self.redaction_version:
                errors.append("real_session tasks must declare redaction_version")
            if not self.episode_type:
                errors.append("real_session tasks must declare episode_type")
            if not self.next_milestone:
                errors.append("real_session tasks must declare next_milestone")
            if not isinstance(self.family_payload.get("episode_bundle"), dict):
                errors.append("real_session tasks must declare family_payload.episode_bundle")
            if not isinstance(self.family_payload.get("milestone_evaluator"), dict):
                errors.append("real_session tasks must declare family_payload.milestone_evaluator")

        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "title": self.title,
            "summary": self.summary,
            "repo": self.repo,
            "ref": self.ref,
            "setup_script": self.setup_script,
            "prompt": self.prompt,
            "allowed_tools": list(self.allowed_tools),
            "time_budget_minutes": self.time_budget_minutes,
            "step_budget": self.step_budget,
            "success_evaluator": dict(self.success_evaluator),
            "artifact_policy": dict(self.artifact_policy),
            "public_release": self.public_release,
            "asset_dir": self.asset_dir,
            "workspace_source": dict(self.workspace_source),
            "family_payload": dict(self.family_payload),
            "seed_patch": self.seed_patch,
            "prompt_forbidden_terms": list(self.prompt_forbidden_terms),
            "allowed_paths": list(self.allowed_paths),
            "visible_tests": list(self.visible_tests),
            "hidden_tests": list(self.hidden_tests),
            "required_files": list(self.required_files),
            "required_symbols": list(self.required_symbols),
            "supporting_spans": [dict(item) for item in self.supporting_spans],
            "answer_contract": self.answer_contract,
            "capture_source": self.capture_source,
            "redaction_version": self.redaction_version,
            "episode_type": self.episode_type,
            "next_milestone": self.next_milestone,
        }


@dataclass(frozen=True)
class BenchmarkCatalog:
    root: str
    lanes: tuple[BenchmarkLane, ...]
    tasks: tuple[BenchmarkTaskManifest, ...]

    def validate(self) -> list[str]:
        errors: list[str] = []
        headline_lanes = [lane for lane in self.lanes if lane.claim_scope == "headline"]
        if len(headline_lanes) != 1:
            errors.append("catalog must contain exactly one headline lane")
        elif headline_lanes[0].id != "production_claude_lane":
            errors.append("headline lane must be production_claude_lane")

        lane_ids: set[str] = set()
        for lane in self.lanes:
            if lane.id in lane_ids:
                errors.append(f"duplicate lane id: {lane.id}")
            lane_ids.add(lane.id)

        task_ids: set[str] = set()
        family_counts = {family: 0 for family in VALID_BENCHMARK_FAMILIES}
        for task in self.tasks:
            if task.id in task_ids:
                errors.append(f"duplicate task id: {task.id}")
            task_ids.add(task.id)
            if task.family in family_counts:
                family_counts[task.family] += 1

        for family, count in family_counts.items():
            if count == 0:
                errors.append(f"catalog is missing tasks for family: {family}")

        return errors

    def family_counts(self) -> dict[str, int]:
        counts = {family: 0 for family in VALID_BENCHMARK_FAMILIES}
        for task in self.tasks:
            counts[task.family] = counts.get(task.family, 0) + 1
        return counts

    def headline_lane(self) -> BenchmarkLane:
        for lane in self.lanes:
            if lane.claim_scope == "headline":
                return lane
        msg = "catalog does not contain a headline lane"
        raise ValueError(msg)

    def compatibility_lanes(self) -> tuple[BenchmarkLane, ...]:
        return tuple(lane for lane in self.lanes if lane.claim_scope == "secondary")

    def lane_by_id(self, lane_id: str) -> BenchmarkLane:
        for lane in self.lanes:
            if lane.id == lane_id:
                return lane
        msg = f"unknown benchmark lane: {lane_id}"
        raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "lanes": [lane.to_dict() for lane in self.lanes],
            "tasks": [task.to_dict() for task in self.tasks],
        }


def load_benchmark_catalog(root: Path = DEFAULT_BENCHMARK_ROOT) -> BenchmarkCatalog:
    lanes: list[BenchmarkLane] = []
    tasks: list[BenchmarkTaskManifest] = []
    lanes_dir = root / "lanes"
    if not lanes_dir.exists():
        msg = f"benchmark lanes directory not found: {lanes_dir}"
        raise ValueError(msg)
    for path in sorted(lanes_dir.glob("*.json")):
        lanes.extend(BenchmarkLane.from_dict(item) for item in _load_json_records(path, key="lanes"))
    for family in sorted(VALID_BENCHMARK_FAMILIES):
        family_dir = root / family
        if not family_dir.exists():
            continue
        for path in sorted(family_dir.glob("*.json")):
            tasks.extend(BenchmarkTaskManifest.from_dict(item) for item in _load_json_records(path, key="tasks"))
    catalog = BenchmarkCatalog(root=str(root), lanes=tuple(lanes), tasks=tuple(tasks))
    errors = catalog.validate()
    if errors:
        msg = f"invalid benchmark catalog: {'; '.join(errors)}"
        raise ValueError(msg)
    return catalog


@dataclass(frozen=True)
class BenchmarkConditionPlan:
    lane_id: str
    condition: str
    runtime_path: str
    transport_shape: str
    model_family: str
    provider: str
    adapter_name: str
    adapter_notes: str
    claim_scope: str
    task_ids: tuple[str, ...]
    runtime_wrapper_active: bool
    candidate_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "condition": self.condition,
            "runtime_path": self.runtime_path,
            "transport_shape": self.transport_shape,
            "model_family": self.model_family,
            "provider": self.provider,
            "adapter_name": self.adapter_name,
            "adapter_notes": self.adapter_notes,
            "claim_scope": self.claim_scope,
            "task_ids": list(self.task_ids),
            "runtime_wrapper_active": self.runtime_wrapper_active,
            "candidate_mode": self.candidate_mode,
        }


def build_condition_plan(
    catalog: BenchmarkCatalog,
    *,
    lane_id: str,
    condition: str,
) -> BenchmarkConditionPlan:
    if condition not in VALID_BENCHMARK_CONDITIONS:
        msg = f"condition must be one of {sorted(VALID_BENCHMARK_CONDITIONS)}"
        raise ValueError(msg)
    lane = catalog.lane_by_id(lane_id)
    task_ids = tuple(sorted(task.id for task in catalog.tasks))
    return BenchmarkConditionPlan(
        lane_id=lane.id,
        condition=condition,
        runtime_path=lane.runtime_path,
        transport_shape=lane.transport_shape,
        model_family=lane.model_family,
        provider=lane.provider,
        adapter_name=lane.adapter_name,
        adapter_notes=lane.adapter_notes,
        claim_scope=lane.claim_scope,
        task_ids=task_ids,
        runtime_wrapper_active=condition == "tok-universal",
        candidate_mode=condition,
    )


@dataclass(frozen=True)
class BenchmarkComparisonRun:
    lane_id: str
    task_id: str
    family: str
    repeat_index: int
    public_release: bool
    baseline_success: bool
    tok_success: bool
    quality_gate_passed: bool
    total_token_delta: int
    latency_delta_ms: float
    reacquisition_events: int = 0
    invalid_tool_calls: int = 0
    paired_result_stable: bool = True
    baseline_grounding_success: bool = False
    tok_grounding_success: bool = False
    baseline_tool_calls: int = 0
    tok_tool_calls: int = 0
    format_contract_violations: tuple[str, ...] = ()
    tool_engagement_stats: dict[str, Any] = field(default_factory=dict)
    matched_completion_pair: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkComparisonRun:
        run = cls(
            lane_id=_clean_str(data.get("lane_id")),
            task_id=_clean_str(data.get("task_id")),
            family=_clean_str(data.get("family")),
            repeat_index=int(data.get("repeat_index", 0) or 0),
            public_release=bool(data.get("public_release")),
            baseline_success=bool(data.get("baseline_success")),
            tok_success=bool(data.get("tok_success")),
            quality_gate_passed=bool(data.get("quality_gate_passed")),
            total_token_delta=int(data.get("total_token_delta", 0) or 0),
            latency_delta_ms=float(data.get("latency_delta_ms", 0.0) or 0.0),
            reacquisition_events=int(data.get("reacquisition_events", 0) or 0),
            invalid_tool_calls=int(data.get("invalid_tool_calls", 0) or 0),
            paired_result_stable=bool(data.get("paired_result_stable", True)),
            baseline_grounding_success=bool(data.get("baseline_grounding_success", False)),
            tok_grounding_success=bool(data.get("tok_grounding_success", False)),
            baseline_tool_calls=int(data.get("baseline_tool_calls", 0) or 0),
            tok_tool_calls=int(data.get("tok_tool_calls", 0) or 0),
            format_contract_violations=_str_tuple(data.get("format_contract_violations")),
            tool_engagement_stats=dict(data.get("tool_engagement_stats") or {}),
            matched_completion_pair=bool(data.get("matched_completion_pair", False)),
        )
        errors = run.validate()
        if errors:
            msg = f"invalid benchmark comparison run '{run.task_id or '<missing>'}': {'; '.join(errors)}"
            raise ValueError(msg)
        return run

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.lane_id:
            errors.append("missing lane_id")
        if not self.task_id:
            errors.append("missing task_id")
        if self.family not in VALID_BENCHMARK_FAMILIES:
            errors.append(f"family must be one of {sorted(VALID_BENCHMARK_FAMILIES)}")
        if self.repeat_index <= 0:
            errors.append("repeat_index must be > 0")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "task_id": self.task_id,
            "family": self.family,
            "repeat_index": self.repeat_index,
            "public_release": self.public_release,
            "baseline_success": self.baseline_success,
            "tok_success": self.tok_success,
            "quality_gate_passed": self.quality_gate_passed,
            "total_token_delta": self.total_token_delta,
            "latency_delta_ms": self.latency_delta_ms,
            "reacquisition_events": self.reacquisition_events,
            "invalid_tool_calls": self.invalid_tool_calls,
            "paired_result_stable": self.paired_result_stable,
            "baseline_grounding_success": self.baseline_grounding_success,
            "tok_grounding_success": self.tok_grounding_success,
            "baseline_tool_calls": self.baseline_tool_calls,
            "tok_tool_calls": self.tok_tool_calls,
            "format_contract_violations": list(self.format_contract_violations),
            "tool_engagement_stats": dict(self.tool_engagement_stats),
            "matched_completion_pair": self.matched_completion_pair,
        }


@dataclass(frozen=True)
class BenchmarkLaneSummary:
    lane: BenchmarkLane
    summary_scope: str
    sample_size: int
    baseline_success_rate: float
    tok_success_rate: float
    token_win_rate: float
    success_delta: float
    median_token_delta: float
    matched_success_token_delta: float | None
    repeat_to_repeat_variance: float
    latency_variance: float
    reacquisition_events: int
    invalid_tool_call_count: int
    consistency_gate_passed: bool
    public_claim_allowed: bool
    baseline_grounding_rate: float = 0.0
    tok_grounding_rate: float = 0.0
    grounding_delta: float = 0.0
    completion_success_rate: dict[str, float] = field(default_factory=dict)
    format_contract_violations: dict[str, int] = field(default_factory=dict)
    tool_engagement_stats: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkLaneSummary:
        lane = BenchmarkLane.from_dict(dict(data.get("lane") or {}))
        summary_scope = _clean_str(data.get("summary_scope"))
        if not summary_scope:
            summary_scope = "public_production" if lane.claim_scope == "headline" else "supplemental"
        return cls(
            lane=lane,
            summary_scope=summary_scope,
            sample_size=int(data.get("sample_size", 0) or 0),
            baseline_success_rate=float(data.get("baseline_success_rate", 0.0) or 0.0),
            tok_success_rate=float(data.get("tok_success_rate", 0.0) or 0.0),
            token_win_rate=float(data.get("token_win_rate", 0.0) or 0.0),
            success_delta=float(data.get("success_delta", 0.0) or 0.0),
            median_token_delta=float(data.get("median_token_delta", 0.0) or 0.0),
            matched_success_token_delta=(
                None
                if data.get("matched_success_token_delta") is None
                else float(data.get("matched_success_token_delta", 0.0) or 0.0)
            ),
            repeat_to_repeat_variance=float(data.get("repeat_to_repeat_variance", 0.0) or 0.0),
            latency_variance=float(data.get("latency_variance", 0.0) or 0.0),
            reacquisition_events=int(data.get("reacquisition_events", 0) or 0),
            invalid_tool_call_count=int(data.get("invalid_tool_call_count", 0) or 0),
            consistency_gate_passed=bool(data.get("consistency_gate_passed")),
            public_claim_allowed=bool(data.get("public_claim_allowed")),
            baseline_grounding_rate=float(data.get("baseline_grounding_rate", 0.0) or 0.0),
            tok_grounding_rate=float(data.get("tok_grounding_rate", 0.0) or 0.0),
            grounding_delta=float(data.get("grounding_delta", 0.0) or 0.0),
            completion_success_rate=dict(data.get("completion_success_rate") or {}),
            format_contract_violations=dict(data.get("format_contract_violations") or {}),
            tool_engagement_stats=dict(data.get("tool_engagement_stats") or {}),
            notes=_str_tuple(data.get("notes")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane.to_dict(),
            "summary_scope": self.summary_scope,
            "sample_size": self.sample_size,
            "baseline_success_rate": self.baseline_success_rate,
            "tok_success_rate": self.tok_success_rate,
            "token_win_rate": self.token_win_rate,
            "success_delta": self.success_delta,
            "median_token_delta": self.median_token_delta,
            "matched_success_token_delta": self.matched_success_token_delta,
            "repeat_to_repeat_variance": self.repeat_to_repeat_variance,
            "latency_variance": self.latency_variance,
            "reacquisition_events": self.reacquisition_events,
            "invalid_tool_call_count": self.invalid_tool_call_count,
            "consistency_gate_passed": self.consistency_gate_passed,
            "public_claim_allowed": self.public_claim_allowed,
            "baseline_grounding_rate": self.baseline_grounding_rate,
            "tok_grounding_rate": self.tok_grounding_rate,
            "grounding_delta": self.grounding_delta,
            "completion_success_rate": dict(self.completion_success_rate),
            "format_contract_violations": dict(self.format_contract_violations),
            "tool_engagement_stats": dict(self.tool_engagement_stats),
            "notes": list(self.notes),
        }


def summarize_lane_runs(
    lane: BenchmarkLane,
    runs: list[BenchmarkComparisonRun],
    *,
    summary_scope: str,
    claimable: bool,
) -> BenchmarkLaneSummary:
    if not runs:
        notes_list = ["no_runs"]
        return BenchmarkLaneSummary(
            lane=lane,
            summary_scope=summary_scope,
            sample_size=0,
            baseline_success_rate=0.0,
            tok_success_rate=0.0,
            token_win_rate=0.0,
            success_delta=0.0,
            median_token_delta=0.0,
            matched_success_token_delta=None,
            repeat_to_repeat_variance=0.0,
            latency_variance=0.0,
            reacquisition_events=0,
            invalid_tool_call_count=0,
            consistency_gate_passed=False,
            public_claim_allowed=False,
            notes=tuple(notes_list),
        )

    baseline_successes = [1.0 if run.baseline_success else 0.0 for run in runs]
    tok_successes = [1.0 if run.tok_success else 0.0 for run in runs]
    token_wins = [1.0 if (run.total_token_delta < 0 and run.tok_success) else 0.0 for run in runs]
    baseline_groundings = [1.0 if run.baseline_grounding_success else 0.0 for run in runs]
    tok_groundings = [1.0 if run.tok_grounding_success else 0.0 for run in runs]
    quality_gate_scores = [1.0 if run.quality_gate_passed else 0.0 for run in runs]
    token_deltas = [float(run.total_token_delta) for run in runs]
    matched_token_deltas = [
        float(run.total_token_delta)
        for run in runs
        if run.matched_completion_pair or (run.baseline_success and run.tok_success)
    ]
    both_fail_count = sum(1 for run in runs if (not run.baseline_success and not run.tok_success))
    all_conditions_failed = both_fail_count == len(runs)
    runs_by_task: dict[str, list[BenchmarkComparisonRun]] = {}
    for run in runs:
        runs_by_task.setdefault(run.task_id, []).append(run)

    baseline_success_rate = round(sum(baseline_successes) / len(baseline_successes), 3)
    tok_success_rate = round(sum(tok_successes) / len(tok_successes), 3)
    token_win_rate = round(sum(token_wins) / len(token_wins), 3)
    success_delta = round(tok_success_rate - baseline_success_rate, 3)
    baseline_grounding_rate = round(sum(baseline_groundings) / len(baseline_groundings), 3)
    tok_grounding_rate = round(sum(tok_groundings) / len(tok_groundings), 3)
    grounding_delta = round(tok_grounding_rate - baseline_grounding_rate, 3)
    quality_gate_rate = round(sum(quality_gate_scores) / len(quality_gate_scores), 3)
    median_token_delta = round(float(statistics.median(token_deltas)), 1)
    matched_success_token_delta = (
        round(float(statistics.median(matched_token_deltas)), 1) if matched_token_deltas else None
    )
    per_task_repeat_variances = [
        float(statistics.pvariance([float(item.total_token_delta) for item in task_runs]))
        for task_runs in runs_by_task.values()
        if len(task_runs) > 1
    ]
    per_task_latency_variances = [
        float(statistics.pvariance([float(item.latency_delta_ms) for item in task_runs]))
        for task_runs in runs_by_task.values()
        if len(task_runs) > 1
    ]
    repeat_variance = round(float(statistics.mean(per_task_repeat_variances)), 2) if per_task_repeat_variances else 0.0
    latency_variance = (
        round(float(statistics.mean(per_task_latency_variances)), 2) if per_task_latency_variances else 0.0
    )
    reacquisition_events = sum(run.reacquisition_events for run in runs)
    invalid_tool_call_count = sum(run.invalid_tool_calls for run in runs)
    stable = all(run.paired_result_stable for run in runs)
    success_stable = all(
        len({(item.baseline_success, item.tok_success, item.quality_gate_passed) for item in task_runs}) == 1
        for task_runs in runs_by_task.values()
    )
    stable = stable and success_stable

    notes: list[str] = []
    if success_delta < 0:
        notes.append("success_regressed_vs_baseline")
    if grounding_delta < 0:
        notes.append("grounding_regressed_vs_baseline")
    if not stable:
        notes.append("paired_result_unstable")
    if invalid_tool_call_count > 0:
        notes.append("invalid_tool_calls_present")
    if tok_success_rate < MIN_ABSOLUTE_SUCCESS_FLOOR:
        notes.append("tok_success_below_absolute_floor")
    if tok_grounding_rate < MIN_ABSOLUTE_GROUNDING_FLOOR:
        notes.append("tok_grounding_below_absolute_floor")
    if quality_gate_rate < MIN_QUALITY_GATE_RATE:
        notes.append("quality_gate_below_absolute_floor")
    if all_conditions_failed:
        notes.append("all_conditions_failed")
    violation_counts: dict[str, int] = {}
    for run in runs:
        for violation in run.format_contract_violations:
            violation_counts[violation] = violation_counts.get(violation, 0) + 1
    if violation_counts:
        notes.append("advisory_format_contract_violations_present")
    baseline_tool_calls = [run.baseline_tool_calls for run in runs]
    tok_tool_calls = [run.tok_tool_calls for run in runs]
    tool_engagement_stats: dict[str, Any] = {
        "baseline_avg_tool_calls": round(sum(baseline_tool_calls) / len(baseline_tool_calls), 2),
        "tok_avg_tool_calls": round(sum(tok_tool_calls) / len(tok_tool_calls), 2),
        "baseline_zero_tool_call_rate": round(
            sum(1 for calls in baseline_tool_calls if calls == 0) / len(baseline_tool_calls), 3
        ),
        "tok_zero_tool_call_rate": round(sum(1 for calls in tok_tool_calls if calls == 0) / len(tok_tool_calls), 3),
    }

    consistency_gate_passed = (
        stable
        and success_delta >= 0
        and grounding_delta >= 0
        and tok_success_rate >= MIN_ABSOLUTE_SUCCESS_FLOOR
        and tok_grounding_rate >= MIN_ABSOLUTE_GROUNDING_FLOOR
        and quality_gate_rate >= MIN_QUALITY_GATE_RATE
        and not all_conditions_failed
    )
    public_claim_allowed = claimable and lane.claim_scope == "headline" and consistency_gate_passed

    return BenchmarkLaneSummary(
        lane=lane,
        summary_scope=summary_scope,
        sample_size=len(runs),
        baseline_success_rate=baseline_success_rate,
        tok_success_rate=tok_success_rate,
        token_win_rate=token_win_rate,
        success_delta=success_delta,
        median_token_delta=median_token_delta,
        matched_success_token_delta=matched_success_token_delta,
        repeat_to_repeat_variance=repeat_variance,
        latency_variance=latency_variance,
        reacquisition_events=reacquisition_events,
        invalid_tool_call_count=invalid_tool_call_count,
        consistency_gate_passed=consistency_gate_passed,
        public_claim_allowed=public_claim_allowed,
        baseline_grounding_rate=baseline_grounding_rate,
        tok_grounding_rate=tok_grounding_rate,
        grounding_delta=grounding_delta,
        completion_success_rate={"baseline": baseline_success_rate, "tok-universal": tok_success_rate},
        format_contract_violations=violation_counts,
        tool_engagement_stats=tool_engagement_stats,
        notes=tuple(notes),
    )


@dataclass(frozen=True)
class BenchmarkReport:
    title: str
    statement: str
    lane_summaries: tuple[BenchmarkLaneSummary, ...]
    notes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkReport:
        report = cls(
            title=_clean_str(data.get("title")) or "Production Tok Benchmark Report",
            statement=_clean_str(data.get("statement")) or BENCHMARK_REPORT_STATEMENT,
            lane_summaries=tuple(
                BenchmarkLaneSummary.from_dict(item)
                for item in data.get("lane_summaries", [])
                if isinstance(item, dict)
            ),
            notes=_str_tuple(data.get("notes")),
        )
        errors = report.validate()
        if errors:
            msg = f"invalid benchmark report: {'; '.join(errors)}"
            raise ValueError(msg)
        return report

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.statement != BENCHMARK_REPORT_STATEMENT:
            errors.append("report statement must match the production-tok benchmark statement")
        headline = [summary for summary in self.lane_summaries if summary.summary_scope == "public_production"]
        if len(headline) != 1:
            errors.append("report must contain exactly one public production lane summary")
        elif headline[0].lane.id != "production_claude_lane":
            errors.append("public production lane summary must be production_claude_lane")
        for summary in self.lane_summaries:
            if summary.summary_scope not in VALID_SUMMARY_SCOPES:
                errors.append(f"invalid summary_scope for lane {summary.lane.id}: {summary.summary_scope}")
            if summary.summary_scope == "supplemental" and summary.public_claim_allowed:
                errors.append(f"supplemental lane {summary.lane.id} must not set public_claim_allowed")
        return errors

    def headline_summary(self) -> BenchmarkLaneSummary:
        for summary in self.lane_summaries:
            if summary.summary_scope == "public_production":
                return summary
        msg = "report does not contain a headline summary"
        raise ValueError(msg)

    def supplemental_summaries(self) -> tuple[BenchmarkLaneSummary, ...]:
        return tuple(summary for summary in self.lane_summaries if summary.summary_scope == "supplemental")

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "statement": self.statement,
            "lane_summaries": [summary.to_dict() for summary in self.lane_summaries],
            "notes": list(self.notes),
        }


def build_benchmark_report(
    catalog: BenchmarkCatalog,
    runs: list[BenchmarkComparisonRun],
    *,
    title: str = "Production Tok Benchmark Report",
    notes: tuple[str, ...] = (),
) -> BenchmarkReport:
    runs_by_lane: dict[str, list[BenchmarkComparisonRun]] = {}
    for run in runs:
        catalog.lane_by_id(run.lane_id)
        runs_by_lane.setdefault(run.lane_id, []).append(run)

    headline_lane = catalog.headline_lane()
    public_runs = [run for run in runs_by_lane.get(headline_lane.id, []) if run.public_release]
    lane_summaries: list[BenchmarkLaneSummary] = [
        summarize_lane_runs(
            headline_lane,
            public_runs,
            summary_scope="public_production",
            claimable=True,
        )
    ]
    for lane in catalog.lanes:
        supplemental_runs = [
            run for run in runs_by_lane.get(lane.id, []) if not (lane.id == headline_lane.id and run.public_release)
        ]
        if not supplemental_runs:
            continue
        lane_summaries.append(
            summarize_lane_runs(
                lane,
                supplemental_runs,
                summary_scope="supplemental",
                claimable=False,
            )
        )
    report = BenchmarkReport(
        title=title,
        statement=BENCHMARK_REPORT_STATEMENT,
        lane_summaries=tuple(lane_summaries),
        notes=notes,
    )
    errors = report.validate()
    if errors:
        msg = f"invalid benchmark report: {'; '.join(errors)}"
        raise ValueError(msg)
    return report


def load_benchmark_report(path: Path) -> BenchmarkReport:
    return BenchmarkReport.from_dict(json.loads(path.read_text()))


def check_benchmark_report(path: Path) -> dict[str, Any]:
    report = load_benchmark_report(path)
    headline = report.headline_summary()
    reason = ""
    if not headline.consistency_gate_passed or not headline.public_claim_allowed:
        reason = ",".join(headline.notes) or "consistency_gate_failed"
    return {
        "path": str(path),
        "headline_lane": headline.lane.id,
        "statement_ok": report.statement == BENCHMARK_REPORT_STATEMENT,
        "consistency_gate_passed": headline.consistency_gate_passed,
        "public_claim_allowed": headline.public_claim_allowed,
        "reason": reason,
        "passed": headline.consistency_gate_passed and headline.public_claim_allowed,
    }


def render_benchmark_report_markdown(report: BenchmarkReport) -> str:
    headline = report.headline_summary()
    supplemental = report.supplemental_summaries()
    lines = [
        f"# {report.title}",
        "",
        f"- {report.statement}",
        "- Headline lane: `production_claude_lane`",
        "- `tok-universal` is the benchmark label for the production bridge-first Tok path.",
        "",
        "## Public Production Lane",
        "",
        "| Lane | Provider | Tok Success | Baseline Success | Success Delta | Tok Grounding | Baseline Grounding | Grounding Delta | Median Token Delta | Token Win Rate | Repeat Variance | Latency Variance | Consistency Gate | Public Claim |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        (
            f"| {headline.lane.id} | {headline.lane.provider} | {headline.tok_success_rate:.3f} | "
            f"{headline.baseline_success_rate:.3f} | {headline.success_delta:+.3f} | "
            f"{headline.tok_grounding_rate:.3f} | {headline.baseline_grounding_rate:.3f} | "
            f"{headline.grounding_delta:+.3f} | "
            f"{headline.median_token_delta:.1f} | "
            f"{headline.token_win_rate:.3f} | {headline.repeat_to_repeat_variance:.2f} | {headline.latency_variance:.2f} | "
            f"{headline.consistency_gate_passed} | {headline.public_claim_allowed} |"
        ),
        "",
        "### Advisory Diagnostics",
        "",
        f"- completion_success_rate: baseline={headline.completion_success_rate.get('baseline', 0.0):.3f}, tok-universal={headline.completion_success_rate.get('tok-universal', 0.0):.3f}",
        f"- format_contract_violations: {headline.format_contract_violations or {}}",
        f"- tool_engagement_stats: {headline.tool_engagement_stats or {}}",
        "",
    ]

    if supplemental:
        lines.extend(
            [
                "## Supplemental Internal/Advisory Tasks",
                "",
                "| Lane | Scope | Provider | Tok Grounding | Baseline Grounding | Grounding Delta | Median Token Delta | Token Win Rate | Consistency Gate | Public Claim | Notes |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
            ]
        )
        for summary in supplemental:
            lines.append(
                f"| {summary.lane.id} | {summary.lane.claim_scope} | {summary.lane.provider} | "
                f"{summary.tok_grounding_rate:.3f} | {summary.baseline_grounding_rate:.3f} | "
                f"{summary.grounding_delta:+.3f} | {summary.median_token_delta:.1f} | {summary.token_win_rate:.3f} | "
                f"{summary.consistency_gate_passed} | {summary.public_claim_allowed} | "
                f"{', '.join(summary.notes) if summary.notes else 'n/a'} |"
            )
            lines.append(
                f"- {summary.lane.id} diagnostics: completion_success_rate={summary.completion_success_rate}, "
                f"format_contract_violations={summary.format_contract_violations}, "
                f"tool_engagement_stats={summary.tool_engagement_stats}"
            )
        lines.append("")
    else:
        lines.extend(["## Supplemental Internal/Advisory Tasks", "", "- none", ""])

    if report.notes:
        lines.extend(["## Notes", ""])
        for note in report.notes:
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines)
