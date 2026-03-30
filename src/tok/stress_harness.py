"""Backward-compatible facade for tok.stress_harness.

This module is a thin compatibility wrapper that re-exports the public API
from tok.testing.stress to ensure backward compatibility for:
- CLI
- Tests
- External users
"""

from __future__ import annotations

from .testing.stress import (
    # Models
    StressTask,
    StressHarnessConfig,
    ValidatedAnchor,
    StressBreakpoint,
    StressTurnRecord,
    StressRunResult,
    StressObservation,
    DEFAULT_REQUIRED_CLASSES,
    READ_ONLY_TOOL_NAMES,
    EXCLUDED_GROUNDED_PATH_FRAGMENTS,
    LATE_FOLLOWTHROUGH_MIN_EVIDENCE_CHARS,
    PRE_PRESSURE_MIN_EVIDENCE_CHARS,
    LATE_STAGED_RETRY_PHASES,
    # Catalog
    TASK_CATALOG,
    # Runtime runner
    StressHarness,
    # Classification
    _followthrough_evidence_sufficient,
    _late_tool_contract_grace_kind,
    _preprocess_runtime_contract_signals,
    _runtime_retry_context_signals,
    _runtime_turn_context_signals,
    classify_breakpoints,
    inferred_cause_for_class,
    required_class_coverage,
    refactor_target_for_class,
    should_stop_run,
    # Tool execution
    ReadOnlyToolExecutor,
    # Report rendering
    render_stress_report,
    render_language_refactor_plan,
    summarize_implicated_files,
    extract_breakpoint_paths,
    write_stress_artifacts,
    default_output_dir,
    _sanitize_tool_use_block,
    _strip_answer_labels,
)

__all__ = [
    # Models
    "StressTask",
    "StressHarnessConfig",
    "ValidatedAnchor",
    "StressBreakpoint",
    "StressTurnRecord",
    "StressRunResult",
    "StressObservation",
    "DEFAULT_REQUIRED_CLASSES",
    "READ_ONLY_TOOL_NAMES",
    "EXCLUDED_GROUNDED_PATH_FRAGMENTS",
    "LATE_FOLLOWTHROUGH_MIN_EVIDENCE_CHARS",
    "PRE_PRESSURE_MIN_EVIDENCE_CHARS",
    "LATE_STAGED_RETRY_PHASES",
    # Catalog
    "TASK_CATALOG",
    # Runtime runner
    "StressHarness",
    # Classification
    "_followthrough_evidence_sufficient",
    "_late_tool_contract_grace_kind",
    "_preprocess_runtime_contract_signals",
    "_runtime_retry_context_signals",
    "_runtime_turn_context_signals",
    "classify_breakpoints",
    "inferred_cause_for_class",
    "required_class_coverage",
    "refactor_target_for_class",
    "should_stop_run",
    # Tool execution
    "ReadOnlyToolExecutor",
    # Report rendering
    "render_stress_report",
    "render_language_refactor_plan",
    "summarize_implicated_files",
    "extract_breakpoint_paths",
    "write_stress_artifacts",
    "default_output_dir",
    "_sanitize_tool_use_block",
    "_strip_answer_labels",
]
