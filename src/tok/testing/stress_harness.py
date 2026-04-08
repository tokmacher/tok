"""
Thin re-export facade for internal stress harness package.

This module is a thin compatibility layer over tok.testing.stress and ensures
that external imports continue to work as expected.
"""

from __future__ import annotations

from .stress import (  # Catalog; Tool execution; Runtime runner; Models; Classification; Report rendering
    DEFAULT_REQUIRED_CLASSES,
    EXCLUDED_GROUNDED_PATH_FRAGMENTS,
    LATE_FOLLOWTHROUGH_MIN_EVIDENCE_CHARS,
    LATE_STAGED_RETRY_PHASES,
    PRE_PRESSURE_MIN_EVIDENCE_CHARS,
    READ_ONLY_TOOL_NAMES,
    TASK_CATALOG,
    ReadOnlyToolExecutor,
    StressBreakpoint,
    StressHarness,
    StressHarnessConfig,
    StressObservation,
    StressRunResult,
    StressTask,
    StressTurnRecord,
    ValidatedAnchor,
    _followthrough_evidence_sufficient,
    _late_tool_contract_grace_kind,
    _preprocess_runtime_contract_signals,
    _runtime_retry_context_signals,
    _runtime_turn_context_signals,
    _sanitize_tool_use_block,
    _strip_answer_labels,
    classify_breakpoints,
    default_output_dir,
    extract_breakpoint_paths,
    inferred_cause_for_class,
    refactor_target_for_class,
    render_language_refactor_plan,
    render_stress_report,
    required_class_coverage,
    should_stop_run,
    summarize_implicated_files,
    write_stress_artifacts,
)

__all__ = [
    "DEFAULT_REQUIRED_CLASSES",
    "EXCLUDED_GROUNDED_PATH_FRAGMENTS",
    "LATE_FOLLOWTHROUGH_MIN_EVIDENCE_CHARS",
    "LATE_STAGED_RETRY_PHASES",
    "PRE_PRESSURE_MIN_EVIDENCE_CHARS",
    "READ_ONLY_TOOL_NAMES",
    # Catalog
    "TASK_CATALOG",
    # Tool execution
    "ReadOnlyToolExecutor",
    "StressBreakpoint",
    # Runtime runner
    "StressHarness",
    "StressHarnessConfig",
    "StressObservation",
    "StressRunResult",
    # Models
    "StressTask",
    "StressTurnRecord",
    "ValidatedAnchor",
    # Classification
    "_followthrough_evidence_sufficient",
    "_late_tool_contract_grace_kind",
    "_preprocess_runtime_contract_signals",
    "_runtime_retry_context_signals",
    "_runtime_turn_context_signals",
    "_sanitize_tool_use_block",
    "_strip_answer_labels",
    "classify_breakpoints",
    "default_output_dir",
    "extract_breakpoint_paths",
    "inferred_cause_for_class",
    "refactor_target_for_class",
    "render_language_refactor_plan",
    # Report rendering
    "render_stress_report",
    "required_class_coverage",
    "should_stop_run",
    "summarize_implicated_files",
    "write_stress_artifacts",
]
