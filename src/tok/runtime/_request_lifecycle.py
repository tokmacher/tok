"""Immutable trace context recording which pipeline stages were entered."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RequestLifecycle:
    """Frozen record of which request-pipeline stages completed.

    Each field corresponds to a named stage in the bridge preparation
    pipeline.  Stages are set to ``True`` after the corresponding code
    section finishes successfully.  Because the dataclass is frozen,
    a new instance is produced at each stage boundary via
    ``dataclasses.replace()``.

    Bridge-layer stages (set by _bridge_runtime_pipeline.prepare_bridge_payload):
        initial_preflight, model_extraction, tool_compatibility_check,
        request_preparation, runtime_preparation, signals_and_metrics,
        prepared_preflight, plan_finalization_guard, final_payload_construction.

    Runtime-internal stages (set inside RuntimeSession.prepare_request; require
    runtime instrumentation to observe — not yet wired as of 0.1.9):
        repeat_target_capture, tool_event_normalization, hot_memory_refresh.
    """

    initial_preflight: bool = False
    model_extraction: bool = False
    tool_compatibility_check: bool = False
    request_preparation: bool = False
    runtime_preparation: bool = False
    signals_and_metrics: bool = False
    prepared_preflight: bool = False
    plan_finalization_guard: bool = False
    final_payload_construction: bool = False
    # Runtime-internal: not yet set by the bridge layer.
    repeat_target_capture: bool = False
    tool_event_normalization: bool = False
    hot_memory_refresh: bool = False
