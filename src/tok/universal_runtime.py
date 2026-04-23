"""Backward-compatible facade for the runtime module layout."""

from __future__ import annotations

from .runtime.core import (
    RuntimeSession,
    UniversalTokRuntime,
    apply_schema_adaptations,
)
from .runtime.memory.session_state import _discover_project_markers
from .runtime.pipeline.request_validation import detect_prompt_bloat
from .runtime.pipeline.response_processing import (
    malformed_tok_signals,
    parse_tok_response,
    response_behavior_signals,
    response_contract_for_mode,
    translate_response_tools,
)
from .runtime.pipeline.tool_processing import (
    build_tool_use_id_to_context,
    collect_behavior_signals,
    normalize_tool_events,
)
from .runtime.policy.macro_handling import (
    _jit_context_matches,
    execute_jit_macro,
)
from .runtime.policy.semantic_validation import (
    SemanticValidator,
    calculate_invisible_pressure,
    calculate_memory_lift,
    calculate_semantic_regression_score,
)
from .runtime.types import (
    NormalizedToolEvent,
    PreparedRuntimeRequest,
    ProcessedRuntimeResponse,
    RuntimeRequest,
)

__all__ = [
    "NormalizedToolEvent",
    "PreparedRuntimeRequest",
    "ProcessedRuntimeResponse",
    "RuntimeRequest",
    "RuntimeSession",
    "SemanticValidator",
    "UniversalTokRuntime",
    "_discover_project_markers",
    "_jit_context_matches",
    "apply_schema_adaptations",
    "build_tool_use_id_to_context",
    "calculate_invisible_pressure",
    "calculate_memory_lift",
    "calculate_semantic_regression_score",
    "collect_behavior_signals",
    "detect_prompt_bloat",
    "execute_jit_macro",
    "malformed_tok_signals",
    "normalize_tool_events",
    "parse_tok_response",
    "response_behavior_signals",
    "response_contract_for_mode",
    "translate_response_tools",
]
