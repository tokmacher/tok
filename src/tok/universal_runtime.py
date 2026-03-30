"""Backward-compatible facade for the runtime module layout."""

from __future__ import annotations

from .runtime.core import (
    RuntimeSession,
    UniversalTokRuntime,
    apply_schema_adaptations,
)
from .runtime.memory.session_helpers import (
    _discover_project_markers,
)  # noqa: F401
from .runtime.pipeline.request_validation import (
    detect_prompt_bloat,
)  # noqa: F401
from .runtime.pipeline.tool_processing import (  # noqa: F401
    build_tool_use_id_to_context,
    collect_behavior_signals,
    normalize_tool_events,
)
from .runtime.pipeline.response_processing import (  # noqa: F401
    response_contract_for_mode,
    response_behavior_signals,
    translate_response_tools,
    parse_tok_response,
    malformed_tok_signals,
)
from .runtime.policy.macro_handling import (  # noqa: F401
    _jit_context_matches,
    execute_jit_macro,
)
from .runtime.policy.semantic_validation import (  # noqa: F401
    calculate_invisible_pressure,
    calculate_memory_lift,
    calculate_semantic_regression_score,
    SemanticValidator,
)
from .runtime.types import (  # noqa: F401
    NormalizedToolEvent,
    PreparedRuntimeRequest,
    ProcessedRuntimeResponse,
    RuntimeRequest,
)

__all__ = [
    "SemanticValidator",
    "calculate_invisible_pressure",
    "calculate_memory_lift",
    "calculate_semantic_regression_score",
    "response_contract_for_mode",
    "response_behavior_signals",
    "translate_response_tools",
    "malformed_tok_signals",
    "parse_tok_response",
    "normalize_tool_events",
    "detect_prompt_bloat",
    "_discover_project_markers",
    "build_tool_use_id_to_context",
    "collect_behavior_signals",
    "_jit_context_matches",
    "execute_jit_macro",
    "RuntimeSession",
    "UniversalTokRuntime",
    "RuntimeRequest",
    "PreparedRuntimeRequest",
    "ProcessedRuntimeResponse",
    "NormalizedToolEvent",
    "apply_schema_adaptations",
]
