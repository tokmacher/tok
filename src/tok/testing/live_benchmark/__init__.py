"""Live benchmark harness for comparing baseline vs Tok runtime behavior.

This package was split out of the previous `tok.testing.live_benchmark` god module.
Public symbols remain import-compatible for downstream callers and tests.
"""

from __future__ import annotations

# Re-exported dependencies that tests monkeypatch by module path.
from tok.gateway._bridge_runtime_pipeline import prepare_bridge_payload as prepare_bridge_payload  # noqa: F401
from tok.gateway._request_policy import default_request_policy as default_request_policy  # noqa: F401

from ._comparison import (  # noqa: F401
    _build_fairness_diagnostics,
    _diagnose_comparison,
    _extract_result_warnings,
    _message_normalization_path,
    compare_results,
    select_preferred_mode,
)
from ._definitions import DEFAULT_BENCHMARKS, load_benchmark_definition  # noqa: F401
from ._evaluation import (  # noqa: F401
    _evaluate_repo_grounded_research_success,
    _evaluate_task_success,
    _extract_labeled_fields,
    _extract_repo_candidates,
    _is_research_benchmark,
    _looks_like_placeholder,
    _message_shape_forensics,
    _repo_python_files,
    _repo_symbol_index,
    _resolve_repo_file,
)
from ._fixtures import (  # noqa: F401
    _chunk_messages,
    _flatten_message_content_for_provider,
    _provider_safe_chat_messages,
    _turn_prompts,
    load_fixture_messages,
    normalize_fixture_messages,
    normalize_fixture_messages_for_bridge,
)
from ._models import (  # noqa: F401
    BenchmarkComparison,
    BenchmarkDefinition,
    BenchmarkResult,
    ConversationTurnResult,
    ProviderUsageSnapshot,
)
from ._openai_tools import (  # noqa: F401
    _adapt_tool_results_for_openai,
    _build_openai_tools_param,
    _convert_openai_tool_calls,
    _detect_tool_protocol_retry_reason,
)
from ._prompting import _minimalize_system_prompt, _system_breakdown, _system_text  # noqa: F401
from ._rendering import (  # noqa: F401
    render_comparison_markdown,
    summarize_compare_runs,
    summarize_compare_triage,
    write_result,
)
from ._runner import LiveBenchmarkRunner  # noqa: F401
from ._stability import check_stability_artifacts, render_stability_markdown  # noqa: F401
from ._utils import (  # noqa: F401
    _content_text,
    _estimate_tokens,
    _success_term_matches,
    _sum_warning_signals,
    _system_to_messages,
)

__all__ = [
    "BenchmarkComparison",
    "BenchmarkDefinition",
    "BenchmarkResult",
    "ConversationTurnResult",
    "DEFAULT_BENCHMARKS",
    "LiveBenchmarkRunner",
    "ProviderUsageSnapshot",
    "check_stability_artifacts",
    "compare_results",
    "load_benchmark_definition",
    "load_fixture_messages",
    "normalize_fixture_messages",
    "normalize_fixture_messages_for_bridge",
    "prepare_bridge_payload",
    "default_request_policy",
    "render_comparison_markdown",
    "render_stability_markdown",
    "select_preferred_mode",
    "summarize_compare_runs",
    "summarize_compare_triage",
    "write_result",
]
