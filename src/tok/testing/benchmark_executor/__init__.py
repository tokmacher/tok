"""Full executor for production benchmark catalog tasks.

This package was split out of the previous monolithic
``tok.testing.benchmark_executor`` module.  Public symbols remain
import-compatible for downstream callers and tests.
"""

from __future__ import annotations

from ._evaluator import FamilyEvaluator  # noqa: F401
from ._loop_executor import ToolLoopExecutor  # noqa: F401
from ._materializer import TaskMaterializer  # noqa: F401
from ._models import (  # noqa: F401
    ASSET_LOCK_FILENAME,
    DEFAULT_EVALUATOR_BUNDLE_DIR,
    BenchmarkTaskRunResult,
    CatalogBenchmarkRun,
    MaterializedBenchmarkTask,
    ShellCommandResult,
    TaskEvaluationResult,
    ToolExecutionRecord,
)
from ._orchestrator import (  # noqa: F401
    _execution_failure_code,
    render_combined_benchmark_summary,
    run_catalog_benchmark_suite,
    select_catalog_tasks,
)
from ._tool_executor import BenchmarkToolExecutor  # noqa: F401
from ._utils import (  # noqa: F401
    _directory_sha256,
    _extract_text_tool_calls,
    _normalize_pytest_command,
)

__all__ = [
    "ASSET_LOCK_FILENAME",
    "DEFAULT_EVALUATOR_BUNDLE_DIR",
    "BenchmarkTaskRunResult",
    "BenchmarkToolExecutor",
    "CatalogBenchmarkRun",
    "FamilyEvaluator",
    "MaterializedBenchmarkTask",
    "ShellCommandResult",
    "TaskEvaluationResult",
    "TaskMaterializer",
    "ToolExecutionRecord",
    "ToolLoopExecutor",
    "render_combined_benchmark_summary",
    "run_catalog_benchmark_suite",
    "select_catalog_tasks",
]
