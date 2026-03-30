from typing import Any

from .profiler import TokProfiler as TokProfiler
from .profiler import UsageRecord as UsageRecord


def __getattr__(name: str) -> Any:
    if name in ("global_integrity_report", "main", "tok_health_check"):
        from .sentinel_dashboard import (
            global_integrity_report,
            main,
            tok_health_check,
        )

        ns = {
            "global_integrity_report": global_integrity_report,
            "main": main,
            "tok_health_check": tok_health_check,
        }
        return ns[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
