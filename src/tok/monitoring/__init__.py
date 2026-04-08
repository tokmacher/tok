"""
Tok monitoring and profiling utilities.

This module provides performance monitoring, profiling, and health checking
functionality for the Tok system. It includes tools for tracking usage,
analyzing performance metrics, and monitoring system integrity.
"""

from .profiler import TokProfiler as TokProfiler
from .profiler import UsageRecord as UsageRecord


def __getattr__(name: str) -> object:
    """
    Lazy import of monitoring functions.

    Args:
        name: Name of the attribute to import.

    Returns:
        The requested attribute from sentinel_dashboard.

    Raises:
        AttributeError: If the attribute is not found.

    """
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
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
