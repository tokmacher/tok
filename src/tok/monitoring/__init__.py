"""
Tok monitoring and profiling utilities.

This module provides performance monitoring, profiling, and health checking
functionality for the Tok system. It includes tools for tracking usage,
analyzing performance metrics, and monitoring system integrity.
"""

from .profiler import TokProfiler as TokProfiler
from .profiler import UsageRecord as UsageRecord


def __getattr__(name: str) -> object:
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
