"""Tok environment-variable configuration with strictness classification.

This module owns the canonical schema for all TOK_* environment variables:
severity, type, default, and validation. Call parse_config() to get a validated
TokConfig snapshot of the current environment.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("tok.config")

_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

# ---------------------------------------------------------------------------
# Schema: maps env-var name → {severity, type, default, description}
# Severity levels:
#   critical  — incorrect value can break the bridge or cause data loss
#   warning   — incorrect value degrades behaviour but bridge keeps running
#   info      — cosmetic / diagnostic tuning; no safety impact
# ---------------------------------------------------------------------------

CONFIG_SCHEMA: dict[str, dict[str, Any]] = {
    # --- critical: bridge safety / operation ---
    "TOK_TOOL_TIMEOUT": {
        "severity": "critical",
        "type": "int",
        "default": 120,
        "min": 1,
        "description": "Seconds before a tool call is timed out",
    },
    "TOK_BRIDGE_HOST": {
        "severity": "critical",
        "type": "str",
        "default": "localhost",
        "description": "Hostname the bridge listens on",
    },
    "TOK_PROJECT_DIR": {
        "severity": "critical",
        "type": "path",
        "default": "",
        "description": "Root directory for tok session files",
    },
    "TOK_DIR": {
        "severity": "critical",
        "type": "path",
        "default": "",
        "description": "Directory for tok state files; overrides TOK_PROJECT_DIR",
    },
    # --- warning: behavioural ---
    "TOK_LOOP_DETECTION_ENABLED": {
        "severity": "warning",
        "type": "bool",
        "default": True,
        "description": "Enable automatic repair-loop detection and breaking",
    },
    "TOK_MACRO_HEAL": {
        "severity": "warning",
        "type": "bool",
        "default": True,
        "description": "Enable macro healing for stuck sessions",
    },
    "TOK_NEURO_REACTOR": {
        "severity": "warning",
        "type": "bool",
        "default": True,
        "description": "Enable neuro reactor for JIT context injection",
    },
    "TOK_SAVINGS_FILE": {
        "severity": "warning",
        "type": "path",
        "default": "",
        "description": "Path to the per-session savings file",
    },
    "TOK_RESOLVER_ROOT": {
        "severity": "warning",
        "type": "path",
        "default": "",
        "description": "Root path for the file resolver cache",
    },
    "TOK_ENABLE_FILE_OVERLAP_DELTA": {
        "severity": "warning",
        "type": "bool",
        "default": True,
        "description": "Enable file-overlap delta compression",
    },
    "TOK_ENABLE_FILE_REREAD_DIFF": {
        "severity": "warning",
        "type": "bool",
        "default": True,
        "description": "Enable file re-read diff compression",
    },
    "TOK_ENABLE_JSON_NONEXPANSION_GUARD": {
        "severity": "warning",
        "type": "bool",
        "default": True,
        "description": "Guard against JSON compression expansion",
    },
    "TOK_ENABLE_SEARCH_OVERLAP_DELTA": {
        "severity": "warning",
        "type": "bool",
        "default": True,
        "description": "Enable search-result overlap delta compression",
    },
    "TOK_ENABLE_STACK_REPEAT_DELTA": {
        "severity": "warning",
        "type": "bool",
        "default": False,
        "description": "Enable stack-repeat delta compression",
    },
    "TOK_FORCE_FILE_CODEC": {
        "severity": "warning",
        "type": "bool",
        "default": False,
        "description": "Force file codec regardless of content type",
    },
    "TOK_SELF_BRIDGED_SESSION": {
        "severity": "warning",
        "type": "bool",
        "default": False,
        "description": "Marks session as self-bridged (Claude Code routing)",
    },
    "TOK_CAPTURE": {
        "severity": "warning",
        "type": "bool",
        "default": False,
        "description": "Capture bridge sessions to the tok sessions directory",
    },
    "TOK_ADAPTER": {
        "severity": "warning",
        "type": "str",
        "default": "claude",
        "description": "Runtime adapter selector; live bridge supports claude only in 0.2.x",
    },
    "TOK_UNSTABLE_ADAPTERS": {
        "severity": "warning",
        "type": "bool",
        "default": False,
        "description": "Allow fixture-only future adapter probes",
    },
    # --- info: diagnostics / telemetry ---
    "TOK_LOG_LEVEL": {
        "severity": "info",
        "type": "log_level",
        "default": "INFO",
        "description": "Python logging level for tok CLI",
    },
    "TOK_DEBUG": {
        "severity": "info",
        "type": "bool",
        "default": False,
        "description": "Enable verbose debug output",
    },
    "TOK_TRACE": {
        "severity": "info",
        "type": "bool",
        "default": False,
        "description": "Enable live trace output",
    },
    "TOK_TRACE_FILE": {
        "severity": "info",
        "type": "path",
        "default": "",
        "description": "Path for trace output file",
    },
    "TOK_TRACE_CAPTURE_ARTIFACTS": {
        "severity": "info",
        "type": "bool",
        "default": False,
        "description": "Capture trace artifacts to disk",
    },
    "TOK_COLLECTOR_HOST": {
        "severity": "info",
        "type": "str",
        "default": "localhost",
        "description": "Telemetry collector host",
    },
    "TOK_COLLECTOR_PORT": {
        "severity": "info",
        "type": "int",
        "default": 8000,
        "min": 1,
        "description": "Telemetry collector port",
    },
    "TOK_COLLECTOR_DB": {
        "severity": "info",
        "type": "path",
        "default": "telemetry.db",
        "description": "Path to the telemetry collector database",
    },
    "TOK_TELEMETRY_URL": {
        "severity": "info",
        "type": "str",
        "default": "",
        "description": "Override URL for the telemetry collector endpoint",
    },
    "TOK_ENABLE_PYTEST_FAIL_COMPRESSION": {
        "severity": "info",
        "type": "bool",
        "default": False,
        "description": "Enable pytest-failure compression (dev/testing only)",
    },
    "TOK_RESET_SESSION": {
        "severity": "info",
        "type": "bool",
        "default": False,
        "description": "Reset session state on bridge start",
    },
}


# ---------------------------------------------------------------------------
# TokConfig dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=True)
class TokConfig:
    # Critical
    tool_timeout: int = 120
    bridge_host: str = "localhost"
    project_dir: str = ""
    tok_dir: str = ""
    # Warning
    loop_detection_enabled: bool = True
    macro_heal: bool = True
    neuro_reactor: bool = True
    savings_file: str = ""
    resolver_root: str = ""
    enable_file_overlap_delta: bool = True
    enable_file_reread_diff: bool = True
    enable_json_nonexpansion_guard: bool = True
    enable_search_overlap_delta: bool = True
    enable_stack_repeat_delta: bool = False
    force_file_codec: bool = False
    self_bridged_session: bool = False
    capture: bool = False
    adapter: str = "claude"
    unstable_adapters: bool = False
    # Info
    log_level: str = "INFO"
    debug: bool = False
    trace: bool = False
    trace_file: str = ""
    trace_capture_artifacts: bool = False
    collector_host: str = "localhost"
    collector_port: int = 8000
    collector_db: str = "telemetry.db"
    telemetry_url: str = ""
    enable_pytest_fail_compression: bool = False
    reset_session: bool = False

    # Extra: store unknown keys for forward compatibility
    _extra: dict[str, str] = field(default_factory=dict, compare=False, hash=False)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(raw: str | None, default: int, min_val: int | None = None) -> int:
    if raw is None or raw.strip() == "":
        return default
    try:
        val = int(raw.strip())
    except ValueError:
        logger.warning("Invalid integer value %r; using default %d", raw, default)
        return default
    if min_val is not None and val < min_val:
        logger.warning("Value %d is below minimum %d; using default %d", val, min_val, default)
        return default
    return val


def _parse_log_level(raw: str | None, default: str) -> str:
    if raw is None:
        return default
    upper = raw.strip().upper()
    if upper not in _VALID_LOG_LEVELS:
        logger.warning("Unknown log level %r; using default %r", raw, default)
        return default
    return upper


def _parse_path(raw: str | None, default: str) -> str:
    if raw is None:
        return default
    stripped = raw.strip()
    return stripped if stripped else default


def parse_config() -> TokConfig:
    """Read TOK_* environment variables and return a validated TokConfig."""

    def _b(key: str, default: bool) -> bool:
        return _parse_bool(os.getenv(key), default)

    def _i(key: str, default: int, min_val: int | None = None) -> int:
        return _parse_int(os.getenv(key), default, min_val)

    def _s(key: str, default: str) -> str:
        raw = os.getenv(key)
        return raw if raw is not None else default

    def _p(key: str, default: str) -> str:
        return _parse_path(os.getenv(key), default)

    return TokConfig(
        # critical
        tool_timeout=_i("TOK_TOOL_TIMEOUT", 120, min_val=1),
        bridge_host=_s("TOK_BRIDGE_HOST", "localhost"),
        project_dir=_p("TOK_PROJECT_DIR", ""),
        tok_dir=_p("TOK_DIR", ""),
        # warning
        loop_detection_enabled=_b("TOK_LOOP_DETECTION_ENABLED", True),
        macro_heal=_b("TOK_MACRO_HEAL", True),
        neuro_reactor=_b("TOK_NEURO_REACTOR", True),
        savings_file=_p("TOK_SAVINGS_FILE", ""),
        resolver_root=_p("TOK_RESOLVER_ROOT", ""),
        enable_file_overlap_delta=_b("TOK_ENABLE_FILE_OVERLAP_DELTA", True),
        enable_file_reread_diff=_b("TOK_ENABLE_FILE_REREAD_DIFF", True),
        enable_json_nonexpansion_guard=_b("TOK_ENABLE_JSON_NONEXPANSION_GUARD", True),
        enable_search_overlap_delta=_b("TOK_ENABLE_SEARCH_OVERLAP_DELTA", True),
        enable_stack_repeat_delta=_b("TOK_ENABLE_STACK_REPEAT_DELTA", False),
        force_file_codec=_b("TOK_FORCE_FILE_CODEC", False),
        self_bridged_session=_b("TOK_SELF_BRIDGED_SESSION", False),
        capture=_b("TOK_CAPTURE", False),
        adapter=_s("TOK_ADAPTER", "claude").strip() or "claude",
        unstable_adapters=_b("TOK_UNSTABLE_ADAPTERS", False),
        # info
        log_level=_parse_log_level(os.getenv("TOK_LOG_LEVEL"), "INFO"),
        debug=_b("TOK_DEBUG", False),
        trace=_b("TOK_TRACE", False),
        trace_file=_p("TOK_TRACE_FILE", ""),
        trace_capture_artifacts=_b("TOK_TRACE_CAPTURE_ARTIFACTS", False),
        collector_host=_s("TOK_COLLECTOR_HOST", "localhost"),
        collector_port=_i("TOK_COLLECTOR_PORT", 8000, min_val=1),
        collector_db=_p("TOK_COLLECTOR_DB", "telemetry.db"),
        telemetry_url=_p("TOK_TELEMETRY_URL", ""),
        enable_pytest_fail_compression=_b("TOK_ENABLE_PYTEST_FAIL_COMPRESSION", False),
        reset_session=_b("TOK_RESET_SESSION", False),
    )


__all__ = ["CONFIG_SCHEMA", "TokConfig", "parse_config"]
