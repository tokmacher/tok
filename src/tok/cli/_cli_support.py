"""Shared CLI constants, display helpers, and process utilities."""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


console = Console()

TOK_DIR = Path.home() / ".tok"
PID_FILE = TOK_DIR / "bridge.pid"
LOG_FILE = TOK_DIR / "bridge.log"

RUNTIME_WARNING_SIGNALS = (
    "non_tok_response",
    "fail_open_compat_response",
    "malformed_tok_response",
    "malformed_tok_hybrid_tool",
    "malformed_tok_non_inverted_msg",
    "malformed_tok_markdown_fallback",
    "malformed_tok_bad_header",
)


def bridge_url(port: int | None = None, path: str = "") -> str:
    """Build the bridge server URL."""
    host = os.getenv("TOK_BRIDGE_HOST", "localhost")
    if port is None:
        port = int(os.getenv("TOK_BRIDGE_PORT", "9090"))
    return f"http://{host}:{port}{path}"


def collector_url(path: str = "") -> str:
    """Build the collector server URL."""
    host = os.getenv("TOK_COLLECTOR_HOST", "localhost")
    port = int(os.getenv("TOK_COLLECTOR_PORT", "8000"))
    return f"http://{host}:{port}{path}"


def msg_text(msg: dict[str, Any]) -> str:
    """Extract text content from a message dict."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", "") or block.get("content", ""))
        return " ".join(parts)
    return str(content)


def read_pid() -> int | None:
    """Read PID from file and validate it's alive."""
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        pass
    with contextlib.suppress(PermissionError):
        PID_FILE.unlink(missing_ok=True)
    return None


def validate_port(port: int) -> bool:
    """Validate port number is within valid range."""
    return isinstance(port, int) and 1 <= port <= 65535


def check_port_python(port: int) -> list[int]:
    """Check if port is in use using Python socket module (safer fallback)."""
    if not validate_port(port):
        return []

    pids: list[int] = []
    try:
        # Try to connect to the port to see if it's in use
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex(("localhost", port))
            if result == 0:
                # Port is in use, but we can't get PID without system calls
                # This is a safe fallback that doesn't use subprocess
                logger.debug(f"Port {port} is in use (detected via socket)")
                # Return empty list since we can't safely get PID without lsof
    except (OSError, ValueError) as e:
        logger.debug(f"Socket check failed for port {port}: {e}")

    return pids


def find_pids_on_port(port: int) -> list[int]:
    """Find PIDs listening on a specific port using safe methods."""
    if not validate_port(port):
        logger.warning(f"Invalid port number: {port}")
        return []

    # First try the Python-based safe method
    pids = check_port_python(port)
    if pids:
        return pids

    # Fallback to lsof with proper validation and safety measures
    try:
        # Validate the port parameter again before using in subprocess
        if not validate_port(port):
            return []

        # Use proper argument list to prevent injection
        cmd = ["lsof", "-i", f":{port}", "-t", "-sTCP:LISTEN"]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,  # Add timeout to prevent hanging
        )

        if result.returncode == 0 and result.stdout:
            # Validate output before processing
            output_lines = result.stdout.strip().split()
            pids = []
            for pid_str in output_lines:
                try:
                    pid = int(pid_str)
                    # Additional validation: PID should be reasonable
                    if 1 <= pid <= 999999:
                        pids.append(pid)
                except ValueError:
                    # Skip non-numeric output
                    continue
            return pids
        # Log non-zero exit codes for debugging
        if result.returncode != 0:
            logger.debug(f"lsof failed with exit code {result.returncode} for port {port}")

    except (subprocess.SubprocessError, ValueError, TypeError) as e:
        logger.debug(f"Subprocess error checking port {port}: {e}")
    except Exception as e:
        logger.debug(f"Unexpected error checking port {port}: {e}")

    return []


def get_running_bridge_pid(port: int) -> int | None:
    """Get the running bridge PID, with fallback to port check and self-healing."""
    pid = read_pid()
    if pid is not None:
        return pid

    on_port = find_pids_on_port(port)
    if on_port:
        pid = on_port[0]
        try:
            TOK_DIR.mkdir(parents=True, exist_ok=True)
            PID_FILE.write_text(str(pid))
        except Exception:
            pass
        return pid

    return None


def memory_root() -> Path:
    project_dir = os.getenv("TOK_PROJECT_DIR", "").strip()
    if project_dir:
        return Path(project_dir) / ".tok"
    return Path.home() / ".tok"


def savings_style(pct: float) -> str:
    if pct >= 40:
        return "bold green"
    if pct >= 15:
        return "bold yellow"
    return "bold red"


def render_stats_panel(
    title: str,
    *,
    headline: str,
    headline_style: str,
    subhead: str,
    rows: list[tuple[str, str]],
    border_style: str,
) -> Panel:
    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column(justify="left", ratio=1)
    grid.add_column(justify="right")
    grid.add_row(f"[{headline_style}]{headline}[/{headline_style}]", "")
    grid.add_row(f"[dim]{subhead}[/dim]", "")
    for label, value in rows:
        grid.add_row(f"[bold]{label}[/bold]", value)
    return Panel.fit(grid, title=title, border_style=border_style, padding=(0, 1))


def savings_verdict(pct: float) -> str:
    if pct >= 40:
        return "Strong savings"
    if pct >= 15:
        return "Solid savings"
    if pct > 0:
        return "Light savings"
    return "No visible savings"


def status_border(verdict_style: str) -> str:
    if "green" in verdict_style:
        return "green"
    if "yellow" in verdict_style:
        return "yellow"
    return "red"


def runtime_verdict(
    *,
    tok_active: bool,
    baseline_only: bool,
    mode: str | None = None,
    tokens_saved: int = 0,
    session_quality: str | None = None,
) -> tuple[str, str]:
    if baseline_only:
        return ("Session degraded to baseline", "bold yellow")
    if not tok_active:
        return ("Tok inactive", "bold red")
    if mode == "baseline":
        return ("Bridge running in baseline mode", "bold yellow")
    if session_quality == "watch":
        return ("Tok active, watch session", "bold yellow")
    if session_quality == "degraded":
        return ("Session degraded to baseline", "bold yellow")
    if tokens_saved > 0:
        return ("Tok active and helping", "bold green")
    return ("Tok active, waiting for first savings", "bold yellow")


def session_signals_text(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    reacq_count = int(payload.get("repeat_search_count", 0)) + int(payload.get("repeat_file_read_count", 0))
    signal_map = (
        ("fallback", int(payload.get("fallback_count", 0))),
        ("drift", int(payload.get("semantic_drift_count", 0))),
        ("compat-fallback", int(payload.get("fail_open_count", 0))),
        ("reacq", reacq_count),
        (
            "shape-orig",
            int(payload.get("preflight_block_original_payload_count", 0)),
        ),
        (
            "shape-rewrite",
            int(payload.get("preflight_block_rewritten_payload_count", 0)),
        ),
        (
            "stream-empty",
            int(payload.get("stream_recovery_empty_success_count", 0)),
        ),
        (
            "stream-read",
            int(payload.get("stream_recovery_read_error_count", 0)),
        ),
        (
            "held",
            int(payload.get("request_policy_held_by_recovery_count", 0)),
        ),
    )
    for label, value in signal_map:
        if value > 0:
            parts.append(f"{label}={value}")
    return "clean" if not parts else ", ".join(parts)


def session_recommendation(
    *,
    baseline_only: bool,
    session_quality: str | None,
) -> str:
    if baseline_only or session_quality == "degraded":
        return "Recommendation: investigate degradation before trusting this session"
    if session_quality == "watch":
        return "Recommendation: keep Tok on, but watch this session"
    return "Recommendation: keep Tok on"


def savings_headline(
    summary: Mapping[str, Any] | None,
    *,
    savings_pct: float | None = None,
    tokens_saved: int | None = None,
) -> tuple[str, str, str]:
    if summary is None:
        pct = 0.0 if savings_pct is None else savings_pct
        token_text = "No session savings recorded yet" if tokens_saved is None else f"{tokens_saved:,} tokens avoided"
        return ("Saved $0.0000", f"{pct:.1f}% saved", token_text)

    savings_pct_val = summary["savings_pct"]
    cost_saved_val = summary["cost_saved_usd"]
    tokens_saved_val = summary["tokens_saved"]
    pct = float(savings_pct_val) if isinstance(savings_pct_val, int | float | str) else 0.0
    saved_usd = float(cost_saved_val) if isinstance(cost_saved_val, int | float | str) else 0.0
    tokens_saved = int(tokens_saved_val) if isinstance(tokens_saved_val, int | float | str) else 0
    return (
        f"Saved ${saved_usd:.4f}",
        f"{pct:.1f}% saved",
        f"{savings_verdict(pct)} • {tokens_saved:,} tokens avoided",
    )


def session_status_rows(
    *,
    summary: Mapping[str, Any] | None,
    tok_active: bool,
    baseline_only: bool,
    mode: str | None = None,
    request_policy: str | None = None,
    fallback_count: int | None = None,
    session_quality: str | None = None,
    degradation_reason: str | None = None,
    session_signals: str | None = None,
) -> list[tuple[str, str]]:
    tokens_saved = 0 if summary is None else int(summary["tokens_saved"])
    verdict, _ = runtime_verdict(
        tok_active=tok_active,
        baseline_only=baseline_only,
        mode=mode,
        tokens_saved=tokens_saved,
        session_quality=session_quality or (str(summary.get("session_quality", "")) if summary is not None else None),
    )
    rows = [
        ("Verdict", verdict),
        (
            "Tok active",
            ("yes" if tok_active and mode != "baseline" and not baseline_only else "no"),
        ),
        ("Degraded to baseline", "yes" if baseline_only else "no"),
    ]
    if mode is not None:
        rows.append(("Mode", mode))
    if request_policy is not None:
        rows.append(("Request policy", request_policy))
    if session_quality or (summary is not None and summary.get("session_quality")):
        rows.append(
            (
                "Session quality",
                str(session_quality or (summary.get("session_quality") if summary is not None else "")),
            )
        )
    if degradation_reason or (summary is not None and summary.get("last_degradation_reason")):
        rows.append(
            (
                "Degradation reason",
                str(degradation_reason or (summary.get("last_degradation_reason") if summary is not None else "")),
            )
        )
    if session_signals is not None:
        rows.append(("Session signals", session_signals))
    if summary is not None:
        request_shape_blocks = int(summary.get("preflight_block_original_payload_count", 0)) + int(
            summary.get("preflight_block_rewritten_payload_count", 0)
        )
        if request_shape_blocks > 0:
            rows.append(
                (
                    "Request-shape blocks",
                    f"{int(summary.get('preflight_block_original_payload_count', 0))} original, {int(summary.get('preflight_block_rewritten_payload_count', 0))} rewritten",
                )
            )
        stream_transport_recoveries = int(summary.get("stream_recovery_empty_success_count", 0)) + int(
            summary.get("stream_recovery_read_error_count", 0)
        )
        if stream_transport_recoveries > 0:
            rows.append(
                (
                    "Stream transport incidents (per-call)",
                    f"{int(summary.get('stream_recovery_empty_success_count', 0))} empty, {int(summary.get('stream_recovery_read_error_count', 0))} read-error",
                )
            )
        held_recovery = int(summary.get("request_policy_held_by_recovery_count", 0))
        if held_recovery > 0:
            rows.append(("Recovery holdovers", str(held_recovery)))
    if summary is not None:
        rows.extend(
            [
                (
                    "With Tok vs without Tok",
                    f"{int(summary['actual_tokens']) if isinstance(summary.get('actual_tokens'), int | float | str) else 0:,} / {int(summary['baseline_tokens']) if isinstance(summary.get('baseline_tokens'), int | float | str) else 0:,} tokens",
                ),
                (
                    "Cost",
                    f"${float(summary['actual_cost_usd']) if isinstance(summary.get('actual_cost_usd'), int | float | str) else 0.0:.4f} / ${float(summary['baseline_cost_usd']) if isinstance(summary.get('baseline_cost_usd'), int | float | str) else 0.0:.4f}",
                ),
            ]
        )
    if fallback_count is None and summary is not None:
        fallback_count_val = summary["fallback_count"]
        fallback_count = int(fallback_count_val) if isinstance(fallback_count_val, int | float | str) else 0
    if fallback_count is not None:
        rows.append(("Fallbacks", str(fallback_count)))
    return rows


def interaction_quality_rows(
    *,
    smoothness_score: int | None = None,
    labour_index: int | None = None,
    current_mode: str | None = None,
    stream_instability_events: int | None = None,
    thinking_mutation_events: int | None = None,
    repeated_active_file_reads: int | None = None,
    task_score: int | None = None,
) -> list[tuple[str, str]]:
    rows = []
    if smoothness_score is not None:
        rows.append(("Smoothness score", str(smoothness_score)))
    if labour_index is not None:
        rows.append(("Labour index", str(labour_index)))
    if current_mode is not None:
        rows.append(("Current mode", current_mode))
    if stream_instability_events is not None:
        rows.append(
            (
                "Stream instability events (per-turn)",
                str(stream_instability_events),
            )
        )
    if thinking_mutation_events is not None:
        rows.append(("Thinking mutation events", str(thinking_mutation_events)))
    if repeated_active_file_reads is not None:
        rows.append(("Repeated active-file reads", str(repeated_active_file_reads)))
    if task_score is not None:
        rows.append(("Task score", str(task_score)))
    return rows


__all__ = [
    "LOG_FILE",
    "PID_FILE",
    "RUNTIME_WARNING_SIGNALS",
    "TOK_DIR",
    "bridge_url",
    "console",
    "find_pids_on_port",
    "get_running_bridge_pid",
    "interaction_quality_rows",
    "memory_root",
    "msg_text",
    "read_pid",
    "render_stats_panel",
    "runtime_verdict",
    "savings_headline",
    "savings_style",
    "savings_verdict",
    "session_recommendation",
    "session_signals_text",
    "session_status_rows",
    "status_border",
]
