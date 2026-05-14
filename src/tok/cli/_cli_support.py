"""Shared CLI constants, display helpers, and process utilities."""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


console = Console()

# Defaults are evaluated at import time but can be monkeypatched in tests.
# The bridge PID/log location should be stable across terminals; use TOK_DIR as
# the explicit override rather than implicitly depending on TOK_PROJECT_DIR.
TOK_DIR = Path(os.getenv("TOK_DIR", str(Path.home() / ".tok")))
PID_FILE = TOK_DIR / "bridge.pid"
LOG_FILE = TOK_DIR / "bridge.log"


def tok_dir() -> Path:
    return PID_FILE.parent


def pid_file() -> Path:
    return PID_FILE


def log_file() -> Path:
    return LOG_FILE


_LOOPBACK_HOST_ALIASES = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}

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
        port = env_int("TOK_BRIDGE_PORT", 9090)
    return f"http://{host}:{port}{path}"


def env_int(name: str, fallback: int) -> int:
    """Read an integer environment variable, warning and falling back when malformed."""
    raw = os.getenv(name)
    if raw is None:
        return fallback
    try:
        return int(raw)
    except ValueError:
        console.print(f"[yellow]Invalid integer config {name}={raw!r}; using fallback {fallback}.[/yellow]")
        return fallback


def _normalize_host(host: str) -> str:
    return host.strip().lower().strip("[]")


def _format_host_for_url(host: str) -> str:
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def bridge_health_urls(port: int | None = None, path: str = "/health") -> list[str]:
    """Build bridge health probe URLs with loopback fallbacks for local hosts."""
    configured_host = os.getenv("TOK_BRIDGE_HOST", "localhost").strip() or "localhost"
    if port is None:
        port = env_int("TOK_BRIDGE_PORT", 9090)

    if _normalize_host(configured_host) in _LOOPBACK_HOST_ALIASES:
        host_candidates = ["127.0.0.1", "localhost", "::1"]
    else:
        host_candidates = [configured_host]

    urls: list[str] = []
    seen: set[str] = set()
    for host in host_candidates:
        key = _normalize_host(host)
        if key in seen:
            continue
        seen.add(key)
        urls.append(f"http://{_format_host_for_url(host)}:{port}{path}")
    return urls


def get_bridge_health_response(
    port: int | None = None,
    *,
    timeout: float = 2.0,
    attempts: int = 2,
    backoff_seconds: float = 0.15,
):
    """Probe bridge health robustly and return the last response or raise connection errors."""
    import httpx

    attempts = max(1, attempts)
    urls = bridge_health_urls(port=port, path="/health")
    last_response = None
    last_exception: Exception | None = None

    for attempt in range(attempts):
        for url in urls:
            try:
                response = httpx.get(url, timeout=timeout)
                if response.status_code == 200:
                    return response
                last_response = response
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_exception = exc
        if attempt + 1 < attempts and backoff_seconds > 0:
            time.sleep(backoff_seconds)

    if last_response is not None:
        return last_response
    if last_exception is not None:
        raise last_exception
    raise RuntimeError("bridge health probe failed without response")


def collector_url(path: str = "") -> str:
    """Build the collector server URL."""
    host = os.getenv("TOK_COLLECTOR_HOST", "localhost")
    port = env_int("TOK_COLLECTOR_PORT", 8000)
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
    path = pid_file()
    if not path.exists():
        return None
    try:
        pid = int(path.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        pass
    with contextlib.suppress(PermissionError):
        path.unlink(missing_ok=True)
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
    # When TOK_PROJECT_DIR is set but TOK_DIR is not, treat the bridge as
    # project-scoped for diagnostics and avoid binding to a global pidfile.
    if os.getenv("TOK_PROJECT_DIR", "").strip() and not os.getenv("TOK_DIR", "").strip():
        return None

    pid = read_pid()
    if pid is not None:
        return pid

    # If the pidfile is missing, only trust a port scan if we can confirm the
    # listener is actually the Tok bridge (via /health). This avoids false
    # positives from unrelated processes that happen to bind the same port.
    try:
        resp = get_bridge_health_response(port, timeout=0.35, attempts=1, backoff_seconds=0)
    except Exception:
        return None
    if resp.status_code != 200:
        return None
    try:
        payload = resp.json()
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("bridge") != "tok":
        return None

    on_port = find_pids_on_port(port)
    if len(on_port) != 1:
        return None
    pid = on_port[0]
    try:
        tok_dir().mkdir(parents=True, exist_ok=True)
        pid_file().write_text(str(pid))
    except Exception:
        pass
    return pid


def memory_root() -> Path:
    project_dir = os.getenv("TOK_PROJECT_DIR", "").strip()
    if project_dir:
        return Path(project_dir) / ".tok"
    return Path.home() / ".tok"


def savings_style(pct: float) -> str:
    """Return rich markup style based on savings percentage.

    Green (>= 40%): Strong savings
    Yellow (15-39%): Solid savings
    Red (< 15%): Light or no savings
    """
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
    grid.add_column(justify="left", ratio=1, overflow="fold")
    grid.add_column(justify="right", overflow="fold")
    grid.add_row(f"[{headline_style}]{headline}[/{headline_style}]", "")
    grid.add_row(f"[dim]{subhead}[/dim]", "")
    for label, value in rows:
        grid.add_row(f"[bold]{label}[/bold]", value)
    return Panel.fit(grid, title=title, border_style=border_style, padding=(0, 1))


def savings_verdict(pct: float) -> str:
    """Return human-readable verdict for savings percentage.

    Strong (>= 40%): Substantial compression benefits.
    Solid (15-39%): Meaningful compression benefits.
    Light (0-14%): Minimal but present compression.
    None (<= 0%): No visible token savings.
    """
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
    exact_count = int(payload.get("evidence_exact_observed_count", payload.get("evidence_exact_observed", 0)))
    nonexact_count = int(
        payload.get("evidence_non_exact_reference_count", payload.get("evidence_non_exact_reference_emitted", 0))
    )
    reacq_required_count = int(
        payload.get(
            "evidence_exact_reacquisition_required_count",
            payload.get("evidence_exact_reacquisition_required", 0),
        )
    )
    reacq_satisfied_count = int(
        payload.get(
            "evidence_exact_reacquisition_satisfied_count",
            payload.get("evidence_exact_reacquisition_satisfied", 0),
        )
    )
    compression_blocked_count = int(
        payload.get(
            "evidence_compression_blocked_for_safety_count",
            payload.get("evidence_compression_blocked_for_safety", 0),
        )
    )
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
        ("exact", exact_count),
        ("nonexact", nonexact_count),
    )
    for label, value in signal_map:
        if value > 0:
            parts.append(f"{label}={value}")
    if reacq_required_count > 0 or reacq_satisfied_count > 0:
        parts.append(f"reacq-safe={reacq_satisfied_count}/{reacq_required_count}")
    if compression_blocked_count > 0:
        parts.append(f"safe-block={compression_blocked_count}")
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


def savings_diagnostic_note(
    *,
    summary: Mapping[str, Any] | None,
    baseline_only: bool,
    mode: str | None = None,
    fallback_count: int | None = None,
) -> str | None:
    """Explain low savings without changing the bridge contract."""
    if summary is None:
        return None
    tokens_saved = int(summary.get("tokens_saved", 0))
    savings_pct = float(summary.get("savings_pct", 0.0))
    calls = int(summary.get("calls", 0))
    fallback_total = fallback_count
    if fallback_total is None:
        fallback_total = int(summary.get("fallback_count", 0))
    safe_blocks = int(summary.get("evidence_compression_blocked_for_safety_count", 0))
    repeated_reads = int(summary.get("repeat_file_read_count", 0)) + int(summary.get("repeat_search_count", 0))

    if tokens_saved > 0 and savings_pct >= 5:
        return None
    if baseline_only:
        return "Session fell back to baseline for safety; inspect doctor/logs before judging savings."
    if mode == "baseline":
        return "Bridge is intentionally in baseline mode, so compression is disabled."
    if fallback_total > 0:
        return "Request-level fallback protected fidelity; inspect doctor/logs before judging savings."
    if safe_blocks > 0 and tokens_saved <= 0:
        return "Tok blocked compression for evidence safety; exactness won over token savings."
    if tokens_saved <= 0 or savings_pct <= 0:
        if calls < 3:
            return "Very short sessions often show no savings; recheck after sustained Claude Code work."
        if repeated_reads <= 0:
            return "Savings usually require repeated history, file reads, searches, or tool output."
        return "Repeated context exists, but Tok may be waiting for exact evidence before compressing."
    if 0 < savings_pct < 5:
        return "Light savings can be normal with provider caching, little repetition, or safety blocks."
    return None


def _redact_api_base(value: str) -> str:
    try:
        parts = urlsplit(value)
    except ValueError:
        return value
    if not parts.scheme or not parts.netloc:
        return value

    host = parts.hostname or ""
    if not host:
        return value
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    if parts.username or parts.password:
        host = f"<redacted>@{host}"
    query = "<redacted>" if parts.query else ""
    fragment = "<redacted>" if parts.fragment else ""
    return urlunsplit((parts.scheme, host, parts.path, query, fragment))


def savings_headline(
    summary: Mapping[str, Any] | None,
    *,
    savings_pct: float | None = None,
    tokens_saved: int | None = None,
) -> tuple[str, str, str]:
    if summary is None:
        pct = 0.0 if savings_pct is None else savings_pct
        ts = 0 if tokens_saved is None else tokens_saved
        cost_pct = pct
        return (
            f"Saved {ts:,} tokens • $0.0000 saved",
            f"{pct:.1f}% token savings • {cost_pct:.1f}% cost savings",
            savings_verdict(cost_pct),
        )

    savings_pct_val = summary["savings_pct"]
    cost_savings_pct_val = summary.get("cost_savings_pct", summary["savings_pct"])
    peak_pct_val = summary.get("peak_savings_pct", savings_pct_val)
    cost_saved_val = summary["cost_saved_usd"]
    tokens_saved_val = summary["tokens_saved"]
    pct = float(savings_pct_val) if isinstance(savings_pct_val, int | float | str) else 0.0
    peak_pct = float(peak_pct_val) if isinstance(peak_pct_val, int | float | str) else pct
    cost_pct = float(cost_savings_pct_val) if isinstance(cost_savings_pct_val, int | float | str) else 0.0
    saved_usd = float(cost_saved_val) if isinstance(cost_saved_val, int | float | str) else 0.0
    tokens_saved = int(tokens_saved_val) if isinstance(tokens_saved_val, int | float | str) else 0
    pct_label = f"{pct:.1f}% token savings • {cost_pct:.1f}% cost savings"
    if peak_pct > pct + 5:
        pct_label = f"{pct:.1f}% token savings (peak {peak_pct:.0f}%) • {cost_pct:.1f}% cost savings"
    return (
        f"Saved {tokens_saved:,} tokens • ${saved_usd:.4f} saved",
        pct_label,
        savings_verdict(peak_pct),
    )


def session_status_rows(
    *,
    summary: Mapping[str, Any] | None,
    tok_active: bool,
    baseline_only: bool,
    mode: str | None = None,
    request_policy: str | None = None,
    api_base: str | None = None,
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
    if api_base:
        rows.append(("API base", _redact_api_base(api_base)))
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
        exact_count = int(summary.get("evidence_exact_observed_count", 0))
        nonexact_count = int(summary.get("evidence_non_exact_reference_count", 0))
        summary_count = int(summary.get("evidence_non_exact_summary_count", 0))
        skeleton_count = int(summary.get("evidence_non_exact_skeleton_count", 0))
        if exact_count > 0 or nonexact_count > 0 or summary_count > 0 or skeleton_count > 0:
            rows.append(
                (
                    "Evidence safety",
                    f"exact={exact_count}, non-exact={nonexact_count}, summaries={summary_count}, skeletons={skeleton_count}",
                )
            )
        reacq_required_count = int(summary.get("evidence_exact_reacquisition_required_count", 0))
        reacq_satisfied_count = int(summary.get("evidence_exact_reacquisition_satisfied_count", 0))
        if reacq_required_count > 0 or reacq_satisfied_count > 0:
            rows.append(
                (
                    "Exact reacquisition",
                    f"required={reacq_required_count}, satisfied={reacq_satisfied_count}",
                )
            )
        compression_blocked_count = int(summary.get("evidence_compression_blocked_for_safety_count", 0))
        if compression_blocked_count > 0:
            rows.append(("Compression safety blocks", str(compression_blocked_count)))
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
                    "Tokens (with Tok / est. no Tok)",
                    f"{int(summary['actual_tokens']) if isinstance(summary.get('actual_tokens'), int | float | str) else 0:,} / {int(summary['baseline_tokens']) if isinstance(summary.get('baseline_tokens'), int | float | str) else 0:,}",
                ),
                (
                    "Cost (with Tok / est. no Tok)",
                    f"${float(summary['actual_cost_usd']) if isinstance(summary.get('actual_cost_usd'), int | float | str) else 0.0:.4f} / ${float(summary['baseline_cost_usd']) if isinstance(summary.get('baseline_cost_usd'), int | float | str) else 0.0:.4f}",
                ),
                (
                    "Cost saved",
                    f"${float(summary['cost_saved_usd']) if isinstance(summary.get('cost_saved_usd'), int | float | str) else 0.0:.4f} ({float(summary['cost_savings_pct']) if isinstance(summary.get('cost_savings_pct'), int | float | str) else 0.0:.1f}%)",
                ),
            ]
        )
    note = savings_diagnostic_note(
        summary=summary,
        baseline_only=baseline_only,
        mode=mode,
        fallback_count=fallback_count,
    )
    if note:
        rows.append(("Savings note", note))
    if fallback_count is None and summary is not None:
        fallback_count_val = summary["fallback_count"]
        fallback_count = int(fallback_count_val) if isinstance(fallback_count_val, int | float | str) else 0
    if fallback_count is not None:
        rows.append(("Fallbacks", str(fallback_count)))
    return rows


def format_savings_line(
    *,
    pct: float,
    actual: float,
    baseline: float,
    is_cost: bool,
) -> tuple[str, str]:
    pct_str = f"{pct:.1f}% less"
    if pct >= 40:
        pct_str = f"[green]{pct_str}[/green]"
    elif pct >= 15:
        pct_str = f"[yellow]{pct_str}[/yellow]"
    else:
        pct_str = f"[red]{pct_str}[/red]"

    if is_cost:
        actual_str = f"${actual:.2f}"
        baseline_str = f"${baseline:.2f}"
    else:
        actual_str = _format_token_count(int(actual))
        baseline_str = _format_token_count(int(baseline))

    subline = f"[dim]{actual_str} with Tok vs {baseline_str} base[/dim]"
    return pct_str, subline


def _format_token_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return f"{n:,}"


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


def reliability_line(
    *,
    smoothness_score: int | None,
    fallback_count: int,
    calls: int,
) -> str:
    smoothness_str = "N/A"
    if smoothness_score is not None:
        smoothness_str = f"{smoothness_score}/100"

    parts = [f"[bold]{smoothness_str}[/bold] smoothness"]
    if fallback_count > 0:
        parts.append(f"[yellow]{fallback_count} fallbacks[/yellow]")
    else:
        parts.append("[green]0 fallbacks[/green]")
    parts.append(f"{calls} calls handled")
    return " · ".join(parts)


def status_sentence(
    *,
    tok_active: bool,
    baseline_only: bool,
    fallback_count: int,
    calls: int = 1,
) -> str:
    if baseline_only:
        return "Tok has degraded to baseline for this session."
    if not tok_active:
        return "Tok is not active for this session."
    if calls <= 0:
        return "Tok is active. No completed calls recorded for this session yet."
    if fallback_count > 0:
        return "Tok is active, with fallback events recorded this session."
    return "Tok is active and handling this session normally."


def json_envelope(
    command: str,
    *,
    ok: bool,
    status: str,
    data: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    next_steps: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "tok-cli-result/v0.1",
        "command": command,
        "ok": ok,
        "status": status,
        "data": data or {},
        "warnings": warnings or [],
        "next_steps": next_steps or [],
    }


__all__ = [
    "RUNTIME_WARNING_SIGNALS",
    "bridge_url",
    "console",
    "find_pids_on_port",
    "format_savings_line",
    "get_running_bridge_pid",
    "interaction_quality_rows",
    "json_envelope",
    "memory_root",
    "msg_text",
    "read_pid",
    "reliability_line",
    "render_stats_panel",
    "runtime_verdict",
    "savings_headline",
    "savings_diagnostic_note",
    "savings_style",
    "savings_verdict",
    "session_recommendation",
    "session_signals_text",
    "session_status_rows",
    "status_border",
    "status_sentence",
    "log_file",
    "pid_file",
    "tok_dir",
]
