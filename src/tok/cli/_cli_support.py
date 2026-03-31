from __future__ import annotations

"""Shared CLI constants, display helpers, and process utilities."""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from collections.abc import Mapping

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


console = Console()

TOK_DIR = Path.home() / ".tok"
PID_FILE = TOK_DIR / "bridge.pid"
LOG_FILE = TOK_DIR / "bridge.log"
COLLECTOR_PID_FILE = TOK_DIR / "collector.pid"
COLLECTOR_LOG_FILE = TOK_DIR / "collector.log"

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
    host = os.getenv("TOK_BRIDGE_HOST", "localhost")
    if port is None:
        port = int(os.getenv("TOK_BRIDGE_PORT", "9090"))
    return f"http://{host}:{port}{path}"


def collector_url(path: str = "") -> str:
    host = os.getenv("TOK_COLLECTOR_HOST", "localhost")
    port = int(os.getenv("TOK_COLLECTOR_PORT", "8000"))
    return f"http://{host}:{port}{path}"


def msg_text(msg: dict[str, Any]) -> str:
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
    try:
        PID_FILE.unlink(missing_ok=True)
    except PermissionError:
        pass
    return None


def find_pids_on_port(port: int) -> list[int]:
    """Find PIDs listening on a specific port using lsof."""
    try:
        result = subprocess.run(
            ["lsof", "-i", f":{port}", "-t", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return [int(p) for p in result.stdout.strip().split() if p.strip()]
    except (subprocess.SubprocessError, ValueError):
        pass
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


def read_collector_pid() -> int | None:
    """Read Collector PID from file and validate it's alive."""
    if not COLLECTOR_PID_FILE.exists():
        return None
    try:
        pid = int(COLLECTOR_PID_FILE.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        pass
    try:
        COLLECTOR_PID_FILE.unlink(missing_ok=True)
    except PermissionError:
        pass
    return None


def start_collector(_debug: bool = False) -> None:
    """Start the telemetry collector in the background."""
    existing = read_collector_pid()
    if existing:
        return

    on_port = find_pids_on_port(8000)
    if on_port:
        COLLECTOR_PID_FILE.write_text(str(on_port[0]))
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.parent.parent)

    log_file = open(COLLECTOR_LOG_FILE, "a")

    try:
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "tok.collector.main:app",
            "--port",
            "8000",
        ]

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )
    finally:
        log_file.close()

    COLLECTOR_PID_FILE.write_text(str(proc.pid))

    for _ in range(10):
        time.sleep(0.2)
        try:
            import httpx

            r = httpx.get(collector_url("/health"), timeout=0.5)
            if r.status_code in (
                200,
                404,
            ):
                return
        except Exception:
            pass


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
    return Panel.fit(
        grid, title=title, border_style=border_style, padding=(0, 1)
    )


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
    reacq_count = int(payload.get("repeat_search_count", 0)) + int(
        payload.get("repeat_file_read_count", 0)
    )
    signal_map = (
        ("fallback", int(payload.get("fallback_count", 0))),
        ("drift", int(payload.get("semantic_drift_count", 0))),
        ("fail-open", int(payload.get("fail_open_count", 0))),
        ("reacq", reacq_count),
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
        token_text = (
            "No session savings recorded yet"
            if tokens_saved is None
            else f"{tokens_saved:,} tokens avoided"
        )
        return ("Saved $0.0000", f"{pct:.1f}% saved", token_text)

    savings_pct_val = summary["savings_pct"]
    cost_saved_val = summary["cost_saved_usd"]
    tokens_saved_val = summary["tokens_saved"]
    pct = (
        float(savings_pct_val)
        if isinstance(savings_pct_val, int | float | str)
        else 0.0
    )
    saved_usd = (
        float(cost_saved_val)
        if isinstance(cost_saved_val, int | float | str)
        else 0.0
    )
    tokens_saved = (
        int(tokens_saved_val)
        if isinstance(tokens_saved_val, int | float | str)
        else 0
    )
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
        session_quality=session_quality
        or (
            str(summary.get("session_quality", ""))
            if summary is not None
            else None
        ),
    )
    rows = [
        ("Verdict", verdict),
        (
            "Tok active",
            (
                "yes"
                if tok_active and mode != "baseline" and not baseline_only
                else "no"
            ),
        ),
        ("Degraded to baseline", "yes" if baseline_only else "no"),
    ]
    if mode is not None:
        rows.append(("Mode", mode))
    if session_quality or (
        summary is not None and summary.get("session_quality")
    ):
        rows.append(
            (
                "Session quality",
                str(
                    session_quality
                    or (
                        summary.get("session_quality")
                        if summary is not None
                        else ""
                    )
                ),
            )
        )
    if degradation_reason or (
        summary is not None and summary.get("last_degradation_reason")
    ):
        rows.append(
            (
                "Degradation reason",
                str(
                    degradation_reason
                    or (
                        summary.get("last_degradation_reason")
                        if summary is not None
                        else ""
                    )
                ),
            )
        )
    if session_signals is not None:
        rows.append(("Session signals", session_signals))
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
        fallback_count = (
            int(fallback_count_val)
            if isinstance(fallback_count_val, int | float | str)
            else 0
        )
    if fallback_count is not None:
        rows.append(("Fallbacks", str(fallback_count)))
    return rows


__all__ = [
    "bridge_url",
    "collector_url",
    "msg_text",
    "console",
    "TOK_DIR",
    "PID_FILE",
    "LOG_FILE",
    "COLLECTOR_PID_FILE",
    "COLLECTOR_LOG_FILE",
    "RUNTIME_WARNING_SIGNALS",
    "read_pid",
    "find_pids_on_port",
    "get_running_bridge_pid",
    "read_collector_pid",
    "start_collector",
    "memory_root",
    "savings_style",
    "render_stats_panel",
    "savings_verdict",
    "status_border",
    "runtime_verdict",
    "session_signals_text",
    "session_recommendation",
    "savings_headline",
    "session_status_rows",
]
