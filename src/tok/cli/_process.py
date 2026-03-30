"""Process management utilities for Tok CLI.

This module contains functions for managing Tok bridge and collector processes,
including PID file handling, port checking, and process lifecycle management.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

TOK_DIR = Path.home() / ".tok"
PID_FILE = TOK_DIR / "bridge.pid"
COLLECTOR_PID_FILE = TOK_DIR / "collector.pid"


def _read_pid() -> int | None:
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


def _find_pids_on_port(port: int) -> list[int]:
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


def _get_running_bridge_pid(port: int) -> int | None:
    """Get the running bridge PID, with fallback to port check and self-healing."""
    pid = _read_pid()
    if pid is not None:
        return pid

    on_port = _find_pids_on_port(port)
    if on_port:
        pid = on_port[0]
        TOK_DIR.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(str(pid))
        return pid

    return None


def _read_collector_pid() -> int | None:
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


def _start_collector(debug: bool = False) -> None:
    """Start telemetry collector in the background."""
    existing = _read_collector_pid()
    if existing:
        return

    on_port = _find_pids_on_port(8000)
    if on_port:
        COLLECTOR_PID_FILE.write_text(str(on_port[0]))
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.parent.parent)

    log_file = open(COLLECTOR_PID_FILE.parent / "collector.log", "a")

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
    COLLECTOR_PID_FILE.write_text(str(proc.pid))

    for _ in range(10):
        time.sleep(0.2)
        try:
            import httpx

            r = httpx.get("http://localhost:8000/health", timeout=0.5)
            if r.status_code in (200, 404):
                return
        except Exception:
            pass
