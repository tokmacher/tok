"""Tok MCP server — exposes bridge control and session savings as MCP tools/resources.

Install the mcp extra to use: pip install tok-protocol[mcp]
Run as stdio server:          tok-mcp  (or: python -m tok.mcp)
Register with Claude Code:    tok mcp install
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import TYPE_CHECKING, Any

try:
    from mcp.server.fastmcp import FastMCP

    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False
    if TYPE_CHECKING:
        from mcp.server.fastmcp import FastMCP


def _require_mcp() -> None:
    if not _MCP_AVAILABLE:
        raise RuntimeError("mcp package not installed; run: pip install tok-protocol[mcp]")


# ---------------------------------------------------------------------------
# Internal helpers (no mcp dependency — testable without mcp installed)
# ---------------------------------------------------------------------------


def _fetch_bridge_status_dict() -> dict[str, Any]:
    """Return bridge health dict. Never raises — returns bridge_running=False on failure."""
    import httpx

    from tok.cli._cli_support import get_bridge_health_response

    port = int(os.getenv("TOK_BRIDGE_PORT", "9090"))
    try:
        resp = get_bridge_health_response(port=port)
        return {"bridge_running": True, **resp}
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError, OSError):
        return {"bridge_running": False}


def _fetch_session_savings_dict() -> dict[str, Any]:
    """Return current session savings from SavingsTracker."""
    from tok.stats import SavingsTracker

    tracker = SavingsTracker()
    return tracker.session_summary() or {}


def _fetch_audit_dict(messages: list[dict[str, Any]], keep_turns: int = 6) -> dict[str, Any]:
    """Run dry-run compression audit and return savings breakdown."""
    from tok.compression._audit import audit_messages

    report = audit_messages(messages, keep_turns=keep_turns)
    return {
        "total_tokens_before": report.total_tokens_before,
        "total_tokens_after": report.total_tokens_after,
        "total_tokens_saved": report.total_tokens_saved,
        "type_breakdown": report.type_breakdown,
    }


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------


def create_server() -> FastMCP:
    """Build and return the Tok MCP server with all tools and resources registered."""
    _require_mcp()

    mcp = FastMCP("tok")

    @mcp.tool()
    def bridge_status() -> dict:
        """Return the current Tok bridge health, port, and session savings."""
        return _fetch_bridge_status_dict()

    @mcp.tool()
    def bridge_start(port: int = 9090) -> dict:
        """Start the Tok bridge server on the given port (1024–65535).

        Idempotent: if the bridge is already running and reachable, returns its
        current status without spawning a new process.
        """
        if not (1024 <= port <= 65535):
            return {"returncode": 1, "stdout": "", "stderr": f"Invalid port {port}: must be 1024–65535"}
        # Idempotency: check before spawning to avoid duplicate processes
        current = _fetch_bridge_status_dict()
        if current.get("bridge_running"):
            return {"returncode": 0, "already_running": True, "status": current}
        try:
            result = subprocess.run(
                [sys.executable, "-m", "tok", "bridge", "start", "--port", str(port)],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return {"returncode": 1, "stdout": "", "stderr": "bridge start timed out after 30 s"}
        return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}

    @mcp.tool()
    def bridge_stop() -> dict:
        """Stop the running Tok bridge server."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "tok", "bridge", "stop"],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return {"returncode": 1, "stdout": "", "stderr": "bridge stop timed out after 30 s"}
        return {"returncode": result.returncode, "stdout": result.stdout.strip()}

    @mcp.tool()
    def session_savings() -> dict:
        """Return current session token savings and cost delta."""
        return _fetch_session_savings_dict()

    @mcp.tool()
    def audit(messages: list[dict], keep_turns: int = 6) -> dict:
        """Dry-run Tok compression on a message list and return token savings breakdown."""
        return _fetch_audit_dict(messages=messages, keep_turns=keep_turns)

    @mcp.resource("tok://bridge/status")
    def bridge_status_resource() -> str:
        """Live bridge health snapshot (JSON)."""
        return json.dumps(_fetch_bridge_status_dict())

    @mcp.resource("tok://session/savings")
    def session_savings_resource() -> str:
        """Cumulative session token savings (JSON)."""
        return json.dumps(_fetch_session_savings_dict())

    return mcp


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the Tok MCP server using stdio transport (compatible with Claude Code)."""
    _require_mcp()
    server = create_server()
    server.run(transport="stdio")
