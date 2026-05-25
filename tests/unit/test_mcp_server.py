"""TDD tests for tok.mcp.server — MCP server module.

Sub-stages:
  1. Module exists and exports create_server, main
  2. _MCP_AVAILABLE flag reflects whether mcp package is installed
  3. create_server() returns a FastMCP instance when mcp is available
  4. main() raises RuntimeError clearly when mcp is not available
  5. bridge_status tool returns dict with bridge_running key
  6. audit tool runs dry-run compression and returns savings dict
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Stage 1: Module existence and exports
# ---------------------------------------------------------------------------


def test_mcp_server_module_importable() -> None:
    """tok.mcp.server must be importable."""
    import tok.mcp.server  # noqa: F401


def test_mcp_server_exports_create_server() -> None:
    from tok.mcp.server import create_server  # noqa: F401

    assert callable(create_server)


def test_mcp_server_exports_main() -> None:
    from tok.mcp.server import main  # noqa: F401

    assert callable(main)


def test_mcp_package_init_exports() -> None:
    from tok.mcp import create_server, main  # noqa: F401

    assert callable(create_server)
    assert callable(main)


# ---------------------------------------------------------------------------
# Stage 2: _MCP_AVAILABLE flag
# ---------------------------------------------------------------------------


def test_mcp_available_flag_is_bool() -> None:
    from tok.mcp.server import _MCP_AVAILABLE

    assert isinstance(_MCP_AVAILABLE, bool)


# ---------------------------------------------------------------------------
# Stage 3: create_server() when mcp IS available
# ---------------------------------------------------------------------------


def test_create_server_raises_when_mcp_missing() -> None:
    """When mcp is not installed, create_server() raises RuntimeError with install hint."""
    import tok.mcp.server as server_mod

    original = server_mod._MCP_AVAILABLE
    try:
        server_mod._MCP_AVAILABLE = False
        with pytest.raises(RuntimeError, match="pip install tok-protocol\\[mcp\\]"):
            server_mod.create_server()
    finally:
        server_mod._MCP_AVAILABLE = original


def test_create_server_returns_fastmcp_instance() -> None:
    """create_server() returns a FastMCP-like object with a run() method.

    Requires mcp>=1.0. It is listed in optional-dependencies.dev so this test
    must not be skipped in CI — if it fails here, add mcp to your dev install:
        pip install tok-protocol[mcp]  or  uv sync --extra dev
    """
    import importlib.util

    if importlib.util.find_spec("mcp") is None:
        pytest.fail("mcp package is not installed. Run: pip install tok-protocol[mcp]  or  uv sync --extra dev")
    from tok.mcp.server import create_server

    server = create_server()
    assert hasattr(server, "run"), "FastMCP server must have a run() method"


# ---------------------------------------------------------------------------
# Stage 4: main() raises when mcp unavailable
# ---------------------------------------------------------------------------


def test_main_raises_when_mcp_missing() -> None:
    import tok.mcp.server as server_mod

    original = server_mod._MCP_AVAILABLE
    try:
        server_mod._MCP_AVAILABLE = False
        with pytest.raises(RuntimeError, match="mcp package not installed"):
            server_mod.main()
    finally:
        server_mod._MCP_AVAILABLE = original


# ---------------------------------------------------------------------------
# Stage 5: bridge_status helper returns expected shape
# ---------------------------------------------------------------------------


def test_fetch_bridge_status_dict_not_running() -> None:
    """_fetch_bridge_status_dict() returns bridge_running=False when bridge is down."""
    import httpx

    from tok.mcp.server import _fetch_bridge_status_dict

    with patch("tok.cli._cli_support.get_bridge_health_response", side_effect=httpx.ConnectError("refused")):
        result = _fetch_bridge_status_dict()

    assert result.get("bridge_running") is False


def test_fetch_bridge_status_dict_running() -> None:
    """_fetch_bridge_status_dict() returns bridge_running=True when bridge responds."""
    from tok.mcp.server import _fetch_bridge_status_dict

    fake_health = {"port": 9090, "uptime_seconds": 42}
    with patch("tok.cli._cli_support.get_bridge_health_response", return_value=fake_health):
        result = _fetch_bridge_status_dict()

    assert result["bridge_running"] is True
    assert result["port"] == 9090


# ---------------------------------------------------------------------------
# Stage 6: audit helper runs dry-run compression
# ---------------------------------------------------------------------------


def test_fetch_audit_dict_returns_savings() -> None:
    """_fetch_audit_dict() returns dict with total_tokens_saved key."""
    from tok.mcp.server import _fetch_audit_dict

    messages = [
        {"role": "user", "content": "Hello world"},
        {"role": "assistant", "content": "Hi there"},
    ]
    result = _fetch_audit_dict(messages=messages, keep_turns=6)
    assert "total_tokens_saved" in result
    assert "total_tokens_before" in result
    assert "total_tokens_after" in result
    assert isinstance(result["total_tokens_saved"], int)


# ---------------------------------------------------------------------------
# Stage 7: session savings helper
# ---------------------------------------------------------------------------


def test_fetch_session_savings_dict_returns_dict() -> None:
    """_fetch_session_savings_dict() returns a dict (even if savings are 0)."""
    from tok.mcp.server import _fetch_session_savings_dict

    mock_summary = {"input_tokens_saved": 0, "output_tokens_saved": 0}
    with patch("tok.utils.savings_tracker.SavingsTracker.session_summary", return_value=mock_summary):
        result = _fetch_session_savings_dict()

    assert isinstance(result, dict)
