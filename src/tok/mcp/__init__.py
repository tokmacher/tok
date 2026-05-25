"""Tok MCP server — exposes bridge control and savings as MCP tools/resources.

Install: pip install tok-protocol[mcp]
Start:   tok-mcp
Setup:   tok mcp install
"""

import sys as _sys

from .server import _fetch_audit_dict, _fetch_bridge_status_dict, _fetch_session_savings_dict, create_server
from .server import main as _server_main


def main() -> None:
    """Console script entry point — prints a clean error if mcp extra is missing."""
    try:
        _server_main()
    except RuntimeError as exc:
        print(f"error: {exc}", file=_sys.stderr)
        _sys.exit(1)


__all__ = [
    "create_server",
    "main",
    "_fetch_bridge_status_dict",
    "_fetch_session_savings_dict",
    "_fetch_audit_dict",
]
