"""Tok Bridge — backwards-compatible entry point.

Delegates to tok.gateway. Use `tok bridge start` instead.
"""

from tok.gateway import run_bridge

if __name__ == "__main__":
    run_bridge()
