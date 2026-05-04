"""Allow running gateway as `python -m tok.gateway`."""

from __future__ import annotations

import sys


def _exit_with_error(message: str) -> None:
    print(f"tok.gateway startup failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def _main() -> None:
    try:
        from tok.gateway import run_bridge
    except ImportError as exc:
        _exit_with_error(f"import error: {exc}")

    try:
        run_bridge()
    except OSError as exc:
        import errno

        if exc.errno == errno.EADDRINUSE:
            _exit_with_error(f"port {_resolve_port()} is already in use")
        elif exc.errno == errno.EACCES:
            _exit_with_error(f"permission denied binding port {_resolve_port()}")
        else:
            _exit_with_error(str(exc) or exc.__class__.__name__)
    except KeyboardInterrupt:
        raise SystemExit(0) from None
    except Exception as exc:
        _exit_with_error(str(exc) or exc.__class__.__name__)


def _resolve_port() -> str:
    import os

    return os.getenv("TOK_BRIDGE_PORT", os.getenv("TOK_PROXY_PORT", "9090"))


if __name__ == "__main__":
    _main()
