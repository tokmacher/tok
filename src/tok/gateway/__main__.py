"""Allow running gateway as `python -m tok.gateway`."""

import sys


def _main() -> None:
    try:
        from tok.gateway import run_bridge
    except ImportError as exc:
        print(
            f"tok: failed to import gateway dependencies: {exc}\n"
            "Ensure all required packages are installed: pip install tok[bridge]",
            file=sys.stderr,
        )
        raise SystemExit(1) from None

    try:
        run_bridge()
    except OSError as exc:
        import errno

        if exc.errno == errno.EADDRINUSE:
            port = _resolve_port()
            print(
                f"tok: port {port} is already in use.\n"
                "Run `tok bridge stop` to shut down the existing bridge, "
                "or use `--port` to select a different port.",
                file=sys.stderr,
            )
        elif exc.errno == errno.EACCES:
            print(
                f"tok: permission denied — {exc}\n"
                "Try using a port number above 1024.",
                file=sys.stderr,
            )
        else:
            print(f"tok: startup failed — {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        print("\ntok: bridge stopped.", file=sys.stderr)
        raise SystemExit(0)
    except Exception as exc:
        print(
            f"tok: unexpected startup failure — {exc}\n"
            "Run with `--debug` for more detail, or check logs with `tok bridge logs`.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


def _resolve_port() -> str:
    import os

    return os.getenv("TOK_BRIDGE_PORT", os.getenv("TOK_PROXY_PORT", "9090"))


if __name__ == "__main__":
    _main()
