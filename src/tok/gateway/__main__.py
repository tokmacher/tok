"""Allow running gateway as `python -m tok.gateway`."""


def _main() -> None:
    try:
        from tok.gateway import run_bridge
    except ImportError:
        raise SystemExit(1) from None

    try:
        run_bridge()
    except OSError as exc:
        import errno

        if exc.errno == errno.EADDRINUSE:
            _resolve_port()
        elif exc.errno == errno.EACCES:
            pass
        else:
            pass
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        raise SystemExit(0) from None
    except Exception:
        raise SystemExit(1) from None


def _resolve_port() -> str:
    import os

    return os.getenv("TOK_BRIDGE_PORT", os.getenv("TOK_PROXY_PORT", "9090"))


if __name__ == "__main__":
    _main()
