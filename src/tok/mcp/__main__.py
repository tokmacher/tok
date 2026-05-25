"""Allow running the Tok MCP server as: python -m tok.mcp"""

import sys


def _main() -> None:
    try:
        from .server import main

        main()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    _main()
