"""Allow running tok as `python -m tok`."""

import sys

if sys.platform == "win32":
    print(
        "Tok requires macOS or Linux. Windows is not supported.",
        file=sys.stderr,
    )
    sys.exit(1)

from tok.cli import app

app()
