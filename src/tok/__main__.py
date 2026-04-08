"""Allow running tok as `python -m tok`."""

import sys

if sys.platform == "win32":
    sys.exit(1)

from tok.cli import app

app()
