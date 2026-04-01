from __future__ import annotations

"""Tok CLI — command-line interface for the Tok bridge and protocol tools."""

import logging
import os

import typer
from dotenv import load_dotenv

from ._bridge_commands import register as register_bridge_commands
from ._init_commands import register as register_init_commands
from ._install_commands import register as register_install_commands
from ._legacy_commands import register as register_legacy_commands
from ._memory_commands import register as register_memory_commands
from ._release_commands import register as register_release_commands
from ._dev import dev_app
from ._metrics import metrics_app

load_dotenv()
logging.basicConfig(level=os.getenv("TOK_LOG_LEVEL", "INFO").upper())

app = typer.Typer(
    help="Tok — bridge-first CLI for Claude Code", add_completion=False
)
bridge_app = typer.Typer(help="Bridge-first workflow commands")
app.add_typer(bridge_app, name="bridge")
app.add_typer(metrics_app, name="metrics", hidden=True)
app.add_typer(dev_app, name="dev", hidden=True)

register_install_commands(app)
register_init_commands(app)
register_bridge_commands(bridge_app)
register_memory_commands(app)
register_release_commands(app)
register_legacy_commands(app)


if __name__ == "__main__":
    app()
