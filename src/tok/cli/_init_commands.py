"""CLI commands for initializing Tok projects."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ._cli_support import console


def _ensure_gitignore_entries(gitignore_path: Path, *, entries: list[str]) -> bool:
    """Ensure Tok entries exist in .gitignore file."""
    if gitignore_path.exists():
        existing = gitignore_path.read_text().splitlines()
    else:
        existing = []

    existing_set = {line.strip() for line in existing}
    to_add = [entry for entry in entries if entry not in existing_set]
    if not to_add:
        return False

    lines = list(existing)
    if lines and lines[-1].strip() != "":
        lines.append("")
    lines.append("# Tok")
    lines.extend(to_add)
    lines.append("")
    gitignore_path.write_text("\n".join(lines))
    return True


def _maybe_create_env_file(env_path: Path) -> bool:
    """Create a default .env file if it doesn't exist."""
    if env_path.exists():
        return False
    env_path.write_text("# Tok\nTOK_PROJECT_DIR=.\nTOK_COLLECTOR_DB=.tok/telemetry.db\n")
    return True


def init(
    project_dir: Annotated[
        Path,
        typer.Argument(help="Project directory to initialize (default: current directory)"),
    ] = Path(),
    gitignore: Annotated[
        bool,
        typer.Option(
            "--gitignore/--no-gitignore",
            help="Add Tok artifacts to .gitignore (if this is a git repo)",
        ),
    ] = True,
    env: Annotated[
        bool,
        typer.Option(
            "--env/--no-env",
            help="Create a minimal .env if one does not exist",
        ),
    ] = True,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Proceed even if .tok already exists",
        ),
    ] = False,
) -> None:
    """Initialize a project-local Tok workspace (.tok/ + optional .env/.gitignore)."""
    root = project_dir.resolve()
    tok_dir = root / ".tok"
    if tok_dir.exists() and not force:
        console.print(f"[yellow]⚠️ Already initialized:[/yellow] {tok_dir} (use --force to continue)")
        raise typer.Exit(1)

    tok_dir.mkdir(parents=True, exist_ok=True)
    (tok_dir / ".gitkeep").write_text("")

    console.print(f"[green]✅ Created:[/green] {tok_dir}")

    gitignore_updated = False
    if gitignore:
        git_dir = root / ".git"
        if git_dir.exists() and git_dir.is_dir():
            gitignore_updated = _ensure_gitignore_entries(
                root / ".gitignore",
                entries=[
                    ".tok/",
                    "telemetry.db",
                ],
            )
            if gitignore_updated:
                console.print(f"[green]✅ Updated:[/green] {root / '.gitignore'}")
        else:
            console.print("[dim]Skipping .gitignore update (no .git directory found).[/dim]")

    env_created = False
    if env:
        env_created = _maybe_create_env_file(root / ".env")
        if env_created:
            console.print(f"[green]✅ Created:[/green] {root / '.env'}")
        else:
            console.print(f"[dim]Leaving existing .env unchanged:[/dim] {root / '.env'}")

    console.print(
        "\n[bold]Next steps:[/bold]\n- `tok install`\n- `tok bridge start`\n- `claude`\n- `tok doctor --report`\n"
    )


def register(app: typer.Typer) -> None:
    """Register init command with the CLI app."""
    app.command()(init)
