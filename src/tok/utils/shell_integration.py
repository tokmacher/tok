from __future__ import annotations

import os
from pathlib import Path

START_MARKER = "# >>> tok shell integration >>>"
END_MARKER = "# <<< tok shell integration <<<"

_DATA_SCRIPT = "tok_claude.sh"


def _bundled_script_path() -> Path:
    """Return path to the packaged tok_claude.sh, whether installed or in-tree."""
    from importlib.resources import files

    ref = files("tok.data").joinpath(_DATA_SCRIPT)
    # In Python 3.9+, files() returns a Traversable; resolve to real Path.
    return Path(str(ref))


def detect_shell(shell_env: str | None = None) -> str:
    raw_shell = shell_env or os.getenv("SHELL") or ""
    shell = raw_shell.strip()
    name = Path(shell).name
    if name in {"zsh", "bash"}:
        return name
    raise RuntimeError(
        "tok install currently supports zsh and bash. "
        "For other shells, source the installed tok_claude.sh path manually."
    )


def rc_path_for_shell(shell: str, home: Path | None = None) -> Path:
    root = home or Path.home()
    if shell == "zsh":
        return root / ".zshrc"
    if shell == "bash":
        return root / ".bashrc"
    raise RuntimeError(f"Unsupported shell: {shell}")


def integration_block(script_path: Path | None = None) -> str:
    resolved = script_path or _bundled_script_path()
    return f'{START_MARKER}\nsource "{resolved}"\n{END_MARKER}\n'


def install(
    *,
    shell_env: str | None = None,
    home: Path | None = None,
    tok_dir: Path | None = None,
) -> Path:
    shell = detect_shell(shell_env)
    rc_path = rc_path_for_shell(shell, home)
    # tok_dir kept for backwards compat but ignored; script resolved from package data.
    _ = tok_dir
    block = integration_block()

    rc_path.parent.mkdir(parents=True, exist_ok=True)
    existing = rc_path.read_text() if rc_path.exists() else ""
    if START_MARKER in existing and END_MARKER in existing:
        return rc_path

    prefix = "" if not existing or existing.endswith("\n") else "\n"
    rc_path.write_text(existing + prefix + block)
    return rc_path


def uninstall(*, home: Path | None = None) -> list[Path]:
    removed: list[Path] = []
    roots = [
        ("zsh", rc_path_for_shell("zsh", home)),
        ("bash", rc_path_for_shell("bash", home)),
    ]
    for _, rc_path in roots:
        if not rc_path.exists():
            continue
        content = rc_path.read_text()
        start = content.find(START_MARKER)
        end = content.find(END_MARKER)
        if start == -1 or end == -1:
            continue
        end += len(END_MARKER)
        remainder = content[:start] + content[end:]
        rc_path.write_text(
            remainder.strip("\n") + ("\n" if remainder.strip("\n") else "")
        )
        removed.append(rc_path)
    return removed
