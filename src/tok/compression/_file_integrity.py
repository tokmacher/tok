"""File integrity manifest for injected system additions."""

from __future__ import annotations


def _short_path(path: str) -> str:
    """Return the last two components of a file path."""
    parts = path.replace("\\", "/").split("/")
    parts = [p for p in parts if p]
    return "/".join(parts[-2:]) if len(parts) >= 2 else (parts[0] if parts else path)


def format_file_integrity_manifest(
    files_read_fingerprints: dict[str, str],
    files_fully_delivered: dict[str, int],
) -> str | None:
    """Build a compact @reads manifest from session file-read state.

    Returns None when no files have been fingerprinted.
    Each line: `short/path.py  t:{turn}  fp:{fp8}`
    Sorted by turn ascending, then short path alphabetically.
    """
    if not files_read_fingerprints:
        return None

    rows: list[tuple[int, str, str]] = []
    for norm_path, fp in files_read_fingerprints.items():
        turn = files_fully_delivered.get(norm_path, 0)
        short = _short_path(norm_path)
        rows.append((turn, short, fp))

    rows.sort(key=lambda r: (r[0], r[1]))
    lines = [f"{short}  t:{turn}  fp:{fp}" for turn, short, fp in rows]
    return "\n".join(lines)
