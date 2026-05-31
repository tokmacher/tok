from __future__ import annotations

import os
from pathlib import Path

from tok.cli._cli_support import _resolve_session_dir


def test_resolve_session_dir_returns_provided_dir() -> None:
    p = Path("/some/arbitrary/path")
    result = _resolve_session_dir(session_dir=p, latest=False)
    assert result == p


def test_resolve_session_dir_returns_none_when_not_latest() -> None:
    result = _resolve_session_dir(session_dir=None, latest=False)
    assert result is None


def test_resolve_session_dir_finds_latest_by_mtime(tmp_path: Path, monkeypatch) -> None:
    env_dir = tmp_path / "tok"
    monkeypatch.setenv("TOK_DIR", str(env_dir))
    sessions = env_dir / "sessions"

    older = sessions / "older-session"
    older.mkdir(parents=True)
    (older / "receipts.jsonl").write_text("{}", encoding="utf-8")
    (older / "receipts.jsonl").touch()
    os.utime(str(older / "receipts.jsonl"), (0, 0))

    newer = sessions / "newer-session"
    newer.mkdir(parents=True)
    (newer / "receipts.jsonl").write_text("{}", encoding="utf-8")

    result = _resolve_session_dir(session_dir=None, latest=True)
    assert result is not None
    assert result == newer
