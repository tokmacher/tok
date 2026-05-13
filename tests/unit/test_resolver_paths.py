from __future__ import annotations

from pathlib import Path


def test_resolver_root_uses_env(monkeypatch) -> None:
    from tok.resolver.paths import resolver_root

    monkeypatch.setenv("TOK_RESOLVER_ROOT", "/tmp/tok-resolver-root")
    assert resolver_root() == Path("/tmp/tok-resolver-root")
