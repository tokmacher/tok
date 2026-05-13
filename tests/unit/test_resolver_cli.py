from __future__ import annotations

from typer.testing import CliRunner

from tok.cli import app

runner = CliRunner()


def test_resolver_help() -> None:
    result = runner.invoke(app, ["resolver", "--help"])
    assert result.exit_code == 0
    assert "Local resolver beta commands" in result.output


def test_resolver_status_missing_manifest(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_RESOLVER_ROOT", str(tmp_path))
    result = runner.invoke(app, ["resolver", "status"])
    assert result.exit_code == 0
    assert "Manifest: missing" in result.output


def test_resolver_init_creates_manifest_and_is_idempotent(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_RESOLVER_ROOT", str(tmp_path))
    first = runner.invoke(app, ["resolver", "init"])
    assert first.exit_code == 0
    assert "Initialized resolver" in first.output

    second = runner.invoke(app, ["resolver", "init"])
    assert second.exit_code == 0
    assert "already exists" in second.output

    status = runner.invoke(app, ["resolver", "status"])
    assert status.exit_code == 0
    assert "Manifest: present" in status.output


def test_resolver_store_counts_objects(tmp_path, monkeypatch) -> None:
    from pathlib import Path

    from tok.resolver.store import ContentStore

    monkeypatch.setenv("TOK_RESOLVER_ROOT", str(tmp_path))
    empty = runner.invoke(app, ["resolver", "store"])
    assert empty.exit_code == 0
    assert "Objects: 0" in empty.output

    store = ContentStore(Path(str(tmp_path)))
    store.put(b"hello")
    one = runner.invoke(app, ["resolver", "store"])
    assert one.exit_code == 0
    assert "Objects: 1" in one.output


def test_resolver_get_invalid_uri_is_clean_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_RESOLVER_ROOT", str(tmp_path))
    result = runner.invoke(app, ["resolver", "get", "not-a-uri"])
    assert result.exit_code == 1
    assert "Unsupported resolver URI" in result.output
    assert "Traceback" not in result.output


def test_resolver_get_missing_object(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_RESOLVER_ROOT", str(tmp_path))
    result = runner.invoke(app, ["resolver", "get", "tok-resolver://sha256:" + "0" * 64])
    assert result.exit_code == 1
    assert "Missing object" in result.output


def test_resolver_get_out_writes_file(tmp_path, monkeypatch) -> None:
    from pathlib import Path

    from tok.resolver.store import ContentStore, format_resolver_uri

    monkeypatch.setenv("TOK_RESOLVER_ROOT", str(tmp_path))
    store = ContentStore(Path(str(tmp_path)))
    digest = store.put(b"hello")
    uri = format_resolver_uri(digest)
    out = tmp_path / "hello.bin"
    result = runner.invoke(app, ["resolver", "get", uri, "--out", str(out)])
    assert result.exit_code == 0
    assert out.read_bytes() == b"hello"
