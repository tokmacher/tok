from __future__ import annotations

from typer.testing import CliRunner

from tok.cli import app

runner = CliRunner()


def test_resolver_help() -> None:
    result = runner.invoke(app, ["resolver", "--help"])
    assert result.exit_code == 0
    assert "Local resolver beta commands" in result.output


def test_resolver_get_help_warns_stdout_is_raw_bytes() -> None:
    result = runner.invoke(app, ["resolver", "get", "--help"])
    assert result.exit_code == 0
    assert "raw bytes" in result.output
    assert "stdout" in result.output


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
    uri = "tok-resolver://sha256:" + "0" * 64
    result = runner.invoke(app, ["resolver", "get", uri])
    assert result.exit_code == 1
    assert f"Missing object for {uri}" in result.output


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


def test_resolver_put_stores_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_RESOLVER_ROOT", str(tmp_path))
    file = tmp_path / "hello.txt"
    file.write_text("hello world")
    result = runner.invoke(app, ["resolver", "put", str(file)])
    assert result.exit_code == 0
    assert "sha256:" in result.output
    assert "tok-resolver://sha256:" in result.output
    assert "Bytes:  11" in result.output


def test_resolver_put_prints_copy_paste_safe_uri_line(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_RESOLVER_ROOT", str(tmp_path))
    file = tmp_path / "hello.txt"
    file.write_text("hello world")

    result = runner.invoke(app, ["resolver", "put", str(file)])

    assert result.exit_code == 0
    uri_lines = [line for line in result.output.splitlines() if line.startswith("URI:")]
    assert len(uri_lines) == 1
    assert uri_lines[0].startswith("URI:    tok-resolver://sha256:")
    assert len(uri_lines[0].rsplit("sha256:", 1)[1]) == 64


def test_resolver_put_warns_for_empty_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_RESOLVER_ROOT", str(tmp_path))
    file = tmp_path / "empty.txt"
    file.write_bytes(b"")

    result = runner.invoke(app, ["resolver", "put", str(file)])

    assert result.exit_code == 0
    assert "Bytes:  0" in result.output
    assert "Warning: stored empty file" in result.output


def test_resolver_put_stored_object_retrievable(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_RESOLVER_ROOT", str(tmp_path))
    file = tmp_path / "data.bin"
    file.write_bytes(b"resolver put test")
    result = runner.invoke(app, ["resolver", "put", str(file)])
    assert result.exit_code == 0
    digest = ""
    for line in result.output.splitlines():
        if line.startswith("Digest:"):
            digest = line.split("Digest:", 1)[1].strip()
            break
    assert digest
    uri = f"tok-resolver://{digest}"
    get_result = runner.invoke(app, ["resolver", "get", uri])
    assert get_result.exit_code == 0
    assert "resolver put test" in get_result.output


def test_resolver_get_stdout_preserves_rich_markup_bytes(tmp_path, monkeypatch) -> None:
    from pathlib import Path

    from tok.resolver.store import ContentStore, format_resolver_uri

    monkeypatch.setenv("TOK_RESOLVER_ROOT", str(tmp_path))
    store = ContentStore(Path(str(tmp_path)))
    data = b"content with [red]markup[/red] and [bold]bold[/bold] tags"
    digest = store.put(data)
    uri = format_resolver_uri(digest)

    result = runner.invoke(app, ["resolver", "get", uri])

    assert result.exit_code == 0
    assert result.stdout_bytes == data


def test_resolver_put_missing_file_exits_cleanly(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_RESOLVER_ROOT", str(tmp_path))
    result = runner.invoke(app, ["resolver", "put", str(tmp_path / "nope.txt")])
    assert result.exit_code == 1
    assert "Traceback" not in result.output


def test_resolver_get_invalid_hex_error_stays_single_line(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOK_RESOLVER_ROOT", str(tmp_path))
    uri = "tok-resolver://sha256:" + "g" * 64

    result = runner.invoke(app, ["resolver", "get", uri])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert f"Invalid resolver URI digest: 'sha256:{'g' * 64}'" in result.output
