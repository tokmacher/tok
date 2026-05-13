from __future__ import annotations

import json
import subprocess


def _load_results(stdout: str) -> list[dict[str, object]]:
    return json.loads(stdout)


def _status_map(results: list[dict[str, object]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in results:
        fixture_id = str(row["id"])
        out[fixture_id] = str(row["status"])
    return out


def test_reader_agrees_with_tok_audit_on_full_fixture_pack() -> None:
    tok = subprocess.run(
        ["uv", "run", "tok", "audit", "docs/spec/fixtures/trace_fixtures.json", "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert tok.returncode == 1

    reader = subprocess.run(
        ["python3", "scripts/tok_trace_reader.py", "docs/spec/fixtures/trace_fixtures.json"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert reader.returncode == 1

    tok_results = _load_results(tok.stdout)
    reader_results = _load_results(reader.stdout)
    assert _status_map(tok_results) == _status_map(reader_results)


def test_reader_agrees_with_tok_audit_on_clean_pack() -> None:
    tok = subprocess.run(
        ["uv", "run", "tok", "audit", "docs/spec/fixtures/clean_trace_fixtures.json", "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert tok.returncode == 0

    reader = subprocess.run(
        ["python3", "scripts/tok_trace_reader.py", "docs/spec/fixtures/clean_trace_fixtures.json"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert reader.returncode == 0

    tok_results = _load_results(tok.stdout)
    reader_results = _load_results(reader.stdout)
    assert _status_map(tok_results) == _status_map(reader_results)


def test_reader_source_does_not_import_tok() -> None:
    source = subprocess.run(
        ["python3", "-c", "print(open('scripts/tok_trace_reader.py','r',encoding='utf-8').read())"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "import tok" not in source
    assert "from tok" not in source
