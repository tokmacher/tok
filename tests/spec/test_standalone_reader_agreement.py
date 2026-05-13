from __future__ import annotations

import subprocess


def test_reader_matches_tok_audit_fixture_summary() -> None:
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

    assert '"id": "first_exact_file_observation"' in reader.stdout
    assert '"status": "pass"' in reader.stdout
    assert '"id": "missing_resolver_cache"' in reader.stdout
    assert '"status": "warn"' in reader.stdout
    assert '"id": "malformed_block_rejection"' in reader.stdout
    assert '"status": "fail"' in reader.stdout


def test_reader_clean_pack_exits_zero() -> None:
    reader = subprocess.run(
        ["python3", "scripts/tok_trace_reader.py", "docs/spec/fixtures/clean_trace_fixtures.json"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert reader.returncode == 0


def test_reader_source_does_not_import_tok() -> None:
    source = subprocess.run(
        ["python3", "-c", "print(open('scripts/tok_trace_reader.py','r',encoding='utf-8').read())"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "import tok" not in source
    assert "from tok" not in source
