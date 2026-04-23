"""Tests for Tok's public entry flow."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

import tok
from tok.runtime.core import RuntimeSession, UniversalTokRuntime
from tok.runtime.types import ProcessedRuntimeResponse, RuntimeRequest


def test_public_bridge_import_succeeds() -> None:
    assert hasattr(tok, "Bridge")
    bridge_cls = tok.Bridge
    assert bridge_cls is not None


def test_experimental_root_imports_raise_attribute_error() -> None:
    for name in ("wrap", "process", "RuntimeSession"):
        with pytest.raises(AttributeError):
            getattr(tok, name)


def test_wrap_via_submodule(tmp_path) -> None:
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    runtime = UniversalTokRuntime()

    request = RuntimeRequest(
        model="claude-sonnet-4",
        messages=[{"role": "user", "content": "Summarize the task"}],
        system="Existing system prompt",
        adapter_kind="wrap",
        tool_compatible=True,
    )
    prepared = runtime.prepare_request(request, session)

    assert prepared.body["messages"][0]["role"] == "user"
    assert prepared.body["system"].startswith("Existing system prompt")
    assert isinstance(prepared.behavior_signals, dict)


def test_process_via_submodule(tmp_path) -> None:
    session = RuntimeSession(memory_dir=tmp_path / ".tok")
    runtime = UniversalTokRuntime()

    result = runtime.process_response(
        ">>> turns:1|goal:ship\n@msg role:assistant\n  |> done",
        model="claude-sonnet-4",
        session=session,
        tool_compatible=True,
    )

    assert isinstance(result, ProcessedRuntimeResponse)
    assert result.mode in {"tok", "tok-empty", "tok-native"}
    assert result.behavior_signals.get("tok_native_response", 0) == 1
    assert isinstance(result.updated_memory, str)


def test_python_m_tok_invokes_cli_help(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["python", "--help"])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("tok", run_name="__main__")

    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "Tok" in out
    assert "Commands" in out
    assert "install" in out


def test_tok_wrap_example_is_syntax_valid() -> None:
    example = Path(__file__).resolve().parents[2] / "examples" / "tok_wrap_example.py"
    compile(example.read_text(), str(example), "exec")
