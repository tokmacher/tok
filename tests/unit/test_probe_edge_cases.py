"""Edge-case tests for probe adapter input parsing and evidence classification."""

from __future__ import annotations

import json

import pytest


@pytest.fixture(autouse=True)
def _enable_unstable(monkeypatch):
    monkeypatch.setenv("TOK_UNSTABLE_ADAPTERS", "1")


# ---------------------------------------------------------------------------
# CodexAdapter.parse_inbound_request — malformed / minimal inputs
# ---------------------------------------------------------------------------


def test_codex_adapter_handles_null_input_field() -> None:
    from tok.adapters.probes import CodexAdapter

    raw = json.dumps({"model": "test-model", "input": None}).encode()
    request = CodexAdapter().parse_inbound_request(raw)
    assert request.messages == []


def test_codex_adapter_handles_missing_input_and_messages() -> None:
    from tok.adapters.probes import CodexAdapter

    raw = json.dumps({"model": "test-model"}).encode()
    request = CodexAdapter().parse_inbound_request(raw)
    assert request.messages == []


def test_codex_adapter_handles_empty_input_list() -> None:
    from tok.adapters.probes import CodexAdapter

    raw = json.dumps({"model": "test-model", "input": []}).encode()
    request = CodexAdapter().parse_inbound_request(raw)
    assert request.messages == []
    assert request.request_policy == "legacy_tool_compatible"


def test_codex_adapter_forces_fallback_for_unknown_tool() -> None:
    from tok.adapters.probes import CodexAdapter

    raw = json.dumps(
        {
            "model": "test-model",
            "input": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "mutate_production_db"}],
        }
    ).encode()
    request = CodexAdapter().parse_inbound_request(raw)

    assert request.request_policy == "forced_baseline"
    last_msg = request.messages[-1]
    assert last_msg.get("tok_probe", {}).get("fallback") is True
    signals = json.loads(request.todo)["diagnostic_signals"]
    assert signals["adapter_probe_fallback"] == 1


def test_codex_adapter_no_fallback_for_all_known_tools() -> None:
    from tok.adapters.probes import CodexAdapter

    raw = json.dumps(
        {
            "model": "test-model",
            "input": [{"role": "user", "content": "search"}],
            "tools": [{"name": "read_file"}, {"name": "search"}],
        }
    ).encode()
    request = CodexAdapter().parse_inbound_request(raw)

    assert request.request_policy == "legacy_tool_compatible"
    signals = json.loads(request.todo)["diagnostic_signals"]
    assert signals["adapter_probe_fallback"] == 0


def test_codex_adapter_forces_fallback_for_unsupported_capability() -> None:
    from tok.adapters.probes import CodexAdapter

    raw = json.dumps(
        {
            "model": "test-model",
            "input": [{"role": "user", "content": "hi"}],
            "capabilities": ["live-codex-cli"],
        }
    ).encode()
    request = CodexAdapter().parse_inbound_request(raw)

    assert request.request_policy == "forced_baseline"
    signals = json.loads(request.todo)["diagnostic_signals"]
    assert signals["adapter_probe_unsupported_capability"] == 1


# ---------------------------------------------------------------------------
# _classify_evidence_form branches
# ---------------------------------------------------------------------------


def test_classify_evidence_form_reference_via_reference_id() -> None:
    from tok.adapters.probes import _classify_evidence_form  # noqa: PLC2701

    result = _classify_evidence_form(tool_name="", content="", message={"reference_id": "tok://x"})
    assert result == "reference"


def test_classify_evidence_form_reference_via_tok_uri_content() -> None:
    from tok.adapters.probes import _classify_evidence_form  # noqa: PLC2701

    result = _classify_evidence_form(tool_name="", content="tok://artifact/abc", message={})
    assert result == "reference"


def test_classify_evidence_form_exact_via_read_file_tool() -> None:
    from tok.adapters.probes import _classify_evidence_form  # noqa: PLC2701

    result = _classify_evidence_form(tool_name="read_file", content="content here", message={})
    assert result == "exact"


def test_classify_evidence_form_exact_via_path_key() -> None:
    from tok.adapters.probes import _classify_evidence_form  # noqa: PLC2701

    result = _classify_evidence_form(tool_name="", content="content", message={"path": "src/foo.py"})
    assert result == "exact"


def test_classify_evidence_form_skeleton_via_outline_tool() -> None:
    from tok.adapters.probes import _classify_evidence_form  # noqa: PLC2701

    result = _classify_evidence_form(tool_name="outline", content="class Foo\n  def bar", message={})
    assert result == "skeleton"


def test_classify_evidence_form_skeleton_via_class_in_content() -> None:
    from tok.adapters.probes import _classify_evidence_form  # noqa: PLC2701

    result = _classify_evidence_form(tool_name="", content="class RuntimeCore:", message={})
    assert result == "skeleton"


def test_classify_evidence_form_summary_via_search_tool() -> None:
    from tok.adapters.probes import _classify_evidence_form  # noqa: PLC2701

    result = _classify_evidence_form(tool_name="grep", content="3 matches found", message={})
    assert result == "summary"


def test_classify_evidence_form_summary_as_default() -> None:
    from tok.adapters.probes import _classify_evidence_form  # noqa: PLC2701

    result = _classify_evidence_form(tool_name="unknown_thing", content="some generic content", message={})
    assert result == "summary"


# ---------------------------------------------------------------------------
# OpenCodeAdapter.parse_inbound_request — minimal inputs
# ---------------------------------------------------------------------------


def test_opencode_adapter_handles_missing_messages() -> None:
    from tok.adapters.probes import OpenCodeAdapter

    raw = json.dumps({"model": "test-model"}).encode()
    request = OpenCodeAdapter().parse_inbound_request(raw)
    assert request.messages == []


def test_opencode_adapter_handles_null_messages() -> None:
    from tok.adapters.probes import OpenCodeAdapter

    raw = json.dumps({"model": "test-model", "messages": None}).encode()
    request = OpenCodeAdapter().parse_inbound_request(raw)
    assert request.messages == []
