from __future__ import annotations

from typing import TYPE_CHECKING

from tok.adapters import OrchestratorAdapter
from tok.stats import SavingsTracker
from tok.universal_runtime import (
    RuntimeRequest,
    RuntimeSession,
    UniversalTokRuntime,
)

if TYPE_CHECKING:
    from pathlib import Path

MODEL = "google/gemini-2.0-flash-lite-001"


def _bridge_prepare(
    *,
    session: RuntimeSession,
    messages: list[dict],
    system: str,
    grammar: str | None = None,
    todo: str | None = None,
    deltas: str | None = None,
):
    runtime = UniversalTokRuntime()
    return runtime.prepare_request(
        RuntimeRequest(
            model=MODEL,
            messages=messages,
            system=system,
            adapter_kind="claude-bridge",
            tool_compatible=False,
            grammar=grammar,
            todo=todo,
            deltas=deltas,
        ),
        session,
    )


def _bridge_finalize(
    *,
    session: RuntimeSession,
    text: str,
    behavior_signals: dict[str, int] | None = None,
):
    runtime = UniversalTokRuntime()
    return runtime.process_response(
        text,
        model=MODEL,
        session=session,
        behavior_signals=behavior_signals,
        tool_compatible=False,
    )


def _orchestrator_prepare(
    *,
    session: RuntimeSession,
    messages: list[dict],
    system: str,
    grammar: str | None = None,
    todo: str | None = None,
    deltas: str | None = None,
):
    adapter = OrchestratorAdapter(session=session)
    _chat_messages, prepared = adapter.prepare_turn(
        model=MODEL,
        system_prompt=system,
        dynamic_messages=messages,
        grammar=grammar,
        todo=todo,
        deltas=deltas,
    )
    return prepared


def _orchestrator_finalize(
    *,
    session: RuntimeSession,
    text: str,
    behavior_signals: dict[str, int] | None = None,
):
    adapter = OrchestratorAdapter(session=session)
    return adapter.finalize(
        text=text,
        model=MODEL,
        behavior_signals=behavior_signals,
    )


def _session_tracker(tmp_path: Path, stem: str) -> SavingsTracker:
    return SavingsTracker(
        savings_file=str(tmp_path / f"{stem}-session.tok"),
        ledger_path=tmp_path / f"{stem}-global.tok",
    )


def _record_summary(
    *,
    tracker: SavingsTracker,
    prepared,
    processed,
    behavior_signals: dict[str, int],
    actual_input: int,
    actual_output: int,
):
    tracker.record_call(
        model=MODEL,
        actual_input=actual_input,
        actual_output=actual_output,
        cache_read=0,
        cache_write=0,
        input_saved=prepared.input_saved_tokens,
        output_saved=processed.output_saved_tokens,
        type_breakdown=prepared.type_breakdown,
        behavior_signals=behavior_signals,
    )
    return tracker.session_summary()


def test_bridge_and_orchestrator_prepare_and_finalize_share_happy_path_semantics(
    tmp_path,
) -> None:
    messages = [
        {"role": "assistant", "content": "Earlier context"},
        {"role": "user", "content": "Audit the codebase"},
    ]
    system = "orchestrator system"
    grammar = "grammar"
    todo = "[ ] audit"
    deltas = ">>> turns:1|goal:audit"
    response_text = ">>> turns:2|goal:audit\n@msg role:assistant\n  |> ok"
    incoming_signals = {"repeat_file_read": 1}

    bridge_session = RuntimeSession(memory_dir=tmp_path / "bridge")
    orchestrator_session = RuntimeSession(memory_dir=tmp_path / "orchestrator")

    bridge_prepared = _bridge_prepare(
        session=bridge_session,
        messages=messages,
        system=system,
        grammar=grammar,
        todo=todo,
        deltas=deltas,
    )
    orchestrator_prepared = _orchestrator_prepare(
        session=orchestrator_session,
        messages=messages,
        system=system,
        grammar=grammar,
        todo=todo,
        deltas=deltas,
    )

    assert bridge_prepared.body == orchestrator_prepared.body
    assert bridge_prepared.behavior_signals == orchestrator_prepared.behavior_signals
    assert bridge_prepared.type_breakdown == orchestrator_prepared.type_breakdown

    bridge_processed = _bridge_finalize(
        session=bridge_session,
        text=response_text,
        behavior_signals=incoming_signals,
    )
    orchestrator_processed = _orchestrator_finalize(
        session=orchestrator_session,
        text=response_text,
        behavior_signals=incoming_signals,
    )

    assert bridge_processed.mode == orchestrator_processed.mode == "tok-native"
    assert bridge_processed.content_blocks == orchestrator_processed.content_blocks
    assert bridge_processed.updated_memory == orchestrator_processed.updated_memory
    assert bridge_processed.behavior_signals == orchestrator_processed.behavior_signals


def test_bridge_and_orchestrator_preserve_memory_carry_contract(
    tmp_path,
) -> None:
    payload = (
        ">>> turns:1|goal:audit\n"
        "@msg role:assistant\n"
        "  |> Verification=The main entry point is the `compress_history` "
        "function in `src/tok/compression.py`."
    )

    bridge_session = RuntimeSession(memory_dir=tmp_path / "bridge")
    orchestrator_session = RuntimeSession(memory_dir=tmp_path / "orchestrator")
    bridge_session.bridge_memory.record_search_snapshot(
        "compress_history",
        "src/tok/compression.py:305: def compress_history(",
    )
    orchestrator_session.bridge_memory.record_search_snapshot(
        "compress_history",
        "src/tok/compression.py:305: def compress_history(",
    )

    _bridge_finalize(session=bridge_session, text=payload)
    _orchestrator_finalize(session=orchestrator_session, text=payload)

    assert bridge_session.bridge_memory.wire_state() == (orchestrator_session.bridge_memory.wire_state())
    assert [entry.value for entry in bridge_session.bridge_memory.hot.get("files", [])] == ["src/tok/compression.py"]
    assert [entry.value for entry in orchestrator_session.bridge_memory.hot.get("files", [])] == [
        "src/tok/compression.py"
    ]


def test_bridge_and_orchestrator_record_equivalent_savings_totals(
    tmp_path,
) -> None:
    messages = [{"role": "user", "content": "Audit the codebase"}]
    system = "orchestrator system"
    payload = ">>> turns:1|goal:audit\n@msg role:assistant\n  |> ok"

    bridge_session = RuntimeSession(memory_dir=tmp_path / "bridge")
    orchestrator_session = RuntimeSession(memory_dir=tmp_path / "orchestrator")
    bridge_prepared = _bridge_prepare(
        session=bridge_session,
        messages=messages,
        system=system,
    )
    orchestrator_prepared = _orchestrator_prepare(
        session=orchestrator_session,
        messages=messages,
        system=system,
    )
    bridge_processed = _bridge_finalize(session=bridge_session, text=payload)
    orchestrator_processed = _orchestrator_finalize(
        session=orchestrator_session,
        text=payload,
    )

    bridge_tracker = _session_tracker(tmp_path, "bridge")
    orchestrator_tracker = _session_tracker(tmp_path, "orchestrator")
    bridge_summary = _record_summary(
        tracker=bridge_tracker,
        prepared=bridge_prepared,
        processed=bridge_processed,
        behavior_signals=bridge_processed.behavior_signals,
        actual_input=180,
        actual_output=20,
    )
    orchestrator_summary = _record_summary(
        tracker=orchestrator_tracker,
        prepared=orchestrator_prepared,
        processed=orchestrator_processed,
        behavior_signals=orchestrator_processed.behavior_signals,
        actual_input=180,
        actual_output=20,
    )

    assert bridge_summary == orchestrator_summary
    assert bridge_summary is not None
    assert int(bridge_summary["actual_tokens"]) == 200
    assert int(bridge_summary["baseline_tokens"]) >= 200
    assert int(bridge_summary["tokens_saved"]) == int(bridge_summary["baseline_tokens"]) - 200


def test_bridge_and_orchestrator_keep_fallback_signals_visible(
    tmp_path,
) -> None:
    payload = "Plain assistant reply"
    fallback_signals = {
        "tok_fallback_activated": 1,
        "processing_error": 1,
    }

    bridge_session = RuntimeSession(memory_dir=tmp_path / "bridge")
    orchestrator_session = RuntimeSession(memory_dir=tmp_path / "orchestrator")

    bridge_processed = _bridge_finalize(
        session=bridge_session,
        text=payload,
        behavior_signals=fallback_signals,
    )
    orchestrator_processed = _orchestrator_finalize(
        session=orchestrator_session,
        text=payload,
        behavior_signals=fallback_signals,
    )

    assert bridge_processed.mode == orchestrator_processed.mode
    assert bridge_processed.behavior_signals == orchestrator_processed.behavior_signals
    assert bridge_processed.behavior_signals["tok_fallback_activated"] == 1
    assert bridge_processed.behavior_signals["processing_error"] == 1
    assert bridge_processed.behavior_signals["fail_open_compat_response"] == 1
    assert bridge_processed.behavior_signals["non_tok_response"] == 1
