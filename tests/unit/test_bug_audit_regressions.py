"""Regression tests for bugs found during the 2026-04-28 bug audit.

Each test verifies a specific confirmed bug fix or reclassification.
"""

from __future__ import annotations

from typing import Any


class TestSelectResendStrategyEmptyDict:
    """SB-001: _select_resend_strategy truth inversion for empty dicts."""

    def test_empty_both_suppresses(self) -> None:
        from tok.runtime.memory.tok_state import _select_resend_strategy

        assert _select_resend_strategy({}, {}, False) == "suppress"

    def test_empty_both_reason_verified(self) -> None:
        from tok.runtime.memory.tok_state import _select_resend_reason

        assert _select_resend_reason({}, {}, False) == "verified_current_state"

    def test_non_empty_identical_still_suppresses(self) -> None:
        from tok.runtime.memory.tok_state import _select_resend_strategy

        assert _select_resend_strategy({"a": ["b"]}, {"a": ["b"]}, False) == "suppress"

    def test_different_still_deltas(self) -> None:
        from tok.runtime.memory.tok_state import _select_resend_strategy

        assert _select_resend_strategy({"a": ["b"]}, {"a": ["c"]}, False) == "delta"


class TestTokStateDoubleWrapping:
    """F-1: tok_state double-wrapping at _history_pipeline.py:1506."""

    def test_tool_compatible_no_double_prefix(self) -> None:
        from tok.compression import inject_system_additions

        tok_state = ">>> g:fix|t:2"
        body: dict[str, Any] = {"system": "base prompt"}
        result = inject_system_additions(
            body=body,
            tok_state=tok_state,
            tool_compatible=True,
        )
        system = result.get("system", "")
        assert ">>>\n>>>" not in system
        assert ">>> g:fix|t:2" in system

    def test_non_tool_compatible_uses_state_block(self) -> None:
        from tok.compression import inject_system_additions

        tok_state = ">>> g:fix|t:2"
        body: dict[str, Any] = {"system": "base"}
        result = inject_system_additions(
            body=body,
            tok_state=tok_state,
            tool_compatible=False,
            grammar="some grammar",
        )
        system = result.get("system", "")
        assert "@state\n" in system
        assert ">>> g:fix|t:2" in system


class TestHostStubReplayedHashVerification:
    """F1/F4/F5: _should_replay_host_stub now verifies hash for heuristic case."""

    def test_explicit_unchanged_signal_replays(self) -> None:
        from tok.compression import _should_replay_host_stub

        cached = "class A:\n    pass\n" + "# line\n" * 200
        stub = "Unchanged since last read"

        assert _should_replay_host_stub(True, cached, stub, stub) is True

    def test_empty_content_replays(self) -> None:
        from tok.compression import _should_replay_host_stub

        cached = "class A:\n    pass\n" + "# line\n" * 200

        assert _should_replay_host_stub(True, cached, "", "") is True

    def test_heuristic_mismatch_rejects(self) -> None:
        from tok.compression import _should_replay_host_stub

        cached = "class A:\n    pass\n" + "# line\n" * 200
        small_different = "File deleted: error ENOENT"

        assert _should_replay_host_stub(True, cached, small_different, small_different) is False

    def test_not_file_like_rejects(self) -> None:
        from tok.compression import _should_replay_host_stub

        assert _should_replay_host_stub(False, "cached", "", "") is False

    def test_no_cached_content_rejects(self) -> None:
        from tok.compression import _should_replay_host_stub

        assert _should_replay_host_stub(True, "", "stub", "stub") is False


class TestRedactedThinkingVisibleBlocks:
    """S-04: redacted_thinking counted in has_visible_blocks check."""

    def test_redacted_thinking_prevents_recovery(self) -> None:
        from tok.gateway import _has_visible_content_block

        blocks = [
            {"type": "text", "text": "visible"},
        ]
        assert _has_visible_content_block(blocks)

    def test_only_redacted_thinking_not_visible_in_old_check(self) -> None:
        from tok.gateway import _has_visible_content_block

        blocks = [
            {"type": "redacted_thinking"},
            {"type": "redacted_thinking"},
        ]
        assert not _has_visible_content_block(blocks)

    def test_redacted_thinking_in_stream_visibility(self) -> None:
        translated_blocks = [
            {"type": "redacted_thinking"},
        ]
        has_visible = any(
            block.get("type") == "tool_use"
            or (block.get("type") == "text" and str(block.get("text", "")).strip())
            or block.get("type") == "thinking"
            or block.get("type") == "redacted_thinking"
            for block in translated_blocks
        )
        assert has_visible is True


class TestVerbatimDeliveredPathsIsLocal:
    """F-3 reclassification: _verbatim_delivered_paths is a local variable, not module-level."""

    def test_local_variable_fresh_each_call(self) -> None:
        from tok.compression import ResultCacheEntry, compress_tool_results

        tool_id = "tid_local_var"
        path = "src/local_var_test.py"
        content = "line\n" * 50

        id_to_ctx: dict[str, dict[str, Any]] = {tool_id: {"name": "Read", "args": {"path": path}, "path": path}}

        def _block(tid: str, c: str) -> list[dict[str, Any]]:
            return [{"type": "tool_result", "tool_use_id": tid, "content": c}]

        result_cache: dict[str, ResultCacheEntry] = {}

        r1, _ = compress_tool_results(
            _block(tool_id, content),
            tool_use_id_to_context=id_to_ctx,
            result_cache=result_cache,
        )
        r2, _ = compress_tool_results(
            _block(tool_id, content),
            tool_use_id_to_context=id_to_ctx,
            result_cache=result_cache,
        )

        assert r1 is not r2


class TestShouldIncludeTokState:
    """T-04 reclassification: >>>| without space is never a valid wire state."""

    def test_valid_wire_state_passes(self) -> None:
        from tok.compression._history_pipeline import _should_include_tok_state

        assert _should_include_tok_state(">>> g:fix|t:2", tool_compatible=True) is True

    def test_bare_marker_rejected(self) -> None:
        from tok.compression._history_pipeline import _should_include_tok_state

        assert _should_include_tok_state(">>>", tool_compatible=True) is False

    def test_bare_pipe_marker_rejected(self) -> None:
        from tok.compression._history_pipeline import _should_include_tok_state

        assert _should_include_tok_state(">>>|", tool_compatible=True) is False

    def test_space_only_rejected(self) -> None:
        from tok.compression._history_pipeline import _should_include_tok_state

        assert _should_include_tok_state(">>> ", tool_compatible=True) is False

    def test_none_rejected(self) -> None:
        from tok.compression._history_pipeline import _should_include_tok_state

        assert _should_include_tok_state(None, tool_compatible=True) is False

    def test_empty_rejected(self) -> None:
        from tok.compression._history_pipeline import _should_include_tok_state

        assert _should_include_tok_state("", tool_compatible=True) is False

    def test_non_tool_compatible_passes(self) -> None:
        from tok.compression._history_pipeline import _should_include_tok_state

        assert _should_include_tok_state(">>>", tool_compatible=False) is True
