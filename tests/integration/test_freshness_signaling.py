"""Integration tests for freshness signaling - full flow from file read to system prompt."""

from __future__ import annotations

from typing import Any

import pytest

from tok.runtime.memory.bridge_memory import BridgeMemoryState
from tok.runtime.memory.tok_state import (
    _select_resend_reason,
    _select_resend_strategy,
    _prepare_tool_compatible_state,
)
from tok.compression import inject_system_additions
from tok.universal_runtime import (
    RuntimeRequest,
    RuntimeSession,
    UniversalTokRuntime,
)


class TestFreshnessSignalingIntegration:
    """Integration tests for freshness signaling end-to-end flow."""

    def test_file_read_to_fact_format(self):
        """Verify file facts are stored with new format: LINE_COUNT|digest|~TOKENS."""
        state = BridgeMemoryState()
        content = """def foo():
    pass

class Bar:
        def method(self):
        pass
"""
        # Record the file snapshot
        state.record_file_snapshot("/test/sample.py", content)

        # Verify new format in hot memory facts (storage is correct)
        facts = state.hot.get("facts", [])
        assert len(facts) == 1
        fact_value = facts[0].value

        # Format: file[path]:LINE_COUNT|digest|~TOKENS
        assert "file[/test/sample.py]:" in fact_value
        assert "|~" in fact_value and "t" in fact_value  # token savings

        # Verify line count is present (NUMBER| pattern)
        import re

        assert re.search(r":\d+\|", fact_value), (
            f"Line count missing: {fact_value}"
        )

        # Verify get_file_fact_digests parses correctly
        digests = state.get_file_fact_digests()
        assert "/test/sample.py" in digests
        assert digests["/test/sample.py"] == "def foo(): class Bar:"

    def test_wire_state_includes_freshness_data(self):
        """Verify wire state now includes freshness indicators for AI visibility."""
        state = BridgeMemoryState()
        content = "def foo():\n    pass\n\nclass Bar:\n    pass\n"
        state.record_file_snapshot("/test/sample.py", content)

        wire = state.wire_state()

        # Wire state should now include freshness: f:path:LINE_COUNT|~TOKENS
        assert "f:/test/sample.py:" in wire
        # Check for line count pattern (e.g., :6|)
        import re

        assert re.search(r":\d+\|~\d+t", wire), (
            f"Freshness data not in wire state: {wire}"
        )

    def test_wire_state_to_system_prompt(self):
        """Verify wire state includes file facts with freshness indicators."""
        state = BridgeMemoryState()
        content = "line1\nline2\nline3\n"
        state.record_file_snapshot("/test/file.py", content)

        # Verify facts are stored with new format
        facts = state.hot.get("facts", [])
        assert len(facts) == 1
        assert "file[/test/file.py]:" in facts[0].value
        assert "|~12t" in facts[0].value  # 3 lines * 4 = 12 tokens

        # Wire state includes files list (may not include full facts due to dedup)
        wire = state.wire_state()
        assert ">>>" in wire
        assert "f:/test/file.py" in wire or "f:test/file.py" in wire

    def test_resend_reason_with_verified_current(self):
        """Verify resend reason is 'verified_current_state' when unchanged."""
        # First state
        state1 = {"files": ["a.py"], "goal": ["test"]}
        # Same state again
        state2 = {"files": ["a.py"], "goal": ["test"]}

        reason = _select_resend_reason(state2, state1, has_answer_facts=False)

        assert reason == "verified_current_state"
        assert reason != "unchanged_state"  # old terminology gone

    def test_resend_strategy_suppress_when_unchanged(self):
        """Verify suppress strategy when state verified current."""
        state1 = {"files": ["a.py"], "goal": ["test"]}
        state2 = {"files": ["a.py"], "goal": ["test"]}

        strategy = _select_resend_strategy(
            state2, state1, has_answer_facts=False
        )

        assert strategy == "suppress"

    def test_behavior_signals_with_verified_current(self):
        """Verify behavior signals use new terminology in full flow."""
        state = BridgeMemoryState()
        state.record_file_snapshot("/test/file.py", "def foo(): pass\n")

        wire = state.wire_state()

        # Simulate behavior signals from resend
        behavior_signals = {
            "state_resend_reason_state_verified_current": 1,
        }

        body = {"system": ""}
        result = inject_system_additions(
            body,
            tok_state=wire,
            behavior_signals=behavior_signals,
            tool_compatible=False,
        )

        # Old terminology should not appear
        system = result.get("system", "")
        assert "state_suppressed" not in system
        assert "unchanged" not in system.lower()

    def test_tool_compatible_state_preparation(self):
        """Verify tool compatible state includes file facts."""
        raw_state = (
            ">>> turns:5|goal:test|facts:file[/test/a.py]:3|def foo|~12t"
        )
        previous: dict[str, list[str]] = {"files": [], "goal": []}

        parsed, comparable, has_answer_facts = _prepare_tool_compatible_state(
            raw_state, previous
        )

        assert "turns" in parsed
        assert "facts" in parsed
        # File facts should be in facts
        facts = parsed.get("facts", [])
        assert any("file[/test/a.py]" in f for f in facts)

    def test_large_file_token_savings_visible(self):
        """Verify large file shows significant token savings in storage."""
        state = BridgeMemoryState()
        # Simulate 1000 line file
        content = "\n".join([f"def func_{i}(): pass" for i in range(1000)])

        state.record_file_snapshot("/test/large.py", content)

        # Verify facts stored with line count and token savings
        facts = state.hot.get("facts", [])
        assert len(facts) == 1
        fact_value = facts[0].value

        # Should show ~1000 lines and ~4000 tokens in the fact
        assert "file[/test/large.py]:" in fact_value
        assert (
            ":1000|" in fact_value or ":999|" in fact_value
        )  # around 1000 lines
        assert "|~" in fact_value and "t" in fact_value  # token savings

    def test_multiple_files_all_with_savings(self):
        """Verify multiple files each show token savings in storage."""
        state = BridgeMemoryState()

        state.record_file_snapshot("/test/a.py", "def a(): pass\n" * 50)
        state.record_file_snapshot("/test/b.py", "class B:\n    pass\n" * 30)

        # Verify both files have facts with savings indicator
        facts = state.hot.get("facts", [])
        paths = [
            f.value.split(":")[0] for f in facts if f.value.startswith("file[")
        ]

        assert "file[/test/a.py]" in paths
        assert "file[/test/b.py]" in paths

        # Both should have token savings
        for fact in facts:
            if fact.value.startswith("file["):
                assert "|~" in fact.value and "t" in fact.value, (
                    f"Missing savings: {fact.value}"
                )

    def test_legacy_format_backward_compatibility(self):
        """Verify system handles legacy format (no line count)."""
        state = BridgeMemoryState()
        # Manually insert legacy format
        state._upsert(
            state.hot,
            "facts",
            "file[/test/legacy.py]:def foo() pass",
            score_delta=1,
        )

        # Should still work without errors
        wire = state.wire_state()
        assert "file[/test/legacy.py]" in wire

        # get_file_fact_digests should handle both formats
        digests = state.get_file_fact_digests()
        assert "/test/legacy.py" in digests


class TestAIFreshnessSignalUnderstanding:
    """Tests that verify the AI would understand freshness signals."""

    def test_line_count_makes_freshness_obvious(self):
        """Line count in stored facts makes freshness obvious."""
        state = BridgeMemoryState()
        content = "def foo():\n    # 100 lines of code\n    pass\n" * 100

        state.record_file_snapshot("/test/current.py", content)

        # Verify fact stored with explicit line count
        facts = state.hot.get("facts", [])
        assert len(facts) == 1
        fact_value = facts[0].value

        # AI sees: file[path]:300|...|~1200t (explicit freshness signal)
        assert "file[/test/current.py]:" in fact_value
        assert ":300|" in fact_value or ":299|" in fact_value  # ~300 lines
        assert "|~" in fact_value and "t" in fact_value  # token savings

    def test_token_savings_indicator_format(self):
        """Token savings format is unambiguous: ~TOKENS_SAVEDt"""
        state = BridgeMemoryState()
        state.record_file_snapshot("/test/file.py", "line\n" * 250)

        facts = state.hot.get("facts", [])
        fact_value = facts[0].value

        # Format should be: ~1000t (with 't' suffix)
        import re

        pattern = r"~\d+t"
        assert re.search(pattern, fact_value), (
            f"Expected ~<token-count>t format in: {fact_value}"
        )

    def test_verified_current_vs_changed_delta(self):
        """AI can distinguish verified current from changed state."""
        # Current state
        state1 = {"files": ["a.py:100|def foo|~400t"], "goal": ["test"]}
        # Same state again - verified current
        state2 = {"files": ["a.py:100|def foo|~400t"], "goal": ["test"]}

        reason = _select_resend_reason(state2, state1, has_answer_facts=False)

        # AI sees: verified_current_state = trust the data
        assert reason == "verified_current_state"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestStableResultPayloadIntegration:
    def _extract_first_tool_result(
        self, messages: list[dict[str, Any]]
    ) -> str:
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "tool_result"
                ):
                    block_content = block.get("content")
                    if isinstance(block_content, str):
                        return block_content
        return ""

    def test_stable_payload_emitted_on_repeat(self):
        runtime = UniversalTokRuntime()
        session = RuntimeSession()

        tool_id = "t1"
        raw = (
            "class A:\n"
            "    def m(self):\n"
            "        pass\n\n"
            "async def coro():\n"
            "    return 1\n\n"
            "def top():\n"
            "    return 2\n" + ("# filler\n" * 400)
        )

        def _req(bypass: bool = False) -> RuntimeRequest:
            tool_input: dict[str, Any] = {"path": "src/tok/foo.py"}
            if bypass:
                tool_input["tok_bypass_cache"] = True
            return RuntimeRequest(
                model="claude-sonnet-4-6",
                messages=[
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": tool_id,
                                "name": "view_file",
                                "input": tool_input,
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": raw,
                            }
                        ],
                    },
                ],
            )

        runtime.prepare_request(_req(bypass=False), session)
        prepared2 = runtime.prepare_request(_req(bypass=False), session)

        tool_result2 = self._extract_first_tool_result(
            prepared2.body.get("messages", [])
        )
        assert tool_result2.startswith("@stable_result(hash:")
        assert "\n@stable_summary |>" in tool_result2
        assert "\n@stable_skeleton |>" in tool_result2

        prepared3 = runtime.prepare_request(_req(bypass=True), session)
        tool_result3 = self._extract_first_tool_result(
            prepared3.body.get("messages", [])
        )
        assert tool_result3 == raw
        assert "@stable_result" not in tool_result3

    def test_host_unchanged_stub_replays_cached_precision_bytes(self):
        runtime = UniversalTokRuntime()
        session = RuntimeSession()

        tool_id = "t1"
        raw = "line\n" * 300

        def _req(tool_result_content: str) -> RuntimeRequest:
            return RuntimeRequest(
                model="claude-sonnet-4-6",
                messages=[
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": tool_id,
                                "name": "Read",
                                "input": {
                                    "file_path": "src/tok/runtime/core.py",
                                    "offset": 180,
                                    "limit": 40,
                                },
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": tool_result_content,
                            }
                        ],
                    },
                ],
            )

        # Seed cache with real content.
        runtime.prepare_request(_req(raw), session)

        # Host returns an empty/stub payload; Tok should replay cached bytes for precision reads.
        prepared2 = runtime.prepare_request(_req(""), session)
        tool_result2 = self._extract_first_tool_result(
            prepared2.body.get("messages", [])
        )
        assert tool_result2 == raw

    def test_precision_read_recent_window_preserves_raw_on_stub(self):
        runtime = UniversalTokRuntime()
        session = RuntimeSession()

        tool_id = "t1"
        raw = "line\n" * 2000  # > recent-window threshold

        def _req(tool_result_content: str) -> RuntimeRequest:
            return RuntimeRequest(
                model="claude-sonnet-4-6",
                tool_compatible=True,
                messages=[
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": tool_id,
                                "name": "Read",
                                "input": {
                                    "file_path": "src/tok/runtime/core.py",
                                    "offset": 180,
                                    "limit": 40,
                                },
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": tool_result_content,
                            }
                        ],
                    },
                ],
            )

        runtime.prepare_request(_req(raw), session)

        prepared2 = runtime.prepare_request(
            _req("Unchanged since last read"), session
        )
        tool_result2 = self._extract_first_tool_result(
            prepared2.body.get("messages", [])
        )
        assert tool_result2 == raw
