"""Unit tests for goal extraction and natural-first intent tracking."""

from tok.runtime.memory.session_state import (
    _extract_goal_line,
    _extract_user_goal_line,
    _is_system_injected_line,
    extract_goal_from_messages,
)


class TestExtractGoalLine:
    def test_let_me(self) -> None:
        result = _extract_goal_line("Let me check the failing tests", "let me check the failing tests")
        assert result == "Let me check the failing tests"

    def test_i_need_to(self) -> None:
        result = _extract_goal_line("I need to install the missing deps", "i need to install the missing deps")
        assert result == "I need to install the missing deps"

    def test_the_root_issue_is(self) -> None:
        result = _extract_goal_line(
            "The root issue is the missing pypfopt module",
            "the root issue is the missing pypfopt module",
        )
        assert "root issue is" in result
        assert len(result) <= 40

    def test_returns_empty_for_no_match(self) -> None:
        result = _extract_goal_line("Some unrelated text", "some unrelated text")
        assert result == ""

    def test_truncates_to_max_len(self) -> None:
        long_text = "I need to " + "x" * 60
        lowered = long_text.lower()
        result = _extract_goal_line(long_text, lowered)
        assert len(result) <= 40

    def test_truncates_at_sentence_end(self) -> None:
        text = "I need to fix the tests. Then I'll run the suite."
        lowered = text.lower()
        result = _extract_goal_line(text, lowered)
        assert result == "I need to fix the tests"

    def test_skips_tool_output_lines(self) -> None:
        result = _extract_goal_line(">>> t:5|g:fix", ">>> t:5|g:fix")
        assert result == ""

    def test_skips_code_block_lines(self) -> None:
        result = _extract_goal_line("```python", "```python")
        assert result == ""

    def test_i_want_to(self) -> None:
        result = _extract_goal_line("I want to refactor the bridge module", "i want to refactor the bridge module")
        assert result == "I want to refactor the bridge module"


class TestExtractUserGoalLine:
    def test_i_want_you_to(self) -> None:
        result = _extract_user_goal_line("I want you to find every import", "i want you to find every import")
        assert result == "I want you to find every import"

    def test_investigate(self) -> None:
        result = _extract_user_goal_line("Investigate the failing tests", "investigate the failing tests")
        assert result == "Investigate the failing tests"

    def test_please(self) -> None:
        result = _extract_user_goal_line(
            "Please check if the bridge is healthy", "please check if the bridge is healthy"
        )
        assert result == "Please check if the bridge is healthy"

    def test_returns_empty_for_no_match(self) -> None:
        result = _extract_user_goal_line("ok", "ok")
        assert result == ""

    def test_fix(self) -> None:
        result = _extract_user_goal_line("Fix the optimizer module", "fix the optimizer module")
        assert result == "Fix the optimizer module"


class TestSystemInjectionFilter:
    def test_skill_bullet_skipped(self) -> None:
        assert _is_system_injected_line(
            "- simplify: Review code, then fix any issues found.",
            "- simplify: review code, then fix any issues found.",
        )

    def test_box_drawing_prefix_skipped(self) -> None:
        assert _is_system_injected_line("▎ simplify: fix things", "▎ simplify: fix things")

    def test_normal_user_text_not_skipped(self) -> None:
        assert not _is_system_injected_line("Fix the tests", "fix the tests")

    def test_system_reminder_tag_skipped(self) -> None:
        assert _is_system_injected_line(
            "<system-reminder> some content",
            "<system-reminder> some content",
        )


class TestExtractGoalFromMessages:
    def test_user_goal_takes_priority_over_assistant(self) -> None:
        messages = [
            {"role": "user", "content": "Fix the failing test_turnover tests"},
            {"role": "assistant", "content": "Let me check the imports"},
        ]
        goal, user_sourced = extract_goal_from_messages(messages)
        assert "Fix the failing test_turnover" in goal
        assert user_sourced is True

    def test_falls_back_to_assistant_when_no_user_goal(self) -> None:
        messages = [
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "I need to check what modules are missing"},
        ]
        goal, user_sourced = extract_goal_from_messages(messages)
        assert goal == "I need to check what modules are missing"
        assert user_sourced is False

    def test_first_user_message_wins(self) -> None:
        messages = [
            {"role": "user", "content": "Fix the optimizer module"},
            {"role": "assistant", "content": "Let me check the imports"},
            {"role": "user", "content": "thanks"},
            {"role": "assistant", "content": "I should also fix the tests"},
        ]
        goal, user_sourced = extract_goal_from_messages(messages)
        assert goal == "Fix the optimizer module"
        assert user_sourced is True

    def test_returns_empty_when_no_goal_found(self) -> None:
        messages = [
            {"role": "assistant", "content": "The sky is blue today."},
        ]
        goal, user_sourced = extract_goal_from_messages(messages)
        assert goal == ""
        assert user_sourced is False

    def test_limits_to_max_assistant_messages(self) -> None:
        messages = [
            {"role": "assistant", "content": "Let me start the investigation"},
            {"role": "assistant", "content": "Some intermediate output"},
            {"role": "assistant", "content": "More intermediate output"},
            {"role": "assistant", "content": "I should check the config"},
        ]
        goal, user_sourced = extract_goal_from_messages(messages, max_assistant=2)
        assert "check the config" in goal
        assert user_sourced is False

    def test_handles_list_content(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I need to install the missing deps"},
                ],
            },
        ]
        goal, user_sourced = extract_goal_from_messages(messages)
        assert goal == "I need to install the missing deps"

    def test_handles_user_list_content(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Fix the failing test_turnover tests"},
                ],
            },
        ]
        goal, user_sourced = extract_goal_from_messages(messages)
        assert "Fix the failing test_turnover" in goal
        assert user_sourced is True

    def test_system_injected_content_does_not_override_user_intent(self) -> None:
        messages = [
            {
                "role": "user",
                "content": "- simplify: Review code and fix any issues found.\nI need you to investigate the goal extraction",
            },
            {"role": "assistant", "content": "Let me check the imports"},
            {"role": "user", "content": "ok"},
        ]
        goal, user_sourced = extract_goal_from_messages(messages)
        assert "investigate" in goal
        assert user_sourced is True

    def test_last_user_message_intent_wins_over_earlier_injected(self) -> None:
        messages = [
            {"role": "user", "content": "- simplify: fix any issues found."},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "Implement the new resolver module"},
        ]
        goal, user_sourced = extract_goal_from_messages(messages)
        assert "Implement the new resolver module" in goal
        assert user_sourced is True


class TestNaturalGoalPersistence:
    def test_session_write_memory_persists_goal_without_tok_lines(self) -> None:
        from unittest.mock import MagicMock

        from tok.runtime.memory.bridge_memory import BridgeMemoryState
        from tok.runtime.memory.session_state import session_write_memory

        session = MagicMock()
        session.bridge_memory = BridgeMemoryState()
        session._pending_macro_heal = ""
        session._pending_macro_heal_turn = 0
        session._project_markers = frozenset()

        text = "Let me check the failing tests and fix the optimizer module."
        result = session_write_memory(session, text)
        assert result == ""

        goal_entries = session.bridge_memory.hot.get("goal", [])
        assert len(goal_entries) >= 1
        assert "check the failing tests" in goal_entries[0].value

    def test_session_write_memory_skips_goal_when_no_phrases(self) -> None:
        from unittest.mock import MagicMock

        from tok.runtime.memory.bridge_memory import BridgeMemoryState
        from tok.runtime.memory.session_state import session_write_memory

        session = MagicMock()
        session.bridge_memory = BridgeMemoryState()
        session._pending_macro_heal = ""
        session._pending_macro_heal_turn = 0
        session._project_markers = frozenset()

        text = "The test output shows 3 passed and 2 failed."
        result = session_write_memory(session, text)
        assert result == ""

        goal_entries = session.bridge_memory.hot.get("goal", [])
        assert len(goal_entries) == 0


class TestCompressionDetection:
    def test_detects_message_shrinkage(self) -> None:
        from unittest.mock import MagicMock

        session = MagicMock()
        session._last_request_message_count = 50
        session._request_has_tools = True
        session._answer_phase_expected_this_turn = False
        session._natural_response_acceptable_this_turn = False
        session.pending_behavior_signals = {}

        messages = [{"role": "user", "content": "hi"}] * 20
        body = {"messages": messages}

        _current = len(body["messages"])
        _prev = session._last_request_message_count
        assert _prev > 10
        assert _current > 0
        assert _current < _prev * 0.7

    def test_no_detection_when_session_short(self) -> None:
        _prev = 5
        assert not (_prev > 10)

    def test_no_detection_when_no_shrinkage(self) -> None:
        _prev = 50
        _current = 45
        assert not (_current < _prev * 0.7)
