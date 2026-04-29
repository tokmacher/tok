"""Tests for tok_state module - freshness signaling and resend strategies."""

from __future__ import annotations

import pytest

from tok.runtime.memory.tok_state import (
    _build_tok_state,
    _canonicalize_tool_compatible_state_fields,
    _delta_tok_state_fields,
    _prepare_tool_compatible_state,
    _select_resend_reason,
    _select_resend_strategy,
    _tool_compatible_has_answer_facts,
)


class TestSelectResendReason:
    """Test suite for _select_resend_reason function - freshness signaling terminology."""

    def test_unchanged_state_returns_verified_current(self) -> None:
        """Verify that identical comparable state returns 'verified_current_state'."""
        current = {"files": ["a.py", "b.py"], "goal": ["fix_tests"]}
        previous = {"files": ["a.py", "b.py"], "goal": ["fix_tests"]}

        result = _select_resend_reason(current, previous, has_answer_facts=False)

        assert result == "verified_current_state"
        # Old terminology "unchanged_state" should NOT appear
        assert result != "unchanged_state"

    def test_changed_state_returns_changed_delta(self) -> None:
        """Verify that changed state returns 'changed_state_delta'."""
        current = {"files": ["a.py", "c.py"], "goal": ["fix_tests"]}
        previous = {"files": ["a.py", "b.py"], "goal": ["fix_tests"]}

        result = _select_resend_reason(current, previous, has_answer_facts=False)

        assert result == "changed_state_delta"

    def test_new_answer_anchor_returns_new_answer_anchor(self) -> None:
        """Verify new answer facts trigger 'new_answer_anchor' reason."""
        current = {"files": ["a.py"], "facts": ["answer_file:foo.py"]}
        previous = {"files": ["a.py"]}  # No answer facts

        result = _select_resend_reason(current, previous, has_answer_facts=True)

        assert result == "new_answer_anchor"

    def test_empty_state_returns_changed_delta(self) -> None:
        """Verify empty state doesn't return verified_current (needs non-empty)."""
        current: dict[str, list[str]] = {}
        previous: dict[str, list[str]] = {"files": ["a.py"]}

        result = _select_resend_reason(current, previous, has_answer_facts=False)

        # Empty current state should not be considered "verified current"
        assert result == "changed_state_delta"


class TestSelectResendStrategy:
    """Test suite for _select_resend_strategy function."""

    def test_identical_state_suppresses(self) -> None:
        """Verify identical state triggers 'suppress' strategy."""
        current = {"files": ["a.py"], "goal": ["test"]}
        previous = {"files": ["a.py"], "goal": ["test"]}

        result = _select_resend_strategy(current, previous, has_answer_facts=False)

        assert result == "suppress"

    def test_changed_state_deltas(self) -> None:
        """Verify changed state triggers 'delta' strategy."""
        current = {"files": ["a.py", "b.py"], "goal": ["test"]}
        previous = {"files": ["a.py"], "goal": ["test"]}

        result = _select_resend_strategy(current, previous, has_answer_facts=False)

        assert result == "delta"

    def test_new_answer_facts_force_full(self) -> None:
        """Verify new answer facts trigger 'full' resend."""
        current = {"files": ["a.py"], "facts": ["answer_file:foo.py"]}
        previous = {"files": ["a.py"]}  # No previous answer facts

        result = _select_resend_strategy(current, previous, has_answer_facts=True)

        assert result == "full"

    def test_empty_comparable_dicts_suppress(self) -> None:
        """Verify identical empty dicts suppress (truth inversion fix)."""
        result = _select_resend_strategy({}, {}, has_answer_facts=False)
        assert result == "suppress"

    def test_empty_comparable_reason_verified(self) -> None:
        """Verify identical empty dicts return verified_current_state reason."""
        result = _select_resend_reason({}, {}, has_answer_facts=False)
        assert result == "verified_current_state"


class TestToolCompatibleHasAnswerFacts:
    """Test suite for _tool_compatible_has_answer_facts function."""

    def test_detects_answer_file_fact(self) -> None:
        """Verify detection of answer_file facts."""
        fields = {"facts": ["answer_file:src/foo.py", "other:thing"]}

        result = _tool_compatible_has_answer_facts(fields)

        assert result is True

    def test_detects_answer_verification_fact(self) -> None:
        """Verify detection of answer_verification facts."""
        fields = {"facts": ["answer_verification:correct"]}

        result = _tool_compatible_has_answer_facts(fields)

        assert result is True

    def test_no_answer_facts_returns_false(self) -> None:
        """Verify false when no answer facts present."""
        fields = {"facts": ["other:thing", "branch:main"]}

        result = _tool_compatible_has_answer_facts(fields)

        assert result is False

    def test_empty_facts_returns_false(self) -> None:
        """Verify false when facts list is empty."""
        fields: dict[str, list[str]] = {"facts": []}

        result = _tool_compatible_has_answer_facts(fields)

        assert result is False


class TestPrepareToolCompatibleState:
    """Test suite for _prepare_tool_compatible_state function."""

    def test_prepares_state_correctly(self) -> None:
        """Verify state preparation returns expected structure."""
        raw_state = ">>> turns:5|goal:test|files:a.py,b.py"
        previous = {"files": ["a.py"], "goal": ["old"]}

        parsed, comparable, has_answer_facts = _prepare_tool_compatible_state(raw_state, previous)

        assert "turns" in parsed
        assert "files" in parsed
        assert "turns" not in comparable  # turns excluded from comparison
        assert isinstance(has_answer_facts, bool)


class TestCanonicalizeToolCompatibleStateFields:
    """Test suite for _canonicalize_tool_compatible_state_fields function."""

    def test_compacts_values(self) -> None:
        """Verify values are compacted and deduplicated."""
        fields = {
            "turns": ["5"],
            "files": ["/long/path/to/file.py", "/long/path/to/file.py"],
        }

        result = _canonicalize_tool_compatible_state_fields(fields)

        assert "turns" in result
        assert "files" in result
        # Should dedupe identical files
        assert len(result["files"]) == 1


class TestBuildTokState:
    """Test suite for _build_tok_state function."""

    def test_builds_valid_state_string(self) -> None:
        """Verify state string is built correctly."""
        fields: dict[str, list[str]] = {
            "turns": ["5"],
            "goal": ["fix_tests"],
            "files": ["a.py", "b.py"],
        }

        result = _build_tok_state(fields)

        assert result.startswith(">>> ")
        assert "g:fix_tests" in result
        assert "f:a.py" in result or "f:" in result

    def test_empty_fields_returns_empty(self) -> None:
        """Verify empty fields returns empty string."""
        fields: dict[str, list[str]] = {}

        result = _build_tok_state(fields)

        assert result == ""


class TestDeltaTokStateFields:
    """Test suite for _delta_tok_state_fields function."""

    def test_returns_only_changed_fields(self) -> None:
        """Verify only changed fields are included in delta."""
        previous = {"files": ["a.py"], "goal": ["old_goal"]}
        current = {
            "files": ["a.py", "b.py"],
            "goal": ["old_goal"],
            "turns": ["6"],
        }

        result = _delta_tok_state_fields(previous, current)

        # Turns always included, files changed (added b.py), goal unchanged
        assert "t:" in result  # turns always appears
        assert "f:a.py,b.py" in result  # files with both entries

    def test_empty_delta_when_only_turns_changed(self) -> None:
        """Verify empty string when only turns changed."""
        previous = {"files": ["a.py"], "goal": ["test"], "turns": ["5"]}
        current = {"files": ["a.py"], "goal": ["test"], "turns": ["6"]}

        result = _delta_tok_state_fields(previous, current)

        assert result == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
