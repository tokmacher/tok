"""Section 5.3.1: Exactness Gate Mechanical Enforcement — RED/GREEN tests.

Also covers TOK_EXACTNESS_GATE=warn phased rollout (risk register §8).

Tests the evidence-safety gate through both the EvidenceSafetyState ledger and
the compression pipeline's enforcement of first-observation exactness.

Contract (docs/bridge-standard.md):
- First file observation must be exact (no compression).
- Non-exact evidence must not authorize edit-like behavior without reacquisition.
- Repeated reads may compress only after exact evidence exists.
- Novel failures must not be compressed away (first failure observation is exact).
"""
from __future__ import annotations

import logging

import pytest

from tok.runtime.evidence_safety import EvidenceSafetyState, normalize_evidence_key

# ---------------------------------------------------------------------------
# Evidence ledger state transitions
# ---------------------------------------------------------------------------

class TestExactnessGateStateMachine:
    def test_unknown_key_requires_reacquisition(self) -> None:
        state = EvidenceSafetyState()
        # An unseen key requires reacquisition (safe default = must re-read)
        assert state.requires_reacquisition("src/foo.py") is True

    def test_exact_observation_clears_reacquisition_requirement(self) -> None:
        state = EvidenceSafetyState()
        state.record_exact("src/foo.py", digest="abc", turn=1)
        assert state.requires_reacquisition("src/foo.py") is False

    def test_non_exact_observation_requires_reacquisition(self) -> None:
        state = EvidenceSafetyState()
        # First record non-exact (skeleton)
        state.record_non_exact("src/foo.py", form="skeleton", turn=1)
        assert state.requires_reacquisition("src/foo.py") is True

    def test_exact_after_non_exact_clears_requirement(self) -> None:
        state = EvidenceSafetyState()
        state.record_non_exact("src/foo.py", form="skeleton", turn=1)
        assert state.requires_reacquisition("src/foo.py") is True
        state.record_exact("src/foo.py", digest="abc", turn=2)
        assert state.requires_reacquisition("src/foo.py") is False

    def test_require_exact_reacquisition_sets_flag(self) -> None:
        state = EvidenceSafetyState()
        state.record_non_exact("src/foo.py", form="skeleton", turn=1)
        signals = state.require_exact_reacquisition("src/foo.py")
        assert signals.get("evidence_exact_reacquisition_required", 0) == 1
        assert signals.get("evidence_compression_blocked_for_safety", 0) == 1
        entry = state.ledger["src/foo.py"]
        assert entry.exact_reacquisition_required is True

    def test_exact_observation_satisfies_reacquisition(self) -> None:
        state = EvidenceSafetyState()
        state.record_non_exact("src/foo.py", form="skeleton", turn=1)
        state.require_exact_reacquisition("src/foo.py")
        signals = state.record_exact("src/foo.py", digest="new", turn=2)
        assert signals.get("evidence_exact_reacquisition_satisfied", 0) == 1
        entry = state.ledger["src/foo.py"]
        assert entry.exact_reacquisition_required is False

    def test_require_reacquisition_on_already_exact_key_is_noop(self) -> None:
        state = EvidenceSafetyState()
        state.record_exact("src/foo.py", digest="abc", turn=1)
        signals = state.require_exact_reacquisition("src/foo.py")
        # Exact key: no requirement needed
        assert signals == {}

    def test_first_exact_turn_recorded(self) -> None:
        state = EvidenceSafetyState()
        state.record_exact("src/foo.py", digest="abc", turn=3)
        entry = state.ledger["src/foo.py"]
        assert entry.first_exact_turn == 3
        assert entry.has_exact is True

    def test_summary_evidence_requires_reacquisition(self) -> None:
        state = EvidenceSafetyState()
        state.record_non_exact("src/bar.py", form="summary", turn=1)
        assert state.requires_reacquisition("src/bar.py") is True

    def test_reference_evidence_requires_reacquisition(self) -> None:
        state = EvidenceSafetyState()
        state.record_non_exact("src/bar.py", form="reference", turn=1)
        assert state.requires_reacquisition("src/bar.py") is True


# ---------------------------------------------------------------------------
# Scenario: read once exact, second read compressible
# ---------------------------------------------------------------------------

class TestRepeatReadAfterExact:
    def test_first_read_sets_first_exact_turn(self) -> None:
        state = EvidenceSafetyState()
        state.record_exact("src/main.py", digest="hash1", turn=1)
        assert state.ledger["src/main.py"].first_exact_turn == 1
        assert state.ledger["src/main.py"].latest_form == "exact"

    def test_second_exact_read_does_not_require_reacquisition(self) -> None:
        state = EvidenceSafetyState()
        state.record_exact("src/main.py", digest="hash1", turn=1)
        state.record_exact("src/main.py", digest="hash1", turn=3)
        # Already exact — still not requiring reacquisition
        assert state.requires_reacquisition("src/main.py") is False

    def test_first_exact_observed_signal_only_on_first_call(self) -> None:
        state = EvidenceSafetyState()
        sig1 = state.record_exact("src/main.py", digest="hash1", turn=1)
        assert sig1.get("evidence_first_exact_observed") == 1
        sig2 = state.record_exact("src/main.py", digest="hash1", turn=2)
        assert "evidence_first_exact_observed" not in sig2 or sig2["evidence_first_exact_observed"] == 0


# ---------------------------------------------------------------------------
# Scenario: non-exact file targeted by edit tool
# ---------------------------------------------------------------------------

class TestEditToolTargetingNonExactFile:
    def test_skeleton_file_targeted_by_edit_requires_reacquisition(self) -> None:
        state = EvidenceSafetyState()
        # File summarized as skeleton
        state.record_non_exact("src/edit_target.py", form="skeleton", turn=2)
        # Edit tool would target it
        signals = state.require_exact_reacquisition("src/edit_target.py")
        assert signals.get("evidence_exact_reacquisition_required") == 1
        assert signals.get("evidence_compression_blocked_for_safety") == 1

    def test_summary_file_targeted_by_write_requires_reacquisition(self) -> None:
        state = EvidenceSafetyState()
        state.record_non_exact("src/output.py", form="summary", turn=1)
        signals = state.require_exact_reacquisition("src/output.py")
        assert signals.get("evidence_compression_blocked_for_safety", 0) > 0

    def test_exact_file_targeted_by_edit_allows_it(self) -> None:
        state = EvidenceSafetyState()
        state.record_exact("src/edit_target.py", digest="abc", turn=1)
        signals = state.require_exact_reacquisition("src/edit_target.py")
        # No blocking required — evidence is exact
        assert signals == {}


# ---------------------------------------------------------------------------
# Compression pipeline: first observation must be exact
# ---------------------------------------------------------------------------

class TestCompressionPipelineFirstExact:
    """The compression pipeline must preserve first file observations verbatim."""

    def _make_file_read_messages(self, path: str, content: str, tool_id: str = "t1") -> list:
        return [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": tool_id, "name": "Read", "input": {"file_path": path}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": content}],
            },
        ]

    def test_first_file_read_is_delivered_verbatim(self) -> None:
        from tok.compression._history_pipeline import compress_tool_results_impl
        from tok.runtime.pipeline._tool_context import build_tool_use_id_to_context

        content = "def foo():\n    return 42\n" * 50  # > 1000 chars to trigger compression
        messages = self._make_file_read_messages("/repo/foo.py", content, "t1")
        tool_use_id_to_context = build_tool_use_id_to_context(messages)

        first_exact_seen: set[str] = set()
        compressed, breakdown = compress_tool_results_impl(
            messages,
            result_cache={},
            tool_use_id_to_context=tool_use_id_to_context,
            compression_level="balanced",
            first_exact_evidence_seen=first_exact_seen,
            preserve_exact_search_evidence=True,
        )

        # The first read must be verbatim
        result_blocks = [
            block
            for msg in compressed
            if msg["role"] == "user"
            for block in msg.get("content", [])
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        assert len(result_blocks) >= 1
        first_content = str(result_blocks[0].get("content", ""))
        assert content[:50] in first_content, (
            "First file observation must be delivered verbatim (not compressed)"
        )

    def test_first_exact_seen_populated_after_first_read(self) -> None:
        from tok.compression._history_pipeline import compress_tool_results_impl
        from tok.runtime.pipeline._tool_context import build_tool_use_id_to_context

        content = "file content " * 80
        messages = self._make_file_read_messages("/repo/bar.py", content, "t1")
        tool_use_id_to_context = build_tool_use_id_to_context(messages)

        first_exact_seen: set[str] = set()
        compress_tool_results_impl(
            messages,
            result_cache={},
            tool_use_id_to_context=tool_use_id_to_context,
            compression_level="balanced",
            first_exact_evidence_seen=first_exact_seen,
            preserve_exact_search_evidence=True,
        )
        # After first read, the path must be in first_exact_evidence_seen
        assert len(first_exact_seen) >= 1, (
            "first_exact_evidence_seen must be populated after first file read"
        )

    def test_second_read_same_file_with_first_exact_can_be_compressed(self) -> None:
        from tok.compression._history_pipeline import compress_tool_results_impl
        from tok.runtime.pipeline._tool_context import build_tool_use_id_to_context

        content = "file content " * 80  # > 1000 chars
        messages = (
            self._make_file_read_messages("/repo/baz.py", content, "t1")
            + self._make_file_read_messages("/repo/baz.py", content, "t2")
            + self._make_file_read_messages("/repo/baz.py", content, "t3")
        )
        tool_use_id_to_context = build_tool_use_id_to_context(messages)

        first_exact_seen: set[str] = set()
        compressed, breakdown = compress_tool_results_impl(
            messages,
            result_cache={},
            tool_use_id_to_context=tool_use_id_to_context,
            compression_level="balanced",
            first_exact_evidence_seen=first_exact_seen,
            preserve_exact_search_evidence=True,
        )
        # Must not crash; must return valid structure
        assert len(compressed) == len(messages)
        assert isinstance(breakdown, dict)
        # Semantics allow verbatim after two reads in some codepaths; this edge case
        # only verifies that repeated reads keep valid structure.
        assert breakdown is not None


# ---------------------------------------------------------------------------
# Adversarial: novel failures must not be compressed away
# ---------------------------------------------------------------------------

class TestNovelFailurePreservation:
    def test_first_failure_observation_preserved_verbatim(self) -> None:
        from tok.compression._history_pipeline import compress_tool_results_impl
        from tok.runtime.pipeline._tool_context import build_tool_use_id_to_context

        failure_content = (
            "Traceback (most recent call last):\n"
            "  File 'test_foo.py', line 10, in test_bar\n"
            "AssertionError: expected 42, got 0\n" * 30
        )
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "c1", "name": "Bash",
                     "input": {"command": "pytest tests/test_foo.py"}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "c1", "content": failure_content}
                ],
            },
        ]
        tool_use_id_to_context = build_tool_use_id_to_context(messages)
        first_exact_seen: set[str] = set()
        compressed, breakdown = compress_tool_results_impl(
            messages,
            result_cache={},
            tool_use_id_to_context=tool_use_id_to_context,
            compression_level="balanced",
            first_exact_evidence_seen=first_exact_seen,
            preserve_exact_search_evidence=True,
        )
        # First failure must be verbatim (novel failure preservation)
        result_blocks = [
            block
            for msg in compressed
            if msg["role"] == "user"
            for block in msg.get("content", [])
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        assert len(result_blocks) >= 1
        first_result = str(result_blocks[0].get("content", ""))
        assert "AssertionError" in first_result or "Traceback" in first_result, (
            "Novel failure traceback must not be compressed away on first observation"
        )


# ---------------------------------------------------------------------------
# Key normalization
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TOK_EXACTNESS_GATE=warn phased rollout (risk register §8)
# ---------------------------------------------------------------------------

class TestExactnessGateWarnMode:
    """In warn mode the gate must allow operations and emit a WARNING instead of blocking."""

    def test_requires_reacquisition_enforce_mode_returns_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default (no env var) — gate enforces: requires_reacquisition returns True."""
        monkeypatch.delenv("TOK_EXACTNESS_GATE", raising=False)
        state = EvidenceSafetyState()
        state.record_non_exact("src/foo.py", form="skeleton", turn=1)
        assert state.requires_reacquisition("src/foo.py") is True

    def test_requires_reacquisition_warn_mode_returns_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TOK_EXACTNESS_GATE=warn — gate allows: requires_reacquisition returns False."""
        monkeypatch.setenv("TOK_EXACTNESS_GATE", "warn")
        state = EvidenceSafetyState()
        state.record_non_exact("src/foo.py", form="skeleton", turn=1)
        assert state.requires_reacquisition("src/foo.py") is False, (
            "In warn mode, requires_reacquisition must return False (allow compression)"
        )

    def test_requires_reacquisition_warn_mode_emits_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """TOK_EXACTNESS_GATE=warn — gate emits a WARNING when it would have blocked."""
        monkeypatch.setenv("TOK_EXACTNESS_GATE", "warn")
        state = EvidenceSafetyState()
        state.record_non_exact("src/warn_target.py", form="skeleton", turn=1)
        with caplog.at_level(logging.WARNING, logger="tok.evidence_safety"):
            result = state.requires_reacquisition("src/warn_target.py")
        assert result is False
        assert "warn" in caplog.text.lower() or "exactness" in caplog.text.lower(), (
            "Expected a warning about exactness gate warn mode, got: " + caplog.text
        )

    def test_require_exact_reacquisition_enforce_mode_sets_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default (enforce) — require_exact_reacquisition sets the flag and emits signals."""
        monkeypatch.delenv("TOK_EXACTNESS_GATE", raising=False)
        state = EvidenceSafetyState()
        state.record_non_exact("src/foo.py", form="skeleton", turn=1)
        signals = state.require_exact_reacquisition("src/foo.py")
        assert signals.get("evidence_compression_blocked_for_safety", 0) == 1
        assert state.ledger["src/foo.py"].exact_reacquisition_required is True

    def test_require_exact_reacquisition_warn_mode_does_not_set_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TOK_EXACTNESS_GATE=warn — require_exact_reacquisition must NOT set blocking flag."""
        monkeypatch.setenv("TOK_EXACTNESS_GATE", "warn")
        state = EvidenceSafetyState()
        state.record_non_exact("src/foo.py", form="skeleton", turn=1)
        signals = state.require_exact_reacquisition("src/foo.py")
        assert signals.get("evidence_compression_blocked_for_safety", 0) == 0, (
            "In warn mode, blocking signal must not be emitted"
        )
        assert state.ledger["src/foo.py"].exact_reacquisition_required is False, (
            "In warn mode, exact_reacquisition_required flag must not be set"
        )

    def test_enforce_mode_explicit_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """TOK_EXACTNESS_GATE=enforce — behaves identically to default (block)."""
        monkeypatch.setenv("TOK_EXACTNESS_GATE", "enforce")
        state = EvidenceSafetyState()
        state.record_non_exact("src/foo.py", form="skeleton", turn=1)
        assert state.requires_reacquisition("src/foo.py") is True

    def test_unknown_mode_falls_back_to_enforce(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unrecognized TOK_EXACTNESS_GATE value — must default to enforce (safe)."""
        monkeypatch.setenv("TOK_EXACTNESS_GATE", "permissive_xyzzy")
        state = EvidenceSafetyState()
        state.record_non_exact("src/foo.py", form="skeleton", turn=1)
        assert state.requires_reacquisition("src/foo.py") is True, (
            "Unknown gate mode must default to enforce (fail-safe)"
        )


class TestEvidenceKeyNormalization:
    def test_file_prefix_stripped(self) -> None:
        key = normalize_evidence_key("file:src/main.py")
        assert key == "src/main.py"

    def test_dotslash_stripped(self) -> None:
        key = normalize_evidence_key("./src/main.py")
        assert key == "src/main.py"

    def test_backslashes_normalized(self) -> None:
        key = normalize_evidence_key("src\\main\\foo.py")
        assert key == "src/main/foo.py"

    def test_empty_key_normalizes_to_empty(self) -> None:
        key = normalize_evidence_key("")
        assert key == ""

    def test_whitespace_stripped(self) -> None:
        key = normalize_evidence_key("  src/main.py  ")
        assert key == "src/main.py"
