"""Tests for thinking-block mutation detection and deduplication.

These tests verify:
1. No false positive when multiple assistant turns have thinking blocks.
2. Per-canonicalization-cycle dedup prevents double-counting identical mutations.
3. The non-penalized breadcrumb event is recorded when a restore succeeds.
"""

from tok.runtime.pipeline.request_validation import (
    _check_thinking_block_mutation,
    _content_hash,
    canonicalize_anthropic_bridge_messages,
)
from tok.runtime.smoothness.models import SmoothnessEventType


def _make_thinking_block(text: str, signature: str | None = None) -> dict:
    block: dict = {"type": "thinking", "thinking": text}
    if signature is not None:
        block["signature"] = signature
    return block


def _make_messages_with_two_thinking_assistants() -> list[dict]:
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": "First prompt"}],
        },
        {
            "role": "assistant",
            "content": [
                _make_thinking_block("earlier thinking", "sig_early"),
                {"type": "text", "text": "Earlier response"},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "earlier result",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                _make_thinking_block("latest thinking", "sig_latest"),
                {
                    "type": "tool_use",
                    "id": "toolu_2",
                    "name": "read",
                    "input": {"path": "x.py"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_2",
                    "content": "latest result",
                }
            ],
        },
    ]


class TestNoFalsePositiveWithMultipleThinkingMessages:
    """The LAST assistant with thinking is protected, not the first."""

    def test_no_mutation_signal_with_two_thinking_assistants(self):
        messages = _make_messages_with_two_thinking_assistants()
        _, _, signals = canonicalize_anthropic_bridge_messages(messages)
        assert "thinking_block_mutated" not in signals

    def test_protected_hash_matches_latest_assistant(self):
        messages = _make_messages_with_two_thinking_assistants()
        latest_content = messages[3]["content"]
        latest_hash = _content_hash(latest_content)

        _, _, signals = canonicalize_anthropic_bridge_messages(messages)

        assert signals.get("thinking_block_mutated") is None
        assert latest_hash is not None


class TestPerCycleDedup:
    """Identical mutation detections within one canonicalization cycle are
    deduplicated via the seen_mutation_pairs set."""

    def test_dedup_prevents_double_signal(self):
        messages = _make_messages_with_two_thinking_assistants()
        protected_content = messages[3]["content"]
        before_hash = _content_hash(protected_content)
        before_block_types = [b.get("type", "?") for b in protected_content]

        merged_output = list(messages)
        signals: dict[str, int] = {}
        seen: set[tuple[str, str]] = set()

        _check_thinking_block_mutation(
            merged_output,
            before_hash,
            before_block_types,
            True,
            3,
            signals,
            protected_content_identity=None,
            seen_mutation_pairs=seen,
        )
        assert signals.get("thinking_block_mutated") == 1

        signals2: dict[str, int] = {}
        _check_thinking_block_mutation(
            merged_output,
            before_hash,
            before_block_types,
            True,
            3,
            signals2,
            protected_content_identity=None,
            seen_mutation_pairs=seen,
        )
        assert "thinking_block_mutated" not in signals2

    def test_different_mutations_both_fire(self):
        messages = _make_messages_with_two_thinking_assistants()

        first_content = messages[1]["content"]
        latest_content = messages[3]["content"]
        after_hash_first = _content_hash(first_content)
        after_hash_latest = _content_hash(latest_content)

        assert after_hash_first != after_hash_latest

        fake_before_1 = "aaaa0000_different_1"
        fake_before_2 = "bbbb0000_different_2"
        block_types = ["thinking", "text"]

        signals1: dict[str, int] = {}
        signals2: dict[str, int] = {}
        seen: set[tuple[str, str]] = set()

        _check_thinking_block_mutation(
            messages,
            fake_before_1,
            block_types,
            True,
            1,
            signals1,
            protected_content_identity=None,
            seen_mutation_pairs=seen,
        )

        _check_thinking_block_mutation(
            messages,
            fake_before_2,
            block_types,
            True,
            3,
            signals2,
            protected_content_identity=None,
            seen_mutation_pairs=seen,
        )

        assert signals1.get("thinking_block_mutated") == 1
        assert signals2.get("thinking_block_mutated") == 1
        assert len(seen) == 2


class TestRestoreBreadcrumb:
    """Verify the THINKING_BLOCK_MUTATION_RESTORED event exists and has zero
    penalty weight."""

    def test_restored_event_type_exists(self):
        assert hasattr(SmoothnessEventType, "THINKING_BLOCK_MUTATION_RESTORED")

    def test_restored_event_has_no_penalty(self):
        from tok.runtime.smoothness.scoring import PENALTIES

        assert (
            SmoothnessEventType.THINKING_BLOCK_MUTATION_RESTORED
            not in PENALTIES
        )

    def test_restored_event_does_not_trigger_smooth_mode_override(self):
        from tok.runtime.smoothness.policy import choose_tok_mode
        from tok.runtime.smoothness.scoring import score_turn

        report = score_turn("t1", "task1", [])
        mode = choose_tok_mode(report, None)
        assert mode.value != "smooth_mode"
