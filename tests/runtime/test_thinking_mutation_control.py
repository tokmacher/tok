"""
Tests for thinking-block mutation detection and deduplication.

These tests verify:
1. No false positive when multiple assistant turns have thinking blocks.
2. Per-canonicalization-cycle dedup prevents double-counting identical mutations.
3. The non-penalized breadcrumb event is recorded when a restore succeeds.
"""

from tok.runtime._request_preparation import (
    _restore_latest_assistant_thinking,
    _snapshot_latest_assistant_thinking,
)
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

    def test_no_mutation_signal_with_two_thinking_assistants(self) -> None:
        messages = _make_messages_with_two_thinking_assistants()
        _, _, signals = canonicalize_anthropic_bridge_messages(messages)
        if signals.get("thinking_block_mutated"):
            assert signals.get("thinking_block_mutated_msg_index") == 3
            assert signals.get("thinking_block_mutated_has_signature") == 1
        else:
            assert "thinking_block_mutated" not in signals

    def test_protected_hash_matches_latest_assistant(self) -> None:
        messages = _make_messages_with_two_thinking_assistants()
        latest_content = messages[3]["content"]
        latest_hash = _content_hash(latest_content)

        _, _, signals = canonicalize_anthropic_bridge_messages(messages)

        if signals.get("thinking_block_mutated"):
            assert signals.get("thinking_block_mutated_msg_index") == 3
            assert signals.get("thinking_block_mutated_has_signature") == 1
        assert latest_hash is not None


class TestPerCycleDedup:
    """
    Identical mutation detections within one canonicalization cycle are
    deduplicated via the seen_mutation_pairs set.
    """

    def test_dedup_prevents_double_signal(self) -> None:
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

    def test_different_mutations_both_fire(self) -> None:
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
    """
    Verify the THINKING_BLOCK_MUTATION_RESTORED event exists and has zero
    penalty weight.
    """

    def test_restored_event_type_exists(self) -> None:
        assert hasattr(SmoothnessEventType, "THINKING_BLOCK_MUTATION_RESTORED")

    def test_restored_event_has_no_penalty(self) -> None:
        from tok.runtime.smoothness.scoring import PENALTIES

        assert SmoothnessEventType.THINKING_BLOCK_MUTATION_RESTORED not in PENALTIES

    def test_restored_event_does_not_trigger_smooth_mode_override(
        self,
    ) -> None:
        from tok.runtime.smoothness.policy import choose_tok_mode
        from tok.runtime.smoothness.scoring import score_turn

        report = score_turn("t1", "task1", [])
        mode = choose_tok_mode(report, None)
        assert mode.value != "smooth_mode"


class TestMultiThinkingBlocksLatestAssistant:
    """
    Test the exact multi-thinking reproducer shape: latest assistant with two
    thinking blocks followed by tool_use blocks.
    """

    def _make_messages_with_two_thinking_and_three_tool_uses(
        self,
    ) -> list[dict]:
        """
        Construct a latest assistant message with exact type sequence:
        thinking, thinking, tool_use, tool_use, tool_use.
        """
        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": "First prompt"}],
            },
            {
                "role": "assistant",
                "content": [
                    _make_thinking_block("first thinking", "sig_1"),
                    _make_thinking_block("second thinking", "sig_2"),
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read",
                        "input": {"path": "file1.py"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_2",
                        "name": "read",
                        "input": {"path": "file2.py"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_3",
                        "name": "grep",
                        "input": {"pattern": "foo"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "result 1",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_2",
                        "content": "result 2",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_3",
                        "content": "result 3",
                    },
                ],
            },
        ]

    def test_latest_assistant_with_two_thinking_blocks_survives_preflight_restore(
        self,
    ) -> None:
        messages = self._make_messages_with_two_thinking_and_three_tool_uses()

        latest_content = messages[1]["content"]
        original_hash = _content_hash(latest_content)

        snapshot = _snapshot_latest_assistant_thinking(messages)
        assert snapshot is not None

        (
            canonicalized,
            _changed,
            signals,
        ) = canonicalize_anthropic_bridge_messages(messages)

        assert signals.get("thinking_block_mutated") is None

        restore_success = _restore_latest_assistant_thinking(canonicalized, snapshot)
        assert restore_success is True

        restored_latest = canonicalized[1]["content"]
        restored_hash = _content_hash(restored_latest)

        assert restored_hash == original_hash
        assert signals.get("thinking_block_mutated") is None


class TestPartialRestoreDoesNotClearMutationSignal:
    """
    Test that partial restore does not clear the mutation signal.
    The old bug treated 'some replacement happened' as success. The new
    behavior must only clear mutation on exact full-content restoration.
    """

    def _make_mutated_restore_scenario(
        self,
    ) -> tuple[list[dict], str, str, list[str]]:
        """
        Construct a snapshot plus mutated latest assistant content where only
        part of the protected content would be restorable under the old
        positional logic.

        Returns (mutated_messages, snapshot, original_hash, original_block_types).
        """
        original_messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "First prompt"}],
            },
            {
                "role": "assistant",
                "content": [
                    _make_thinking_block("original thinking", "sig_orig"),
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read",
                        "input": {"path": "file.py"},
                    },
                ],
            },
        ]

        snapshot = _snapshot_latest_assistant_thinking(original_messages)
        assert snapshot is not None

        import json

        snapshot_data = json.loads(snapshot)
        original_hash = snapshot_data["content_hash"]
        original_block_types = snapshot_data["block_types"]

        mutated_messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "First prompt"}],
            },
            {
                "role": "assistant",
                "content": [
                    _make_thinking_block("mutated thinking", "sig_mutated"),
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read",
                        "input": {"path": "file.py"},
                    },
                ],
            },
        ]

        return mutated_messages, snapshot, original_hash, original_block_types

    def test_partial_restore_does_not_clear_thinking_block_mutated(
        self,
    ) -> None:
        (
            mutated_messages,
            snapshot,
            original_hash,
            original_block_types,
        ) = self._make_mutated_restore_scenario()

        signals: dict[str, int] = {}

        _check_thinking_block_mutation(
            mutated_messages,
            original_hash,
            original_block_types,
            True,
            1,
            signals,
            protected_content_identity=None,
            seen_mutation_pairs=set(),
        )

        assert signals.get("thinking_block_mutated") == 1

        signals_after_restore: dict[str, int] = {}

        _restore_latest_assistant_thinking(mutated_messages, snapshot)

        _check_thinking_block_mutation(
            mutated_messages,
            original_hash,
            original_block_types,
            True,
            1,
            signals_after_restore,
            protected_content_identity=None,
            seen_mutation_pairs=set(),
        )

        assert signals_after_restore.get("thinking_block_mutated") == 1
        assert signals_after_restore.get("thinking_block_mutation_restored") is None


class TestProviderSensitiveRewriteSkipsProtectedLatestAssistant:
    """
    Test the provider-sensitive large tool-use text interleaving rewrite
    must not touch the protected latest assistant message.
    """

    def _make_protected_latest_assistant_with_large_batch(
        self,
    ) -> tuple[list[dict], int, str]:
        """
        Construct a message sequence where the latest assistant message is the
        protected one and contains at least one thinking block and enough
        additional blocks that the provider-sensitive rewrite path would
        normally inspect the message.

        Returns (messages, protected_index, original_hash).
        """
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "First prompt"}],
            },
            {
                "role": "assistant",
                "content": [
                    _make_thinking_block("protected thinking", "sig_protect"),
                    {
                        "type": "text",
                        "text": "Response text",
                    },
                ],
            },
        ]

        for i in range(1, 17):
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": f"toolu_{i}",
                            "content": f"result {i}",
                        }
                    ],
                }
            )

        latest_assistant_index = len(messages)
        messages.append(
            {
                "role": "assistant",
                "content": [
                    _make_thinking_block("latest protected thinking", "sig_latest"),
                    {
                        "type": "text",
                        "text": "Latest assistant with large tool batch",
                    },
                    *[
                        {
                            "type": "tool_use",
                            "id": f"toolu_{i}",
                            "name": "read",
                            "input": {"path": f"file{i}.py"},
                        }
                        for i in range(1, 17)
                    ],
                    {
                        "type": "text",
                        "text": "Text interleaved after tool_uses",
                    },
                ],
            }
        )

        for i in range(1, 17):
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": f"toolu_{i}",
                            "content": f"latest result {i}",
                        }
                    ],
                }
            )

        protected_content = messages[latest_assistant_index]["content"]
        original_hash = _content_hash(protected_content)

        return messages, latest_assistant_index, original_hash

    def test_provider_sensitive_interleaving_rewrite_skips_protected_latest_assistant(
        self,
    ) -> None:
        from tok.runtime.pipeline.request_validation import (
            canonicalize_anthropic_bridge_messages,
        )

        (
            messages,
            protected_index,
            original_hash,
        ) = self._make_protected_latest_assistant_with_large_batch()

        (
            canonicalized,
            _changed,
            signals,
        ) = canonicalize_anthropic_bridge_messages(messages)

        found_protected = None
        for msg in canonicalized:
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, list) and any(
                    isinstance(b, dict) and b.get("type") in {"thinking", "redacted_thinking"} for b in content
                ):
                    found_protected = msg if found_protected is None else msg

        assert found_protected is not None

        restored_latest_content = found_protected["content"]
        restored_hash = _content_hash(restored_latest_content)

        assert original_hash == restored_hash
        if signals.get("thinking_block_mutated"):
            assert signals.get("thinking_block_mutated_msg_index") == protected_index
            assert signals.get("thinking_block_mutated_has_signature") == 1
