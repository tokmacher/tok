"""Regression tests for audit/dry-run mode — mutation safety and invariants.

Stage 7: audit_messages does NOT mutate the original messages list.
Plus cross-stage invariant regressions from stages 1, 3, 4, 5, 6, 8.
"""

from __future__ import annotations

import copy
import inspect
from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_user_message(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def _make_assistant_message(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def _make_long_convo(n_turns: int, content: str = "short") -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    for i in range(n_turns):
        msgs.append({"role": "user", "content": f"{content} user {i}"})
        msgs.append({"role": "assistant", "content": f"{content} assistant {i}"})
    return msgs


# ---------------------------------------------------------------------------
# Stage 1 regressions — dataclass invariants
# ---------------------------------------------------------------------------


class TestAuditDataclassesRegression:
    def test_stage_hit_equality(self) -> None:
        from tok.compression._audit import StageHit

        assert StageHit(key="a", tokens_saved=1) == StageHit(key="a", tokens_saved=1)
        assert StageHit(key="a", tokens_saved=1) != StageHit(key="b", tokens_saved=1)

    def test_message_audit_is_frozen(self) -> None:
        from tok.compression._audit import MessageAudit

        ma = MessageAudit(index=0, role="user", tokens_before=10, tokens_after=8, tokens_saved=2)
        raised = False
        try:
            ma.index = 99  # type: ignore[misc]
        except Exception:
            raised = True
        assert raised, "MessageAudit should be frozen"

    def test_audit_report_messages_is_tuple(self) -> None:
        from tok.compression._audit import AuditReport

        report = AuditReport(
            total_tokens_before=0,
            total_tokens_after=0,
            total_tokens_saved=0,
            messages=(),
            type_breakdown={},
        )
        assert isinstance(report.messages, tuple)


# ---------------------------------------------------------------------------
# Stage 3 regressions — aggregate invariants
# ---------------------------------------------------------------------------


class TestAggregateTokensRegression:
    def test_total_saved_is_never_negative(self) -> None:
        from tok.compression._audit import audit_messages

        msgs = [_make_user_message("x")]
        result = audit_messages(msgs, keep_turns=8)
        assert result.total_tokens_saved >= 0

    def test_single_message_list_aggregate(self) -> None:
        from tok.compression._audit import audit_messages
        from tok.utils.token_utils import count_tokens

        msgs = [_make_user_message("just one message here")]
        result = audit_messages(msgs, keep_turns=8)
        assert result.total_tokens_before == count_tokens("just one message here")
        assert result.total_tokens_saved >= 0


# ---------------------------------------------------------------------------
# Stage 4 regressions — per-message count invariants
# ---------------------------------------------------------------------------


class TestPerMessageBeforeRegression:
    def test_message_count_equals_input_length(self) -> None:
        from tok.compression._audit import audit_messages

        for n in (0, 1, 3, 7):
            msgs = [_make_user_message(f"msg {i}") for i in range(n)]
            result = audit_messages(msgs, keep_turns=8)
            assert len(result.messages) == n


# ---------------------------------------------------------------------------
# Stage 5 regressions — alignment invariants
# ---------------------------------------------------------------------------


class TestPerMessageAfterRegression:
    def test_sum_of_per_message_tokens_after_equals_total_after(self) -> None:
        from tok.compression._audit import audit_messages

        msgs = _make_long_convo(6)
        result = audit_messages(msgs, keep_turns=2)
        assert sum(m.tokens_after for m in result.messages) == result.total_tokens_after

    def test_sum_of_per_message_tokens_before_equals_total_before(self) -> None:
        from tok.compression._audit import audit_messages

        msgs = _make_long_convo(4)
        result = audit_messages(msgs, keep_turns=8)
        assert sum(m.tokens_before for m in result.messages) == result.total_tokens_before

    def test_no_message_has_negative_tokens_saved(self) -> None:
        from tok.compression._audit import audit_messages

        msgs = _make_long_convo(5)
        result = audit_messages(msgs, keep_turns=2)
        assert all(m.tokens_saved >= 0 for m in result.messages)


# ---------------------------------------------------------------------------
# Stage 6 regressions — breakdown invariants
# ---------------------------------------------------------------------------


class TestTypeBreakdownRegression:
    def test_type_breakdown_values_nonnegative(self) -> None:
        from tok.compression._audit import audit_messages

        large = "line\n" * 100
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "bash",
                        "input": {"command": "ls"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": large,
                    }
                ],
            },
            _make_user_message("done"),
        ]
        result = audit_messages(msgs)
        assert all(v >= 0 for v in result.type_breakdown.values())


# ---------------------------------------------------------------------------
# Stage 7: Mutation safety
# ---------------------------------------------------------------------------


class TestAuditMutationSafety:
    def test_original_messages_not_mutated(self) -> None:
        from tok.compression._audit import audit_messages

        large_content = "file content\n" * 100
        original_msgs: list[dict[str, Any]] = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "bash",
                        "input": {"command": "cat foo.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": large_content,
                    }
                ],
            },
            _make_user_message("done"),
        ]
        snapshot_before = copy.deepcopy(original_msgs)
        audit_messages(original_msgs, keep_turns=8)
        assert original_msgs == snapshot_before

    def test_original_messages_content_not_mutated(self) -> None:
        from tok.compression._audit import audit_messages

        content = "x\n" * 200
        msgs: list[dict[str, Any]] = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "u1",
                        "name": "bash",
                        "input": {"command": "ls"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "u1",
                        "content": content,
                    }
                ],
            },
        ]
        original_content = msgs[1]["content"][0]["content"]
        audit_messages(msgs, keep_turns=8)
        assert msgs[1]["content"][0]["content"] == original_content

    def test_calling_audit_twice_gives_same_result(self) -> None:
        from tok.compression._audit import audit_messages

        msgs = _make_long_convo(4)
        result1 = audit_messages(msgs, keep_turns=2)
        result2 = audit_messages(msgs, keep_turns=2)
        assert result1.total_tokens_before == result2.total_tokens_before
        assert result1.total_tokens_after == result2.total_tokens_after


# ---------------------------------------------------------------------------
# Stage 8 regressions — public API invariants
# ---------------------------------------------------------------------------


class TestPublicAPIRegression:
    def test_audit_messages_signature_accepts_all_params(self) -> None:
        from tok.compression import audit_messages

        sig = inspect.signature(audit_messages)
        params = set(sig.parameters)
        assert "messages" in params
        assert "system_prompt" in params
        assert "keep_turns" in params

    def test_audit_messages_in_compression_all(self) -> None:
        import tok.compression

        assert "audit_messages" in tok.compression.__all__
