"""TDD tests for tok.compression._audit — audit/dry-run mode.

Sub-stages:
  1. Dataclasses exist with correct fields (RED → GREEN → regression)
  2. audit_messages exists and returns AuditReport
  3. Aggregate token counts correct
  4. Per-message tokens_before populated
  5. Per-message tokens_after populated (dropped messages = 0)
  6. type_breakdown populated from pipeline
  8. Public API: from tok.compression import audit_messages
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _make_user_message(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def _make_assistant_message(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def _make_tool_use_message(tool_use_id: str, tool_name: str, command: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": tool_use_id,
                "name": tool_name,
                "input": {"command": command},
            }
        ],
    }


def _make_tool_result_message(tool_use_id: str, content: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content,
            }
        ],
    }


def _make_long_convo(n_turns: int, content: str = "short") -> list[dict[str, Any]]:
    """Build n_turns user+assistant pairs."""
    msgs: list[dict[str, Any]] = []
    for i in range(n_turns):
        msgs.append({"role": "user", "content": f"{content} user {i}"})
        msgs.append({"role": "assistant", "content": f"{content} assistant {i}"})
    return msgs


# ---------------------------------------------------------------------------
# Stage 1: Dataclasses exist with correct fields
# ---------------------------------------------------------------------------


class TestAuditDataclasses:
    def test_stage_hit_has_key_and_tokens_saved(self) -> None:
        from tok.compression._audit import StageHit

        sh = StageHit(key="command_cached", tokens_saved=120)
        assert sh.key == "command_cached"
        assert sh.tokens_saved == 120

    def test_stage_hit_is_frozen(self) -> None:
        from tok.compression._audit import StageHit

        sh = StageHit(key="x", tokens_saved=0)
        raised = False
        try:
            sh.key = "y"  # type: ignore[misc]
        except Exception:
            raised = True
        assert raised, "StageHit should be frozen"

    def test_message_audit_fields(self) -> None:
        from tok.compression._audit import MessageAudit

        ma = MessageAudit(index=0, role="user", tokens_before=100, tokens_after=80, tokens_saved=20)
        assert ma.index == 0
        assert ma.role == "user"
        assert ma.tokens_before == 100
        assert ma.tokens_after == 80
        assert ma.tokens_saved == 20

    def test_audit_report_fields(self) -> None:
        from tok.compression._audit import AuditReport, MessageAudit

        ma = MessageAudit(index=0, role="user", tokens_before=10, tokens_after=8, tokens_saved=2)
        report = AuditReport(
            total_tokens_before=10,
            total_tokens_after=8,
            total_tokens_saved=2,
            messages=(ma,),
            type_breakdown={"command_cached": 5},
        )
        assert report.total_tokens_before == 10
        assert report.total_tokens_after == 8
        assert report.total_tokens_saved == 2
        assert len(report.messages) == 1
        assert report.type_breakdown == {"command_cached": 5}


# ---------------------------------------------------------------------------
# Stage 2: audit_messages exists and returns AuditReport
# ---------------------------------------------------------------------------


class TestAuditFunction:
    def test_returns_audit_report(self) -> None:
        from tok.compression._audit import AuditReport, audit_messages

        msgs = [_make_user_message("hello")]
        result = audit_messages(msgs)
        assert isinstance(result, AuditReport)

    def test_accepts_system_prompt(self) -> None:
        from tok.compression._audit import AuditReport, audit_messages

        msgs = [_make_user_message("hello")]
        result = audit_messages(msgs, system_prompt="You are helpful.")
        assert isinstance(result, AuditReport)

    def test_accepts_keep_turns(self) -> None:
        from tok.compression._audit import AuditReport, audit_messages

        msgs = [_make_user_message("hello")]
        result = audit_messages(msgs, keep_turns=4)
        assert isinstance(result, AuditReport)


# ---------------------------------------------------------------------------
# Stage 3: Aggregate token counts correct
# ---------------------------------------------------------------------------


class TestAuditAggregateTokens:
    def test_total_tokens_before_counts_all_messages(self) -> None:
        from tok.compression._audit import audit_messages
        from tok.utils.token_utils import count_tokens

        msgs = [
            _make_user_message("hello world"),
            _make_assistant_message("hi there"),
        ]
        result = audit_messages(msgs, keep_turns=8)
        expected = count_tokens("hello world") + count_tokens("hi there")
        assert result.total_tokens_before == expected

    def test_empty_message_list_returns_zeros(self) -> None:
        from tok.compression._audit import audit_messages

        result = audit_messages([])
        assert result.total_tokens_before == 0
        assert result.total_tokens_after == 0
        assert result.total_tokens_saved == 0

    def test_total_tokens_saved_equals_before_minus_after(self) -> None:
        from tok.compression._audit import audit_messages

        msgs = [
            _make_user_message("hello"),
            _make_assistant_message("hi"),
        ]
        result = audit_messages(msgs, keep_turns=8)
        assert result.total_tokens_saved == result.total_tokens_before - result.total_tokens_after


# ---------------------------------------------------------------------------
# Stage 4: Per-message tokens_before populated
# ---------------------------------------------------------------------------


class TestAuditPerMessageBefore:
    def test_per_message_tokens_before_populated(self) -> None:
        from tok.compression._audit import audit_messages
        from tok.utils.token_utils import count_tokens

        msgs = [
            _make_user_message("the first user message"),
            _make_assistant_message("the assistant reply"),
        ]
        result = audit_messages(msgs, keep_turns=8)
        assert len(result.messages) == 2
        assert result.messages[0].tokens_before == count_tokens("the first user message")
        assert result.messages[1].tokens_before == count_tokens("the assistant reply")

    def test_per_message_roles_populated(self) -> None:
        from tok.compression._audit import audit_messages

        msgs = [
            _make_user_message("hello"),
            _make_assistant_message("hi"),
        ]
        result = audit_messages(msgs, keep_turns=8)
        assert result.messages[0].role == "user"
        assert result.messages[1].role == "assistant"

    def test_per_message_index_populated(self) -> None:
        from tok.compression._audit import audit_messages

        msgs = [
            _make_user_message("a"),
            _make_assistant_message("b"),
            _make_user_message("c"),
        ]
        result = audit_messages(msgs, keep_turns=8)
        assert [m.index for m in result.messages] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Stage 5: Per-message tokens_after populated (dropped = 0)
# ---------------------------------------------------------------------------


class TestAuditPerMessageAfter:
    def test_kept_messages_have_correct_tokens_after(self) -> None:
        from tok.compression._audit import audit_messages
        from tok.utils.token_utils import count_tokens

        msgs = [
            _make_user_message("keep this message"),
            _make_assistant_message("also keep this"),
        ]
        result = audit_messages(msgs, keep_turns=8)
        # Short history — nothing dropped
        assert result.messages[0].tokens_after == count_tokens("keep this message")
        assert result.messages[1].tokens_after == count_tokens("also keep this")

    def test_dropped_messages_have_tokens_after_zero(self) -> None:
        from tok.compression._audit import audit_messages

        # 6 turns = 12 messages; keep_turns=2 should drop old ones
        msgs = _make_long_convo(6)
        result = audit_messages(msgs, keep_turns=2)
        dropped = [m for m in result.messages if m.tokens_after == 0]
        assert len(dropped) > 0

    def test_kept_messages_at_tail_have_nonzero_tokens_after(self) -> None:
        from tok.compression._audit import audit_messages

        msgs = _make_long_convo(6)
        result = audit_messages(msgs, keep_turns=2)
        # The last 2 turns (4 messages) should be kept
        tail = result.messages[-4:]
        assert all(m.tokens_after > 0 for m in tail)

    def test_tokens_saved_per_message_equals_before_minus_after(self) -> None:
        from tok.compression._audit import audit_messages

        msgs = _make_long_convo(5)
        result = audit_messages(msgs, keep_turns=2)
        for ma in result.messages:
            assert ma.tokens_saved == ma.tokens_before - ma.tokens_after


# ---------------------------------------------------------------------------
# Stage 6: type_breakdown populated from pipeline
# ---------------------------------------------------------------------------


class TestAuditTypeBreakdown:
    def test_type_breakdown_is_dict(self) -> None:
        from tok.compression._audit import audit_messages

        result = audit_messages([_make_user_message("hi")])
        assert isinstance(result.type_breakdown, dict)

    def test_type_breakdown_empty_for_plain_messages(self) -> None:
        from tok.compression._audit import audit_messages

        msgs = [_make_user_message("hello"), _make_assistant_message("hi")]
        result = audit_messages(msgs)
        # Plain text messages produce no compression breakdown
        assert result.type_breakdown == {}

    def test_type_breakdown_populated_for_repeated_tool_result(self) -> None:
        from tok.compression._audit import audit_messages

        large_content = "output line\n" * 100
        msgs = [
            _make_tool_use_message("t1", "bash", "ls /tmp"),
            _make_tool_result_message("t1", large_content),
            _make_tool_use_message("t2", "bash", "ls /tmp"),
            _make_tool_result_message("t2", large_content),
            _make_user_message("what did you find?"),
        ]
        result = audit_messages(msgs)
        assert len(result.type_breakdown) > 0

    def test_type_breakdown_values_are_ints(self) -> None:
        from tok.compression._audit import audit_messages

        large_content = "x\n" * 200
        msgs = [
            _make_tool_use_message("t1", "bash", "echo test"),
            _make_tool_result_message("t1", large_content),
            _make_user_message("ok"),
        ]
        result = audit_messages(msgs)
        for v in result.type_breakdown.values():
            assert isinstance(v, int)


# ---------------------------------------------------------------------------
# Stage 8: Public API exposure
# ---------------------------------------------------------------------------


class TestAuditPublicAPI:
    def test_importable_from_tok_compression(self) -> None:
        from tok.compression import audit_messages  # noqa: F401

        assert callable(audit_messages)

    def test_audit_report_importable_from_audit_module(self) -> None:
        from tok.compression._audit import AuditReport  # noqa: F401

        assert True

    def test_audit_messages_returns_audit_report_when_imported_from_public(self) -> None:
        from tok.compression import audit_messages
        from tok.compression._audit import AuditReport

        msgs = [{"role": "user", "content": "hello"}]
        result = audit_messages(msgs)
        assert isinstance(result, AuditReport)
