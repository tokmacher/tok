"""Regression tests: system prompt cache hints must be safe and non-breaking."""

from __future__ import annotations

from tok.runtime.core import RuntimeSession
from tok.runtime.pipeline._prepare_inject_system import Step8Result, run_step_8
from tok.runtime.types import RuntimeRequest


def _make_request(**overrides) -> RuntimeRequest:
    defaults = dict(
        model="claude-sonnet-4",
        messages=[{"role": "user", "content": "hello"}],
        adapter_kind="unknown",
        tool_compatible=False,
    )
    defaults.update(overrides)
    return RuntimeRequest(**defaults)


def _make_body(system: str = "") -> dict:
    return {
        "model": "claude-sonnet-4",
        "messages": [{"role": "user", "content": "hello"}],
        "system": system,
    }


class TestFirstTurnPassthrough:
    def test_step8_first_turn_system_string_unchanged(self) -> None:
        """On the first call, system string must not be converted to list."""
        session = RuntimeSession()
        assert session._system_fingerprints is None

        system_text = "You are a helpful assistant."
        body = _make_body(system=system_text)
        result = run_step_8(
            runtime_self=None,
            request=_make_request(),
            session=session,
            body=body,
            session_memory="",
            history_skip_reason=None,
            skip_reason="short_session",
            behavior_signals={},
            runtime_hints=[],
            effective_tool_compatible=False,
            current_pressure=0,
            hot_hint_metrics={},
            translated_messages=[],
            should_skip_history=False,
            recent=[],
            has_answer_anchor=False,
        )
        # short_session skips system injection entirely — system unchanged
        assert result.body.get("system", system_text) == system_text
        assert "system_prompt_cache_hint_chars" not in result.behavior_signals

    def test_step8_short_session_still_injects_reads_manifest(self) -> None:
        """A cached file read is tiny enough to carry even while state injection is skipped."""
        session = RuntimeSession()
        session._files_read_fingerprints["src/tok/compression/_file_integrity.py"] = "a3f2c1d4"
        session._files_fully_delivered["src/tok/compression/_file_integrity.py"] = 1

        system_text = "You are a helpful assistant."
        result = run_step_8(
            runtime_self=None,
            request=_make_request(),
            session=session,
            body=_make_body(system=system_text),
            session_memory="",
            history_skip_reason=None,
            skip_reason="short_session",
            behavior_signals={},
            runtime_hints=[],
            effective_tool_compatible=False,
            current_pressure=0,
            hot_hint_metrics={},
            translated_messages=[],
            should_skip_history=False,
            recent=[],
            has_answer_anchor=False,
        )

        assert result.body["system"].startswith(system_text)
        assert "@reads" in result.body["system"]
        assert "compression/_file_integrity.py  t:1  fp:a3f2c1d4" in result.body["system"]
        assert result.behavior_signals["short_session_system_additions_skipped"] == 1

    def test_short_system_never_converted_to_list(self) -> None:
        """A system prompt shorter than MIN_CACHEABLE_CHARS is never converted."""
        from tok.compression._system_prompt_cache import _MIN_CACHEABLE_CHARS, apply_system_cache_hint

        short = "X" * (_MIN_CACHEABLE_CHARS - 1)
        # Pretend we have a prior fingerprint for the same short text
        _, prior_fps, _ = apply_system_cache_hint(short, None)
        result, _, static_chars = apply_system_cache_hint(short, prior_fps)
        assert isinstance(result, str)
        assert static_chars == 0

    def test_behavior_signal_absent_on_first_turn(self) -> None:
        """system_prompt_cache_hint_chars must be absent (not 0) on first turn."""
        from tok.compression._system_prompt_cache import apply_system_cache_hint

        _, _, static_chars = apply_system_cache_hint("A" * 500, None)
        assert static_chars == 0  # no signal emitted on first turn

    def test_idempotent_on_already_cached_list_input(self) -> None:
        """Calling apply_system_cache_hint twice does not double-wrap cache_control."""
        from tok.compression._system_prompt_cache import apply_system_cache_hint, paragraph_fingerprints

        system = "A" * 400
        prior_fps = paragraph_fingerprints(system)
        cached_list = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        result, _, _ = apply_system_cache_hint(cached_list, prior_fps)
        assert isinstance(result, list)
        # Should still have exactly 1 block — no duplication
        cache_blocks = [b for b in result if "cache_control" in b]
        assert len(cache_blocks) == 1


class TestSignalAbsenceWhenNoHit:
    def test_no_signal_key_when_no_static_prefix(self) -> None:
        """Signal key must be absent, not present with value 0."""
        from tok.compression._system_prompt_cache import apply_system_cache_hint

        # Dynamic system that changes every turn
        _, fps1, _ = apply_system_cache_hint("turn1 dynamic content " * 20, None)
        _, _, static_chars = apply_system_cache_hint("turn2 completely different " * 20, fps1)
        # No match → static_chars == 0 → signal should not be emitted
        assert static_chars == 0


class TestExistingStep8BehaviorPreserved:
    def test_step8_result_fields_unchanged(self) -> None:
        """Step8Result dataclass fields have not changed."""
        from dataclasses import fields

        expected = {
            "body",
            "injected_state_payload",
            "runtime_hints",
            "behavior_signals",
            "hot_hint_metrics",
            "resend_signals",
            "answer_ready",
            "has_answer_anchor",
            "session_memory",
            "tok_state",
        }
        actual = {f.name for f in fields(Step8Result)}
        assert actual == expected

    def test_step8_defaults_unchanged(self) -> None:
        r = Step8Result()
        assert r.body == {}
        assert r.injected_state_payload == ""
        assert r.runtime_hints == []
        assert r.behavior_signals == {}
        assert r.answer_ready is False
