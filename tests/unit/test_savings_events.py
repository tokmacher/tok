"""Section 5.2.2: Per-Request Savings Event JSONL — RED tests.

Verifies SavingsEvent schema correctness, JSONL round-trip, and append durability.
"""

from __future__ import annotations

import dataclasses
import json
import uuid
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestSavingsEventSchema:
    """SavingsEvent must declare all required fields from docs/savings-accounting.md."""

    _REQUIRED_FIELDS = [
        "schema",
        "event_id",
        "session_id",
        "request_id",
        "timestamp",
        "model",
        "mode",
        "request_policy",
        "baseline_input_tokens",
        "actual_input_tokens",
        "input_tokens_saved",
        "baseline_output_tokens",
        "actual_output_tokens",
        "output_tokens_saved",
        "cache_read_tokens",
        "cache_write_tokens",
        "baseline_cost_usd",
        "actual_cost_usd",
        "cost_saved_usd",
        "fallback",
        "degraded_to_baseline",
        "compression_paths",
        "non_headline_estimates",
    ]

    def test_savings_event_declares_all_required_fields(self) -> None:
        from tok.utils.savings_event import SavingsEvent

        fields = {f.name for f in dataclasses.fields(SavingsEvent)}
        missing = [f for f in self._REQUIRED_FIELDS if f not in fields]
        assert not missing, f"SavingsEvent is missing fields: {missing}"

    def test_savings_event_schema_field_default(self) -> None:
        from tok.utils.savings_event import SavingsEvent

        ev = SavingsEvent(
            event_id=str(uuid.uuid4()),
            session_id="abc12345",
            request_id="turn-1-claude-sonnet",
            timestamp="2026-05-20T00:00:00Z",
            model="claude-sonnet-4",
        )
        assert ev.schema == "tok-savings-event/v1"

    def test_savings_event_token_fields_default_to_zero(self) -> None:
        from tok.utils.savings_event import SavingsEvent

        ev = SavingsEvent(
            event_id=str(uuid.uuid4()),
            session_id="abc12345",
            request_id="turn-1",
            timestamp="2026-05-20T00:00:00Z",
            model="claude-sonnet-4",
        )
        assert ev.baseline_input_tokens == 0
        assert ev.actual_input_tokens == 0
        assert ev.input_tokens_saved == 0
        assert ev.baseline_output_tokens == 0
        assert ev.actual_output_tokens == 0
        assert ev.output_tokens_saved == 0
        assert ev.cache_read_tokens == 0
        assert ev.cache_write_tokens == 0

    def test_savings_event_cost_fields_default_to_zero(self) -> None:
        from tok.utils.savings_event import SavingsEvent

        ev = SavingsEvent(
            event_id=str(uuid.uuid4()),
            session_id="abc12345",
            request_id="turn-1",
            timestamp="2026-05-20T00:00:00Z",
            model="claude-sonnet-4",
        )
        assert ev.baseline_cost_usd == 0.0
        assert ev.actual_cost_usd == 0.0
        assert ev.cost_saved_usd == 0.0

    def test_savings_event_boolean_fields_default_false(self) -> None:
        from tok.utils.savings_event import SavingsEvent

        ev = SavingsEvent(
            event_id=str(uuid.uuid4()),
            session_id="abc12345",
            request_id="turn-1",
            timestamp="2026-05-20T00:00:00Z",
            model="claude-sonnet-4",
        )
        assert ev.fallback is False
        assert ev.degraded_to_baseline is False

    def test_savings_event_map_fields_default_empty(self) -> None:
        from tok.utils.savings_event import SavingsEvent

        ev = SavingsEvent(
            event_id=str(uuid.uuid4()),
            session_id="abc12345",
            request_id="turn-1",
            timestamp="2026-05-20T00:00:00Z",
            model="claude-sonnet-4",
        )
        assert ev.compression_paths == {}
        assert ev.non_headline_estimates == {}

    def test_savings_event_with_full_data(self) -> None:
        from tok.utils.savings_event import SavingsEvent

        ev = SavingsEvent(
            event_id="evt-001",
            session_id="sess-001",
            request_id="turn-5-claude-sonnet-4",
            timestamp="2026-05-20T12:00:00Z",
            model="claude-sonnet-4",
            mode="tool-compatible",
            request_policy="natural_first",
            baseline_input_tokens=5000,
            actual_input_tokens=3500,
            input_tokens_saved=1500,
            baseline_output_tokens=500,
            actual_output_tokens=450,
            output_tokens_saved=50,
            cache_read_tokens=2000,
            cache_write_tokens=100,
            baseline_cost_usd=0.025,
            actual_cost_usd=0.020,
            cost_saved_usd=0.005,
            fallback=False,
            degraded_to_baseline=False,
            compression_paths={"file": 800, "semantic_dedup": 400},
            non_headline_estimates={"reacquisition_tokens_avoided_estimate": 200},
        )
        assert ev.baseline_input_tokens == 5000
        assert ev.input_tokens_saved == 1500
        assert ev.compression_paths["file"] == 800


# ---------------------------------------------------------------------------
# JSONL serialization tests
# ---------------------------------------------------------------------------


class TestSavingsEventJSONL:
    """SavingsEvent must support JSONL serialization and round-trip."""

    def _make_event(self, event_id: str = "evt-001") -> Any:
        from tok.utils.savings_event import SavingsEvent

        return SavingsEvent(
            event_id=event_id,
            session_id="sess-abc",
            request_id="turn-1",
            timestamp="2026-05-20T00:00:00Z",
            model="claude-sonnet-4",
            mode="tool-compatible",
            request_policy="natural_first",
            baseline_input_tokens=1000,
            actual_input_tokens=700,
            input_tokens_saved=300,
            baseline_cost_usd=0.010,
            actual_cost_usd=0.007,
            cost_saved_usd=0.003,
            compression_paths={"file": 300},
        )

    def test_savings_event_to_dict(self) -> None:
        ev = self._make_event()
        d = ev.to_dict()
        assert isinstance(d, dict)
        assert d["schema"] == "tok-savings-event/v1"
        assert d["event_id"] == "evt-001"
        assert d["input_tokens_saved"] == 300

    def test_savings_event_to_jsonl_line(self) -> None:
        ev = self._make_event()
        line = ev.to_jsonl_line()
        assert isinstance(line, str)
        assert line.endswith("\n")
        parsed = json.loads(line.strip())
        assert parsed["schema"] == "tok-savings-event/v1"
        assert parsed["event_id"] == "evt-001"

    def test_savings_event_round_trip(self) -> None:
        from tok.utils.savings_event import SavingsEvent

        ev = self._make_event()
        line = ev.to_jsonl_line()
        parsed = json.loads(line.strip())
        ev2 = SavingsEvent.from_dict(parsed)
        assert ev2.event_id == ev.event_id
        assert ev2.input_tokens_saved == ev.input_tokens_saved
        assert ev2.compression_paths == ev.compression_paths
        assert ev2.schema == ev.schema

    def test_savings_event_from_dict_with_all_fields(self) -> None:
        from tok.utils.savings_event import SavingsEvent

        data = {
            "schema": "tok-savings-event/v1",
            "event_id": "evt-002",
            "session_id": "sess-xyz",
            "request_id": "turn-2",
            "timestamp": "2026-05-20T01:00:00Z",
            "model": "claude-haiku-4-5",
            "mode": "natural",
            "request_policy": "forced_baseline",
            "baseline_input_tokens": 2000,
            "actual_input_tokens": 2000,
            "input_tokens_saved": 0,
            "baseline_output_tokens": 200,
            "actual_output_tokens": 200,
            "output_tokens_saved": 0,
            "cache_read_tokens": 500,
            "cache_write_tokens": 50,
            "baseline_cost_usd": 0.002,
            "actual_cost_usd": 0.002,
            "cost_saved_usd": 0.000,
            "fallback": True,
            "degraded_to_baseline": True,
            "compression_paths": {},
            "non_headline_estimates": {"reacquisition_tokens_avoided_estimate": 0},
        }
        ev = SavingsEvent.from_dict(data)
        assert ev.fallback is True
        assert ev.degraded_to_baseline is True
        assert ev.input_tokens_saved == 0

    def test_jsonl_line_is_single_line(self) -> None:
        ev = self._make_event()
        line = ev.to_jsonl_line()
        # Must be exactly one newline at the end and no newlines inside
        assert line.count("\n") == 1
        assert line[-1] == "\n"


# ---------------------------------------------------------------------------
# Append durability tests
# ---------------------------------------------------------------------------


class TestSavingsEventAppend:
    """append_savings_event must write and persist JSONL without corruption."""

    def test_append_single_event_creates_file(self, tmp_path: Path) -> None:
        from tok.utils.savings_event import SavingsEvent, append_savings_event

        ev = SavingsEvent(
            event_id="evt-001",
            session_id="sess-001",
            request_id="turn-1",
            timestamp="2026-05-20T00:00:00Z",
            model="claude-sonnet-4",
        )
        out_file = tmp_path / "savings_events.jsonl"
        append_savings_event(ev, out_file)
        assert out_file.exists()
        lines = out_file.read_text().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["event_id"] == "evt-001"

    def test_append_multiple_events_accumulates(self, tmp_path: Path) -> None:
        from tok.utils.savings_event import SavingsEvent, append_savings_event

        out_file = tmp_path / "savings_events.jsonl"
        for i in range(5):
            ev = SavingsEvent(
                event_id=f"evt-{i:03d}",
                session_id="sess-001",
                request_id=f"turn-{i}",
                timestamp="2026-05-20T00:00:00Z",
                model="claude-sonnet-4",
                input_tokens_saved=i * 100,
            )
            append_savings_event(ev, out_file)

        lines = out_file.read_text().splitlines()
        assert len(lines) == 5
        ids = [json.loads(line)["event_id"] for line in lines]
        assert ids == [f"evt-{i:03d}" for i in range(5)]

    def test_append_creates_parent_directory(self, tmp_path: Path) -> None:
        from tok.utils.savings_event import SavingsEvent, append_savings_event

        nested = tmp_path / "sessions" / "sess-abc" / "savings_events.jsonl"
        ev = SavingsEvent(
            event_id="evt-001",
            session_id="sess-abc",
            request_id="turn-1",
            timestamp="2026-05-20T00:00:00Z",
            model="claude-sonnet-4",
        )
        append_savings_event(ev, nested)
        assert nested.exists()

    def test_append_does_not_truncate_existing(self, tmp_path: Path) -> None:
        from tok.utils.savings_event import SavingsEvent, append_savings_event

        out_file = tmp_path / "savings_events.jsonl"
        ev1 = SavingsEvent(
            event_id="evt-001",
            session_id="s",
            request_id="r1",
            timestamp="2026-05-20T00:00:00Z",
            model="claude-sonnet-4",
        )
        append_savings_event(ev1, out_file)

        ev2 = SavingsEvent(
            event_id="evt-002",
            session_id="s",
            request_id="r2",
            timestamp="2026-05-20T00:01:00Z",
            model="claude-sonnet-4",
        )
        append_savings_event(ev2, out_file)

        lines = out_file.read_text().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["event_id"] == "evt-001"
        assert json.loads(lines[1])["event_id"] == "evt-002"

    def test_each_appended_line_is_valid_json(self, tmp_path: Path) -> None:
        from tok.utils.savings_event import SavingsEvent, append_savings_event

        out_file = tmp_path / "savings_events.jsonl"
        for i in range(3):
            ev = SavingsEvent(
                event_id=f"evt-{i}",
                session_id="s",
                request_id=f"r{i}",
                timestamp="2026-05-20T00:00:00Z",
                model="claude-sonnet-4",
                compression_paths={"file": i * 10},
                non_headline_estimates={"reacquisition_tokens_avoided_estimate": i},
            )
            append_savings_event(ev, out_file)

        for line in out_file.read_text().splitlines():
            parsed = json.loads(line)
            assert parsed["schema"] == "tok-savings-event/v1"


# ---------------------------------------------------------------------------
# Crash-safety tests (RED for step 3)
# ---------------------------------------------------------------------------


class TestSavingsEventCrashSafety:
    """Previous events must survive a partial write."""

    def test_existing_events_survive_partial_line(self, tmp_path: Path) -> None:
        from tok.utils.savings_event import SavingsEvent, append_savings_event

        out_file = tmp_path / "savings_events.jsonl"

        # Write two valid events
        for i in range(2):
            ev = SavingsEvent(
                event_id=f"evt-{i}",
                session_id="s",
                request_id=f"r{i}",
                timestamp="2026-05-20T00:00:00Z",
                model="claude-sonnet-4",
            )
            append_savings_event(ev, out_file)

        # Simulate crash: append a partial (non-JSON) line
        with out_file.open("a") as f:
            f.write('{"schema": "tok-savings-event/v1", "event_id": "partial-cr')
            # No closing brace, no newline

        # Previous complete events must still be readable
        lines = out_file.read_text().splitlines()
        complete_lines = [ln for ln in lines if ln.strip()]
        recoverable = 0
        for line in complete_lines:
            try:
                json.loads(line)
                recoverable += 1
            except json.JSONDecodeError:
                pass
        assert recoverable >= 2, f"At least 2 complete events must survive crash; got {recoverable}"

    def test_read_events_skips_corrupt_lines(self, tmp_path: Path) -> None:
        """read_savings_events must skip corrupt lines and return valid ones."""
        from tok.utils.savings_event import SavingsEvent, append_savings_event, read_savings_events

        out_file = tmp_path / "savings_events.jsonl"
        ev = SavingsEvent(
            event_id="good-001",
            session_id="s",
            request_id="r1",
            timestamp="2026-05-20T00:00:00Z",
            model="claude-sonnet-4",
        )
        append_savings_event(ev, out_file)

        # Inject corrupt line
        with out_file.open("a") as f:
            f.write("not-json\n")

        ev2 = SavingsEvent(
            event_id="good-002",
            session_id="s",
            request_id="r2",
            timestamp="2026-05-20T00:01:00Z",
            model="claude-sonnet-4",
        )
        append_savings_event(ev2, out_file)

        events = read_savings_events(out_file)
        assert len(events) == 2
        assert events[0].event_id == "good-001"
        assert events[1].event_id == "good-002"

    def test_read_events_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        from tok.utils.savings_event import read_savings_events

        out_file = tmp_path / "savings_events.jsonl"
        out_file.write_text("")
        events = read_savings_events(out_file)
        assert events == []

    def test_read_events_missing_file_returns_empty_list(self, tmp_path: Path) -> None:
        from tok.utils.savings_event import read_savings_events

        out_file = tmp_path / "nonexistent.jsonl"
        events = read_savings_events(out_file)
        assert events == []


class TestEmitSavingsEventCostAccounting:
    """emit_savings_event must populate cost fields using model pricing."""

    def _make_mock_session(self, tmp_path: Path) -> object:
        import types

        session = types.SimpleNamespace()
        session.memory_dir = tmp_path
        session.session_id = "live:test_cost_accounting"
        session._savings_event_emitted = False
        session._operation_receipt_emitted = False
        return session

    def test_emit_savings_event_populates_cost_fields_for_known_model(self, tmp_path: Path) -> None:
        from tok.gateway._operation_artifacts import emit_savings_event
        from tok.utils.savings_event import read_savings_events

        session = self._make_mock_session(tmp_path)
        emit_savings_event(
            session,
            model="claude-sonnet-4-6",
            usage={"input_tokens": 1000, "output_tokens": 200},
            request_policy="natural_first",
            compressed=True,
            fallback=False,
            input_saved=500,
            output_saved=10,
            tool_breakdown=None,
            prompt_metrics=None,
        )
        from tok.receipt import _session_id_from_session, bridge_receipt_path

        events_path = bridge_receipt_path(
            memory_dir=tmp_path,
            session_id=_session_id_from_session(session),
        ).with_name("savings_events.jsonl")
        assert events_path.exists(), "savings_events.jsonl was not created"
        events = read_savings_events(events_path)
        assert len(events) == 1
        ev = events[0]
        assert ev.actual_cost_usd > 0.0, "actual_cost_usd must be non-zero for known model"
        assert ev.baseline_cost_usd > ev.actual_cost_usd, (
            "baseline_cost_usd must exceed actual_cost_usd when tokens were saved"
        )
        assert ev.cost_saved_usd > 0.0, "cost_saved_usd must be non-zero when tokens were saved"

    def test_emit_savings_event_zeroes_cost_on_fallback(self, tmp_path: Path) -> None:
        from tok.gateway._operation_artifacts import emit_savings_event
        from tok.utils.savings_event import read_savings_events

        session = self._make_mock_session(tmp_path)
        emit_savings_event(
            session,
            model="claude-sonnet-4-6",
            usage={"input_tokens": 1000, "output_tokens": 200},
            request_policy="natural_first",
            compressed=False,
            fallback=True,
            input_saved=500,
            output_saved=10,
            tool_breakdown=None,
            prompt_metrics=None,
        )
        from tok.receipt import _session_id_from_session, bridge_receipt_path

        events_path = bridge_receipt_path(
            memory_dir=tmp_path,
            session_id=_session_id_from_session(session),
        ).with_name("savings_events.jsonl")
        assert events_path.exists()
        events = read_savings_events(events_path)
        assert len(events) == 1
        ev = events[0]
        assert ev.cost_saved_usd == 0.0, "cost_saved_usd must be zero when fallback=True"
        assert ev.baseline_cost_usd == 0.0, "baseline_cost_usd must be zero when fallback=True"
        assert ev.actual_cost_usd == 0.0, "actual_cost_usd must be zero when fallback=True"
