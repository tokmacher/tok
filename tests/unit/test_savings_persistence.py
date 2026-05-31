"""Persistence and atomicity tests for savings session stats (S8).

Verifies:
- save_stats writes atomically via temp file + rename
- Existing stats survive a new write cycle (no temp-file leftover)
- Stats round-trip correctly across fresh SavingsTracker instances
- Per-request JSONL append preserves insertion order

References: docs/plans/0.2.0/savings-ledger-hardening-plan.md Packet S8
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tok.stats import SavingsTracker
from tok.utils.savings_event import SavingsEvent, append_savings_event, read_savings_events

_MODEL = "claude-haiku-4-5-20251001"


@pytest.fixture
def savings_file(tmp_path: Path) -> Path:
    return tmp_path / "tok_savings.tok"


@pytest.fixture
def tracker(tmp_path: Path) -> SavingsTracker:
    return SavingsTracker(
        savings_file=str(tmp_path / "tok_savings.tok"),
        ledger_path=tmp_path / "global_savings.tok",
    )


class TestSessionStatsPersistence:
    def test_record_call_creates_stats_file(self, tmp_path: Path, savings_file: Path) -> None:
        t = SavingsTracker(
            savings_file=str(savings_file),
            ledger_path=tmp_path / "global.tok",
        )
        t.record_call(
            model=_MODEL,
            actual_input=100,
            actual_output=20,
            cache_read=0,
            cache_write=0,
            input_saved=10,
            output_saved=0,
        )
        assert savings_file.exists()

    def test_save_stats_does_not_leave_temp_file(self, tmp_path: Path, savings_file: Path) -> None:
        t = SavingsTracker(
            savings_file=str(savings_file),
            ledger_path=tmp_path / "global.tok",
        )
        t.record_call(
            model=_MODEL,
            actual_input=100,
            actual_output=20,
            cache_read=0,
            cache_write=0,
            input_saved=10,
            output_saved=0,
        )
        temp = Path(str(savings_file) + ".tmp")
        assert not temp.exists(), "temp file must be removed after atomic rename"

    def test_stats_survive_across_fresh_tracker_instance(self, tmp_path: Path, savings_file: Path) -> None:
        t1 = SavingsTracker(
            savings_file=str(savings_file),
            ledger_path=tmp_path / "global.tok",
        )
        t1.record_call(
            model=_MODEL,
            actual_input=100,
            actual_output=20,
            cache_read=0,
            cache_write=0,
            input_saved=30,
            output_saved=5,
        )
        saved_tokens = t1.session_summary()["tokens_saved"]

        t2 = SavingsTracker(
            savings_file=str(savings_file),
            ledger_path=tmp_path / "global.tok",
        )
        assert t2.session_summary()["tokens_saved"] == saved_tokens

    def test_multiple_calls_accumulate_in_saved_file(self, tmp_path: Path, savings_file: Path) -> None:
        t = SavingsTracker(
            savings_file=str(savings_file),
            ledger_path=tmp_path / "global.tok",
        )
        for _ in range(5):
            t.record_call(
                model=_MODEL,
                actual_input=100,
                actual_output=20,
                cache_read=0,
                cache_write=0,
                input_saved=20,
                output_saved=0,
            )

        t2 = SavingsTracker(
            savings_file=str(savings_file),
            ledger_path=tmp_path / "global.tok",
        )
        assert t2.session_summary()["tokens_saved"] == 100  # 5 × 20

    def test_type_breakdown_persists_across_instances(self, tmp_path: Path, savings_file: Path) -> None:
        t = SavingsTracker(
            savings_file=str(savings_file),
            ledger_path=tmp_path / "global.tok",
        )
        t.record_call(
            model=_MODEL,
            actual_input=200,
            actual_output=40,
            cache_read=0,
            cache_write=0,
            input_saved=80,
            output_saved=0,
            type_breakdown={"file_read": 320},
        )

        t2 = SavingsTracker(
            savings_file=str(savings_file),
            ledger_path=tmp_path / "global.tok",
        )
        stats = t2.load_stats()
        models = stats.get("models", {})
        model_stats = next(iter(models.values()), {})
        bd = model_stats.get("type_breakdown", {})
        assert bd.get("file_read", 0) == 320

    def test_multiple_save_cycles_do_not_corrupt_file(self, tmp_path: Path, savings_file: Path) -> None:
        t = SavingsTracker(
            savings_file=str(savings_file),
            ledger_path=tmp_path / "global.tok",
        )
        for i in range(1, 4):
            t.record_call(
                model=_MODEL,
                actual_input=100,
                actual_output=10,
                cache_read=0,
                cache_write=0,
                input_saved=i * 10,
                output_saved=0,
            )
        summary = t.session_summary()
        # 10 + 20 + 30 = 60
        assert summary["tokens_saved"] == 60
        assert summary["calls"] == 3


class TestSavingsJSONLDurabilityComplement:
    def test_jsonl_survives_partial_trailing_data(self, tmp_path: Path) -> None:
        path = tmp_path / "savings_events.jsonl"
        event = SavingsEvent(
            event_id="e1",
            session_id="s1",
            request_id="r1",
            timestamp="2026-01-01T00:00:00+00:00",
            model=_MODEL,
            input_tokens_saved=10,
        )
        append_savings_event(event, path)

        # Simulate crash mid-write: partial JSON appended at end
        with path.open("a") as fh:
            fh.write('{"event_id": "partial"')  # no closing brace

        events = read_savings_events(path)
        assert len(events) == 1, "partial trailing line must be skipped"
        assert events[0].event_id == "e1"

    def test_append_events_preserve_insertion_order(self, tmp_path: Path) -> None:
        path = tmp_path / "savings_events.jsonl"
        ids = [f"event-{i}" for i in range(5)]
        for eid in ids:
            event = SavingsEvent(
                event_id=eid,
                session_id="s1",
                request_id=eid,
                timestamp="2026-01-01T00:00:00+00:00",
                model=_MODEL,
            )
            append_savings_event(event, path)

        recovered = read_savings_events(path)
        assert [e.event_id for e in recovered] == ids

    def test_events_from_multiple_models_are_all_readable(self, tmp_path: Path) -> None:
        path = tmp_path / "savings_events.jsonl"
        models = ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-7"]
        for i, model in enumerate(models):
            event = SavingsEvent(
                event_id=f"e-{i}",
                session_id="s1",
                request_id=f"r-{i}",
                timestamp="2026-01-01T00:00:00+00:00",
                model=model,
                input_tokens_saved=i * 10,
            )
            append_savings_event(event, path)

        events = read_savings_events(path)
        assert len(events) == 3
        assert [e.model for e in events] == models
