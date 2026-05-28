"""Coverage matrix tests verifying per-path savings tracking (S7).

For each savings path (file_read, search, command, semantic_dedup) verifies:
- the path appears in type_breakdown after record_call
- values accumulate correctly across multiple calls
- SavingsEvent.compression_paths mirrors tool_breakdown

References: docs/plans/0.2.0/savings-ledger-hardening-plan.md Packet S7
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tok.stats import SavingsTracker
from tok.utils.savings_event import SavingsEvent, append_savings_event, read_savings_events

_MODEL = "claude-haiku-4-5-20251001"


@pytest.fixture
def tracker(tmp_path: Path) -> SavingsTracker:
    return SavingsTracker(
        savings_file=str(tmp_path / "tok_savings.tok"),
        ledger_path=tmp_path / "global_savings.tok",
    )


def _model_breakdown(tracker: SavingsTracker) -> dict[str, int]:
    stats = tracker.load_stats()
    models = stats.get("models", {})
    if not models:
        return {}
    return dict(next(iter(models.values())).get("type_breakdown", {}))


class TestCompressionPathTypeBreakdown:
    def test_file_read_appears_in_type_breakdown(self, tracker: SavingsTracker) -> None:
        tracker.record_call(
            model=_MODEL,
            actual_input=100,
            actual_output=20,
            cache_read=0,
            cache_write=0,
            input_saved=50,
            output_saved=0,
            type_breakdown={"file_read": 200},
        )
        bd = _model_breakdown(tracker)
        assert "file_read" in bd, "file_read path must appear in type_breakdown"
        assert bd["file_read"] == 200

    def test_search_appears_in_type_breakdown(self, tracker: SavingsTracker) -> None:
        tracker.record_call(
            model=_MODEL,
            actual_input=100,
            actual_output=20,
            cache_read=0,
            cache_write=0,
            input_saved=40,
            output_saved=0,
            type_breakdown={"search": 150},
        )
        bd = _model_breakdown(tracker)
        assert "search" in bd, "search path must appear in type_breakdown"
        assert bd["search"] == 150

    def test_command_cached_appears_in_type_breakdown(self, tracker: SavingsTracker) -> None:
        tracker.record_call(
            model=_MODEL,
            actual_input=100,
            actual_output=20,
            cache_read=0,
            cache_write=0,
            input_saved=30,
            output_saved=0,
            type_breakdown={"command_cached": 120},
        )
        bd = _model_breakdown(tracker)
        assert "command_cached" in bd, "command_cached path must appear in type_breakdown"
        assert bd["command_cached"] == 120

    def test_semantic_dedup_appears_in_type_breakdown(self, tracker: SavingsTracker) -> None:
        tracker.record_call(
            model=_MODEL,
            actual_input=100,
            actual_output=20,
            cache_read=0,
            cache_write=0,
            input_saved=25,
            output_saved=0,
            type_breakdown={"semantic_dedup": 100},
        )
        bd = _model_breakdown(tracker)
        assert "semantic_dedup" in bd, "semantic_dedup path must appear in type_breakdown"
        assert bd["semantic_dedup"] == 100

    def test_multiple_paths_accumulate_across_calls(self, tracker: SavingsTracker) -> None:
        tracker.record_call(
            model=_MODEL,
            actual_input=100,
            actual_output=20,
            cache_read=0,
            cache_write=0,
            input_saved=50,
            output_saved=0,
            type_breakdown={"file_read": 200},
        )
        tracker.record_call(
            model=_MODEL,
            actual_input=100,
            actual_output=20,
            cache_read=0,
            cache_write=0,
            input_saved=40,
            output_saved=0,
            type_breakdown={"file_read": 160},
        )
        bd = _model_breakdown(tracker)
        assert bd.get("file_read", 0) == 360, "file_read should accumulate across calls"

    def test_mixed_paths_all_appear(self, tracker: SavingsTracker) -> None:
        tracker.record_call(
            model=_MODEL,
            actual_input=200,
            actual_output=40,
            cache_read=0,
            cache_write=0,
            input_saved=60,
            output_saved=0,
            type_breakdown={"file_read": 100, "search": 80, "command_cached": 60},
        )
        bd = _model_breakdown(tracker)
        assert bd.get("file_read") == 100
        assert bd.get("search") == 80
        assert bd.get("command_cached") == 60

    def test_empty_type_breakdown_does_not_error(self, tracker: SavingsTracker) -> None:
        tracker.record_call(
            model=_MODEL,
            actual_input=100,
            actual_output=20,
            cache_read=0,
            cache_write=0,
            input_saved=0,
            output_saved=0,
            type_breakdown={},
        )
        assert tracker.session_summary() is not None

    def test_none_type_breakdown_treated_as_empty(self, tracker: SavingsTracker) -> None:
        tracker.record_call(
            model=_MODEL,
            actual_input=100,
            actual_output=20,
            cache_read=0,
            cache_write=0,
            input_saved=0,
            output_saved=0,
            type_breakdown=None,
        )
        assert tracker.session_summary() is not None

    def test_type_breakdown_values_nonnegative(self, tracker: SavingsTracker) -> None:
        tracker.record_call(
            model=_MODEL,
            actual_input=100,
            actual_output=20,
            cache_read=0,
            cache_write=0,
            input_saved=10,
            output_saved=0,
            type_breakdown={"file_read": 40, "search": 0},
        )
        bd = _model_breakdown(tracker)
        for key, val in bd.items():
            assert val >= 0, f"type_breakdown[{key!r}] must be non-negative"


class TestCompressionPathsInSavingsEvent:
    def test_compression_paths_populated_directly(self) -> None:
        event = SavingsEvent(
            event_id="e1",
            session_id="s1",
            request_id="r1",
            timestamp="2026-01-01T00:00:00+00:00",
            model=_MODEL,
            input_tokens_saved=50,
            compression_paths={"file_read": 200, "search": 100},
        )
        assert event.compression_paths == {"file_read": 200, "search": 100}

    def test_compression_paths_values_are_nonnegative(self) -> None:
        event = SavingsEvent(
            event_id="e1",
            session_id="s1",
            request_id="r1",
            timestamp="2026-01-01T00:00:00+00:00",
            model=_MODEL,
            compression_paths={"file_read": 200, "search": 50, "command_cached": 0},
        )
        for path, count in event.compression_paths.items():
            assert count >= 0, f"{path!r} has negative count {count}"

    def test_multiple_path_types_in_single_event(self) -> None:
        event = SavingsEvent(
            event_id="e1",
            session_id="s1",
            request_id="r1",
            timestamp="2026-01-01T00:00:00+00:00",
            model=_MODEL,
            compression_paths={
                "file_read": 200,
                "search": 100,
                "command_cached": 50,
                "semantic_dedup": 30,
            },
        )
        assert len(event.compression_paths) == 4
        assert event.compression_paths["file_read"] == 200
        assert event.compression_paths["search"] == 100
        assert event.compression_paths["command_cached"] == 50
        assert event.compression_paths["semantic_dedup"] == 30

    def test_compression_paths_survive_jsonl_round_trip(self, tmp_path: Path) -> None:
        original = SavingsEvent(
            event_id="e-roundtrip",
            session_id="s1",
            request_id="r1",
            timestamp="2026-01-01T00:00:00+00:00",
            model=_MODEL,
            input_tokens_saved=75,
            compression_paths={"file_read": 300, "search": 200},
        )
        path = tmp_path / "events.jsonl"
        append_savings_event(original, path)
        recovered = read_savings_events(path)
        assert len(recovered) == 1
        assert recovered[0].compression_paths == {"file_read": 300, "search": 200}

    def test_fallback_event_has_empty_compression_paths(self, tmp_path: Path) -> None:
        event = SavingsEvent(
            event_id="e-fallback",
            session_id="s1",
            request_id="r1",
            timestamp="2026-01-01T00:00:00+00:00",
            model=_MODEL,
            fallback=True,
            input_tokens_saved=0,
            compression_paths={},
        )
        path = tmp_path / "savings_events.jsonl"
        append_savings_event(event, path)
        events = read_savings_events(path)
        assert len(events) == 1
        assert events[0].fallback is True
        assert events[0].compression_paths == {}

    def test_empty_compression_paths_default(self) -> None:
        event = SavingsEvent(
            event_id="e2",
            session_id="s1",
            request_id="r1",
            timestamp="2026-01-01T00:00:00+00:00",
            model=_MODEL,
        )
        assert event.compression_paths == {}
