"""Section 5.2.3: Ledger-Derived Session & Lifetime Summaries — RED/GREEN tests.

Creates synthetic SavingsEvent JSONL files and verifies that
reconstruct_session_summary() and reconstruct_lifetime_summary() compute
correct totals.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _make_event(
    event_id: str,
    session_id: str,
    request_id: str,
    *,
    input_tokens_saved: int = 0,
    output_tokens_saved: int = 0,
    baseline_input_tokens: int = 1000,
    actual_input_tokens: int = 1000,
    baseline_output_tokens: int = 200,
    actual_output_tokens: int = 200,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    baseline_cost_usd: float = 0.010,
    actual_cost_usd: float = 0.010,
    cost_saved_usd: float = 0.0,
    fallback: bool = False,
    degraded_to_baseline: bool = False,
    compression_paths: dict | None = None,
) -> Any:
    from tok.utils.savings_event import SavingsEvent

    return SavingsEvent(
        event_id=event_id,
        session_id=session_id,
        request_id=request_id,
        timestamp="2026-05-20T00:00:00Z",
        model="claude-sonnet-4",
        input_tokens_saved=input_tokens_saved,
        output_tokens_saved=output_tokens_saved,
        baseline_input_tokens=baseline_input_tokens,
        actual_input_tokens=actual_input_tokens,
        baseline_output_tokens=baseline_output_tokens,
        actual_output_tokens=actual_output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        baseline_cost_usd=baseline_cost_usd,
        actual_cost_usd=actual_cost_usd,
        cost_saved_usd=cost_saved_usd,
        fallback=fallback,
        degraded_to_baseline=degraded_to_baseline,
        compression_paths=compression_paths or {},
    )


def _write_events(path: Path, events: list) -> None:
    from tok.utils.savings_event import append_savings_event

    for ev in events:
        append_savings_event(ev, path)


# ---------------------------------------------------------------------------
# SavingsSummary dataclass
# ---------------------------------------------------------------------------


class TestSavingsSummaryDataclass:
    def test_savings_summary_exists(self) -> None:
        from tok.utils.savings_reconstruction import SavingsSummary

        s = SavingsSummary()
        assert hasattr(s, "calls")
        assert hasattr(s, "input_tokens_saved")
        assert hasattr(s, "output_tokens_saved")
        assert hasattr(s, "baseline_input_tokens")
        assert hasattr(s, "actual_input_tokens")
        assert hasattr(s, "baseline_cost_usd")
        assert hasattr(s, "actual_cost_usd")
        assert hasattr(s, "cost_saved_usd")
        assert hasattr(s, "fallback_count")
        assert hasattr(s, "degraded_count")

    def test_savings_summary_defaults_to_zero(self) -> None:
        from tok.utils.savings_reconstruction import SavingsSummary

        s = SavingsSummary()
        assert s.calls == 0
        assert s.input_tokens_saved == 0
        assert s.cost_saved_usd == 0.0
        assert s.fallback_count == 0


# ---------------------------------------------------------------------------
# reconstruct_session_summary
# ---------------------------------------------------------------------------


class TestReconstructSessionSummary:
    def test_empty_events_returns_zero_summary(self) -> None:
        from tok.utils.savings_reconstruction import reconstruct_session_summary

        summary = reconstruct_session_summary([])
        assert summary.calls == 0
        assert summary.input_tokens_saved == 0

    def test_single_event_totals(self) -> None:
        from tok.utils.savings_reconstruction import reconstruct_session_summary

        ev = _make_event(
            "e1",
            "s1",
            "r1",
            input_tokens_saved=300,
            baseline_input_tokens=1000,
            actual_input_tokens=700,
            baseline_cost_usd=0.010,
            actual_cost_usd=0.007,
            cost_saved_usd=0.003,
        )
        summary = reconstruct_session_summary([ev])
        assert summary.calls == 1
        assert summary.input_tokens_saved == 300
        assert summary.baseline_input_tokens == 1000
        assert summary.actual_input_tokens == 700
        assert abs(summary.cost_saved_usd - 0.003) < 1e-9

    def test_multiple_events_sum_correctly(self) -> None:
        from tok.utils.savings_reconstruction import reconstruct_session_summary

        events = [
            _make_event("e1", "s1", "r1", input_tokens_saved=100, cost_saved_usd=0.001),
            _make_event("e2", "s1", "r2", input_tokens_saved=200, cost_saved_usd=0.002),
            _make_event("e3", "s1", "r3", input_tokens_saved=150, cost_saved_usd=0.0015),
        ]
        summary = reconstruct_session_summary(events)
        assert summary.calls == 3
        assert summary.input_tokens_saved == 450
        assert abs(summary.cost_saved_usd - 0.0045) < 1e-9

    def test_fallback_events_counted(self) -> None:
        from tok.utils.savings_reconstruction import reconstruct_session_summary

        events = [
            _make_event("e1", "s1", "r1", fallback=False),
            _make_event("e2", "s1", "r2", fallback=True),
            _make_event("e3", "s1", "r3", fallback=True),
        ]
        summary = reconstruct_session_summary(events)
        assert summary.fallback_count == 2
        assert summary.calls == 3

    def test_degraded_events_counted(self) -> None:
        from tok.utils.savings_reconstruction import reconstruct_session_summary

        events = [
            _make_event("e1", "s1", "r1", degraded_to_baseline=True),
            _make_event("e2", "s1", "r2", degraded_to_baseline=False),
        ]
        summary = reconstruct_session_summary(events)
        assert summary.degraded_count == 1

    def test_cache_tokens_summed(self) -> None:
        from tok.utils.savings_reconstruction import reconstruct_session_summary

        events = [
            _make_event("e1", "s1", "r1", cache_read_tokens=500, cache_write_tokens=50),
            _make_event("e2", "s1", "r2", cache_read_tokens=300, cache_write_tokens=30),
        ]
        summary = reconstruct_session_summary(events)
        assert summary.cache_read_tokens == 800
        assert summary.cache_write_tokens == 80

    def test_output_tokens_summed(self) -> None:
        from tok.utils.savings_reconstruction import reconstruct_session_summary

        events = [
            _make_event("e1", "s1", "r1", output_tokens_saved=10),
            _make_event("e2", "s1", "r2", output_tokens_saved=20),
        ]
        summary = reconstruct_session_summary(events)
        assert summary.output_tokens_saved == 30


# ---------------------------------------------------------------------------
# reconstruct_session_summary_from_file
# ---------------------------------------------------------------------------


class TestReconstructFromFile:
    def test_reconstruct_from_jsonl_file(self, tmp_path: Path) -> None:
        from tok.utils.savings_reconstruction import reconstruct_session_summary_from_file

        events_file = tmp_path / "savings_events.jsonl"
        events = [
            _make_event("e1", "s1", "r1", input_tokens_saved=100),
            _make_event("e2", "s1", "r2", input_tokens_saved=200),
        ]
        _write_events(events_file, events)

        summary = reconstruct_session_summary_from_file(events_file)
        assert summary.calls == 2
        assert summary.input_tokens_saved == 300

    def test_missing_file_returns_zero_summary(self, tmp_path: Path) -> None:
        from tok.utils.savings_reconstruction import reconstruct_session_summary_from_file

        summary = reconstruct_session_summary_from_file(tmp_path / "nonexistent.jsonl")
        assert summary.calls == 0

    def test_corrupt_lines_skipped_gracefully(self, tmp_path: Path) -> None:
        from tok.utils.savings_reconstruction import reconstruct_session_summary_from_file

        events_file = tmp_path / "savings_events.jsonl"
        ev = _make_event("e1", "s1", "r1", input_tokens_saved=100)
        _write_events(events_file, [ev])

        with events_file.open("a") as f:
            f.write("not-valid-json\n")
            f.write('{"incomplete": true}\n')  # valid JSON but missing required fields

        ev2 = _make_event("e2", "s1", "r2", input_tokens_saved=50)
        _write_events(events_file, [ev2])

        summary = reconstruct_session_summary_from_file(events_file)
        assert summary.calls == 2
        assert summary.input_tokens_saved == 150

    def test_missing_fields_handled_defensively(self, tmp_path: Path) -> None:
        from tok.utils.savings_reconstruction import reconstruct_session_summary_from_file

        events_file = tmp_path / "savings_events.jsonl"
        # Write a minimal-valid SavingsEvent, then a dict with missing fields
        ev = _make_event("e1", "s1", "r1", input_tokens_saved=50)
        _write_events(events_file, [ev])

        # Append a partial record (missing required event_id, session_id, etc.)
        partial = {"schema": "tok-savings-event/v1", "input_tokens_saved": 999}
        with events_file.open("a") as f:
            f.write(json.dumps(partial) + "\n")

        # Should not raise; partial record that can't construct SavingsEvent is skipped
        summary = reconstruct_session_summary_from_file(events_file)
        assert summary.calls >= 1


# ---------------------------------------------------------------------------
# reconstruct_lifetime_summary
# ---------------------------------------------------------------------------


class TestReconstructLifetimeSummary:
    def test_reconstruct_lifetime_empty_returns_zero(self, tmp_path: Path) -> None:
        from tok.utils.savings_reconstruction import reconstruct_lifetime_summary

        summary = reconstruct_lifetime_summary(tmp_path)
        assert summary.calls == 0

    def test_reconstruct_lifetime_single_session(self, tmp_path: Path) -> None:
        from tok.utils.savings_reconstruction import reconstruct_lifetime_summary

        sess_dir = tmp_path / "sess-abc"
        sess_dir.mkdir()
        events_file = sess_dir / "savings_events.jsonl"
        events = [
            _make_event("e1", "sess-abc", "r1", input_tokens_saved=100),
            _make_event("e2", "sess-abc", "r2", input_tokens_saved=200),
        ]
        _write_events(events_file, events)

        summary = reconstruct_lifetime_summary(tmp_path)
        assert summary.calls == 2
        assert summary.input_tokens_saved == 300

    def test_reconstruct_lifetime_multiple_sessions(self, tmp_path: Path) -> None:
        from tok.utils.savings_reconstruction import reconstruct_lifetime_summary

        for i, (sess_id, saved) in enumerate([("sess-a", 100), ("sess-b", 200), ("sess-c", 150)]):
            sess_dir = tmp_path / sess_id
            sess_dir.mkdir()
            ev = _make_event(f"e{i}", sess_id, f"r{i}", input_tokens_saved=saved)
            _write_events(sess_dir / "savings_events.jsonl", [ev])

        summary = reconstruct_lifetime_summary(tmp_path)
        assert summary.calls == 3
        assert summary.input_tokens_saved == 450

    def test_reconstruct_lifetime_skips_directories_without_events(self, tmp_path: Path) -> None:
        from tok.utils.savings_reconstruction import reconstruct_lifetime_summary

        # Session with events
        sess_dir = tmp_path / "sess-valid"
        sess_dir.mkdir()
        ev = _make_event("e1", "sess-valid", "r1", input_tokens_saved=100)
        _write_events(sess_dir / "savings_events.jsonl", [ev])

        # Session dir with no events file
        empty_dir = tmp_path / "sess-empty"
        empty_dir.mkdir()

        summary = reconstruct_lifetime_summary(tmp_path)
        assert summary.calls == 1
        assert summary.input_tokens_saved == 100


# ---------------------------------------------------------------------------
# Invariant checks on reconstructed summaries
# ---------------------------------------------------------------------------


class TestReconstructedSummaryInvariants:
    """Invariants from docs/savings-accounting.md must hold for reconstructed totals."""

    def test_input_tokens_saved_nonnegative(self) -> None:
        from tok.utils.savings_reconstruction import reconstruct_session_summary

        events = [
            _make_event("e1", "s1", "r1", input_tokens_saved=0, fallback=True),
        ]
        summary = reconstruct_session_summary(events)
        assert summary.input_tokens_saved >= 0

    def test_tokens_saved_is_zero_for_pure_fallback_session(self) -> None:
        from tok.utils.savings_reconstruction import reconstruct_session_summary

        events = [
            _make_event("e1", "s1", "r1", input_tokens_saved=0, fallback=True),
            _make_event("e2", "s1", "r2", input_tokens_saved=0, fallback=True),
        ]
        summary = reconstruct_session_summary(events)
        assert summary.input_tokens_saved == 0
        assert summary.fallback_count == 2
