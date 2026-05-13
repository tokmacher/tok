"""Invariant tests for SavingsTracker accounting formulas."""

from __future__ import annotations

import pytest

from tok.stats import SavingsTracker


@pytest.fixture
def tracker(tmp_path):
    return SavingsTracker(
        savings_file=str(tmp_path / "tok_savings.tok"),
        ledger_path=tmp_path / "global_savings.tok",
    )


def test_saved_tokens_formula_includes_reacquisition_and_hot_hints(tracker: SavingsTracker) -> None:
    tracker.record_call(
        model="claude-sonnet-4",
        actual_input=100,
        actual_output=50,
        cache_read=0,
        cache_write=0,
        input_saved=20,
        output_saved=10,
        prompt_metrics={
            "hot_hint_tokens_added": 7,
            "reacquisition_tokens_avoided_estimate": 30,
        },
    )
    summary = tracker.session_summary()
    assert summary is not None
    assert summary["tokens_saved"] == 20 + 10 + 30 - 7


def test_baseline_tokens_equals_actual_plus_saved(tracker: SavingsTracker) -> None:
    tracker.record_call(
        model="claude-sonnet-4",
        actual_input=10,
        actual_output=5,
        cache_read=2,
        cache_write=1,
        input_saved=3,
        output_saved=4,
    )
    summary = tracker.session_summary()
    assert summary is not None
    assert summary["baseline_tokens"] == summary["actual_tokens"] + summary["tokens_saved"]


def test_savings_pct_matches_saved_over_baseline(tracker: SavingsTracker) -> None:
    tracker.record_call(
        model="claude-sonnet-4",
        actual_input=1000,
        actual_output=0,
        cache_read=0,
        cache_write=0,
        input_saved=1000,
        output_saved=0,
    )
    summary = tracker.session_summary()
    assert summary is not None
    expected = summary["tokens_saved"] / summary["baseline_tokens"] * 100
    assert abs(summary["savings_pct"] - expected) < 1e-9


def test_cost_savings_pct_matches_cost_saved_over_baseline_cost(tracker: SavingsTracker) -> None:
    tracker.record_call(
        model="claude-sonnet-4",
        actual_input=1000,
        actual_output=100,
        cache_read=50,
        cache_write=25,
        input_saved=200,
        output_saved=0,
    )
    summary = tracker.session_summary()
    assert summary is not None
    expected = summary["cost_saved_usd"] / summary["baseline_cost_usd"] * 100
    assert summary["cost_savings_pct"] == round(expected, 1)


def test_session_totals_equal_sum_of_record_calls(tracker: SavingsTracker) -> None:
    tracker.record_call(
        model="claude-sonnet-4",
        actual_input=10,
        actual_output=1,
        cache_read=0,
        cache_write=0,
        input_saved=2,
        output_saved=0,
    )
    tracker.record_call(
        model="claude-sonnet-4",
        actual_input=20,
        actual_output=2,
        cache_read=0,
        cache_write=0,
        input_saved=3,
        output_saved=4,
        prompt_metrics={
            "hot_hint_tokens_added": 1,
            "reacquisition_tokens_avoided_estimate": 5,
        },
    )
    summary = tracker.session_summary()
    assert summary is not None
    assert summary["calls"] == 2
    assert summary["tokens_saved"] == (2 + 0) + (3 + 4 + 5 - 1)
