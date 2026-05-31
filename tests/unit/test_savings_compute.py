"""Tests for the typed session-savings computation.

Savings math (gross vs net vs overhead vs avoided-estimate) was computed inline
in SavingsTracker.session_summary with the relationships left implicit. These
tests pin the single typed surface and its core invariant: net never exceeds
gross.
"""

from __future__ import annotations

from tok.utils._savings_compute import SessionSavings, compute_session_savings


def test_gross_includes_avoided_estimate_minus_overhead() -> None:
    models = {
        "m": {
            "input_saved_tokens": 100,
            "output_saved_tokens": 20,
            "reacquisition_tokens_avoided_estimate": 30,
            "hot_hint_tokens_added": 10,
        }
    }
    savings = compute_session_savings(models, {"reacquisition_cost_tokens": 25})

    assert savings.gross_tokens_saved == 100 + 20 + 30 - 10  # 140
    assert savings.reacquisition_avoided_estimate == 30
    assert savings.hot_hint_overhead_tokens == 10
    assert savings.reacquisition_cost_tokens == 25
    assert savings.net_tokens_saved == 140 - 25  # 115


def test_net_never_exceeds_gross_property() -> None:
    # reacquisition_cost is a non-negative count, so net <= gross always holds.
    for cost in (0, 5, 1000):
        models = {"m": {"input_saved_tokens": 100}}
        savings = compute_session_savings(models, {"reacquisition_cost_tokens": cost})
        assert savings.net_tokens_saved <= savings.gross_tokens_saved


def test_empty_inputs_are_zero() -> None:
    savings = compute_session_savings({}, {})
    assert savings.gross_tokens_saved == 0
    assert savings.net_tokens_saved == 0
    assert savings.reacquisition_cost_tokens == 0


def test_sums_across_models() -> None:
    models = {
        "a": {"input_saved_tokens": 50, "hot_hint_tokens_added": 5},
        "b": {"input_saved_tokens": 70, "reacquisition_tokens_avoided_estimate": 10},
    }
    savings = compute_session_savings(models, {})
    assert savings.gross_tokens_saved == (50 - 5) + (70 + 10)  # 125
    assert isinstance(savings, SessionSavings)
