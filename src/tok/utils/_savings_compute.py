"""Typed session-savings computation.

The gross/net/overhead/avoided-estimate relationships were previously computed
inline inside ``SavingsTracker.session_summary`` with the relationships left
implicit and re-derived by callers. This module is the single place that turns
per-model token tallies + behavior signals into a typed :class:`SessionSavings`,
so the invariants (notably ``net <= gross``) live in one auditable spot.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SessionSavings:
    """Reconciled token-savings figures for a session.

    ``gross_tokens_saved`` is the optimistic figure: request/response tokens
    saved by compression, plus an *estimate* of reacquisition tokens avoided,
    minus the overhead Tok itself added (hot hints). ``net_tokens_saved`` then
    subtracts the tokens actually spent re-acquiring exact evidence. Because
    ``reacquisition_cost_tokens`` is a non-negative count, ``net <= gross``
    always holds -- ``net`` is the conservative, credible figure to report.
    """

    gross_tokens_saved: int
    reacquisition_cost_tokens: int
    reacquisition_avoided_estimate: int
    hot_hint_overhead_tokens: int

    @property
    def net_tokens_saved(self) -> int:
        return self.gross_tokens_saved - self.reacquisition_cost_tokens


def _sum(models: Mapping[str, Mapping[str, Any]], key: str) -> int:
    return sum(int(m.get(key, 0) or 0) for m in models.values())


def compute_session_savings(
    models: Mapping[str, Mapping[str, Any]],
    signals: Mapping[str, Any],
) -> SessionSavings:
    """Reconcile per-model tallies + behavior signals into typed savings.

    *models* maps model name -> the per-model stats dict accumulated by
    ``SavingsTracker`` (``input_saved_tokens``, ``output_saved_tokens``,
    ``reacquisition_tokens_avoided_estimate``, ``hot_hint_tokens_added``).
    *signals* is the session behavior-signal map (for ``reacquisition_cost_tokens``).
    """
    avoided_estimate = _sum(models, "reacquisition_tokens_avoided_estimate")
    hot_hint_overhead = _sum(models, "hot_hint_tokens_added")
    gross = (
        _sum(models, "input_saved_tokens") + _sum(models, "output_saved_tokens") + avoided_estimate - hot_hint_overhead
    )
    reacquisition_cost = int(signals.get("reacquisition_cost_tokens", 0) or 0)
    return SessionSavings(
        gross_tokens_saved=gross,
        reacquisition_cost_tokens=reacquisition_cost,
        reacquisition_avoided_estimate=avoided_estimate,
        hot_hint_overhead_tokens=hot_hint_overhead,
    )
