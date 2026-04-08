from __future__ import annotations

from tok.gateway import stats as gateway_stats
from tok.utils import savings_tracker as canonical_savings_tracker
from tok.utils._savings_persistence import (
    GLOBAL_LEDGER_FILENAME as CANONICAL_GLOBAL_LEDGER_FILENAME,
)


def test_stats_modules_re_export_canonical_symbols() -> None:
    assert gateway_stats.SavingsTracker is canonical_savings_tracker.SavingsTracker

    assert gateway_stats.GLOBAL_LEDGER_FILENAME == CANONICAL_GLOBAL_LEDGER_FILENAME


def test_stats_modules_share_quality_helpers() -> None:
    assert (
        gateway_stats._session_quality(
            {"tok_fallback_activated": 1},
            baseline_only=False,
        )
        == "watch"
    )
