from __future__ import annotations

from tok.gateway import stats as gateway_stats
from tok import savings_tracker as savings_tracker_shim
from tok import stats as stats_shim
from tok.utils import savings_tracker as canonical_savings_tracker
from tok.utils._savings_persistence import (
    GLOBAL_LEDGER_FILENAME as CANONICAL_GLOBAL_LEDGER_FILENAME,
    SESSION_STATS_FILENAME as CANONICAL_SESSION_STATS_FILENAME,
)
from tok.utils._savings_quality import (
    BASELINE_ONLY_SIGNAL as CANONICAL_BASELINE_ONLY_SIGNAL,
    FALLBACK_SIGNAL as CANONICAL_FALLBACK_SIGNAL,
)


def test_stats_shims_re_export_canonical_symbols() -> None:
    assert (
        stats_shim.SavingsTracker is canonical_savings_tracker.SavingsTracker
    )
    assert (
        savings_tracker_shim.SavingsTracker
        is canonical_savings_tracker.SavingsTracker
    )
    assert (
        gateway_stats.SavingsTracker
        is canonical_savings_tracker.SavingsTracker
    )

    assert stats_shim.BASELINE_ONLY_SIGNAL == CANONICAL_BASELINE_ONLY_SIGNAL
    assert savings_tracker_shim.FALLBACK_SIGNAL == CANONICAL_FALLBACK_SIGNAL
    assert (
        gateway_stats.GLOBAL_LEDGER_FILENAME
        == CANONICAL_GLOBAL_LEDGER_FILENAME
    )
    assert (
        stats_shim.SESSION_STATS_FILENAME == CANONICAL_SESSION_STATS_FILENAME
    )


def test_stats_shims_share_quality_helpers() -> None:
    signals = {"repeat_file_read": 1}

    assert (
        stats_shim._degradation_reason(signals, baseline_only=False)
        == "context reacquisition"
    )
    assert (
        gateway_stats._session_quality(
            {"tok_fallback_activated": 1},
            baseline_only=False,
        )
        == "watch"
    )
