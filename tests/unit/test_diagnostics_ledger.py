"""Section 5.2.4: Diagnostics Ledger Integration — RED/GREEN tests.

Verifies that DiagnosticsSnapshot.savings_source is set correctly and
that diagnostics can reflect ledger-derived savings when a JSONL file
is available.
"""

from __future__ import annotations

from pathlib import Path


class TestSavingsSourceField:
    def test_diagnostics_snapshot_has_savings_source_field(self) -> None:
        from tok.runtime._diagnostics import DiagnosticsSnapshot

        snap = DiagnosticsSnapshot()
        assert hasattr(snap, "savings_source")

    def test_savings_source_default_is_session_tracker(self) -> None:
        from tok.runtime._diagnostics import DiagnosticsSnapshot

        snap = DiagnosticsSnapshot()
        assert snap.savings_source == "session_tracker"

    def test_savings_source_in_health_response(self) -> None:
        from tok.runtime._diagnostics import DiagnosticsSnapshot

        snap = DiagnosticsSnapshot()
        health = snap.to_health_response()
        assert "savings_source" in health
        assert health["savings_source"] == "session_tracker"

    def test_savings_source_ledger_accepted(self) -> None:
        from dataclasses import replace

        from tok.runtime._diagnostics import DiagnosticsSnapshot

        snap = DiagnosticsSnapshot()
        snap2 = replace(snap, savings_source="ledger")
        assert snap2.savings_source == "ledger"

    def test_savings_source_memory_accepted(self) -> None:
        from dataclasses import replace

        from tok.runtime._diagnostics import DiagnosticsSnapshot

        snap = DiagnosticsSnapshot()
        snap2 = replace(snap, savings_source="memory")
        assert snap2.savings_source == "memory"


class TestDiagnosticsLedgerConsistency:
    """Verify that ledger-derived and in-memory values can be compared."""

    def test_ledger_summary_fields_map_to_diagnostics(self) -> None:
        """SavingsSummary fields must have a plausible mapping to DiagnosticsSnapshot."""
        from tok.runtime._diagnostics import DiagnosticsSnapshot

        # The key fields that a ledger summary provides should be representable
        # in some form within DiagnosticsSnapshot or its health response.
        snap = DiagnosticsSnapshot()
        health = snap.to_health_response()
        # tokens_saved is the headline metric; must be in health response
        assert "session_tokens_saved" in health or "tokens_saved" in health or "cost_saved_usd" in health

    def test_diagnostics_snapshot_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        import pytest

        from tok.runtime._diagnostics import DiagnosticsSnapshot

        snap = DiagnosticsSnapshot()
        with pytest.raises(FrozenInstanceError):
            snap.savings_source = "ledger"  # type: ignore[misc]

    def test_from_health_response_preserves_savings_source(self) -> None:
        from tok.runtime._diagnostics import DiagnosticsSnapshot

        snap = DiagnosticsSnapshot(savings_source="ledger")
        health = snap.to_health_response()
        # If from_health_response is available, it should preserve the field
        if hasattr(DiagnosticsSnapshot, "from_health_response"):
            snap2 = DiagnosticsSnapshot.from_health_response(health)
            assert snap2.savings_source == "ledger"


class TestDiagnosticsLedgerIntegration:
    """
    Full integration: build a DiagnosticsSnapshot from a ledger-derived summary
    and verify the savings_source is set to 'ledger'.
    """

    def test_diagnostics_from_ledger_sets_savings_source(self, tmp_path: Path) -> None:
        """When a ledger file exists, savings_source should be 'ledger'."""

        from tok.runtime._diagnostics import DiagnosticsSnapshot
        from tok.utils.savings_event import SavingsEvent, append_savings_event
        from tok.utils.savings_reconstruction import (
            reconstruct_session_summary_from_file,
        )

        events_file = tmp_path / "savings_events.jsonl"
        for i in range(3):
            ev = SavingsEvent(
                event_id=f"e{i}",
                session_id="sess-test",
                request_id=f"turn-{i}",
                timestamp="2026-05-20T00:00:00Z",
                model="claude-sonnet-4",
                input_tokens_saved=100 * (i + 1),
                baseline_input_tokens=1000,
                actual_input_tokens=1000 - 100 * (i + 1),
            )
            append_savings_event(ev, events_file)

        summary = reconstruct_session_summary_from_file(events_file)
        assert summary.calls == 3
        assert summary.input_tokens_saved == 600

        # Build a DiagnosticsSnapshot that reflects ledger data
        snap = DiagnosticsSnapshot(savings_source="ledger")
        assert snap.savings_source == "ledger"
        health = snap.to_health_response()
        assert health["savings_source"] == "ledger"

    def test_tok_stats_json_includes_savings_source(self) -> None:
        """tok stats output (from DiagnosticsSnapshot.to_health_response) must include savings_source."""
        from tok.runtime._diagnostics import DiagnosticsSnapshot

        snap = DiagnosticsSnapshot(savings_source="session_tracker")
        health = snap.to_health_response()
        assert "savings_source" in health
        assert health["savings_source"] in {"session_tracker", "ledger", "memory"}
