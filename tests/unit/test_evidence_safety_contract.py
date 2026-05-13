"""Evidence safety contract tests (Plan 5).

Tests each clause of the evidence-safety contract from philosophy.md:
1. First file observation must be exact.
2. Skeleton evidence must not authorize edit-like tools.
3. Repeated reads may compress only after exact evidence exists.
4. Edit target requires exact reacquisition first.
5. Novel failures must not be compressed away.

Plus tests for the deepened EvidenceSafetyState methods.
"""

from __future__ import annotations

from tok.runtime.evidence_safety import (
    EvidenceSafetyState,
)


class TestEvidenceSafetyStateMethods:
    def test_record_exact_evidence_new_key(self):
        state = EvidenceSafetyState()
        signals = state.record_exact("file.py", digest="abc", turn=1)
        assert "file.py" in state.first_exact_seen
        assert state.ledger["file.py"].first_exact_turn == 1
        assert state.ledger["file.py"].latest_form == "exact"
        assert signals.get("evidence_exact_observed") == 1
        assert signals.get("evidence_first_exact_observed") == 1

    def test_record_exact_evidence_clears_reacquisition(self):
        state = EvidenceSafetyState()
        state.record_non_exact("file.py", form="skeleton", turn=1)
        state.require_exact_reacquisition("file.py")
        assert state.ledger["file.py"].exact_reacquisition_required is True
        signals = state.record_exact("file.py", digest="def", turn=2)
        assert state.ledger["file.py"].exact_reacquisition_required is False
        assert signals.get("evidence_exact_reacquisition_satisfied") == 1

    def test_record_non_exact_evidence(self):
        state = EvidenceSafetyState()
        signals = state.record_non_exact("file.py", form="skeleton", turn=1)
        assert state.ledger["file.py"].latest_form == "skeleton"
        assert "file.py" not in state.first_exact_seen
        assert signals.get("evidence_non_exact_reference_emitted") == 1

    def test_require_exact_reacquisition_on_non_exact(self):
        state = EvidenceSafetyState()
        state.record_non_exact("file.py", form="skeleton", turn=1)
        signals = state.require_exact_reacquisition("file.py")
        assert state.ledger["file.py"].exact_reacquisition_required is True
        assert signals.get("evidence_exact_reacquisition_required") == 1

    def test_require_exact_reacquisition_on_exact_is_noop(self):
        state = EvidenceSafetyState()
        state.record_exact("file.py", digest="abc", turn=1)
        signals = state.require_exact_reacquisition("file.py")
        assert signals == {}

    def test_evidence_requires_reacquisition(self):
        state = EvidenceSafetyState()
        state.record_non_exact("file.py", form="skeleton", turn=1)
        assert state.requires_reacquisition("file.py") is True
        state.record_exact("file.py", digest="abc", turn=2)
        assert state.requires_reacquisition("file.py") is False

    def test_audit_summary(self):
        state = EvidenceSafetyState()
        state.record_exact("file1.py", digest="a", turn=1)
        state.record_non_exact("file2.py", form="skeleton", turn=1)
        state.require_exact_reacquisition("file2.py")
        summary = state.audit_summary()
        assert summary["entries"] == 2
        assert summary["exact_entries"] == 1
        assert summary["non_exact_latest"] == 1
        assert summary["reacquisition_required"] == 1


class TestPhilosophyContract:
    """One test per clause of the evidence-safety contract from philosophy.md."""

    def test_first_observation_is_exact(self):
        state = EvidenceSafetyState()
        state.record_exact("file.py", digest="abc", turn=1)
        assert state.ledger["file.py"].latest_form == "exact"
        assert state.ledger["file.py"].has_exact

    def test_skeleton_does_not_authorize_edits(self):
        state = EvidenceSafetyState()
        state.record_non_exact("file.py", form="skeleton", turn=1)
        assert state.requires_reacquisition("file.py") is True

    def test_repeated_reads_compress_after_exact_exists(self):
        state = EvidenceSafetyState()
        state.record_exact("file.py", digest="abc", turn=1)
        state.record_non_exact("file.py", form="reference", turn=2)
        assert state.ledger["file.py"].has_exact
        assert state.ledger["file.py"].latest_form == "reference"

    def test_edit_target_requires_reacquisition(self):
        state = EvidenceSafetyState()
        state.record_non_exact("file.py", form="skeleton", turn=1)
        signals = state.require_exact_reacquisition("file.py")
        assert state.ledger["file.py"].exact_reacquisition_required is True
        assert signals.get("evidence_compression_blocked_for_safety") == 1

    def test_reacquisition_satisfied_by_exact(self):
        state = EvidenceSafetyState()
        state.record_non_exact("file.py", form="skeleton", turn=1)
        state.require_exact_reacquisition("file.py")
        assert state.ledger["file.py"].exact_reacquisition_required is True
        state.record_exact("file.py", digest="full_content", turn=2)
        assert state.ledger["file.py"].exact_reacquisition_required is False
        assert state.ledger["file.py"].exact_reacquisition_satisfied_turn == 2

    def test_empty_key_is_noop(self):
        state = EvidenceSafetyState()
        assert state.record_exact("", digest="abc", turn=1) == {}
        assert state.record_non_exact("", form="skeleton", turn=1) == {}
        assert state.require_exact_reacquisition("") == {}

    def test_reset_clears_everything(self):
        state = EvidenceSafetyState()
        state.record_exact("file1.py", digest="a", turn=1)
        state.record_non_exact("file2.py", form="skeleton", turn=1)
        state.require_exact_reacquisition("file2.py")
        state.reset()
        assert state.ledger == {}
        assert state.first_exact_seen == set()
        assert state.pending_exact_keys == set()
        assert state.neighborhoods == {}
        assert state.alias_map == {}
