"""Exact patch and receipt tests for tok.deterministic."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Test 1: Receipt construction — apply_exact_patch happy path
# ---------------------------------------------------------------------------


def test_receipt_construction_exact_patch() -> None:
    from tok.deterministic.exact_patch import apply_exact_patch
    from tok.deterministic.models import ExactPatchPreconditions

    original = b"def foo():\n    return 1\n"
    before_hash = hashlib.sha256(original).hexdigest()

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        target = workspace / "foo.py"
        target.write_bytes(original)

        preconditions = ExactPatchPreconditions(
            path="foo.py",
            expected_before_hash=before_hash,
            search="return 1",
            replace="return 42",
            workspace_root=str(workspace),
        )

        receipt = apply_exact_patch(preconditions, workspace)

    assert receipt.path == "foo.py"
    assert receipt.before_hash == before_hash
    assert receipt.after_hash is not None
    assert receipt.after_hash != before_hash
    assert receipt.status == "applied"
    assert receipt.exactness_level == "exact_patch"
    assert receipt.rollback_ref is not None
    assert len(receipt.rollback_ref) > 0
    assert receipt.action_id != ""
    assert not receipt.lossy


# ---------------------------------------------------------------------------
# Test 2: Hash precondition rejection — wrong hash
# ---------------------------------------------------------------------------


def test_hash_precondition_wrong_hash_rejected() -> None:
    from tok.deterministic.exact_patch import apply_exact_patch
    from tok.deterministic.models import ExactPatchPreconditions

    original = b"def bar():\n    pass\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        target = workspace / "bar.py"
        target.write_bytes(original)

        preconditions = ExactPatchPreconditions(
            path="bar.py",
            expected_before_hash="deadbeef" * 8,  # definitely wrong
            search="pass",
            replace="return None",
            workspace_root=str(workspace),
        )

        receipt = apply_exact_patch(preconditions, workspace)

    assert receipt.status == "rejected"
    assert receipt.rejection_reason is not None
    assert "hash" in receipt.rejection_reason.lower() or "mismatch" in receipt.rejection_reason.lower()


# ---------------------------------------------------------------------------
# Test 3: Ambiguity rejection — search string appears twice
# ---------------------------------------------------------------------------


def test_ambiguous_patch_rejected() -> None:
    from tok.deterministic.exact_patch import apply_exact_patch
    from tok.deterministic.models import ExactPatchPreconditions

    original = b"x = 1\nx = 1\n"
    before_hash = hashlib.sha256(original).hexdigest()

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        target = workspace / "dup.py"
        target.write_bytes(original)

        preconditions = ExactPatchPreconditions(
            path="dup.py",
            expected_before_hash=before_hash,
            search="x = 1",
            replace="x = 2",
            workspace_root=str(workspace),
        )

        receipt = apply_exact_patch(preconditions, workspace)

    assert receipt.status == "rejected"
    assert receipt.rejection_reason is not None
    assert "ambiguous" in receipt.rejection_reason.lower()


# ---------------------------------------------------------------------------
# Test 4: Vague semantic request not executable
# ---------------------------------------------------------------------------


def test_vague_semantic_request_not_executable() -> None:
    from tok.deterministic.action_decision import classify_action

    decision = classify_action({"kind": "semantic", "instruction": "fix this function"})

    assert decision.should_execute is False
    assert decision.should_reject is True


# ---------------------------------------------------------------------------
# Test 5: Verification receipt round-trip
# ---------------------------------------------------------------------------


def test_verification_receipt_compact() -> None:
    from tok.deterministic.models import DeterministicActionReceipt

    receipt = DeterministicActionReceipt(
        action_id="test-001",
        action_kind="exact_verification",
        provider_surface="claude_code",
        path=None,
        paths=[],
        before_hash=None,
        after_hash=None,
        changed_ranges=[],
        exactness_level="exact_verification",
        lossy=False,
        commands_run=["python -m py_compile foo.py"],
        verification_results={"exit_code": 0, "stdout": ""},
        rollback_ref=None,
        exact_diff_ref=None,
        status="verified",
        rejection_reason=None,
        created_at="2026-05-24T00:00:00Z",
    )

    d = receipt.to_dict()
    restored = DeterministicActionReceipt.from_dict(d)

    assert restored.action_id == receipt.action_id
    assert restored.action_kind == receipt.action_kind
    assert restored.commands_run == receipt.commands_run
    assert restored.verification_results == receipt.verification_results
    assert restored.status == receipt.status
    assert restored.lossy is False
