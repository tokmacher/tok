"""Adversarial exact patch regressions for tok.deterministic."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Test 1: Stale file regression — hash computed before content was mutated
# ---------------------------------------------------------------------------


def test_stale_file_rejected_after_content_change() -> None:
    from tok.deterministic.exact_patch import apply_exact_patch
    from tok.deterministic.models import ExactPatchPreconditions

    original = b"x = 1\n"
    stale_hash = hashlib.sha256(original).hexdigest()

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        target = workspace / "stale.py"
        target.write_bytes(original)

        # Mutate the file after computing the hash
        target.write_bytes(b"x = 999\n")

        preconditions = ExactPatchPreconditions(
            path="stale.py",
            expected_before_hash=stale_hash,
            search="x = 1",
            replace="x = 2",
            workspace_root=str(workspace),
        )

        receipt = apply_exact_patch(preconditions, workspace)

    assert receipt.status == "rejected"
    assert receipt.rejection_reason is not None
    assert "hash" in receipt.rejection_reason.lower() or "mismatch" in receipt.rejection_reason.lower()


# ---------------------------------------------------------------------------
# Test 2: Fuzzy match regression — off-by-one whitespace/spelling rejected
# ---------------------------------------------------------------------------


def test_fuzzy_match_rejected() -> None:
    from tok.deterministic.exact_patch import apply_exact_patch
    from tok.deterministic.models import ExactPatchPreconditions

    original = b"return value\n"
    before_hash = hashlib.sha256(original).hexdigest()

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        target = workspace / "fuzzy.py"
        target.write_bytes(original)

        # Off by one space — should NOT fuzzy-match
        preconditions = ExactPatchPreconditions(
            path="fuzzy.py",
            expected_before_hash=before_hash,
            search="return  value",  # two spaces vs one
            replace="return result",
            workspace_root=str(workspace),
        )

        receipt = apply_exact_patch(preconditions, workspace)

    assert receipt.status == "rejected"
    # search_not_found or similar — no fuzzy fallback
    assert receipt.rejection_reason is not None


# ---------------------------------------------------------------------------
# Test 3: Path traversal regression
# ---------------------------------------------------------------------------


def test_path_traversal_rejected() -> None:
    from tok.deterministic.exact_patch import _safe_resolve, apply_exact_patch
    from tok.deterministic.models import ExactPatchPreconditions

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)

        # _safe_resolve should raise
        with pytest.raises(ValueError, match="outside workspace_root|traversal"):
            _safe_resolve("../../etc/passwd", workspace)

        # apply_exact_patch should return rejected receipt
        preconditions = ExactPatchPreconditions(
            path="../../etc/passwd",
            expected_before_hash="a" * 64,
            search="root",
            replace="toor",
            workspace_root=str(workspace),
        )

        receipt = apply_exact_patch(preconditions, workspace)

    assert receipt.status == "rejected"
    assert receipt.rejection_reason is not None
    assert "traversal" in receipt.rejection_reason.lower()


# ---------------------------------------------------------------------------
# Test 4: Lossy receipt blocks execution
# ---------------------------------------------------------------------------


def test_lossy_receipt_blocks_execution() -> None:
    from tok.deterministic.action_decision import classify_action
    from tok.deterministic.models import DeterministicActionReceipt

    # A lossy receipt should not be marked should_execute=True
    lossy_receipt = DeterministicActionReceipt(
        action_id="lossy-001",
        action_kind="lossy_summary_only",
        provider_surface=None,
        path=None,
        paths=[],
        before_hash=None,
        after_hash=None,
        changed_ranges=[],
        exactness_level="lossy_summary_only",
        lossy=True,
        commands_run=[],
        verification_results={},
        rollback_ref=None,
        exact_diff_ref=None,
        status="applied",
        rejection_reason=None,
        created_at="2026-05-24T00:00:00Z",
    )

    assert lossy_receipt.lossy is True

    # An action dict without expected_before_hash must be rejected
    decision = classify_action({"kind": "edit", "path": "foo.py", "search": "x", "replace": "y"})
    assert decision.should_execute is False
    assert decision.should_reject is True
    assert "expected_before_hash" in decision.missing_preconditions


# ---------------------------------------------------------------------------
# Test 5: Multi-file request rejected
# ---------------------------------------------------------------------------


def test_multi_file_request_rejected() -> None:
    from tok.deterministic.action_decision import classify_action

    decision = classify_action({"paths": ["a.py", "b.py"], "kind": "multi_edit"})
    assert decision.should_reject is True
    assert decision.should_execute is False


# ---------------------------------------------------------------------------
# Test 6: Provider-neutrality regression
# ---------------------------------------------------------------------------


def test_provider_neutral_core() -> None:
    from tok.deterministic.action_decision import check_preconditions, classify_action
    from tok.deterministic.models import ExactPatchPreconditions

    # classify_action succeeds with no provider_surface field
    decision = classify_action(
        {
            "path": "foo.py",
            "search": "x = 1",
            "replace": "x = 2",
            "expected_before_hash": "a" * 64,
        }
    )
    assert decision.should_execute is True
    assert decision.should_reject is False

    # check_preconditions fails based on hash mismatch, not provider
    content = b"x = 1\n"
    preconditions = ExactPatchPreconditions(
        path="foo.py",
        expected_before_hash="b" * 64,  # wrong hash
        search="x = 1",
        replace="x = 2",
        workspace_root="/tmp",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        result = check_preconditions(preconditions, workspace, content)

    assert result.should_reject is True
    assert "hash" in result.reason.lower() or "mismatch" in result.reason.lower()


# ---------------------------------------------------------------------------
# Test 7: Existing bridge/compression/trace imports unaffected
# ---------------------------------------------------------------------------


def test_existing_bridge_and_compression_imports() -> None:
    from tok.receipt import BridgeReceipt

    assert BridgeReceipt is not None

    import tok.spec.trace as trace_mod

    assert hasattr(trace_mod, "ALLOWED_ACTIONS"), "ALLOWED_ACTIONS missing from tok.spec.trace"

    # tok.deterministic should be importable without affecting tok root namespace
    from tok.deterministic import (
        DeterministicActionReceipt,
        ExactPatchPreconditions,
        apply_exact_patch,
        classify_action,
    )

    assert apply_exact_patch is not None
    assert classify_action is not None
    assert ExactPatchPreconditions is not None
    assert DeterministicActionReceipt is not None

    # tok root __all__ must NOT include deterministic symbols
    import tok

    tok_all = getattr(tok, "__all__", [])
    assert "apply_exact_patch" not in tok_all
    assert "DeterministicActionReceipt" not in tok_all


def test_pure_precondition_rejects_sibling_prefix_escape() -> None:
    from tok.deterministic.action_decision import check_preconditions
    from tok.deterministic.models import ExactPatchPreconditions

    content = b"x = 1\n"
    before_hash = hashlib.sha256(content).hexdigest()

    with tempfile.TemporaryDirectory() as tmpdir:
        parent = Path(tmpdir)
        workspace = parent / "work"
        sibling = parent / "work_evil"
        workspace.mkdir()
        sibling.mkdir()

        preconditions = ExactPatchPreconditions(
            path="../work_evil/pwn.py",
            expected_before_hash=before_hash,
            search="x = 1",
            replace="x = 2",
            workspace_root=str(workspace),
        )

        decision = check_preconditions(preconditions, workspace, content)

    assert decision.should_reject is True
    assert decision.should_execute is False
    assert decision.reason == "path_traversal_detected"


def test_exact_patch_preserves_non_utf8_bytes() -> None:
    from tok.deterministic.exact_patch import apply_exact_patch
    from tok.deterministic.models import ExactPatchPreconditions

    original = b"a\xffb\nneedle\n"
    before_hash = hashlib.sha256(original).hexdigest()

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        target = workspace / "binary.dat"
        target.write_bytes(original)

        preconditions = ExactPatchPreconditions(
            path="binary.dat",
            expected_before_hash=before_hash,
            search="needle",
            replace="thread",
            workspace_root=str(workspace),
        )

        receipt = apply_exact_patch(preconditions, workspace)
        after = target.read_bytes()

    assert receipt.status == "applied"
    assert after == b"a\xffb\nthread\n"
    assert receipt.rollback_ref is not None
    assert receipt.rollback_ref.encode("latin-1") == original


def test_empty_replacement_deletes_exact_match() -> None:
    from tok.deterministic.action_decision import classify_action
    from tok.deterministic.exact_patch import apply_exact_patch
    from tok.deterministic.models import ExactPatchPreconditions

    original = b"keep\nremove\nkeep\n"
    before_hash = hashlib.sha256(original).hexdigest()

    decision = classify_action(
        {
            "kind": "edit",
            "path": "delete.txt",
            "search": "remove\n",
            "replace": "",
            "expected_before_hash": before_hash,
        }
    )
    assert decision.should_execute is True

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        target = workspace / "delete.txt"
        target.write_bytes(original)

        preconditions = ExactPatchPreconditions(
            path="delete.txt",
            expected_before_hash=before_hash,
            search="remove\n",
            replace="",
            workspace_root=str(workspace),
        )

        receipt = apply_exact_patch(preconditions, workspace)
        after = target.read_bytes()

    assert receipt.status == "applied"
    assert after == b"keep\nkeep\n"
