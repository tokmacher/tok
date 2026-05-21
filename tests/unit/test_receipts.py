"""Phase 0.2.4: Bridge operation receipt tests."""

from __future__ import annotations

import json
from pathlib import Path


def test_bridge_receipt_digest_is_deterministic() -> None:
    from tok.receipt import BridgeReceipt

    receipt = BridgeReceipt(
        receipt_id="r1",
        session_id="s1",
        turn=2,
        step=3,
        request_policy="natural_first",
        compression_applied=True,
        evidence_summary={"exact_entries": 1},
        savings={"tokens_saved": 12},
        fallback=False,
    )

    assert receipt.compute_digest() == receipt.with_digest().compute_digest()
    assert receipt.with_digest().digest.startswith("sha256:")


def test_bridge_receipt_json_round_trip() -> None:
    from tok.receipt import BridgeReceipt

    receipt = BridgeReceipt(
        receipt_id="r1",
        session_id="s1",
        turn=1,
        step=9,
        request_policy="forced_baseline",
        compression_applied=False,
        evidence_summary={"reacquisition_required": 1},
        savings={"tokens_saved": 0, "cost_saved_usd": 0.0},
        fallback=True,
    ).with_digest()

    loaded = BridgeReceipt.from_dict(json.loads(json.dumps(receipt.to_dict())))
    assert loaded == receipt


def test_append_bridge_receipt_writes_jsonl(tmp_path: Path) -> None:
    from tok.receipt import BridgeReceipt, append_bridge_receipt, read_bridge_receipts

    receipt = BridgeReceipt(
        receipt_id="r1",
        session_id="s1",
        turn=1,
        step=1,
        request_policy="natural_first",
        compression_applied=True,
        evidence_summary={"exact_entries": 2},
        savings={"tokens_saved": 100},
        fallback=False,
    )

    path = tmp_path / "receipts.jsonl"
    append_bridge_receipt(path, receipt)

    records = read_bridge_receipts(path)
    assert len(records) == 1
    assert records[0].digest.startswith("sha256:")
    assert records[0].receipt_id == "r1"


def test_receipt_path_uses_session_directory(tmp_path: Path) -> None:
    from tok.receipt import bridge_receipt_path

    path = bridge_receipt_path(memory_dir=tmp_path, session_id="live:abc/unsafe")
    assert path == tmp_path / "sessions" / "live_abc_unsafe" / "receipts.jsonl"


def test_emit_bridge_receipt_records_fallback(tmp_path: Path) -> None:
    from tok.receipt import emit_bridge_receipt, read_bridge_receipts

    path = emit_bridge_receipt(
        memory_dir=tmp_path,
        session_id="s1",
        turn=5,
        step=6,
        request_policy="natural_first",
        compression_applied=False,
        evidence_summary={"exact_entries": 0},
        savings={"tokens_saved": 0},
        fallback=True,
    )

    receipts = read_bridge_receipts(path)
    assert len(receipts) == 1
    assert receipts[0].fallback is True
    assert receipts[0].compression_applied is False


def test_read_bridge_receipts_skips_partial_crash_line(tmp_path: Path) -> None:
    from tok.receipt import BridgeReceipt, append_bridge_receipt, read_bridge_receipts

    path = tmp_path / "receipts.jsonl"
    append_bridge_receipt(
        path,
        BridgeReceipt(
            receipt_id="r1",
            session_id="s1",
            turn=1,
            step=1,
            request_policy="natural_first",
            compression_applied=True,
            evidence_summary={},
            savings={"tokens_saved": 1},
            fallback=False,
        ),
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"receipt_id": "half-written"')

    receipts = read_bridge_receipts(path)
    assert [receipt.receipt_id for receipt in receipts] == ["r1"]
