"""Bridge operation receipts.

Receipts are durable, local JSONL records of bridge operations. They complement
Tok Trace sidecars: traces describe structure and timing, while receipts record
the operation outcome, evidence counters, savings, and fallback state.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BridgeReceipt:
    receipt_id: str
    session_id: str
    turn: int
    step: int
    request_policy: str
    compression_applied: bool
    evidence_summary: dict[str, Any] = field(default_factory=dict)
    savings: dict[str, Any] = field(default_factory=dict)
    fallback: bool = False
    digest: str = ""
    signature: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "session_id": self.session_id,
            "turn": int(self.turn),
            "step": int(self.step),
            "request_policy": self.request_policy,
            "compression_applied": bool(self.compression_applied),
            "evidence_summary": dict(self.evidence_summary),
            "savings": dict(self.savings),
            "fallback": bool(self.fallback),
            "digest": self.digest,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BridgeReceipt:
        return cls(
            receipt_id=str(data.get("receipt_id", "")),
            session_id=str(data.get("session_id", "")),
            turn=int(data.get("turn", 0) or 0),
            step=int(data.get("step", 0) or 0),
            request_policy=str(data.get("request_policy", "")),
            compression_applied=bool(data.get("compression_applied", False)),
            evidence_summary=dict(data.get("evidence_summary") or {}),
            savings=dict(data.get("savings") or {}),
            fallback=bool(data.get("fallback", False)),
            digest=str(data.get("digest", "")),
            signature=str(data.get("signature", "")),
        )

    def compute_digest(self) -> str:
        payload = self.to_dict()
        payload["digest"] = ""
        payload["signature"] = ""
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return "sha256:" + sha256(encoded).hexdigest()

    def with_digest(self) -> BridgeReceipt:
        return BridgeReceipt(
            receipt_id=self.receipt_id,
            session_id=self.session_id,
            turn=self.turn,
            step=self.step,
            request_policy=self.request_policy,
            compression_applied=self.compression_applied,
            evidence_summary=dict(self.evidence_summary),
            savings=dict(self.savings),
            fallback=self.fallback,
            digest=self.compute_digest(),
            signature=self.signature,
        )


def bridge_receipt_path(*, memory_dir: Path, session_id: str) -> Path:
    return memory_dir / "sessions" / _safe_session_id(session_id) / "receipts.jsonl"


def append_bridge_receipt(path: Path, receipt: BridgeReceipt) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = receipt.with_digest()
    line = json.dumps(completed.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix="tok-receipt-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        with path.open("a", encoding="utf-8") as out:
            out.write(Path(tmp_name).read_text(encoding="utf-8"))
            out.flush()
            os.fsync(out.fileno())
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def read_bridge_receipts(path: Path) -> list[BridgeReceipt]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    receipts: list[BridgeReceipt] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            receipts.append(BridgeReceipt.from_dict(data))
    return receipts


def emit_bridge_receipt(
    *,
    memory_dir: Path,
    session_id: str,
    turn: int,
    step: int,
    request_policy: str,
    compression_applied: bool,
    evidence_summary: dict[str, Any],
    savings: dict[str, Any],
    fallback: bool,
) -> Path:
    receipt = BridgeReceipt(
        receipt_id=str(uuid.uuid4()),
        session_id=session_id,
        turn=turn,
        step=step,
        request_policy=request_policy,
        compression_applied=compression_applied,
        evidence_summary=evidence_summary,
        savings=savings,
        fallback=fallback,
    )
    path = bridge_receipt_path(memory_dir=memory_dir, session_id=session_id)
    append_bridge_receipt(path, receipt)
    return path


def emit_bridge_receipt_for_session(
    session: Any,
    *,
    request_policy: str,
    compression_applied: bool,
    evidence_summary: dict[str, Any],
    savings: dict[str, Any],
    fallback: bool,
) -> Path:
    memory_dir = Path(getattr(session, "memory_dir", None) or Path.home() / ".tok")
    session_id = _session_id_from_session(session)
    runtime_session = getattr(session, "runtime_session", None)
    bridge_memory = getattr(runtime_session, "bridge_memory", None)
    turn = int(getattr(bridge_memory, "turn", 0) or 0)
    step = int(getattr(session, "_receipt_step", 0) or 0) + 1
    try:
        session._receipt_step = step
    except Exception:
        pass
    return emit_bridge_receipt(
        memory_dir=memory_dir,
        session_id=session_id,
        turn=turn,
        step=step,
        request_policy=request_policy,
        compression_applied=compression_applied,
        evidence_summary=evidence_summary,
        savings=savings,
        fallback=fallback,
    )


def _safe_session_id(session_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id).strip("_") or "default"


def _session_id_from_session(session: Any) -> str:
    instance = str(getattr(session, "_live_trace_instance_id", "") or "standalone")
    key = str(getattr(session, "_active_session_key", "") or "default")
    return "live:" + sha256(f"{instance}:{key}".encode()).hexdigest()[:24]


__all__ = [
    "BridgeReceipt",
    "append_bridge_receipt",
    "bridge_receipt_path",
    "emit_bridge_receipt",
    "emit_bridge_receipt_for_session",
    "read_bridge_receipts",
]
