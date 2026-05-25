"""JSONL persistence for DeterministicActionReceipt — mirrors src/tok/receipt.py.

Experimental — not part of the 0.2.x public API.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from tok.deterministic.models import DeterministicActionReceipt


def deterministic_receipt_path(memory_dir: Path) -> Path:
    return memory_dir / "deterministic_receipts.jsonl"


def append_deterministic_receipt(path: Path, receipt: DeterministicActionReceipt) -> None:
    """Atomically append a receipt line with fsync."""
    line = json.dumps(receipt.to_dict(), separators=(",", ":")) + "\n"
    encoded = line.encode("utf-8")

    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    fd, tmp = tempfile.mkstemp(dir=parent, prefix=".tok_det_receipt_")
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)

    # Append by reading existing + writing combined, or just appending directly
    # Using direct append mode for simplicity and atomicity per line
    try:
        os.unlink(tmp)
    except OSError:
        pass

    with open(path, "ab") as f:
        f.write(encoded)
        f.flush()
        os.fsync(f.fileno())


def read_deterministic_receipts(path: Path) -> list[DeterministicActionReceipt]:
    if not path.exists():
        return []
    receipts: list[DeterministicActionReceipt] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                receipts.append(DeterministicActionReceipt.from_dict(data))
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
    return receipts
