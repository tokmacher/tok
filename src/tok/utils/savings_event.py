"""Per-request savings event schema and JSONL I/O.

Provides the canonical ``SavingsEvent`` dataclass (schema tok-savings-event/v1)
and helpers for append-only JSONL persistence.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA = "tok-savings-event/v1"

__all__ = [
    "SavingsEvent",
    "append_savings_event",
    "read_savings_events",
]


@dataclass
class SavingsEvent:
    """Immutable record of per-request savings metrics.

    Fields follow docs/savings-accounting.md schema tok-savings-event/v1.
    Map fields (compression_paths, non_headline_estimates) use default_factory so
    each instance gets its own dict.
    """

    # Required identity fields
    event_id: str
    session_id: str
    request_id: str
    timestamp: str
    model: str

    # Fixed schema discriminator
    schema: str = _SCHEMA

    # Optional context
    mode: str = ""
    request_policy: str = ""

    # Token accounting
    baseline_input_tokens: int = 0
    actual_input_tokens: int = 0
    input_tokens_saved: int = 0
    baseline_output_tokens: int = 0
    actual_output_tokens: int = 0
    output_tokens_saved: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    # Cost accounting (USD)
    baseline_cost_usd: float = 0.0
    actual_cost_usd: float = 0.0
    cost_saved_usd: float = 0.0

    # Failure flags
    fallback: bool = False
    degraded_to_baseline: bool = False

    # Breakdown maps
    compression_paths: dict[str, int] = field(default_factory=dict)
    non_headline_estimates: dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_jsonl_line(self) -> str:
        """Return the event as a single JSON line ending with newline."""
        return json.dumps(self.to_dict(), separators=(",", ":")) + "\n"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SavingsEvent:
        """Construct from a deserialized dict, ignoring unknown keys."""
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


def append_savings_event(event: SavingsEvent, path: Path) -> None:
    """Append one savings event to a JSONL file.

    Creates the parent directory if needed. Uses atomic rename for durability
    only when appending to a new file; for existing files falls back to direct
    append (rename would lose prior data for a single-append case and is
    unnecessary — POSIX guarantees that short writes to O_APPEND are atomic up
    to PIPE_BUF, and our lines are always well under that limit).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = event.to_jsonl_line()
    if not path.exists():
        # Atomic create: write to a temp file in the same directory then rename.
        try:
            fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".savings_tmp_")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(line)
                os.replace(tmp_name, path)
            except Exception:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
        except OSError:
            # Fallback: direct write if temp/rename fails (e.g. cross-device)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line)
    else:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)


def read_savings_events(path: Path) -> list[SavingsEvent]:
    """Read all valid SavingsEvent records from a JSONL file.

    Lines that are empty or fail JSON parsing are silently skipped (crash-safe).
    Returns an empty list if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return []
    events: list[SavingsEvent] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("savings_events: skipping corrupt line %d in %s", lineno, path)
                continue
            if not isinstance(data, dict):
                continue
            try:
                events.append(SavingsEvent.from_dict(data))
            except (TypeError, ValueError) as exc:
                logger.debug("savings_events: skipping invalid record at line %d: %s", lineno, exc)
    return events
