"""File delivery and fidelity tracking state extracted from RuntimeSession."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FileDeliveryState:
    """Tracks file reads, delivery status, fidelity overrides, and skeleton tracking."""

    files_read_this_session: set[str] = field(default_factory=set)
    files_fully_delivered: dict[str, int] = field(default_factory=dict)
    fidelity_overrides: dict[str, int] = field(default_factory=dict)
    file_reads_by_turn: dict[str, int] = field(default_factory=dict)
    last_elevated_path: str = ""
    skeleton_delivered_paths: set[str] = field(default_factory=set)

    def reset(self) -> None:
        self.files_read_this_session.clear()
        self.files_fully_delivered.clear()
        self.fidelity_overrides.clear()
        self.file_reads_by_turn.clear()
        self.last_elevated_path = ""
        self.skeleton_delivered_paths.clear()
