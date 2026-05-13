from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProjectState:
    markers: frozenset[str] = field(default_factory=frozenset)
    files_read: set[str] = field(default_factory=set)
    files_fully_delivered: dict[str, int] = field(default_factory=dict)
    recently_edited_files: dict[str, int] = field(default_factory=dict)
    skeleton_delivered_paths: set[str] = field(default_factory=set)

    def mark_file_edited(self, norm_path: str, step_count: int) -> None:
        self.recently_edited_files[norm_path] = step_count

    def is_recently_edited(self, norm_path: str, step_count: int, window: int = 2) -> bool:
        edit_step = self.recently_edited_files.get(norm_path)
        if edit_step is None:
            return False
        return (step_count - edit_step) < window

    def reset(self) -> None:
        self.files_read.clear()
        self.files_fully_delivered.clear()
        self.recently_edited_files.clear()
        self.skeleton_delivered_paths.clear()
