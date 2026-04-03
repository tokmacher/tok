"""PointerRegistry for relational memory pointers extracted from bridge_memory."""

from __future__ import annotations

from ..utils.event_logging import log_pointer_created


class PointerRegistry:
    """Registry for relational memory pointers (*A, *B, etc.)."""

    def __init__(self) -> None:
        self.map: dict[str, str] = {}
        self.reverse_map: dict[str, str] = {}
        self._next_idx: int = 0

    def get_pointer(self, value: str) -> str:
        """Get or create a pointer for a value."""
        if value in self.reverse_map:
            return self.reverse_map[value]

        ptr = f"*{chr(65 + (self._next_idx % 26))}"
        if self._next_idx >= 26:
            ptr += str(self._next_idx // 26)

        self._next_idx += 1
        self.map[ptr] = value
        self.reverse_map[value] = ptr
        log_pointer_created(ptr, value)
        return ptr

    def resolve(self, ptr: str) -> str | None:
        return self.map.get(ptr)

    def to_tok(self) -> str:
        if not self.map:
            return ""
        lines = ["@pointers"]
        for ptr, val in sorted(self.map.items()):
            lines.append(f"  |> {ptr}={val}")
        return "\n".join(lines) + "\n"

    @classmethod
    def from_tok(cls, text: str) -> PointerRegistry:
        registry = cls()
        in_pointers: bool = False
        max_idx: int = -1
        for line in text.splitlines():
            stripped = line.strip()
            if stripped == "@pointers":
                in_pointers = True
                continue
            if in_pointers and stripped.startswith("|> "):
                payload = stripped[3:].strip()
                if "=" in payload:
                    ptr, val = payload.split("=", 1)
                    ptr = ptr.strip()
                    registry.map[ptr] = val.strip()
                    registry.reverse_map[val.strip()] = ptr

                    # Update next_idx to avoid collisions
                    # Simple A-Z pointers (*A, *B, etc.) and overflow (*A1, *B1, etc.)
                    if ptr.startswith("*"):
                        base = ptr[1]
                        if len(ptr) == 2:
                            idx = ord(base) - 65
                        else:
                            try:
                                cycle = int(ptr[2:])
                                idx = (cycle * 26) + (ord(base) - 65)
                            except ValueError:
                                idx = len(registry.map)
                            max_idx = max(max_idx, idx)
            elif stripped.startswith("@"):
                in_pointers = False
        registry._next_idx = max_idx + 1
        return registry
