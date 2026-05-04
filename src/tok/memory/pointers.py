"""PointerRegistry for relational memory pointers extracted from bridge_memory."""

from __future__ import annotations

from tok.utils.event_logging import log_pointer_created

_MAX_POINTERS = 52  # 26 base + 26 overflow slots; prune beyond this


class PointerRegistry:
    """Registry for relational memory pointers (*A, *B, etc.)."""

    def __init__(self) -> None:
        self.map: dict[str, str] = {}
        self.reverse_map: dict[str, str] = {}
        self.scores: dict[str, int] = {}  # ptr -> score
        self._next_idx: int = 0

    def get_pointer(self, value: str, score: int = 1) -> str:
        """Get or create a pointer for a value, bumping its score on each access."""
        if value in self.reverse_map:
            ptr = self.reverse_map[value]
            self.scores[ptr] = self.scores.get(ptr, 1) + score
            return ptr

        ptr = f"*{chr(65 + (self._next_idx % 26))}"
        if self._next_idx >= 26:
            ptr += str(self._next_idx // 26)

        # Only intern if it actually saves space
        if len(value) <= len(ptr):
            return value

        self._next_idx += 1
        self.map[ptr] = value
        self.reverse_map[value] = ptr
        self.scores[ptr] = score
        log_pointer_created(ptr, value)

        # Evict lowest-scored pointer if over limit
        if len(self.map) > _MAX_POINTERS:
            self._evict_lowest()

        return ptr

    def _evict_lowest(self) -> None:
        if not self.map:
            return
        evict_ptr = min(self.map, key=lambda p: self.scores.get(p, 0))
        evict_val = self.map.pop(evict_ptr)
        self.reverse_map.pop(evict_val, None)
        self.scores.pop(evict_ptr, None)

    def resolve(self, ptr: str) -> str | None:
        return self.map.get(ptr)

    def to_tok(self) -> str:
        if not self.map:
            return ""
        lines = ["@pointers"]
        # Emit highest-scored pointers first so the session-start hint is signal-dense
        for ptr, val in sorted(self.map.items(), key=lambda item: -self.scores.get(item[0], 0)):
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
