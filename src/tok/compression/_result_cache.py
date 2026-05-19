from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CacheEntry:
    key: str
    content_hash: str
    compressed_content: str
    timestamp: float = field(default_factory=time.time)
    tool_name: str = ""
    turn_stored: int = 0


@dataclass
class ResultCache:
    entries: dict[str, CacheEntry] = field(default_factory=dict)

    def lookup(self, key: str) -> CacheEntry | None:
        return self.entries.get(key)

    def store(self, key: str, entry: CacheEntry) -> None:
        self.entries[key] = entry

    def invalidate(self, key: str) -> None:
        self.entries.pop(key, None)

    def should_dedup(self, tool_name: str, content_hash: str, key: str) -> bool:
        entry = self.entries.get(key)
        if entry is None:
            return False
        return entry.content_hash == content_hash

    @staticmethod
    def compute_key(tool_name: str, args: dict, content: str) -> str:
        payload = json.dumps(
            {"tool": tool_name, "args": args, "content_hash": hashlib.sha256(content.encode()).hexdigest()},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:32]
