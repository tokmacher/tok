from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CacheState:
    result_cache: dict[str, Any] = field(default_factory=dict)
    semantic_hash_cache: dict[str, str] = field(default_factory=dict)
    observed_tool_result_ids: dict[str, None] = field(default_factory=dict)
    prepared_prompt_token_cache: dict[str, int] = field(default_factory=dict)
    predictive_cache_warm_keys: set[str] = field(default_factory=set)

    def reset(self) -> None:
        self.result_cache.clear()
        self.semantic_hash_cache.clear()
        self.observed_tool_result_ids.clear()
        self.prepared_prompt_token_cache.clear()
        self.predictive_cache_warm_keys.clear()
