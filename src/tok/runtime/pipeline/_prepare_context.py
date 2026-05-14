from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tok.runtime.core import RuntimeSession


@dataclass
class PreparePipelineContext:
    body: dict[str, Any]
    original_body: dict[str, Any]
    compressed: bool = False
    saved_tokens: int = 0
    type_breakdown: dict[str, int] = field(default_factory=dict)
    behavior_signals: dict[str, int] = field(default_factory=dict)
    hot_hint_metrics: dict[str, int] = field(default_factory=dict)
    pre_existing_session_signals: dict[str, int] = field(default_factory=dict)

    def add_saved_tokens(self, saved_tokens: int) -> None:
        if saved_tokens <= 0:
            return
        self.saved_tokens += saved_tokens
        self.compressed = True

    def bump_signal(self, key: str, value: int = 1) -> None:
        if value:
            self.behavior_signals[key] = self.behavior_signals.get(key, 0) + value

    def merge_signals(self, signals: dict[str, int]) -> None:
        for key, value in signals.items():
            self.bump_signal(key, value)

    def merge_hot_hint_metrics(self) -> None:
        for key, value in self.hot_hint_metrics.items():
            self.bump_signal(key, value)

    def merge_new_session_signals(self, session: RuntimeSession) -> None:
        for key, value in session.pending_behavior_signals.items():
            previous = self.pre_existing_session_signals.get(key, 0)
            if value and value > previous:
                self.bump_signal(key, value - previous)

    def account_prompt_savings(self, session: RuntimeSession) -> tuple[int, int, int]:
        prepared_prompt_tokens = session.prepared_prompt_tokens(self.body)
        baseline_prompt_tokens = session.prepared_prompt_tokens(self.original_body)
        saved_prompt_tokens = max(0, baseline_prompt_tokens - prepared_prompt_tokens)
        self.add_saved_tokens(saved_prompt_tokens)
        return baseline_prompt_tokens, prepared_prompt_tokens, saved_prompt_tokens
