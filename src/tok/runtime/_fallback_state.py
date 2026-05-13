from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import _FALLBACK_THRESHOLD

logger = logging.getLogger("tok.runtime.fallback_state")


@dataclass
class FallbackState:
    consecutive_count: int = 0
    baseline_only: bool = False
    persistence_failures: int = 0
    invalid_tool_history_recovery_count: int = 0

    def record_fallback_event(self) -> None:
        self.consecutive_count += 1
        if self.consecutive_count >= _FALLBACK_THRESHOLD and not self.baseline_only:
            self.baseline_only = True
            logger.warning(
                "tok_fallback_activated: session degraded to baseline after %d consecutive fallback events",
                self.consecutive_count,
            )

    def reset_fallback_count(self) -> None:
        self.consecutive_count = 0
        self.baseline_only = False

    def reset(self) -> None:
        self.consecutive_count = 0
        self.baseline_only = False
        self.persistence_failures = 0
