"""TokProfiler: Utility for tracking token usage and API costs."""

import time
from dataclasses import dataclass, field


@dataclass
class UsageRecord:
    turn: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model: str
    timestamp: float = field(default_factory=time.time)


class TokProfiler:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.history: list[UsageRecord] = []
        self.start_time = time.time()
