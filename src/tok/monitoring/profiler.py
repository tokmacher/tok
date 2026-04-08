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

    def log_turn(self, turn: int, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
        """Record usage for a specific turn."""
        record = UsageRecord(
            turn=turn,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            model=self.model_name,
        )
        self.history.append(record)

    def get_total_cost(self) -> float:
        """Calculate total session cost."""
        return sum(r.cost_usd for r in self.history)

    def get_total_tokens(self) -> dict[str, int]:
        """Calculate total tokens."""
        return {
            "input": sum(r.input_tokens for r in self.history),
            "output": sum(r.output_tokens for r in self.history),
        }

    def get_summary(self) -> str:
        """Generate a markdown summary report."""
        total_cost = self.get_total_cost()
        tokens = self.get_total_tokens()
        duration = time.time() - self.start_time

        report = [
            "# Tok Session Report",
            f"**Model**: {self.model_name}",
            f"**Duration**: {duration:.2f}s",
            f"**Total Turns**: {len(self.history)}",
            f"**Total Cost**: ${total_cost:.6f}",
            f"**Input Tokens**: {tokens['input']}",
            f"**Output Tokens**: {tokens['output']}",
            "\n## Turn Details",
            "| Turn | Input | Output | Cost |",
            "| :--- | :--- | :--- | :--- |",
        ]

        for r in self.history:
            report.append(f"| {r.turn} | {r.input_tokens} | {r.output_tokens} | ${r.cost_usd:.6f} |")

        return "\n".join(report)
