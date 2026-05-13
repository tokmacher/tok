from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TelemetryState:
    step_count: int = 0
    tool_names_seen: set[str] = field(default_factory=set)
    token_count: int = 0
    tool_density: float = 0.0
    context_char_count: int = 0
    invisible_pressure: int = 0
    active_tools: list[str] = field(default_factory=list)
    last_tool_compatible_state: str = ""
    last_tool_compatible_state_fields: dict[str, list[str]] = field(default_factory=dict)
    suppressed_failure_markers: frozenset[str] = field(default_factory=frozenset)
    response_word_samples: list[int] = field(default_factory=list)
    is_first_request: bool = True
    tok_memory_snap_triggered: int = 0
    last_mode: str = ""

    def reset(self) -> None:
        self.step_count = 0
        self.tool_names_seen.clear()
        self.token_count = 0
        self.tool_density = 0.0
        self.context_char_count = 0
        self.invisible_pressure = 0
        self.active_tools.clear()
        self.last_tool_compatible_state = ""
        self.last_tool_compatible_state_fields.clear()
        self.suppressed_failure_markers = frozenset()
        self.response_word_samples.clear()
        self.is_first_request = True
        self.tok_memory_snap_triggered = 0
        self.last_mode = ""
