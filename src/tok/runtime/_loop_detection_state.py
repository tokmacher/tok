from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .config import TOK_LOOP_DETECTION_ENABLED, TOK_LOOP_DETECTION_THRESHOLD

logger = logging.getLogger("tok.runtime.loop_detection")


@dataclass
class LoopDetectionState:
    window: list[tuple[str, str]] = field(default_factory=list)
    detected: bool = False

    def observe_tool_action(self, tool_name: str, tool_input_key: str) -> bool:
        if not TOK_LOOP_DETECTION_ENABLED:
            return False
        action_key = f"{tool_name}:{tool_input_key}"
        self.window.append((tool_name, action_key))
        window_size = TOK_LOOP_DETECTION_THRESHOLD * 3
        if len(self.window) > window_size:
            self.window = self.window[-window_size:]
        if len(self.window) < TOK_LOOP_DETECTION_THRESHOLD:
            return False
        recent = self.window[-TOK_LOOP_DETECTION_THRESHOLD:]
        if all(action_key == recent[0][1] for _, action_key in recent):
            self.detected = True
            logger.warning(
                "tok_loop_detected: %d consecutive identical actions: %s",
                TOK_LOOP_DETECTION_THRESHOLD,
                recent[0][1],
            )
            return True
        return False

    def consume_loop_detected(self) -> bool:
        was_detected = self.detected
        self.detected = False
        return was_detected

    def reset(self) -> None:
        self.window.clear()
        self.detected = False
