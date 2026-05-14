from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, cast

from tok.compression import text_of
from tok.runtime._context_fidelity import prompt_optimization_materially_degrades_context
from tok.runtime.core import RuntimeSession
from tok.runtime.memory.bridge_memory import clean_system_context
from tok.runtime.pipeline.request_validation import detect_prompt_bloat
from tok.runtime.types import RuntimeRequest

logger = logging.getLogger("tok.runtime.pipeline._prepare_optimize_prompt")


@dataclass
class Step2Result:
    body: dict[str, Any] = field(default_factory=dict)
    compressed: bool = False


def run_step_2(
    request: RuntimeRequest,
    session: RuntimeSession,
    body: dict[str, Any],
    last_user_msg: str,
    is_bridge_adapter: bool,
    compressed: bool,
) -> Step2Result:
    if detect_prompt_bloat(body.get("system"), last_user_msg):
        session.pending_behavior_signals["tok_prompt_bloat_detected"] = 1
        if is_bridge_adapter:
            session.pending_behavior_signals["tok_prompt_optimization_skipped_bridge"] = 1
            current_sys = cast("Any", body.get("system", ""))
            dry_run_memory = type(session.bridge_memory)(load_global_macros=False)
            cleaned_sys = clean_system_context(dry_run_memory, current_sys)
            if cleaned_sys and cleaned_sys != current_sys:
                original_chars = len(text_of(current_sys) if isinstance(current_sys, list) else str(current_sys))
                cleaned_chars = len(text_of(cleaned_sys) if isinstance(cleaned_sys, list) else str(cleaned_sys))
                suppressed_chars = max(0, original_chars - cleaned_chars)
                if suppressed_chars:
                    session.pending_behavior_signals["tok_prompt_optimization_suppressed_chars"] = (
                        session.pending_behavior_signals.get("tok_prompt_optimization_suppressed_chars", 0)
                        + suppressed_chars
                    )
                    session.pending_behavior_signals["tok_prompt_optimization_suppressed_tokens"] = (
                        session.pending_behavior_signals.get("tok_prompt_optimization_suppressed_tokens", 0)
                        + suppressed_chars // 4
                    )
            logger.info(
                "tok_prompt_optimization_skipped_bridge: adapter_kind=%s, skipping clean_system_context",
                request.adapter_kind,
            )
        else:
            current_sys = cast("Any", body.get("system", ""))
            cleaned_sys = clean_system_context(session.bridge_memory, current_sys)
            if cleaned_sys and cleaned_sys != current_sys:
                degraded, degrade_reason = prompt_optimization_materially_degrades_context(
                    current_sys,
                    cleaned_sys,
                    last_user_msg,
                )
                if degraded:
                    session.pending_behavior_signals["tok_prompt_optimization_blocked"] = 1
                    session.pending_behavior_signals[f"tok_prompt_optimization_blocked_{degrade_reason}"] = 1
                    logger.info(
                        "tok_prompt_optimization_blocked: reason=%s original_chars=%d optimized_chars=%d",
                        degrade_reason,
                        len(text_of(current_sys) if isinstance(current_sys, list) else str(current_sys)),
                        len(text_of(cleaned_sys) if isinstance(cleaned_sys, list) else str(cleaned_sys)),
                    )
                else:
                    body["system"] = cleaned_sys
                    session.pending_behavior_signals["tok_prompt_optimized"] = 1
                    if session.bridge_memory.top_hot_files(1):
                        session.pending_behavior_signals["smoothness_prompt_optimization_active_task"] = 1
                    compressed = True
                    logger.warning(
                        "tok_prompt_optimized: system prompt reduced from %d to %d chars",
                        len(text_of(current_sys) if isinstance(current_sys, list) else str(current_sys)),
                        len(text_of(cleaned_sys) if isinstance(cleaned_sys, list) else str(cleaned_sys)),
                    )

    return Step2Result(
        body=body,
        compressed=compressed,
    )
