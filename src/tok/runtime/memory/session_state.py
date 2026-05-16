"""Helper functions for RuntimeSession state management."""

import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tok.compression import text_of
from tok.macros.integration import distill_bridge_history
from tok.runtime.config import _PROJECT_MARKER_FILES
from tok.runtime.policy.smart_policy import (
    advance_state,
    initial_state,
    policy_for_model,
)

if TYPE_CHECKING:
    from tok.runtime.core import RuntimeSession

logger = logging.getLogger("tok.runtime")


def calculate_reasoning_depth(session: "RuntimeSession") -> float:
    """Compute reasoning diversity per token consumed."""
    if session._token_count == 0:
        return 0.0
    tool_diversity = len(session._tool_names_seen) or 1
    return round((session._step_count * tool_diversity) / session._token_count, 4)


def update_session_family_mode(session: "RuntimeSession", model: str, signals: dict[str, int]) -> str:
    """Advance the family adaptive state and return the new mode."""
    if not model:
        return ""
    policy = policy_for_model(model)
    current = session.family_states.setdefault(policy.family.key, initial_state(policy))
    # Pass tool names for task type detection
    tool_names = list(session._tool_names_seen) if session._tool_names_seen else None
    next_state = advance_state(policy, current, signals, tool_names=tool_names)
    session.family_states[policy.family.key] = next_state
    return next_state.mode


def _persist_natural_goal(session: "RuntimeSession", text: str) -> None:
    goal, _ = extract_goal_from_messages([{"role": "assistant", "content": text}])
    if not goal:
        return
    existing = session.bridge_memory.hot.get("goal", [])
    if existing:
        return
    session.bridge_memory._upsert(session.bridge_memory.hot, "goal", goal, score_delta=1)
    session._save_bridge_memory()
    logger.debug("Natural goal persisted: %s", goal[:60])


def session_write_memory(session: "RuntimeSession", text: str) -> str:
    """Ingest Tok state from response text and update memory."""
    from tok.runtime.policy.macro_handling import _heal_macro_from_repair

    tok_lines: list[str] = re.findall(r"^>>>.*$", text, re.MULTILINE)
    if not tok_lines:
        logger.debug("No Tok state lines found in response text")
        _persist_natural_goal(session, text)
        return ""
    latest_state: str = tok_lines[-1]
    logger.debug("Writing memory from Tok state: %s", latest_state[:100])
    session.fallback_memory = latest_state
    metrics = session.bridge_memory.ingest_wire_state(latest_state)
    logger.debug("Memory ingestion metrics: %s", metrics)
    session._bump_signals(metrics)
    session._save_fallback_memory()

    is_error = bool(re.search(r"errs:[^|]+", latest_state))
    if session._pending_macro_heal and int(os.getenv("TOK_MACRO_HEAL", "1")) and not is_error:
        _heal_macro_from_repair(
            session._pending_macro_heal,
            session.bridge_memory,
            heal_turn=session._pending_macro_heal_turn,
        )
    session._pending_macro_heal = ""

    if os.getenv("TOK_NEURO_REACTOR", "1") == "1":
        discovered = distill_bridge_history(
            session.bridge_memory,
            project_markers=session._project_markers,
        )
        if discovered:
            logger.info("NeuroReactor: Discovered %d macros this turn", len(discovered))

    session._save_bridge_memory()
    session._save_result_cache()
    return latest_state


def get_adaptive_keep_turns(session: "RuntimeSession") -> int:
    """
    Determine how many history turns to keep based on session age.

    Raw runtime sessions retain the young-session stability floor.  Bridge
    sessions mark keep_turns as explicit so user-facing configuration such as
    ``--keep-turns 0`` is honored exactly.
    """
    configured_keep_turns = max(0, int(getattr(session, "keep_turns", 2)))
    if getattr(session, "_keep_turns_explicit", False):
        return configured_keep_turns
    if session._step_count < 3:
        return max(configured_keep_turns, 3)
    return configured_keep_turns


def _discover_project_markers(cwd: Path | None = None) -> frozenset[str]:
    """
    Non-recursively scan the current working directory for project-type markers.

    Returns only the filenames that actually exist (e.g. ``'package.json'``).
    Errors are silently swallowed so a missing CWD never crashes session init.
    """
    try:
        base = cwd if cwd is not None else Path.cwd()
        return frozenset(name for name in _PROJECT_MARKER_FILES if (base / name).exists())
    except Exception:
        return frozenset()


_GOAL_MAX_LEN = 40
_GOAL_PHRASES = (
    "let me ",
    "i need to ",
    "i'll ",
    "i will ",
    "next, i ",
    "the root issue is",
    "the real problem is",
    "the problem is",
    "the issue is",
    "root cause is",
    "my goal is",
    "the goal is",
    "the plan is",
    "i'm going to ",
    "i am going to ",
    "the next step is",
    "we need to ",
    "i want to ",
    "i should ",
    "let's ",
    "now i ",
    "first, ",
    "then i ",
)
_GOAL_SKIP_PREFIXES = (
    "```",
    "<tool",
    "<TOOL",
    "@",
    ">>>",
)


def _extract_goal_line(stripped: str, lowered: str) -> str:
    for phrase in _GOAL_PHRASES:
        idx = lowered.find(phrase)
        if idx == -1:
            continue
        candidate = stripped[idx:].strip()
        if not candidate:
            continue
        if len(candidate) > _GOAL_MAX_LEN:
            sentence_end = -1
            for sep in (".", "!", "?", ";"):
                pos = candidate.find(sep)
                if pos != -1 and (sentence_end == -1 or pos < sentence_end):
                    sentence_end = pos
            if sentence_end > 0:
                candidate = candidate[:sentence_end].strip()
        if len(candidate) > _GOAL_MAX_LEN:
            candidate = candidate[:_GOAL_MAX_LEN].strip()
        if candidate and not any(candidate.startswith(p) for p in _GOAL_SKIP_PREFIXES):
            return candidate
    return ""


_USER_GOAL_PHRASES = (
    "i want you to ",
    "i want to ",
    "i need you to ",
    "please ",
    "i need ",
    "help me ",
    "can you ",
    "could you ",
    "your task is",
    "the task is",
    "the goal is",
    "the objective is",
    "i'm trying to ",
    "i am trying to ",
    "make sure ",
    "ensure that ",
    "we need to ",
    "fix ",
    "implement ",
    "investigate ",
    "refactor ",
    "debug ",
    "resolve ",
)


_USER_GOAL_SKIP_PATTERNS = (
    "- ",
    "▎",
    "▗",
    "▘",
    "system-reminder",
    "<system-reminder",
)


def _is_system_injected_line(stripped: str, lowered: str) -> bool:
    if lowered.startswith("simplify:") or lowered.startswith("review "):
        return True
    for pattern in _USER_GOAL_SKIP_PATTERNS:
        if lowered.startswith(pattern) or stripped.startswith(pattern):
            return True
    return False


def _extract_user_goal_line(stripped: str, lowered: str) -> str:
    for phrase in _USER_GOAL_PHRASES:
        idx = lowered.find(phrase)
        if idx == -1:
            continue
        candidate = stripped[idx:].strip()
        if not candidate:
            continue
        if len(candidate) > _GOAL_MAX_LEN:
            sentence_end = -1
            for sep in (".", "!", "?", ";", "\n"):
                pos = candidate.find(sep)
                if pos != -1 and (sentence_end == -1 or pos < sentence_end):
                    sentence_end = pos
            if sentence_end > 0:
                candidate = candidate[:sentence_end].strip()
        if len(candidate) > _GOAL_MAX_LEN:
            candidate = candidate[:_GOAL_MAX_LEN].strip()
        if candidate and not any(candidate.startswith(p) for p in _GOAL_SKIP_PREFIXES):
            return candidate
    return ""


def _first_user_goal(messages: list[dict[str, Any]]) -> str:
    best_goal = ""
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if msg.get("role") != "user":
            continue
        msg_text = text_of(msg.get("content", ""))
        for line in msg_text.splitlines():
            stripped = line.strip()
            lowered = stripped.lower()
            if not stripped:
                continue
            if _is_system_injected_line(stripped, lowered):
                continue
            goal = _extract_user_goal_line(stripped, lowered)
            if goal:
                best_goal = goal
    return best_goal


def extract_goal_from_messages(messages: list[dict[str, Any]], max_assistant: int = 3) -> tuple[str, bool]:
    user_goal = _first_user_goal(messages)
    if user_goal:
        return user_goal, True
    best_goal = ""
    best_idx = -1
    assistant_seen = 0
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if msg.get("role") != "assistant":
            continue
        assistant_seen += 1
        if assistant_seen > max_assistant:
            break
        msg_text = text_of(msg.get("content", ""))
        for line in msg_text.splitlines():
            stripped = line.strip()
            lowered = stripped.lower()
            if not stripped:
                continue
            goal = _extract_goal_line(stripped, lowered)
            if goal and idx > best_idx:
                best_goal = goal
                best_idx = idx
    return best_goal, False


_HYPOTHESIS_QUESTION_WORDS = (
    "what if",
    "should we",
    "how to",
    "why does",
    "can we",
    "is there",
)

_HYPOTHESIS_PHRASES = (
    "need to",
    "should check",
    "might need",
    "could try",
)


def _check_blocker_line(lowered: str, blockers: list[str]) -> None:
    """Check a lowered line for blocker phrases and append if found."""
    if "blocked on " in lowered:
        blocker = lowered.split("blocked on", 1)[1].strip()
        if blocker and len(blocker) < 100:
            blockers.append(f"blocked_on:{blocker}")
    elif "blocked by " in lowered:
        blocker = lowered.split("blocked by", 1)[1].strip()
        if blocker and len(blocker) < 100:
            blockers.append(f"blocked_by:{blocker}")


def _check_hypothesis_line(stripped: str, lowered: str, hypotheses: list[str]) -> None:
    """Check a line for hypothesis/question phrases and append if found."""
    if stripped.endswith("?") and len(stripped) < 150:
        if any(word in lowered for word in _HYPOTHESIS_QUESTION_WORDS):
            hypotheses.append(f"question:{stripped[:100]}")
    elif any(phrase in lowered for phrase in _HYPOTHESIS_PHRASES):
        if len(stripped) < 150:
            hypotheses.append(f"hypothesis:{stripped[:100]}")


def extract_memory_items(
    messages: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Extract blocker and hypothesis strings for memory injection."""
    blockers: list[str] = []
    hypotheses: list[str] = []
    for msg in messages:
        msg_text = text_of(msg.get("content", ""))
        for line in msg_text.splitlines():
            stripped = line.strip()
            lowered = stripped.lower()
            _check_blocker_line(lowered, blockers)
            _check_hypothesis_line(stripped, lowered, hypotheses)
    return blockers, hypotheses


__all__ = [
    "_discover_project_markers",
    "calculate_reasoning_depth",
    "extract_goal_from_messages",
    "extract_memory_items",
    "get_adaptive_keep_turns",
    "session_write_memory",
    "update_session_family_mode",
]
