from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from tok.compression import text_of


@dataclass(frozen=True)
class ContextDependencyDecision:
    depends_on_context: bool = False
    kind: str = ""
    protected_suffix_start: int | None = None
    reason: str = ""


_SHORT_HANDOFF_MAX_WORDS = 14
_RECENT_CONTEXT_WINDOW = 12
_SELF_CONTAINED_MARKERS = (":", "\n", "```", "<proposed_plan>")
_HANDOFF_VERBS = frozenset(
    {
        "accept",
        "accepted",
        "approved",
        "continue",
        "do",
        "execute",
        "go",
        "implement",
        "ok",
        "okay",
        "proceed",
        "ship",
        "start",
        "yes",
    }
)
_HANDOFF_REFERENTS = frozenset({"above", "it", "plan", "that", "this", "your"})
_PLAN_PATH_RE = re.compile(r"(?i)(?:^|[/\\])(?:\.claude[/\\])?plans[/\\].+\.md\b")


def classify_context_dependency(messages: list[dict[str, Any]]) -> ContextDependencyDecision:
    latest_index, latest_user = _latest_user_message(messages)
    if latest_index < 0 or latest_user is None:
        return ContextDependencyDecision()
    if not _is_short_referential_handoff(latest_user):
        return ContextDependencyDecision()

    evidence_index = _find_recent_plan_evidence(messages, latest_index)
    if evidence_index is None:
        return ContextDependencyDecision()

    protected_start = protected_suffix_start_for_tool_pairs(messages, evidence_index)
    if protected_start is None:
        return ContextDependencyDecision(
            depends_on_context=True,
            kind="plan_handoff",
            protected_suffix_start=None,
            reason="plan_evidence_without_safe_tool_pair_boundary",
        )
    return ContextDependencyDecision(
        depends_on_context=True,
        kind="plan_handoff",
        protected_suffix_start=protected_start,
        reason="short_referential_turn_after_plan_evidence",
    )


def protected_suffix_start_for_tool_pairs(messages: list[dict[str, Any]], start: int) -> int | None:
    if start < 0:
        return None
    protected_start = min(start, len(messages))
    while True:
        tool_use_ids = _tool_use_ids_in_slice(messages[protected_start:])
        earliest_missing_tool_use_index: int | None = None
        for tool_result_id in _tool_result_ids_in_slice(messages[protected_start:]):
            if not tool_result_id or tool_result_id in tool_use_ids:
                continue
            tool_use_index = _find_tool_use_message_index(messages, tool_result_id)
            if tool_use_index is None:
                return None
            if earliest_missing_tool_use_index is None or tool_use_index < earliest_missing_tool_use_index:
                earliest_missing_tool_use_index = tool_use_index
        if earliest_missing_tool_use_index is None or earliest_missing_tool_use_index >= protected_start:
            return protected_start
        protected_start = earliest_missing_tool_use_index


def suffix_preserves_tool_pairs(messages: list[dict[str, Any]], start: int | None) -> bool:
    if start is None or start < 0 or start > len(messages):
        return False
    tool_use_ids = _tool_use_ids_in_slice(messages[start:])
    for tool_result_id in _tool_result_ids_in_slice(messages[start:]):
        if tool_result_id and tool_result_id not in tool_use_ids:
            return False
    return True


def context_dependency_signals(decision: ContextDependencyDecision) -> dict[str, int]:
    if not decision.depends_on_context:
        return {}
    signals = {
        "context_dependency_turn": 1,
        f"context_dependency_kind_{decision.kind}": 1,
        "plan_finalization_turn": 1,
    }
    if decision.protected_suffix_start is not None:
        signals["context_dependency_protected_suffix_start"] = int(decision.protected_suffix_start)
    return signals


def _latest_user_message(messages: list[dict[str, Any]]) -> tuple[int, dict[str, Any] | None]:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, dict) and message.get("role") == "user":
            return index, message
    return -1, None


def _is_short_referential_handoff(message: dict[str, Any]) -> bool:
    content = message.get("content", "")
    if _message_has_tool_result(content):
        return False
    text = text_of(content).strip()
    if not text:
        return False
    lowered = re.sub(r"\s+", " ", text.lower()).strip()
    if any(marker in text for marker in _SELF_CONTAINED_MARKERS):
        return False
    words = re.findall(r"[a-z0-9']+", lowered)
    if len(words) > _SHORT_HANDOFF_MAX_WORDS:
        return False
    word_set = set(words)
    if word_set & _HANDOFF_VERBS and word_set & _HANDOFF_REFERENTS:
        return True
    return lowered in {
        "accept",
        "accepted",
        "approved",
        "continue",
        "do it",
        "go ahead",
        "ok",
        "okay",
        "proceed",
        "ship it",
        "start",
        "yes",
    }


def _find_recent_plan_evidence(messages: list[dict[str, Any]], before_index: int) -> int | None:
    lower_bound = max(0, before_index - _RECENT_CONTEXT_WINDOW)
    for index in range(before_index - 1, lower_bound - 1, -1):
        message = messages[index]
        if not isinstance(message, dict):
            continue
        if _message_contains_plan_evidence(message):
            return index
    return None


def _message_contains_plan_evidence(message: dict[str, Any]) -> bool:
    content = message.get("content", "")
    if _is_plan_like_text(text_of(content)):
        return True
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result" and _is_plan_like_text(text_of(block.get("content", ""))):
                return True
            if block.get("type") == "tool_use" and _tool_use_mentions_plan_artifact(block):
                return True
    return False


def _is_plan_like_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if _PLAN_PATH_RE.search(stripped):
        return True
    if "proposed_plan" in lowered or "implementation plan" in lowered:
        return True
    if lowered.startswith(("plan:", "# plan", "## plan")) and len(stripped) >= 40:
        return True
    if "plan" not in lowered:
        return False
    structural_markers = sum(
        marker in lowered
        for marker in (
            "\n- ",
            "\n1.",
            "\n2.",
            "phase ",
            "step ",
            "test plan",
            "key changes",
            "implementation",
        )
    )
    return len(stripped) >= 120 and structural_markers >= 2


def _tool_use_mentions_plan_artifact(block: dict[str, Any]) -> bool:
    raw_input = block.get("input")
    if not isinstance(raw_input, dict):
        return False
    for key in ("file_path", "path", "filename"):
        raw_path = raw_input.get(key)
        if raw_path and _PLAN_PATH_RE.search(str(raw_path)):
            return True
    return False


def _message_has_tool_result(content: Any) -> bool:
    if not isinstance(content, list):
        return False
    return any(isinstance(block, dict) and block.get("type") == "tool_result" for block in content)


def _tool_use_ids_in_slice(messages: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tool_id = str(block.get("id") or "").strip()
                if tool_id:
                    ids.add(tool_id)
    return ids


def _tool_result_ids_in_slice(messages: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for message in messages:
        if message.get("role") == "tool_result":
            tool_id = str(message.get("tool_use_id") or "").strip()
            if tool_id:
                ids.add(tool_id)
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                tool_id = str(block.get("tool_use_id") or "").strip()
                if tool_id:
                    ids.add(tool_id)
    return ids


def _find_tool_use_message_index(messages: list[dict[str, Any]], tool_use_id: str) -> int | None:
    for index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and str(block.get("id") or "").strip() == tool_use_id
            ):
                return index
    return None
