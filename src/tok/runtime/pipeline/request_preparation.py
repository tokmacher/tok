"""Helper functions for preparing runtime requests."""

from __future__ import annotations

import logging
import re
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

_logger = logging.getLogger("tok.runtime.pipeline.request_preparation")

# Matches grep output lines: path:lineno: content
_GREP_PATH_RE = re.compile(r"^([^:\s][^:]+\.[a-zA-Z]{1,8}):\d+:")

from pydantic import BaseModel, ConfigDict

from tok.compression import FILE_LIKE_TOOLS, inject_system_additions, text_of
from tok.runtime.config import (
    _TOOL_REQUIRED_PROMPT_PATTERNS,
    ANSWER_READY_REPAIR_HINT,
    ANSWER_READY_RUNTIME_HINT,
    LATE_ANSWER_ASSEMBLY_ANSWER_ONLY_REPAIR_HINT,
    LATE_ANSWER_ASSEMBLY_TOOL_ONLY_REPAIR_HINT,
    LATE_ANSWER_FOLLOWTHROUGH_HINT,
)

if TYPE_CHECKING:
    from tok.runtime.core import RuntimeSession

from tok.runtime.repeat_targets import (
    SEARCH_LIKE_TOOLS,
    EvidenceIntent,
    evidence_identity_key,
    extract_git_history_path,
    extract_metadata_probe,
    extract_shell_file_read_path,
    extract_shell_search_params,
    resolve_evidence_intent,
)

from .request_validation import validate_anthropic_request_body
from .tool_processing import (
    _HARD_BLOCKER_PHRASES_SET,
    _TRANSIENT_ERROR_PHRASES_SET,
    _iter_tool_results,
)


def _process_content_list(
    content: list[dict[str, Any]],
) -> str | list[dict[str, Any]]:
    """Process a list content block and return adapted content."""
    if not content:
        return " "
    if all(isinstance(b, dict) and b.get("type") == "text" for b in content):
        # Pure text array -> flatten to string for Bedrock/OpenAI compatibility
        return "\n".join(b.get("text", "") for b in content).strip() or " "
    # Ensure tool-only assistant messages have at least a placeholder space
    return content


def apply_schema_adaptations(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Ensure message list is compatible with strict providers (Bedrock, Gemini).

    1. Flattens single-item or text-only arrays into strings.
    2. Ensures no message has empty content (injects a placeholder).
    """
    if not isinstance(messages, list) or not messages:
        return messages

    adapted = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        new_msg = dict(msg)
        content = msg.get("content")

        # Guard against None or empty values
        if content is None:
            content = " "
        elif isinstance(content, list):
            content = _process_content_list(content)
        elif isinstance(content, str) and not content.strip():
            content = " "

        new_msg["content"] = content
        adapted.append(new_msg)

    return adapted


def mutation_signals(original_body: dict[str, Any], mutated_body: dict[str, Any]) -> dict[str, int]:
    """Identify structural mutations between original and prepared request bodies."""
    signals: dict[str, int] = {}
    original_failures = validate_anthropic_request_body(original_body)
    mutated_failures = validate_anthropic_request_body(mutated_body)
    if mutated_failures and not original_failures:
        signals["tok_preflight_rejected"] = 1
    original_messages = original_body.get("messages", [])
    mutated_messages = mutated_body.get("messages", [])
    if (
        isinstance(original_messages, list)
        and isinstance(mutated_messages, list)
        and len(original_messages) != len(mutated_messages)
    ):
        signals["tok_structural_mutation"] = 1
    if original_body.get("system") != mutated_body.get("system"):
        signals["tok_structural_mutation"] = signals.get("tok_structural_mutation", 0) + 1
    return signals


def collect_transient_error_snippets(
    messages: list[dict[str, Any]],
) -> list[str]:
    """Extract transient-error snippets from messages for injection into hot['errs']."""
    seen: set[str] = set()
    result: list[str] = []
    for msg in messages:
        msg_text = text_of(msg.get("content", ""))
        for line in msg_text.splitlines():
            stripped = line.strip()
            lowered = stripped.lower()
            if any(phrase in lowered for phrase in _HARD_BLOCKER_PHRASES_SET):
                continue
            if any(phrase in lowered for phrase in _TRANSIENT_ERROR_PHRASES_SET):
                snippet = stripped[:96]
                if snippet and snippet not in seen:
                    seen.add(snippet)
                    result.append(snippet)
    return result


def _latest_user_message(
    messages: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the last user message in the conversation."""
    for message in reversed(messages):
        if message.get("role") == "user":
            return message
    return None


def _message_has_tool_results(message: dict[str, Any] | None) -> bool:
    """Return True if the message contains tool results."""
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(block, dict) and block.get("type") == "tool_result" for block in content)


def _message_user_text(message: dict[str, Any] | None) -> str:
    """Return only user-authored text from a message, excluding tool results."""
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and str(block.get("text", "")).strip():
            parts.append(str(block.get("text", "")).strip())
    return "\n".join(parts)


_READ_ONLY_AUDIT_HINT_PATTERNS = (
    "read-only",
    "no edits",
    "no tests",
    "no installs",
    "no network",
    "tool budget",
    "use these anchors",
)
_READ_ONLY_AUDIT_SCOPE_PATTERNS = ("audit", "explor", "stress", "investigat")
_NO_ANSWER_PATTERNS = ("do not answer", "do not answer yet")
_ANSWER_READY_TEXT_PATTERNS = (
    "use the evidence",
    "use the evidence you just retrieved",
    "answer now",
    "reply now",
    "summarize your findings",
    "summarize the findings",
    "what did you find",
    "what's the verdict",
    "what is the verdict",
    "what's the answer",
    "what is the answer",
    "confirm",
)
_FINALIZATION_TEXT_PATTERNS = (
    "write the plan",
    "finish the plan",
    "finalize the plan",
    "finalise the plan",
    "proposed_plan",
    "produce the plan",
    "summarize your findings",
    "summarize the findings",
    "summarise your findings",
    "summarise the findings",
    "final answer",
    "answer now",
    "reply now",
    "what did you find",
    "what's the verdict",
    "what is the verdict",
    "what's the answer",
    "what is the answer",
)


class AuditTurnIntent(BaseModel):
    model_config = ConfigDict(frozen=True)

    read_only: bool = False
    no_answer: bool = False
    no_network: bool = False
    tool_budget: bool = False
    audit_scope: bool = False

    @property
    def is_read_only_audit(self) -> bool:
        return self.no_answer or (self.read_only and self.audit_scope and (self.no_network or self.tool_budget))


def _audit_turn_intent(messages: list[dict[str, Any]]) -> AuditTurnIntent:
    """Return a typed classification of the latest user turn's audit intent."""
    latest_user = _latest_user_message(messages)
    lowered = _message_user_text(latest_user).lower()
    if not lowered:
        return AuditTurnIntent()
    return AuditTurnIntent(
        read_only="read-only" in lowered,
        no_answer=any(pattern in lowered for pattern in _NO_ANSWER_PATTERNS),
        no_network="no network" in lowered,
        tool_budget="tool budget" in lowered,
        audit_scope=any(pattern in lowered for pattern in _READ_ONLY_AUDIT_SCOPE_PATTERNS),
    )


def _is_read_only_audit_turn(messages: list[dict[str, Any]]) -> bool:
    """Return True when the latest user turn explicitly requests read-only exploration."""
    latest_user = _latest_user_message(messages)
    lowered = _message_user_text(latest_user).lower()
    if not lowered:
        return False
    intent = _audit_turn_intent(messages)
    if intent.is_read_only_audit:
        return True
    has_read_only_contract = any(pattern in lowered for pattern in _READ_ONLY_AUDIT_HINT_PATTERNS)
    if has_read_only_contract and intent.audit_scope:
        return True
    # Exploration mode: audit/investigation that requires documenting the process itself
    if intent.audit_scope:
        # Meta-observation: prompt asks to document HOW tools are used, not just results
        # Detected by: process-verbs + tool-usage reference
        process_verbs = ("narrate", "document", "record", "note", "describe", "list")
        tool_refs = ("tool call", "each call", "every call", "as you", "for each")
        has_process_verb = any(v in lowered for v in process_verbs)
        has_tool_ref = any(r in lowered for r in tool_refs)
        if has_process_verb and has_tool_ref:
            return True
        # Self-assessment patterns: evaluating quality of evidence/tool-output
        self_assess = ("what i got back", "immediately useful", "re-query", "re-read", "workaround")
        if any(s in lowered for s in self_assess):
            return True
    return False


def _message_explicitly_requests_answer(
    message: dict[str, Any] | None,
) -> bool:
    """Return True when the latest user-authored text explicitly asks for an answer."""
    lowered = _message_user_text(message).lower()
    if not lowered:
        return False
    if any(pattern in lowered for pattern in _NO_ANSWER_PATTERNS):
        return False
    return any(pattern in lowered for pattern in _ANSWER_READY_TEXT_PATTERNS)


def is_plan_or_answer_finalization_turn(messages: list[dict[str, Any]]) -> bool:
    """Return True when the latest user turn asks for a final plan/answer, not new tool work."""
    latest_user = _latest_user_message(messages)
    lowered = _message_user_text(latest_user).lower()
    if not lowered:
        return False
    if _is_read_only_audit_turn(messages):
        return False
    if _has_unresolved_tool_required_conditions(messages):
        return False
    return any(pattern in lowered for pattern in _FINALIZATION_TEXT_PATTERNS)


def _has_unresolved_tool_required_conditions(
    messages: list[dict[str, Any]],
) -> bool:
    """Return True if the user prompt implies a tool is needed but not yet used."""
    latest_user = _latest_user_message(messages)
    if not latest_user or _message_has_tool_results(latest_user):
        return False
    lowered = text_of(latest_user.get("content", "")).lower()
    return any(pattern in lowered for pattern in _TOOL_REQUIRED_PROMPT_PATTERNS)


def _is_answer_ready_turn(
    messages: list[dict[str, Any]],
    *,
    tool_compatible: bool,
    has_answer_anchor: bool,
    baseline_only: bool,
) -> bool:
    """Return True if the current turn context suggests an answer is expected next."""
    if not tool_compatible or baseline_only:
        return False
    latest_user = _latest_user_message(messages)
    if not latest_user:
        return False
    if _is_read_only_audit_turn(messages):
        return False
    if _message_has_tool_results(latest_user):
        if _message_explicitly_requests_answer(latest_user):
            return True
        if not has_answer_anchor:
            return False
        return not _has_unresolved_tool_required_conditions(messages)
    if not has_answer_anchor:
        return False
    return not _has_unresolved_tool_required_conditions(messages)


def _answer_ready_runtime_hints(*, answer_ready: bool) -> list[str]:
    """Return hints for turns where an answer is expected."""
    if not answer_ready:
        return []
    return [ANSWER_READY_RUNTIME_HINT]


def _answer_ready_repair_hints(*, repair_active: bool) -> list[str]:
    """Return hints for turns where an answer repair is requested."""
    if not repair_active:
        return []
    return [ANSWER_READY_REPAIR_HINT]


def _late_answer_assembly_repair_hints(*, repair_mode: str) -> list[str]:
    """Return hints for late-session answer assembly repairs."""
    if repair_mode == "tool_only":
        return [LATE_ANSWER_ASSEMBLY_TOOL_ONLY_REPAIR_HINT]
    if repair_mode == "answer_only":
        return [LATE_ANSWER_ASSEMBLY_ANSWER_ONLY_REPAIR_HINT]
    return []


def _late_answer_followthrough_hints(*, active: bool) -> list[str]:
    """Return hints for late-session followthrough turns."""
    if not active:
        return []
    return [LATE_ANSWER_FOLLOWTHROUGH_HINT]


def _runtime_hints_for_turn(
    *,
    answer_ready: bool,
    answer_ready_repair_active: bool,
    late_answer_followthrough_active: bool,
    late_answer_assembly_repair_mode: str,
) -> list[str]:
    """Return a consolidated list of runtime strategy hints for the current turn.

    Steering hints (answer-ready pressure, latch escalation, repair urgency) are
    deliberately not injected here. Tok is an invisible bridge: the model decides
    when it is done and what tools to use. See docs/philosophy.md.
    """
    return []


def _annotate_reacquisition_diagnostics(
    behavior_signals: dict[str, int],
    *,
    answer_ready: bool,
    answer_ready_repair_active: bool,
    exploration_mode: bool = False,
) -> None:
    """Set diagnostic signals for anchor reacquisition attempts."""
    if not behavior_signals.get("answer_anchor_present", 0):
        return
    if not (
        behavior_signals.get("repeat_file_read", 0) > 0
        or behavior_signals.get("repeat_search", 0) > 0
        or behavior_signals.get("cached_file_read", 0) > 0
        or behavior_signals.get("cached_search", 0) > 0
    ):
        return
    # During exploration mode, reacquisition is expected - don't flag as problematic
    if exploration_mode:
        behavior_signals["exploration_reacquisition_expected"] = 1
        return
    behavior_signals["answer_anchor_reacquisition_attempt"] = 1
    if answer_ready:
        behavior_signals["answer_ready_reacquisition_attempt"] = 1
    elif answer_ready_repair_active:
        behavior_signals["repair_phase_reacquisition_attempt"] = 1
    else:
        behavior_signals["benign_reverification_attempt"] = 1


def _looks_like_shell_read_error(command: str, text: str) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return True
    try:
        command_name = Path(shlex.split(command)[0]).name.lower()
    except Exception:
        command_name = ""
    prefixes = (
        f"{command_name}:",
        "no such file",
        "cannot open",
        "permission denied",
        "not found",
        "is a directory",
        "usage:",
    )
    return any(lowered.startswith(prefix) or prefix in lowered for prefix in prefixes)


def _process_single_tool_snapshot(
    session: RuntimeSession,
    tool_name: str,
    path: str | None,
    query: str | None,
    command: str | None,
    tool_args: dict[str, Any] | None,
    snippet: str,
    blocker_rediscovery: bool,
    captured: list[int],
) -> dict[str, int]:
    """Helper to process a single tool result snapshot."""
    signals: dict[str, int] = {}
    shell_file_path = extract_shell_file_read_path(command or "") if command else None

    evidence_intent = resolve_evidence_intent(tool_name, path=path, query=query, command=command)

    if evidence_intent:
        _apply_evidence_intent_snapshot(
            session,
            evidence_intent,
            path,
            query,
            command,
            snippet,
            shell_file_path,
            captured,
            signals,
        )
    else:
        _apply_default_snapshots(
            session,
            tool_name,
            path,
            query,
            command,
            snippet,
            shell_file_path,
            captured,
        )

    exact_key = evidence_identity_key(
        tool_name,
        path=path,
        query=query,
        command=command,
        args=tool_args,
    )

    observed = session.observe_repeat_target_result(
        tool_id="",
        tool_name=tool_name,
        path=path,
        query=query,
        command=command,
        raw_content=snippet,
        tool_args=tool_args,
        exact_evidence_key=exact_key,
        blocker_rediscovery=blocker_rediscovery,
    )
    for key, value in observed.items():
        signals[key] = signals.get(key, 0) + value

    return signals


def _apply_evidence_intent_snapshot(
    session: RuntimeSession,
    evidence_intent: EvidenceIntent,
    path: str | None,
    query: str | None,
    command: str | None,
    snippet: str,
    shell_file_path: str | None,
    captured: list[int],
    signals: dict[str, int],
) -> None:
    if evidence_intent.domain == "file_history" and command:
        _record_file_history_snapshot(session, command, snippet, captured)
        return
    if evidence_intent.domain in {"file_metadata", "listing"} and command:
        _record_file_metadata_snapshot(session, path, command, snippet)
        return
    if evidence_intent.domain == "listing" and path and not command:
        session.record_metadata_snapshot(path, "listing", snippet)
        return
    if evidence_intent.domain == "file_current":
        _record_file_current_snapshot(
            session,
            evidence_intent,
            path,
            command,
            snippet,
            shell_file_path,
            captured,
            signals,
        )
        return
    if evidence_intent.domain == "search":
        _record_search_snapshot(session, query, command, snippet, evidence_intent, captured)


def _apply_default_snapshots(
    session: RuntimeSession,
    tool_name: str,
    path: str | None,
    query: str | None,
    command: str | None,
    snippet: str,
    shell_file_path: str | None,
    captured: list[int],
) -> None:
    if tool_name in FILE_LIKE_TOOLS and path:
        if session.record_file_snapshot(path, snippet):
            captured[0] += 1
    elif (
        shell_file_path
        and tool_name in {"bash", "sh", "run_terminal", "computer"}
        and not _looks_like_shell_read_error(command or "", snippet)
    ) and session.record_file_snapshot(shell_file_path, snippet):
        captured[0] += 1
        captured[2] += 1
    if tool_name in SEARCH_LIKE_TOOLS and query:
        if session.record_search_snapshot(query, snippet):
            captured[1] += 1
        session.record_symbol_locations(snippet)
        # Bump heat for every file referenced in the grep output
        for line in snippet.splitlines():
            m = _GREP_PATH_RE.match(line)
            if m:
                session.bridge_memory.bump_file_heat(m.group(1), weight=0.3)


def _record_file_history_snapshot(session: RuntimeSession, command: str, snippet: str, captured: list[int]) -> None:
    git_path, git_rev = extract_git_history_path(command)
    if git_path and session.record_history_snapshot(git_path, git_rev, snippet):
        captured[0] += 1


def _record_file_metadata_snapshot(session: RuntimeSession, path: str | None, command: str, snippet: str) -> None:
    meta_subtype = extract_metadata_probe(command)
    if meta_subtype:
        meta_path = path or ""
        session.record_metadata_snapshot(meta_path, meta_subtype, snippet)


def _record_file_current_snapshot(
    session: RuntimeSession,
    evidence_intent: EvidenceIntent,
    path: str | None,
    command: str | None,
    snippet: str,
    shell_file_path: str | None,
    captured: list[int],
    signals: dict[str, int],
) -> None:
    if evidence_intent.source_kind == "native_tool" and path:
        if session.record_file_snapshot(path, snippet):
            captured[0] += 1
            if evidence_intent.variant == "copy":
                _apply_temp_copy_alias(session, path, snippet, signals)
    elif (
        shell_file_path and command and not _looks_like_shell_read_error(command, snippet)
    ) and session.record_file_snapshot(shell_file_path, snippet):
        captured[0] += 1
        captured[2] += 1
        if evidence_intent.source_kind == "temp_copy":
            _apply_temp_copy_alias(session, shell_file_path, snippet, signals)


def _apply_temp_copy_alias(
    session: RuntimeSession,
    path: str,
    snippet: str,
    signals: dict[str, int],
) -> None:
    alias = session.check_temp_copy_alias(path, snippet)
    if alias:
        signals["derived_copy_alias_applied"] = 1
        signals["evidence_alias_resolved"] = 1


def _record_search_snapshot(
    session: RuntimeSession,
    query: str | None,
    command: str | None,
    snippet: str,
    evidence_intent: EvidenceIntent,
    captured: list[int],
) -> None:
    search_query = query
    if evidence_intent.source_kind == "shell_search" and command:
        sq, _ = extract_shell_search_params(command)
        if sq:
            search_query = sq
    if search_query and session.record_search_snapshot(search_query, snippet):
        captured[1] += 1
    session.record_symbol_locations(snippet)
    for line in snippet.splitlines():
        m = _GREP_PATH_RE.match(line)
        if m:
            session.bridge_memory.bump_file_heat(m.group(1), weight=0.3)


def _capture_repeat_target_snapshots(
    messages: list[dict[str, Any]],
    id_to_context: dict[str, dict[str, Any]],
    session: RuntimeSession,
) -> dict[str, int]:
    """Scan tool_results and record bounded snapshots for repeat-target control."""
    captured = [0, 0, 0]  # file, search, shell_file
    signals: dict[str, int] = {}
    blocker_rediscovery = bool(session.pending_behavior_signals.get("blocker_rediscovery", 0))

    for tool_id, snippet in _iter_tool_results(messages):
        context = id_to_context.get(tool_id)
        if not context:
            continue
        tool_name = str(context.get("name", "")).lower()
        path = str(context.get("path") or "").strip() or None
        query = str(context.get("query") or "").strip() or None
        args = context.get("args") or {}
        command = str(args.get("command") or args.get("cmd") or "").strip() or None

        step_signals = _process_single_tool_snapshot(
            session,
            tool_name,
            path,
            query,
            command,
            args if isinstance(args, dict) else None,
            snippet,
            blocker_rediscovery,
            captured,
        )
        for k, v in step_signals.items():
            signals[k] = signals.get(k, 0) + v

        # Feature: traceback → errs facts
        if "Traceback (most recent call last):" in snippet or 'File "' in snippet:
            n = session.record_traceback_errors(snippet)
            if n:
                signals["traceback_errs_recorded"] = signals.get("traceback_errs_recorded", 0) + n

        # Feature: Edit observation — treat edit result as updated file snapshot
        from tok.compression import EDIT_LIKE_TOOLS

        if tool_name in EDIT_LIKE_TOOLS and path and snippet:
            session.bridge_memory.bump_file_heat(path, weight=1.0)
            session.record_file_snapshot(path, snippet)

    if captured[0]:
        signals["file_snapshot_captured"] = captured[0]
    if captured[2]:
        signals["shell_file_snapshot_captured"] = captured[2]
    if captured[1]:
        signals["search_snapshot_captured"] = captured[1]
    return signals


def _annotate_full_turn_resend(
    behavior_signals: dict[str, int],
    resend_signals: dict[str, int],
    *,
    resend_reason: str | None,
    skip_reason_hint: str | None,
    tok_history_compression_skipped: bool,
    tok_history_cut_point_missing: bool,
    tool_compatible_compression: bool,
) -> None:
    """Helper for applying full turn resend diagnostics."""
    if resend_signals.get("state_resend_reason_answer_ready_forced_full"):
        behavior_signals["state_resend_reason_answer_ready_forced_full"] = 1
        return
    if resend_reason == "new_answer_anchor":
        behavior_signals["state_resend_reason_answer_anchor_present_kept_full"] = 1
        behavior_signals["answer_anchor_forced_full_resend"] = 1
        return
    if resend_signals.get("state_resend_reason_delta_not_smaller"):
        behavior_signals["state_resend_reason_delta_not_smaller"] = 1
        return
    if skip_reason_hint or tok_history_compression_skipped:
        behavior_signals["state_resend_reason_history_compression_skipped"] = 1
        return
    if tok_history_cut_point_missing:
        behavior_signals["tok_history_cut_point_missing"] = 1
    if tool_compatible_compression:
        behavior_signals["state_resend_reason_tool_compatible_compression_without_resend_change"] = 1
        return
    behavior_signals["state_resend_reason_full_default"] = 1


def _apply_tool_compatible_resend_diagnostics(
    behavior_signals: dict[str, int],
    loaded_memory: str,
    resend_signals: dict[str, int],
    *,
    has_answer_anchor: bool = False,
    resend_reason: str | None = None,
    skip_reason_hint: str | None = None,
    tok_history_compression_skipped: bool = False,
    tok_history_cut_point_missing: bool = False,
    tool_compatible_compression: bool = False,
) -> None:
    """Set diagnostic signals for tool-compatible state resends."""
    if has_answer_anchor:
        behavior_signals["answer_anchor_present"] = 1
    behavior_signals["state_payload_chars"] = len(loaded_memory)

    if resend_signals.get("state_resend_delta_turn"):
        behavior_signals["state_resend_reason_delta_selected"] = 1
        if has_answer_anchor:
            behavior_signals["answer_anchor_delta_allowed"] = 1
        return

    if resend_signals.get("state_resend_suppressed_turn"):
        behavior_signals["state_resend_reason_state_verified_current"] = (
            1  # Changed: state is verified current, not "suppressed"
        )
        if has_answer_anchor:
            behavior_signals["answer_anchor_verified_current"] = 1  # Changed: verified current, not "suppressed"
        return

    if resend_signals.get("state_resend_full_turn"):
        _annotate_full_turn_resend(
            behavior_signals,
            resend_signals,
            resend_reason=resend_reason,
            skip_reason_hint=skip_reason_hint,
            tok_history_compression_skipped=tok_history_compression_skipped,
            tok_history_cut_point_missing=tok_history_cut_point_missing,
            tool_compatible_compression=tool_compatible_compression,
        )


def _inject_system(
    current_body: dict[str, Any],
    current_memory: str,
    current_runtime_hints: list[str],
    *,
    tool_compatible: bool,
    grammar: bool,
    todo: str,
    deltas: bool,
    pressure: int,
    behavior_signals: dict[str, int],
    current_turn: int | None = None,
    session: Any | None = None,
) -> dict[str, Any]:
    """Inject Tok state and hints into the system prompt."""
    from tok.runtime.config import (
        _SHORT_MEMORY_TURN_CEILING,
        _SHORT_SESSION_THRESHOLD,
        TOOL_COMPAT_MEMORY_PROFILE,
        TOOL_COMPAT_MEMORY_PROFILE_SHORT,
    )

    tok_state: str | Any = current_memory
    if hasattr(current_memory, "wire_state"):
        from typing import cast

        if tool_compatible:
            effective_turn = current_turn if current_turn is not None else 999
            if _SHORT_SESSION_THRESHOLD <= effective_turn <= _SHORT_MEMORY_TURN_CEILING:
                profile = TOOL_COMPAT_MEMORY_PROFILE_SHORT
            else:
                profile = TOOL_COMPAT_MEMORY_PROFILE
        else:
            profile = None

        tok_state = cast("Any", current_memory).wire_state(
            profile=profile,
            markers=behavior_signals.get("_project_markers_proxy"),
        )

    # Append verbosity signal to the >>> state line when the rolling average of
    # visible response word counts is elevated (≥120 words over last 5 turns).
    if isinstance(tok_state, str) and tok_state.startswith(">>>") and session is not None:
        samples = getattr(session, "_response_word_samples", [])
        if len(samples) >= 3 and sum(samples) / len(samples) >= 120:
            tok_state = tok_state + "|verbose:high"

    kwargs: dict[str, Any] = {
        "body": current_body,
        "tok_state": tok_state,
        "tool_compatible": tool_compatible,
        "grammar": grammar,
        "todo": todo,
        "deltas": deltas,
        "pressure": pressure,
        "behavior_signals": behavior_signals,
    }
    if current_runtime_hints:
        kwargs["runtime_hints"] = current_runtime_hints
    try:
        return inject_system_additions(**kwargs)
    except TypeError as exc:
        if "runtime_hints" not in str(exc):
            raise
        _logger.warning("runtime_hints rejected by inject_system_additions; retrying without")
        kwargs.pop("runtime_hints", None)
        return inject_system_additions(**kwargs)
