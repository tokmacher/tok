"""Tok response parsing, translation, and contract validation logic."""

from __future__ import annotations

import copy
import logging
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tok.protocol.models import TokNode
from tok.protocol.parser import TokParser, serialize
from tok.runtime.memory.answer_memory import extract_structured_answer_memory
from tok.runtime.policy.translator import IS_TOK, postprocess_response
from tok.runtime.types import ProcessedRuntimeResponse

from ._bridge_wire_models import normalize_tool_use_blocks
from ._response_processing_patterns import (
    IDENTIFIER_RE as _IDENTIFIER_RE,
)
from ._response_processing_patterns import (
    PATH_RE as _PATH_RE,
)
from ._response_processing_patterns import (
    STRUCTURED_FIELD_NAMES as _STRUCTURED_FIELD_NAMES,
)
from ._response_processing_patterns import (
    STRUCTURED_LABEL_RE as _STRUCTURED_LABEL_RE,
)
from ._response_processing_patterns import (
    TOOL_INTENT_TEXT_RE as _TOOL_INTENT_TEXT_RE,
)
from ._response_processing_patterns import (
    VERIFICATION_STOPWORDS as _VERIFICATION_STOPWORDS,
)

if TYPE_CHECKING:
    from tok.runtime.core import RuntimeSession

logger = logging.getLogger("tok.runtime")


def heal_drift(
    text: str,
    behavior_signals: dict[str, int],
    *,
    tool_compatible: bool = False,
) -> str:
    """Wrap raw prose drift in a Tok envelope to ensure protocol adherence."""
    if tool_compatible:
        return text
    if behavior_signals.get("semantic_drift_detected") or behavior_signals.get("non_tok_response"):
        if ">>>" not in text:
            header = ">>> t:1|s:drift_healed"
            lines = text.strip().splitlines()
            if len(lines) > 1:
                body = "@msg role:assistant\n" + "\n".join(f"  |> {ln}" for ln in lines)
            else:
                body = f"@msg role:assistant\n  |> {text.strip()}"
            return f"{header}\n{body}"
    return text


def response_behavior_signals(
    text: str,
    *,
    tool_compatible: bool = False,
    session: RuntimeSession | None = None,
) -> dict[str, int]:
    """Detect response-side protocol drift."""
    expected_labels = _expected_structured_labels(session)
    if expected_labels and _is_strict_structured_answer_response(
        text,
        expected_labels=expected_labels,
    ):
        return {}
    if not tool_compatible and text.strip() and not IS_TOK.search(text):
        return {"non_tok_response": 1}
    return {}


def _visible_text_from_content_blocks(
    content_blocks: list[dict[str, Any]],
) -> str:
    return "\n".join(
        str(block.get("text", "")).strip()
        for block in content_blocks
        if block.get("type") == "text" and str(block.get("text", "")).strip()
    ).strip()


def _is_answer_like_visible_text(text: str) -> bool:
    if not text.strip():
        return False
    lowered = text.lower()
    if "file=" in lowered or "verification=" in lowered:
        return True
    fields = extract_structured_answer_memory(text)
    return bool(fields.get("files")) or any(
        fact.startswith(("answer_file:", "answer_verification:")) for fact in fields.get("facts", [])
    )


def _explicit_answer_phase_context(session: RuntimeSession | None) -> bool:
    if session is None:
        return False
    return bool(
        getattr(session, "_answer_ready_repair_active", False)
        or getattr(session, "_late_answer_followthrough_active", False)
        or getattr(session, "_late_answer_assembly_repair_active", False)
    )


def _answer_phase_context_active(session: RuntimeSession | None) -> bool:
    if session is None:
        return False
    return bool(getattr(session, "_answer_phase_expected_this_turn", False) or _explicit_answer_phase_context(session))


def _answer_phase_fallback_allowed(session: RuntimeSession | None) -> bool:
    if session is None or not _answer_phase_context_active(session):
        return False
    if _explicit_answer_phase_context(session):
        return True
    return not bool(getattr(session, "_request_has_tools", False))


def _looks_like_tool_intent_text(text: str) -> bool:
    return bool(text.strip() and _TOOL_INTENT_TEXT_RE.search(text))


def is_safe_visible_contract_output(
    visible_text: str,
    *,
    content_blocks: list[dict[str, Any]],
    expected_labels: tuple[str, ...],
    session: RuntimeSession | None,
) -> bool:
    if not visible_text.strip():
        return False
    if any(block.get("type") == "tool_use" for block in content_blocks):
        return False
    if _looks_like_tool_intent_text(visible_text):
        return False
    if expected_labels:
        return _is_strict_structured_answer_response(visible_text, expected_labels=expected_labels)
    if _answer_phase_context_active(session):
        return _is_answer_like_visible_text(visible_text)
    return True


def _is_tool_intent_without_answer(
    content_blocks: list[dict[str, Any]],
    visible_text: str,
    *,
    raw_text: str,
) -> bool:
    if _is_answer_like_visible_text(visible_text):
        return False
    has_tool_blocks = any(block.get("type") == "tool_use" for block in content_blocks)
    if has_tool_blocks:
        return True
    return _looks_like_tool_intent_text(visible_text) or _looks_like_tool_intent_text(raw_text)


def _expected_structured_labels(session: RuntimeSession | None) -> tuple[str, ...]:
    if session is None:
        return ()
    labels = tuple(getattr(session, "_last_user_prompt_labels", ()) or ())
    if labels:
        return labels
    prompt_text = str(getattr(session, "_last_user_prompt_text", "") or "")
    if not prompt_text.strip():
        return ()
    ordered: list[str] = []
    seen: set[str] = set()
    for match in _STRUCTURED_LABEL_RE.finditer(prompt_text):
        label = match.group(1).lower()
        if label in seen:
            continue
        seen.add(label)
        ordered.append(label)
    return tuple(ordered)


def _extract_labeled_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in _STRUCTURED_LABEL_RE.finditer(text):
        key = match.group(1).lower()
        value = match.group(2).strip().rstrip(".,;")
        if value:
            fields[key] = value
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("|>"):
            line = line[2:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lower()
        if key not in _STRUCTURED_FIELD_NAMES:
            continue
        cleaned = value.strip().rstrip(".,;")
        if cleaned and key not in fields:
            fields[key] = cleaned
    return fields


def _canonicalize_repo_path(candidate: str, known_files: list[str]) -> str:
    cleaned = candidate.strip().strip("'\"")
    if not cleaned:
        return ""
    cleaned = re.sub(r"[:#]L?\d+$", "", cleaned).removeprefix("./")
    candidate_path = Path.cwd() / cleaned
    if candidate_path.is_file():
        return cleaned
    if cleaned.startswith("src/"):
        alt_path = Path.cwd() / cleaned[4:]
        if alt_path.is_file():
            return cleaned
    package_forms: list[str] = []
    if cleaned.endswith(".py"):
        module_base = cleaned[:-3]
        package_forms = [f"{module_base}/__init__.py"]
        if module_base.startswith("src/"):
            package_forms.append(f"{module_base[4:]}/__init__.py")
        else:
            package_forms.append(f"src/{module_base}/__init__.py")
        for form in package_forms:
            if (Path.cwd() / form).is_file():
                return f"{cleaned} ({form})"
    if cleaned in known_files:
        return cleaned
    if cleaned.startswith("src/") and cleaned[4:] in known_files:
        return cleaned[4:]
    for known in known_files:
        if known.endswith(cleaned) or cleaned.endswith(known):
            return known
    base = cleaned.rsplit("/", 1)[-1]
    candidates = [known for known in known_files if known.rsplit("/", 1)[-1] == base]
    if len(candidates) == 1:
        return candidates[0]
    if package_forms:
        for form in package_forms:
            if (Path.cwd() / form).is_file():
                return form
        for form in package_forms:
            if form in known_files:
                return form
        for known in known_files:
            if any(known.endswith(form) for form in package_forms):
                return known
    return ""


def _extract_session_answer_anchors(
    session: RuntimeSession,
) -> tuple[list[str], list[str]]:
    file_entries: list[tuple[int, int, str]] = []
    verification_entries: list[tuple[int, int, str]] = []
    for bucket in (session.bridge_memory.hot, session.bridge_memory.durable):
        for entry in bucket.get("facts", []):
            value = str(getattr(entry, "value", "")).strip()
            score = int(getattr(entry, "score", 0))
            last_seen = int(getattr(entry, "last_seen_turn", 0))
            if value.startswith("answer_file:"):
                file_value = value.split(":", 1)[1].strip()
                if file_value:
                    file_entries.append((score, last_seen, file_value))
            elif value.startswith("answer_verification:"):
                verification_value = value.split(":", 1)[1].strip()
                if verification_value:
                    verification_entries.append((score, last_seen, verification_value))
        for entry in bucket.get("files", []):
            value = str(getattr(entry, "value", "")).strip()
            if not value:
                continue
            score = int(getattr(entry, "score", 0))
            last_seen = int(getattr(entry, "last_seen_turn", 0))
            file_entries.append((score, last_seen, value))
    file_entries.sort(key=lambda item: (-item[0], -item[1], item[2]))
    verification_entries.sort(key=lambda item: (-item[0], -item[1], item[2]))
    dedup_files: list[str] = []
    seen_files: set[str] = set()
    for _score, _last_seen, value in file_entries:
        if value in seen_files:
            continue
        seen_files.add(value)
        dedup_files.append(value)
    dedup_verifications: list[str] = []
    seen_verifications: set[str] = set()
    for _score, _last_seen, value in verification_entries:
        if value in seen_verifications:
            continue
        seen_verifications.add(value)
        dedup_verifications.append(value)
    return dedup_files, dedup_verifications


def _normalize_verification_value(value: str, known_verifications: list[str]) -> str:
    cleaned = value.strip()
    if not cleaned:
        return ""

    def _tokenize_phrase(text: str) -> list[str]:
        tokens: list[str] = []
        for token in _IDENTIFIER_RE.findall(text):
            lowered_token = token.lower()
            if lowered_token in _VERIFICATION_STOPWORDS:
                continue
            tokens.append(lowered_token)
        return tokens

    def _split_identifier_parts(token: str) -> list[str]:
        chunks = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", token).replace("_", " ").split()
        return [chunk.lower() for chunk in chunks if chunk]

    def _is_meaningful_anchor(anchor: str) -> bool:
        normalized = anchor.strip().strip("'\"`")
        if not normalized:
            return False
        parts = _tokenize_phrase(normalized)
        if not parts:
            return False
        if len(parts) == 1 and (parts[0] in _VERIFICATION_STOPWORDS or len(parts[0]) <= 2):
            return False
        return True

    def _fuzzy_token_match(left: str, right: str) -> bool:
        if left == right:
            return True
        if len(left) >= 4 and right.startswith(left):
            return True
        if len(right) >= 4 and left.startswith(right):
            return True
        return False

    def _anchor_supported_by_phrase(anchor: str, phrase_tokens: list[str]) -> bool:
        anchor_tokens = _IDENTIFIER_RE.findall(anchor)
        if not anchor_tokens:
            return False
        anchor_parts: list[str] = []
        for token in anchor_tokens:
            anchor_parts.extend(_split_identifier_parts(token))
        anchor_parts = [part for part in anchor_parts if part and part not in _VERIFICATION_STOPWORDS]
        if not anchor_parts:
            return False
        return all(any(_fuzzy_token_match(part, token) for token in phrase_tokens) for part in anchor_parts)

    def _single_identifier_reference(text: str, anchor: str) -> bool:
        phrase_tokens = _tokenize_phrase(text)
        if len(phrase_tokens) != 1:
            return False
        anchor_tokens = _tokenize_phrase(anchor)
        return len(anchor_tokens) == 1 and phrase_tokens[0] == anchor_tokens[0]

    lowered = cleaned.lower()
    phrase_tokens = _tokenize_phrase(cleaned)
    usable_anchors = [anchor for anchor in known_verifications if _is_meaningful_anchor(anchor)]

    exact_matches: list[str] = []
    for anchor in usable_anchors:
        if anchor.lower() in lowered:
            exact_matches.append(anchor)

    inferred_matches: list[str] = []
    for anchor in usable_anchors:
        if anchor in exact_matches:
            continue
        if _anchor_supported_by_phrase(anchor, phrase_tokens):
            inferred_matches.append(anchor)

    if len(exact_matches) >= 2:
        return cleaned
    if len(exact_matches) == 1 and inferred_matches:
        return f"{exact_matches[0]}, {inferred_matches[0]}"
    if len(exact_matches) == 1:
        if _single_identifier_reference(cleaned, exact_matches[0]):
            return exact_matches[0]
        return cleaned
    if len(inferred_matches) >= 2:
        return f"{inferred_matches[0]}, {inferred_matches[1]}"
    if len(inferred_matches) == 1:
        return inferred_matches[0]

    if phrase_tokens:
        best_anchor = ""
        best_score = 0.0
        for anchor in usable_anchors:
            anchor_tokens = _IDENTIFIER_RE.findall(anchor)
            if not anchor_tokens:
                continue
            anchor_parts: list[str] = []
            for token in anchor_tokens:
                anchor_parts.extend(_split_identifier_parts(token))
            anchor_parts = [part for part in anchor_parts if part and part not in _VERIFICATION_STOPWORDS]
            if not anchor_parts:
                continue
            overlap = sum(
                1 for part in anchor_parts if any(_fuzzy_token_match(part, phrase) for phrase in phrase_tokens)
            )
            score = overlap / max(1, len(anchor_parts))
            if score > best_score:
                best_score = score
                best_anchor = anchor
        if best_anchor and best_score >= 0.5:
            return best_anchor

    if len(phrase_tokens) > 2:
        return cleaned

    backtick_match = re.search(r"`([A-Za-z_][A-Za-z0-9_]*)`", cleaned)
    if backtick_match:
        return backtick_match.group(1)
    definition_match = re.search(r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\b", cleaned, re.IGNORECASE)
    if definition_match:
        return definition_match.group(1)
    call_match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", cleaned)
    if call_match:
        return call_match.group(1)
    filtered_identifiers = [
        candidate
        for candidate in _IDENTIFIER_RE.findall(cleaned)
        if candidate.lower() not in _VERIFICATION_STOPWORDS and candidate.lower() != "line"
    ]
    return filtered_identifiers[0] if filtered_identifiers else cleaned


def _looks_like_placeholder_value(value: str) -> bool:
    lowered = value.strip().lower()
    if not lowered:
        return True
    return any(
        marker in lowered
        for marker in (
            "<the",
            "<file",
            "<verification",
            "specific filename from the code change",
            "command run or test outcome",
        )
    )


def _file_from_verification_value(verification: str, known_files: list[str]) -> str:
    for match in _PATH_RE.finditer(verification):
        candidate = match.group(1).strip()
        if known_files:
            resolved = _canonicalize_repo_path(candidate, known_files)
            if resolved:
                return resolved
        elif candidate:
            return re.sub(r"[:#]L?\d+$", "", candidate).removeprefix("./")
    lowered = verification.lower()
    for known in known_files:
        if known.lower() in lowered or known.rsplit("/", 1)[-1].lower() in lowered:
            return known
    return ""


def _is_strict_structured_answer_response(
    text: str,
    *,
    expected_labels: tuple[str, ...],
) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if "@tool" in stripped.lower() or "tool use" in stripped.lower():
        return False
    fields = _extract_labeled_fields(stripped)
    if not fields:
        return False
    if expected_labels and any(label not in fields for label in expected_labels):
        return False
    allowed_labels = set(expected_labels) if expected_labels else set(_STRUCTURED_FIELD_NAMES)
    for line in stripped.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        if cleaned.startswith("|>"):
            cleaned = cleaned[2:].strip()
        key_match = re.match(r"^([A-Za-z_][A-Za-z0-9_-]*)\s*[:=]", cleaned)
        if key_match and key_match.group(1).lower() not in allowed_labels:
            return False
    return True


def _repair_structured_answer_text(
    visible_text: str,
    *,
    expected_labels: tuple[str, ...],
    session: RuntimeSession | None,
) -> tuple[str, dict[str, int]]:
    if not expected_labels:
        return visible_text, {}
    fields = _extract_labeled_fields(visible_text)
    if not fields and not visible_text.strip():
        return visible_text, {"structured_answer_repair_failed": 1}
    if session is None:
        return visible_text, {}

    known_files, known_verifications = _extract_session_answer_anchors(session)
    repaired = dict(fields)
    backfilled = False
    changed = False
    verification_value = repaired.get("verification", "")
    file_value = repaired.get("file", "")
    verification_placeholder = _looks_like_placeholder_value(verification_value) if verification_value else False
    file_placeholder = _looks_like_placeholder_value(file_value) if file_value else False

    if "verification" in expected_labels and verification_value and not verification_placeholder:
        normalized_verification = _normalize_verification_value(repaired["verification"], known_verifications)
        if normalized_verification and normalized_verification != repaired["verification"]:
            repaired["verification"] = normalized_verification
            changed = True

    if "file" in expected_labels and file_value and not file_placeholder:
        resolved = _canonicalize_repo_path(repaired["file"], known_files)
        if resolved and resolved != repaired["file"]:
            repaired["file"] = resolved
            changed = True

    if (
        "verification" in expected_labels
        and (not repaired.get("verification") or verification_placeholder)
        and known_verifications
    ):
        repaired["verification"] = known_verifications[0]
        backfilled = True
        changed = True

    if "file" in expected_labels and (not repaired.get("file") or file_placeholder) and known_files:
        repaired["file"] = known_files[0]
        backfilled = True
        changed = True

    if "file" in expected_labels and repaired.get("verification") and known_files:
        inferred_file = _file_from_verification_value(repaired["verification"], known_files)
        if inferred_file and inferred_file != repaired.get("file", ""):
            repaired["file"] = inferred_file
            changed = True

    rebuilt_lines: list[str] = []
    for label in expected_labels:
        value = repaired.get(label, "").strip()
        if value:
            rebuilt_lines.append(f"{label.capitalize()}={value}")

    if not rebuilt_lines:
        return visible_text, {"structured_answer_repair_failed": 1}

    repaired_text = "\n".join(rebuilt_lines)
    signals: dict[str, int] = {}
    missing_expected = [
        label for label in expected_labels if label not in repaired or not repaired.get(label, "").strip()
    ]
    if missing_expected:
        signals["structured_answer_repair_failed"] = 1
    if changed or repaired_text != visible_text.strip():
        signals["structured_answer_repaired"] = 1
    if backfilled:
        signals["structured_answer_backfilled"] = 1
        repaired_text = "[memory-derived]\n" + repaired_text
    return repaired_text, signals


def _synthesize_answer_phase_fallback_text(
    session: RuntimeSession | None,
) -> str:
    """Placeholder for answer-phase fallback text synthesis.

    Returns empty string by design. Callers check for truthiness before
    using the result. This will be wired to bridge memory anchors in a
    future release.
    """
    return ""


def _tool_compatible_mixed_turn_signals(
    tool_blocks: list[dict[str, Any]],
    visible_text: str,
    *,
    tool_compatible: bool,
) -> dict[str, int]:
    if not tool_compatible:
        return {}
    has_tool = any(block.get("type") == "tool_use" for block in tool_blocks)
    if not has_tool or not visible_text.strip():
        return {}
    signals = {"mixed_tool_visible_text": 1}
    if _is_answer_like_visible_text(visible_text):
        signals["mixed_answer_tool_event"] = 1
        signals["tool_contract_failure"] = 1
    return signals


def has_visible_content_block(content_blocks: list[dict[str, Any]]) -> bool:
    for block in content_blocks:
        if block.get("type") == "tool_use":
            return True
        if block.get("type") == "text" and str(block.get("text", "")).strip():
            return True
    return False


def has_forbidden_tok_hybrid_patterns(text: str) -> bool:
    lowered = text.lower()
    return any(
        pattern in lowered
        for pattern in (
            "@tool(json=",
            "@tool(json:",
            "@tool({",
            "@tool(",
            '"type": "tool_use"',  # Raw JSON tool blobs
        )
    )


def has_non_inverted_assistant_message(text: str) -> bool:
    in_msg_assistant = False
    block_is_inverted = False

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">>>"):
            in_msg_assistant = False
            block_is_inverted = False
            continue
        if stripped.startswith("@"):
            in_msg_assistant = stripped.startswith("@msg") and "role:assistant" in stripped
            block_is_inverted = False
            continue

        if in_msg_assistant:
            if re.match(r"^\s+\|[#\d]?>", line):
                block_is_inverted = True
                continue

            # If we haven't seen an inversion marker yet, or if it's a completely
            # un-prefixed line before a marker, it's non-inverted.
            # BUT once the block is inverted, we allow following lines to be un-prefixed.
            if not block_is_inverted:
                # Malformed: text before any inversion marker in an assistant msg
                return True

    return False


def has_markdown_fallback_after_tok_header(text: str) -> bool:
    saw_header = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">>>"):
            saw_header = True
            continue
        if saw_header and re.match(r"^#{1,6}\s+", stripped):
            return True
    return False


def has_bad_tok_header_shape(text: str) -> bool:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith(">>>"):
            continue
        payload = stripped[3:].strip()
        if not payload:
            return True
        fields = payload.split("|")
        if any(
            not field.strip()
            or ":" not in field
            or not field.split(":", 1)[0].strip()
            or not field.split(":", 1)[1].strip()
            for field in fields
        ):
            return True
    return False


def malformed_tok_signals(text: str) -> dict[str, int]:
    signals: dict[str, int] = {}
    if has_forbidden_tok_hybrid_patterns(text):
        signals["malformed_tok_hybrid_tool"] = 1
    if has_non_inverted_assistant_message(text):
        signals["malformed_tok_non_inverted_msg"] = 1
    if has_markdown_fallback_after_tok_header(text):
        signals["malformed_tok_markdown_fallback"] = 1
    if has_bad_tok_header_shape(text):
        signals["malformed_tok_bad_header"] = 1
    if signals:
        signals["malformed_tok_response"] = 1
    return signals


def has_well_formed_tok_blocks(content_blocks: list[dict[str, Any]]) -> bool:
    """Validate that Tok blocks are well-formed and safe to return."""
    if not content_blocks:
        return False

    has_tool_blocks = False
    for block in content_blocks:
        block_type = block.get("type")
        if block_type == "tool_use":
            has_tool_blocks = True
            # Tool blocks must have required fields
            if not block.get("id") or not block.get("name"):
                return False
            # Tool blocks must have a valid name (not "unknown")
            if block.get("name") == "unknown":
                return False
            tool_name = block.get("name", "").lower()
            input_data = block.get("input")
            if input_data is not None and not isinstance(input_data, dict):
                return False
            # File operations should have path attribute
            if tool_name in {"edit", "write", "read"} and not (input_data and input_data.get("path")):
                return False
        elif block_type == "text":
            # Text blocks should have non-empty content
            if not str(block.get("text", "")).strip():
                return False
        else:
            # Unknown block type
            return False

    # If we have tool blocks, at least one should be well-formed
    return has_tool_blocks or any(b.get("type") == "text" for b in content_blocks)


def translate_request_results(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Translate @Result blocks in user messages to tool_result blocks."""
    parser = TokParser()
    new_messages = []

    for msg in copy.deepcopy(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str) and "@Result" in content:
                nodes = parser.parse(content)
                new_content = []
                last_text = ""

                for node in nodes:
                    if node.type.lower() == "result":
                        if last_text.strip():
                            new_content.append({"type": "text", "text": last_text.strip()})
                            last_text = ""
                        tool_id = node.attrs.get("id") or node.label or "unknown"
                        new_content.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": node.text.strip(),
                            }
                        )
                    else:
                        last_text += serialize([node]) + "\n"

                if last_text.strip():
                    new_content.append({"type": "text", "text": last_text.strip()})

                if new_content:
                    if len(new_content) == 1 and new_content[0].get("type") == "text":
                        msg["content"] = new_content[0].get("text", "")
                    else:
                        msg["content"] = new_content
        new_messages.append(msg)

    return new_messages


def _parse_json_tool_data(data: dict[str, Any]) -> TokNode:
    name = (data.get("name") or data.get("tool") or data.get("action") or "unknown").lower()
    args = (
        data.get("arguments")
        or data.get("input")
        or {k: v for k, v in data.items() if k not in ("name", "tool", "action")}
    )
    return TokNode(type="tool", label=name, attrs=args, text="")


def _parse_json_code_blocks(text: str) -> list[TokNode]:
    import json

    nodes = []
    json_blocks = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    for block in json_blocks:
        try:
            data = json.loads(block)
            if isinstance(data, dict):
                nodes.append(_parse_json_tool_data(data))
        except (json.JSONDecodeError, ValueError):
            continue
    return nodes


def _parse_inline_json(text: str) -> list[TokNode]:
    import json

    nodes = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}") and any(k in line for k in ('"name"', '"tool"', '"action"')):
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    nodes.append(_parse_json_tool_data(data))
            except (json.JSONDecodeError, ValueError):
                continue
    return nodes


def _parse_hybrid_tools(text: str) -> list[TokNode]:
    import json

    nodes = []
    hybrid_matches = re.finditer(r"@Tool\s+([a-zA-Z0-9_-]+)\s*(\{.*?\})", text, re.DOTALL)
    for match in hybrid_matches:
        name = match.group(1).lower()
        if name == "readfile":
            name = "read"
        if name == "viewfile":
            name = "view_file"
        if name == "listdir":
            name = "list_dir"
        if name == "grepsearch":
            name = "grep_search"

        json_str = match.group(2)
        try:
            args = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            try:
                repaired = re.sub(r"([{,]\s*)([a-zA-Z0-9_-]+):", r'\1"\2":', json_str)
                args = json.loads(repaired)
            except Exception:
                logger.debug("Failed to repair hybrid tool JSON: %s", json_str[:80])
                continue

        if isinstance(args, dict):
            nodes.append(TokNode(type="tool", label=name, attrs=args, text=""))
    return nodes


def _extract_json_tools(text: str) -> list[TokNode]:
    """Find and extract JSON tool calls from raw text (Orchestrator extraction logic)."""
    nodes = []
    nodes.extend(_parse_json_code_blocks(text))
    nodes.extend(_parse_inline_json(text))
    nodes.extend(_parse_hybrid_tools(text))
    return nodes


def _preprocess_cleaned_text(text: str) -> str:
    cleaned = re.sub(r"```json\s*(\{.*?\})\s*```", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"@Tool\s+[a-zA-Z0-9_-]+\s*\{.*?\}", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"```[\w]*\n?(.*?)```", r"\1", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"```[\w]*\n?", "", cleaned)

    if (
        not IS_TOK.search(text)
        or (not re.search(r"^>>>", cleaned, re.MULTILINE) and not re.search(r"^@msg", cleaned, re.MULTILINE))
    ) and (cleaned.strip().startswith("@Tool") or cleaned.strip().startswith("|>")):
        cleaned = "@msg role:assistant\n  " + cleaned.replace("\n", "\n  ")
    return cleaned


def _apply_drift_guard(node: TokNode, tool_input: dict[str, Any]) -> None:
    drift_detected = False
    for k in (
        "text",
        "content",
        "replace",
        "search",
        "cmd",
        "command",
    ):
        val = tool_input.get(k)
        if isinstance(val, str) and ("\n" in val or "\\n" in val):
            node.text = (node.text + "\n" + val.replace("\\n", "\n")).strip()
            tool_input.pop(k)
            drift_detected = True

    if drift_detected:
        node.text = re.sub(r"^\|\>\s*", "", node.text, flags=re.MULTILINE)
        node.text = re.sub(r"^>\s*", "", node.text, flags=re.MULTILINE)


def _cleanup_edit_tool_children(node: TokNode, tool_input: dict[str, Any]) -> None:
    if not node.children:
        return
    for child in node.children:
        ctype = child.type.lower()
        if ctype == "search":
            tool_input["search"] = child.text.strip()
            child._processed_as_attr = True
        elif ctype == "replace":
            tool_input["replace"] = child.text.strip()
            child._processed_as_attr = True


def _cleanup_tool_input(node: TokNode, tool_name: str, tool_input: dict[str, Any]) -> None:
    for key in ("id", "name", "trust"):
        tool_input.pop(key, None)

    if node.text.strip():
        if tool_name in ("write", "edit"):
            if tool_name == "write" and "content" not in tool_input:
                tool_input["content"] = node.text.strip()
            elif tool_name == "edit" and "replace" not in tool_input:
                tool_input["replace"] = node.text.strip()
        elif "text" not in tool_input:
            tool_input["text"] = node.text.strip()

    if tool_name == "edit":
        _cleanup_edit_tool_children(node, tool_input)


def _process_tool_node(
    node: TokNode,
    content_blocks: list[dict[str, Any]],
    current_text: list[str],
) -> None:
    ntype = node.type.lower()
    if ntype == "tool":
        if current_text:
            full_text = "\n".join(current_text).strip()
            if full_text:
                content_blocks.append({"type": "text", "text": full_text})
            current_text.clear()

        tool_name = node.label or node.attrs.get("name", "unknown")
        tool_id = node.attrs.get("id") or f"tool_{uuid.uuid4().hex[:8]}"
        tool_input = dict(node.attrs)

        _apply_drift_guard(node, tool_input)
        _cleanup_tool_input(node, tool_name, tool_input)

        content_blocks.append(
            {
                "type": "tool_use",
                "id": tool_id,
                "name": tool_name,
                "input": tool_input,
            }
        )
    elif ntype in {"thought", "end"} or getattr(node, "_processed_as_attr", False):
        pass
    else:
        node_text = node.text.strip()
        if node_text:
            lines = [line for line in node_text.split("\n") if not line.strip().startswith(">>")]
            if lines:
                current_text.append("\n".join(lines))


def translate_response_tools(text: str) -> list[dict[str, Any]]:
    """Identify @Tool blocks in Tok response and map to tool_use blocks."""
    json_nodes = _extract_json_tools(text)
    cleaned = _preprocess_cleaned_text(text)

    parser = TokParser()
    nodes = parser.parse(cleaned)
    nodes.extend(json_nodes)

    all_nodes = []

    def _collect(nl: list[TokNode]) -> None:
        for n in nl:
            all_nodes.append(n)
            _collect(n.children)

    _collect(nodes)

    content_blocks: list[dict[str, Any]] = []
    current_text: list[str] = []

    for node in all_nodes:
        _process_tool_node(node, content_blocks, current_text)

    if current_text:
        full_text = "\n".join(current_text).strip()
        if full_text:
            content_blocks.append({"type": "text", "text": full_text})

    normalized_blocks, _ = normalize_tool_use_blocks(content_blocks, seed_prefix="toolu_rsp")
    return normalized_blocks


def parse_tok_response(
    text: str, session: RuntimeSession | None = None
) -> tuple[list[dict[str, Any]], dict[str, int], str]:
    """Parse a Tok response into Anthropic-compatible content blocks."""
    # Phase 7: Deterministic parser with malformed signal surfacing
    tok_blocks = translate_response_tools(text)
    malformed_signals = malformed_tok_signals(text)

    # Adaptive mode tracking
    fallback_mode = ""
    if session:
        fallback_mode = session._last_mode

    return tok_blocks, malformed_signals, fallback_mode


def response_contract(text: str) -> ProcessedRuntimeResponse:
    """Classify a model response under the bridge-first success contract."""
    return response_contract_for_mode(text, tool_compatible=False)


def response_contract_for_mode(
    text: str,
    tool_compatible: bool = False,
    _family: str = "",
    _model: str = "",
    session: RuntimeSession | None = None,
) -> ProcessedRuntimeResponse:
    """Analyze a response and determine if it follows the expected protocol contract."""
    if not text.strip():
        return ProcessedRuntimeResponse(
            content_blocks=[],
            output_saved_tokens=0,
            behavior_signals={},
            mode="empty",
            family_mode="",
            updated_memory="",
        )

    # Ignore REPL prompts embedded in fenced code blocks when deciding whether
    # the response is trying to speak Tok protocol.
    contract_text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    tok_detection_text = contract_text if contract_text.strip() else ""
    has_tok_protocol = bool(IS_TOK.search(tok_detection_text))

    tok_blocks, malformed_signals, fallback_mode = parse_tok_response(tok_detection_text, session=session)
    visible_text = ""
    mode = ""
    signals: dict[str, int] = {}
    content_blocks: list[dict[str, Any]] = []

    if (
        has_tok_protocol
        and not malformed_signals
        and has_visible_content_block(tok_blocks)
        and has_well_formed_tok_blocks(tok_blocks)
    ):
        (
            visible_text,
            signals,
            content_blocks,
            mode,
        ) = _tok_native_path(tok_blocks, tool_compatible=tool_compatible)
    else:
        readable, fallback_mode = postprocess_response(text)
        fallback_blocks: list[dict[str, Any]] = [{"type": "text", "text": readable}] if readable else []
        if tool_compatible:
            (
                visible_text,
                signals,
                content_blocks,
                mode,
            ) = _tool_compatible_path(
                tok_blocks,
                fallback_blocks,
                malformed_signals,
                has_tok_protocol,
                tool_compatible=tool_compatible,
            )
        else:
            (
                visible_text,
                signals,
                content_blocks,
                mode,
            ) = _standard_fallback_path(
                tok_blocks,
                fallback_blocks,
                malformed_signals,
                fallback_mode,
                has_tok_protocol,
            )

    expected_labels = _expected_structured_labels(session)
    has_tool_blocks = any(block.get("type") == "tool_use" for block in content_blocks)
    if expected_labels and not has_tool_blocks:
        repaired_text, repair_signals = _repair_structured_answer_text(
            visible_text,
            expected_labels=expected_labels,
            session=session,
        )
        if repair_signals:
            signals.update(repair_signals)
        if repaired_text.strip():
            visible_text = repaired_text.strip()
            content_blocks = [{"type": "text", "text": visible_text}]

    answer_phase_active = tool_compatible and _answer_phase_context_active(session)
    fallback_allowed = tool_compatible and _answer_phase_fallback_allowed(session)
    tool_intent_without_answer = _is_tool_intent_without_answer(
        content_blocks,
        visible_text,
        raw_text=text,
    )
    if answer_phase_active and fallback_allowed and tool_intent_without_answer:
        synthesized = _synthesize_answer_phase_fallback_text(session)
        if synthesized:
            visible_text = synthesized
            content_blocks = [{"type": "text", "text": visible_text}]
            signals["answer_phase_tool_intent_quarantined"] = 1
        else:
            signals["answer_phase_fallback_failed_no_anchor"] = 1
    elif (
        answer_phase_active
        and fallback_allowed
        and not expected_labels
        and (bool(malformed_signals) or not _is_answer_like_visible_text(visible_text))
    ):
        synthesized = _synthesize_answer_phase_fallback_text(session)
        if synthesized:
            visible_text = synthesized
            content_blocks = [{"type": "text", "text": visible_text}]
            signals["answer_phase_non_labeled_fallback_applied"] = 1
        else:
            signals["answer_phase_fallback_failed_no_anchor"] = 1

    if session and hasattr(session, "_last_mode"):
        session._last_mode = mode

    # Note: SemanticValidator is applied in universal_runtime.py process_response

    from .tool_processing import count_tokens

    return ProcessedRuntimeResponse(
        content_blocks=content_blocks,
        output_saved_tokens=max(0, count_tokens(text) - count_tokens(visible_text)),
        behavior_signals=signals,
        mode=mode,
        family_mode="",
        updated_memory="",
    )


def _tok_native_path(
    tok_blocks: list[dict[str, Any]],
    tool_compatible: bool,
) -> tuple[str, dict[str, int], list[dict[str, Any]], str]:
    visible_text = "".join(block.get("text", "") for block in tok_blocks if block.get("type") == "text")
    contract_signals = _tool_compatible_mixed_turn_signals(tok_blocks, visible_text, tool_compatible=tool_compatible)
    signals = {"tok_native_response": 1, **contract_signals}
    return visible_text, signals, tok_blocks, "tok-native"


def _tool_compatible_path(
    tok_blocks: list[dict[str, Any]],
    fallback_blocks: list[dict[str, Any]],
    malformed_signals: dict[str, int],
    has_tok_protocol: bool,
    tool_compatible: bool,
) -> tuple[str, dict[str, int], list[dict[str, Any]], str]:
    has_tool_blocks = any(block.get("type") == "tool_use" for block in tok_blocks)
    content_blocks = tok_blocks if has_tool_blocks else fallback_blocks
    visible_text = _visible_text_from_content_blocks(content_blocks)

    if not has_tok_protocol:
        contract_signals = _tool_compatible_mixed_turn_signals(
            tok_blocks, visible_text, tool_compatible=tool_compatible
        )
        signals = {"tool_compatible_response": 1, **contract_signals} if content_blocks else contract_signals
    else:
        signals = dict(malformed_signals) if malformed_signals else {}
        if signals and fallback_blocks:
            signals["fail_open_compat_response"] = 1
        signals.update(_tool_compatible_mixed_turn_signals(tok_blocks, visible_text, tool_compatible=tool_compatible))
    return visible_text, signals, content_blocks, "tool-compatible"


def _standard_fallback_path(
    tok_blocks: list[dict[str, Any]],
    fallback_blocks: list[dict[str, Any]],
    malformed_signals: dict[str, int],
    fallback_mode: str,
    has_tok_protocol: bool,
) -> tuple[str, dict[str, int], list[dict[str, Any]], str]:
    signals = {}
    if has_tok_protocol:
        signals.update(malformed_signals or {"malformed_tok_response": 1})
        if fallback_blocks:
            signals["fail_open_compat_response"] = 1
    else:
        signals["non_tok_response"] = 1
        if fallback_blocks:
            signals["fail_open_compat_response"] = 1

    content_blocks = fallback_blocks or tok_blocks
    visible_text = "".join(block.get("text", "") for block in content_blocks if block.get("type") == "text")

    mode = fallback_mode
    if has_tok_protocol and malformed_signals:
        if fallback_mode in ("tok-empty", "markdown"):
            mode = fallback_mode
        elif malformed_signals.get("malformed_tok_markdown_fallback"):
            mode = "tok-empty"
        else:
            mode = "tok"

    return visible_text, signals, content_blocks, mode
