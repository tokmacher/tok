"""Answer memory extraction, grounding, and compaction logic."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from .tok_state import _tool_compat_compact_value

if TYPE_CHECKING:
    from ..core import RuntimeSession

AnswerMemory = dict[str, list[str]]


def _add_file_to_fields(fields: dict[str, list[str]], path: str) -> None:
    cleaned = path.strip()
    if not cleaned:
        return
    file_list = fields.setdefault("files", [])
    if cleaned not in file_list:
        file_list.append(cleaned[:96])
    file_fact = f"answer_file:{cleaned[:96]}"
    facts = fields.setdefault("facts", [])
    if file_fact not in facts:
        facts.append(file_fact)


def _add_verification_to_fields(
    fields: dict[str, list[str]], value: str
) -> None:
    cleaned = value.strip()
    if not cleaned:
        return
    fact = f"answer_verification:{cleaned[:96]}"
    facts = fields.setdefault("facts", [])
    if fact not in facts:
        facts.append(fact)


_PATH_PATTERN = re.compile(
    r"([\w./-]+\.(?:py|ts|tsx|js|jsx|json|md|toml|yaml|yml|sh|txt|css|html|sql|rs|go|rb))(?:[:#]L?\d+)?"
)

_IMPLEMENTATION_MARKERS = (
    "main implementation entry",
    "main entry point",
    "implemented in",
    "is in `",
    " in `src/",
)

_IDENTIFIER_MARKERS = (
    "main implementation entry",
    "main entry point",
    "function",
    "class",
)


def _process_kv_line(
    fields: dict[str, list[str]], key: str, value: str
) -> None:
    """Process a key=value line inside extract_structured_answer_memory."""
    if key == "file":
        path_match = _PATH_PATTERN.search(value)
        if path_match:
            _add_file_to_fields(fields, path_match.group(1))
    elif key == "verification":
        path_match = _PATH_PATTERN.search(value)
        if path_match:
            _add_file_to_fields(fields, path_match.group(1))
        _add_verification_to_fields(fields, value)
    elif key == "related":
        fields.setdefault("facts", []).append(f"answer_related:{value[:96]}")


def _process_unstructured_line(
    fields: dict[str, list[str]], line: str
) -> None:
    """Process a non-KV line inside extract_structured_answer_memory."""
    lowered = line.lower()
    path_match = _PATH_PATTERN.search(line)
    if path_match and any(m in lowered for m in _IMPLEMENTATION_MARKERS):
        _add_file_to_fields(fields, path_match.group(1))

    identifier_match = re.search(r"`([A-Za-z_][A-Za-z0-9_]*)`", line)
    if identifier_match and any(m in lowered for m in _IDENTIFIER_MARKERS):
        _add_verification_to_fields(fields, identifier_match.group(1))


def extract_structured_answer_memory(text: str) -> dict[str, list[str]]:
    fields: dict[str, list[str]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if value:
                _process_kv_line(fields, key, value)
            continue
        _process_unstructured_line(fields, line)
    return fields


def _grounded_file_paths(session: RuntimeSession) -> set[str]:
    grounded: set[str] = set()
    path_pattern = re.compile(
        r"[\w./-]+\.(?:py|ts|tsx|js|jsx|json|md|toml|yaml|yml|sh|txt|css|html|sql|rs|go|rb)"
    )

    for bucket in (session.bridge_memory.hot, session.bridge_memory.durable):
        for entry in bucket.get("files", []):
            if entry.value:
                grounded.add(entry.value)
        for entry in bucket.get("facts", []):
            for match in path_pattern.findall(entry.value):
                grounded.add(match)

    return grounded


def _resolve_grounded_path(candidate: str, grounded_paths: set[str]) -> str:
    if candidate in grounded_paths:
        return candidate

    candidate_name = Path(candidate).name.lower()
    candidate_stem = Path(candidate).stem.lower()
    matches = [
        path
        for path in grounded_paths
        if Path(path).name.lower() == candidate_name
        or (candidate_stem and Path(path).stem.lower() == candidate_stem)
    ]
    if matches:
        return max(matches, key=len)

    suffix_matches = [
        path for path in grounded_paths if path.endswith(candidate)
    ]
    if suffix_matches:
        return max(suffix_matches, key=len)
    return ""


def ground_structured_answer_memory(
    session: RuntimeSession, fields: dict[str, list[str]]
) -> dict[str, list[str]]:
    if not fields:
        return {}

    grounded_paths = _grounded_file_paths(session)
    grounded: dict[str, list[str]] = {}

    for path in fields.get("files", []):
        resolved = _resolve_grounded_path(path, grounded_paths)
        if resolved:
            grounded.setdefault("files", []).append(resolved)

    for fact in fields.get("facts", []):
        if fact.startswith("answer_file:"):
            path = fact.split(":", 1)[1].strip()
            resolved = _resolve_grounded_path(path, grounded_paths)
            if resolved:
                grounded.setdefault("facts", []).append(
                    f"answer_file:{resolved}"
                )
            continue
        if fact.startswith("answer_related:"):
            related_value = fact.split(":", 1)[1].strip()
            resolved = _resolve_grounded_path(related_value, grounded_paths)
            if resolved:
                grounded.setdefault("facts", []).append(
                    f"answer_related:{resolved}"
                )
            continue
        grounded.setdefault("facts", []).append(fact)

    return grounded


def extract_answers(text: str) -> AnswerMemory:
    """Compatibility alias for older runtime imports."""
    return extract_structured_answer_memory(text)


def ground_answers_in_memory(
    session: RuntimeSession, fields: AnswerMemory
) -> AnswerMemory:
    """Compatibility alias for older runtime imports."""
    return ground_structured_answer_memory(session, fields)


def _latest_answer_file(session: RuntimeSession) -> str:
    for bucket in (session.bridge_memory.hot, session.bridge_memory.durable):
        for entry in bucket.get("facts", []):
            if entry.value.startswith("answer_file:"):
                return entry.value.split(":", 1)[1].strip()
    for bucket in (session.bridge_memory.hot, session.bridge_memory.durable):
        files = bucket.get("files", [])
        if files:
            return files[0].value
    return ""


def _latest_answer_verification(session: RuntimeSession) -> str:
    for bucket in (session.bridge_memory.hot, session.bridge_memory.durable):
        for entry in bucket.get("facts", []):
            if entry.value.startswith("answer_verification:"):
                return entry.value.split(":", 1)[1].strip()
    return ""


def _verification_specificity(value: str) -> tuple[int, int]:
    lowered = value.lower()
    score = 0
    if "compress_history" in lowered:
        score += 4
    if "function" in lowered:
        score += 1
    if "(" in value and ")" in value:
        score += 1
    return (score, len(value))


def reinforce_structured_answer_memory(
    session: RuntimeSession, fields: dict[str, list[str]]
) -> dict[str, list[str]]:
    if not fields:
        return {}
    reinforced = {
        key: list(values) for key, values in fields.items() if values
    }
    if "files" not in reinforced and any(
        fact.startswith("answer_verification:")
        for fact in reinforced.get("facts", [])
    ):
        prior_file = _latest_answer_file(session)
        if prior_file:
            reinforced.setdefault("files", []).append(prior_file)
            answer_file_fact = f"answer_file:{prior_file}"
            existing_facts = reinforced.setdefault("facts", [])
            if answer_file_fact not in existing_facts:
                existing_facts.append(answer_file_fact)

    existing_facts = reinforced.setdefault("facts", [])
    current_verifications = [
        fact.split(":", 1)[1].strip()
        for fact in existing_facts
        if fact.startswith("answer_verification:")
    ]
    prior_verification = _latest_answer_verification(session)
    if prior_verification:
        if not current_verifications or _verification_specificity(
            prior_verification
        ) > max(_verification_specificity(v) for v in current_verifications):
            existing_facts.insert(
                0, f"answer_verification:{prior_verification}"
            )
    return reinforced


def compact_structured_answer_memory(
    fields: dict[str, list[str]],
) -> dict[str, list[str]]:
    if not fields:
        return {}
    compacted: dict[str, list[str]] = {}
    for key, values in fields.items():
        seen: set[str] = set()
        compact_values: list[str] = []
        for value in values:
            compact = _tool_compat_compact_value(key, value)
            if not compact or compact in seen:
                continue
            seen.add(compact)
            compact_values.append(compact)
        if compact_values:
            compacted[key] = compact_values
    return compacted


def _should_persist_to_durable(field: str, value: str) -> bool:
    """Return True if this field/value pair should be written to durable memory."""
    return field == "files" or (
        field == "facts" and value.startswith("answer_")
    )


def _process_answer_memory(
    session: RuntimeSession, visible_text: str
) -> dict[str, list[str]]:
    """Run the full answer-memory pipeline and return structured fields.

    Stage order is load-bearing and must not be changed:
      1. extract  — parse raw answer facts from visible response text
      2. ground   — resolve file paths against session memory
      3. reinforce — fill in missing file/verification from prior session state
      4. compact  — deduplicate and truncate values for wire-state efficiency
    """
    return compact_structured_answer_memory(
        reinforce_structured_answer_memory(
            session,
            ground_structured_answer_memory(
                session, extract_structured_answer_memory(visible_text)
            ),
        )
    )


def _is_answer_like_visible_text(text: str) -> bool:
    if not text.strip():
        return False
    lowered = text.lower()
    if "file=" in lowered or "verification=" in lowered:
        return True
    fields = extract_structured_answer_memory(text)
    return bool(fields.get("files")) or any(
        fact.startswith("answer_file:")
        or fact.startswith("answer_verification:")
        for fact in fields.get("facts", [])
    )
