"""Tok state parsing, building, and manipulation logic."""

from __future__ import annotations

import re
from pathlib import Path

from tok.compression import TOK_FIELD_ALIAS, TOK_REVERSE_ALIAS
from tok.macros.memory import (
    ConstraintMemory,
    EpisodeMemory,
    LessonMemory,
    RepairMemory,
    TokMemory,
)
from tok.runtime.config import (
    TOOL_COMPAT_DELTA_KEYS,
    TOOL_COMPAT_MAX_FILES,
    TOOL_COMPAT_STICKY_KEYS,
)

from .bridge_memory import BridgeMemoryState


def _tool_compatible_has_answer_facts(fields: dict[str, list[str]]) -> bool:
    return any(fact.startswith("answer_") for fact in fields.get("facts", []) if fact)


def _parse_tok_state_fields(tok_state: str) -> dict[str, list[str]]:
    line = tok_state.strip()
    if line.startswith(">>>"):
        line = line[3:].strip()
    if not line:
        return {}
    canonical_multi_keys = {
        "files",
        "tests",
        "errs",
        "constraints",
        "questions",
        "cmds",
        "blockers",
        "facts",
    }
    result: dict[str, list[str]] = {}
    for part in line.split("|"):
        if ":" not in part:
            continue
        key, raw_value = part.split(":", 1)
        key = key.strip()
        key = TOK_REVERSE_ALIAS.get(key, key)
        raw_value = raw_value.strip()
        if not key or not raw_value:
            continue
        if key in {"turns", "goal", "next"}:
            result[key] = [raw_value]
        elif key in canonical_multi_keys:
            result[key] = [item.strip() for item in raw_value.split(",") if item.strip()]
        else:
            result.setdefault("facts", []).append(f"{key}:{raw_value}")
    return result


_COMPACT_PATH_PATTERN = re.compile(r"[\w./-]+\.(?:py|ts|tsx|js|jsx|json|md|toml|yaml|yml|sh|txt|css|html|sql|rs|go|rb)")

_COMPACT_SUFFIXES = (
    "_NotImplementedError",
    "_AssertionError",
    "_failed",
    "_passed",
)


def _compact_common(
    compact: str,
) -> tuple[str, str, re.Match[str] | None, str]:
    """Return (lowered, test_anchor, path_match, fix_target)."""
    lowered = compact.lower()
    path_match = _COMPACT_PATH_PATTERN.search(compact)
    fix_prefix = "Fix_failing_tests_in_"
    fix_target = ""
    if compact.startswith(fix_prefix):
        fix_target = compact[len(fix_prefix) :].strip("_")
        fix_target = fix_target.split("|", 1)[0].strip()
    test_match = re.search(r"tests/[\w./-]+::[\w./-]+", compact)
    test_anchor = test_match.group(0) if test_match else ""
    for suffix in _COMPACT_SUFFIXES:
        if suffix in test_anchor:
            test_anchor = test_anchor.split(suffix, 1)[0]
            break
    return lowered, test_anchor, path_match, fix_target


def _compact_goal(
    compact: str,
    lowered: str,
    test_anchor: str,
    path_match: re.Match[str] | None,
    fix_target: str,
) -> str:
    if "1_passed" in lowered:
        return "1_passed"
    if "file_updated_successfully" in lowered:
        return "file_updated"
    if test_anchor:
        return test_anchor[:40]
    if fix_target:
        return fix_target[:40]
    if path_match:
        return path_match.group(0)[:40]
    return compact[:40]


def _compact_tests(
    compact: str,
    lowered: str,
    test_anchor: str,
    path_match: re.Match[str] | None,
) -> str:
    passed_match = re.search(r"(\d+)_passed", lowered)
    if passed_match:
        return f"{passed_match.group(1)}_passed"
    failed_match = re.search(r"(\d+)_failed", lowered)
    if failed_match:
        return f"{failed_match.group(1)}_failed"
    if test_anchor:
        return test_anchor[:56]
    if path_match:
        path = path_match.group(0)
        if "fail" in lowered:
            return f"{path}_failed"[:56]
        if "pass" in lowered:
            return f"{path}_passed"[:56]
        return path[:56]
    return compact[:56]


def _compact_errs(
    compact: str,
    lowered: str,
    test_anchor: str,
    path_match: re.Match[str] | None,
) -> str:
    if test_anchor:
        if "notimplementederror" in lowered:
            return f"{test_anchor}_NotImplementedError"[:64]
        if "fail" in lowered:
            return f"{test_anchor}_failed"[:64]
        return test_anchor[:64]
    if path_match:
        path = path_match.group(0)
        if "notimplementederror" in lowered:
            return f"{path}_NotImplementedError"[:64]
        if "fail" in lowered:
            return f"{path}_failed"[:64]
        return path[:64]
    return compact[:64]


def _compact_files(
    compact: str,
    path_match: re.Match[str] | None,
) -> str:
    if compact.startswith("*") and "." not in compact:
        return ""
    if path_match:
        return path_match.group(0)[:48]
    return compact[:48]


def _compact_facts(
    compact: str,
    path_match: re.Match[str] | None,
) -> str:
    fact_key = ""
    fact_value = compact
    if ":" in compact:
        fact_key, fact_value = compact.split(":", 1)
        fact_key = fact_key.strip().lower()
        fact_value = fact_value.strip()
    if fact_key in {"answer_verification", "answer_related"}:
        return _compact_answer_fact(fact_key, fact_value, path_match)
    if fact_key == "answer_file" and path_match:
        return f"{fact_key}:{path_match.group(0)[:48]}"
    return compact[:64]


def _compact_answer_fact(
    fact_key: str,
    fact_value: str,
    path_match: re.Match[str] | None,
) -> str:
    if fact_key == "answer_related" and path_match:
        return f"{fact_key}:{path_match.group(0)[:48]}"
    code_identifier = re.search(r"`([A-Za-z_][A-Za-z0-9_]*)`", fact_value)
    if code_identifier:
        return f"{fact_key}:{code_identifier.group(1)[:48]}"
    symbol_definition = re.search(
        r"\b(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\b",
        fact_value,
        re.IGNORECASE,
    )
    if symbol_definition:
        return f"{fact_key}:{symbol_definition.group(1)[:48]}"
    named_symbol = re.search(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s+(?:function|class|method)\b",
        fact_value,
        re.IGNORECASE,
    )
    if named_symbol:
        return f"{fact_key}:{named_symbol.group(1)[:48]}"
    identifier_match = re.search(
        r"\b(?!def\b|class\b|function\b|method\b|the\b|a\b|an\b)"
        r"([A-Za-z_][A-Za-z0-9_]*)\b",
        fact_value,
        re.IGNORECASE,
    )
    if identifier_match:
        return f"{fact_key}:{identifier_match.group(1)[:48]}"
    return f"{fact_key}:{fact_value[:48]}" if fact_key else fact_value[:48]


def _tool_compat_compact_value(key: str, value: str) -> str:
    compact = value.strip()
    if not compact:
        return ""

    compact = re.sub(r"Tool_result_\(c\d+\):_?", "", compact)
    compact = compact.replace("_-_", "_")
    compact = re.sub(r"_+", "_", compact).strip("_")
    lowered, test_anchor, path_match, fix_target = _compact_common(compact)

    if key == "goal":
        return _compact_goal(compact, lowered, test_anchor, path_match, fix_target)
    if key == "tests":
        return _compact_tests(compact, lowered, test_anchor, path_match)
    if key == "errs":
        return _compact_errs(compact, lowered, test_anchor, path_match)
    if key == "files":
        return _compact_files(compact, path_match)
    if key == "facts":
        return _compact_facts(compact, path_match)
    return compact[:48]


def _canonicalize_tool_compatible_state_fields(
    fields: dict[str, list[str]],
) -> dict[str, list[str]]:
    if not fields:
        return {}

    canonical: dict[str, list[str]] = {}
    for key, values in fields.items():
        if not values:
            continue
        if key == "turns":
            canonical[key] = [values[0]]
            continue
        if key not in TOOL_COMPAT_DELTA_KEYS:
            continue
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            compact = _tool_compat_compact_value(key, value)
            if not compact or compact in seen:
                continue
            seen.add(compact)
            normalized.append(compact)
        if normalized:
            canonical[key] = normalized
    return canonical


def _apply_tool_compatible_sticky_fields(
    previous: dict[str, list[str]], current: dict[str, list[str]]
) -> dict[str, list[str]]:
    if not previous:
        return current
    merged = dict(current)
    for key in TOOL_COMPAT_STICKY_KEYS:
        if merged.get(key):
            continue
        previous_values = previous.get(key)
        if previous_values:
            merged[key] = previous_values
    if merged.get("files"):
        merged["files"] = _select_tool_compatible_files(previous, merged)
    return merged


def _rank_file_candidates(
    candidates: list[str],
    previous: dict[str, list[str]],
    current: dict[str, list[str]],
) -> list[str]:
    """Score and rank file candidates by relevance to tests/errors/goal."""
    test_err_text = " ".join(
        [
            *current.get("tests", []),
            *current.get("errs", []),
            *previous.get("tests", []),
            *previous.get("errs", []),
        ]
    ).lower()
    goal_text = " ".join([*current.get("goal", []), *previous.get("goal", [])]).lower()
    prev_files = previous.get("files") or []

    def _score(path: str) -> tuple[int, int]:
        stem = Path(path).stem.lower()
        score = 0
        if stem and stem in test_err_text:
            score += 4
        if stem and stem in goal_text:
            score += 1
        if prev_files and path == prev_files[0]:
            score += 2
        elif path in prev_files:
            score += 1
        return (score, len(path))

    ranked = sorted(candidates, key=_score, reverse=True)
    top_score = _score(ranked[0])[0]
    second_score = _score(ranked[1])[0]
    if top_score >= 4 and top_score > second_score:
        return [ranked[0]]
    return ranked[:TOOL_COMPAT_MAX_FILES]


def _select_tool_compatible_files(previous: dict[str, list[str]], current: dict[str, list[str]]) -> list[str]:
    candidates: list[str] = []
    seen_candidates: set[str] = set()
    for path in [
        *(current.get("files") or []),
        *(previous.get("files") or []),
    ]:
        if not path or path in seen_candidates:
            continue
        seen_candidates.add(path)
        candidates.append(path)
    if len(candidates) <= 1:
        return candidates
    return _rank_file_candidates(candidates, previous, current)


def _build_tok_state(fields: dict[str, list[str]]) -> str:
    if not fields:
        return ""
    ordered_keys = [
        "turns",
        "goal",
        "next",
        "blockers",
        "files",
        "facts",
        "tests",
        "errs",
        "constraints",
        "questions",
        "cmds",
    ]
    parts: list[str] = []
    for key in ordered_keys:
        values = fields.get(key)
        if not values:
            continue
        alias = TOK_FIELD_ALIAS.get(key, key)
        if key in {"turns", "goal", "next"}:
            parts.append(f"{alias}:{values[0]}")
        else:
            parts.append(f"{alias}:{','.join(values)}")
    for key, values in fields.items():
        if key in ordered_keys or not values:
            continue
        alias = TOK_FIELD_ALIAS.get(key, key)
        parts.append(f"{alias}:{','.join(values)}")
    return ">>> " + "|".join(parts) if parts else ""


def _extract_answer_file_paths(
    facts: list[str],
) -> set[str]:
    """Extract file paths from answer_file: facts."""
    answer_files: set[str] = set()
    for fact in facts:
        if fact.startswith("answer_file:"):
            file_path = fact.split(":", 1)[1].strip()
            if file_path:
                answer_files.add(file_path)
    return answer_files


def _should_include_delta_field(
    key: str,
    values: list[str],
    previous: dict[str, list[str]],
    has_answer_facts: bool,
    answer_files: set[str],
    has_answer_file_facts: bool,
) -> bool:
    """Determine if a field should be included in the delta state."""
    if key == "files" and has_answer_facts:
        return previous.get(key) != values or any(f in answer_files for f in values)
    if key == "tests":
        return previous.get(key) != values
    if key == "facts" and has_answer_facts:
        return previous.get(key) != values or has_answer_file_facts
    return previous.get(key) != values


def _delta_tok_state_fields(previous: dict[str, list[str]], current: dict[str, list[str]]) -> str:
    if not current:
        return ""
    delta: dict[str, list[str]] = {}
    current_facts = current.get("facts", [])
    previous_facts = previous.get("facts", [])
    has_answer_facts = any(fact.startswith("answer_") for fact in [*current_facts, *previous_facts])
    answer_files = _extract_answer_file_paths(current_facts)
    has_answer_file_facts = bool(answer_files)

    if "turns" in current:
        delta["turns"] = current["turns"]
    for key, values in current.items():
        if key == "turns" or key not in TOOL_COMPAT_DELTA_KEYS:
            continue
        if _should_include_delta_field(
            key,
            values,
            previous,
            has_answer_facts,
            answer_files,
            has_answer_file_facts,
        ):
            delta[key] = values
    if list(delta.keys()) == ["turns"]:
        return ""
    return _build_tok_state(delta)


def _prepare_tool_compatible_state(
    raw_state: str,
    previous_fields: dict[str, list[str]],
) -> tuple[dict[str, list[str]], dict[str, list[str]], bool]:
    """
    Parse, canonicalize, and apply sticky fields to a raw tool-compatible state string.

    Returns (parsed_fields, comparable_fields, has_answer_facts).
    comparable_fields excludes the 'turns' key (used for suppression comparison).
    """
    parsed = _canonicalize_tool_compatible_state_fields(_parse_tok_state_fields(raw_state))
    parsed = _apply_tool_compatible_sticky_fields(previous_fields, parsed)
    comparable = {k: v for k, v in parsed.items() if k != "turns"}
    has_answer_facts = _tool_compatible_has_answer_facts(parsed)
    return parsed, comparable, has_answer_facts


def _select_resend_strategy(
    comparable: dict[str, list[str]],
    previous_comparable: dict[str, list[str]],
    has_answer_facts: bool,
) -> str:
    """
    Return the resend strategy: 'full', 'suppress', or 'delta'.

    Ordering is load-bearing:
      - unchanged state suppresses
      - first appearance of answer-bearing state forces full resend
      - other changed state may delta
    """
    if comparable == previous_comparable and comparable:
        return "suppress"
    previous_has_answer_facts = _tool_compatible_has_answer_facts(previous_comparable)
    if has_answer_facts and not previous_has_answer_facts:
        return "full"
    return "delta"


def _select_resend_reason(
    comparable: dict[str, list[str]],
    previous_comparable: dict[str, list[str]],
    has_answer_facts: bool,
) -> str:
    if comparable == previous_comparable and comparable:
        return "verified_current_state"  # Changed from "unchanged_state" - data is verified fresh
    previous_has_answer_facts = _tool_compatible_has_answer_facts(previous_comparable)
    if has_answer_facts and not previous_has_answer_facts:
        return "new_answer_anchor"
    return "changed_state_delta"


__all__ = [
    "BridgeMemoryState",
    "ConstraintMemory",
    "EpisodeMemory",
    "LessonMemory",
    "RepairMemory",
    "TokMemory",
    "_build_tok_state",
    "_delta_tok_state_fields",
    "_prepare_tool_compatible_state",
    "_select_resend_reason",
    "_select_resend_strategy",
]
