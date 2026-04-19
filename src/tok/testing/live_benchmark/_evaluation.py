from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from tok.runtime.core import RuntimeSession

from ._models import BenchmarkDefinition
from ._utils import _success_term_matches


def _extract_labeled_fields(text: str, session: RuntimeSession | None = None) -> dict[str, str]:
    fields: dict[str, str] = {}
    # Search for labels anywhere in the text (last occurrence wins)
    labels = ["file", "verification", "related"]
    for label in labels:
        # Match "File=..." or "file: ..." or "|> File=..."
        # Capture until end of line or pipe separator, allowing spaces
        # Guard against matching substrings inside other identifiers (e.g. "proFILE=...").
        pattern = rf"(?:\|>\s*)?(?<![\w-]){label}(?![\w-])\s*[:=]\s*([^|\n]+)"
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for m in matches:
            # Strip whitespace and trailing punctuation for more flexible matching
            value = m.group(1).strip().rstrip(".,;")
            fields[label.lower()] = value

    # Fallback to line-by-line for non-standard keys or if regex missed something
    for line in text.splitlines():
        cleaned = line.strip()
        if cleaned.startswith("|>"):
            cleaned = cleaned[2:].strip()
        if "=" not in cleaned:
            continue
        key, value = cleaned.split("=", 1)
        k = key.strip().lower()
        if k not in fields:
            # Strip whitespace and trailing punctuation
            fields[k] = value.strip().rstrip(".,;")

    # Resolve Tok v7 Macro Pointers (*A, *B, etc.)
    if session and session.bridge_memory:
        pointers = session.bridge_memory.pointers
        for key, val in fields.items():
            if str(val).startswith("*"):
                resolved = pointers.resolve(val)
                if resolved:
                    fields[key] = resolved
    return fields


def _looks_like_placeholder(value: str) -> bool:
    lowered = value.strip().lower()
    if not lowered:
        return True
    return any(
        marker in lowered
        for marker in (
            "<the",
            "<the specific",
            "<the command",
            "<the file",
            "<the result",
            "<the specific filename",
            "specific filename from the code change",
            "command run or test outcome",
        )
    )


def _is_research_benchmark(definition: BenchmarkDefinition) -> bool:
    return definition.name.startswith("research-loop")


def _repo_python_files(repo_root: Path) -> set[str]:
    src_root = repo_root / "src"
    if not src_root.exists():
        return set()
    return {str(path.relative_to(repo_root)).replace("\\", "/") for path in src_root.rglob("*.py") if path.is_file()}


def _repo_symbol_index(repo_root: Path) -> set[str]:
    symbols: set[str] = set()
    src_root = repo_root / "src"
    if not src_root.exists():
        return symbols
    pattern = re.compile(r"^\s*(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\b")
    for path in src_root.rglob("*.py"):
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                match = pattern.match(line)
                if match:
                    symbols.add(match.group(1))
    return symbols


def _extract_repo_candidates(value: str) -> list[str]:
    cleaned = value.strip().strip("'\"")
    if not cleaned:
        return []
    candidates: list[str] = [cleaned]
    for match in re.findall(r"(?:src/)?[A-Za-z0-9_./-]+\.py(?::\d+)?", cleaned):
        candidates.append(match)
    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        c = candidate.strip().strip("'\"")
        c = re.sub(r":\d+$", "", c)
        c = c.removeprefix("./")
        if c and c not in seen:
            seen.add(c)
            normalized.append(c)
    return normalized


def _resolve_repo_file(
    raw_value: str,
    *,
    repo_files: set[str],
    aliases: dict[str, str],
) -> tuple[str | None, list[str]]:
    warnings: list[str] = []
    basenames: dict[str, list[str]] = {}
    for repo_path in repo_files:
        basenames.setdefault(Path(repo_path).name, []).append(repo_path)

    for candidate in _extract_repo_candidates(raw_value):
        normalized = candidate.replace("\\", "/")
        candidate_forms = [normalized]
        if not normalized.startswith("src/"):
            candidate_forms.append(f"src/{normalized}")
        for form in candidate_forms:
            if form in repo_files:
                return form, warnings
            if form in aliases:
                mapped = aliases[form]
                if mapped in repo_files:
                    warnings.append("fixture_reference_stale")
                    return mapped, warnings
        base = Path(normalized).name
        if base and base in basenames and len(basenames[base]) == 1:
            return basenames[base][0], warnings
    return None, warnings


def _evaluate_repo_grounded_research_success(
    definition: BenchmarkDefinition,
    visible_response: str,
    *,
    repo_root: Path,
    session: RuntimeSession | None = None,
) -> tuple[bool, list[str], list[str]]:
    fields = _extract_labeled_fields(visible_response, session=session)
    failures: list[str] = []
    warnings: list[str] = []

    file_value = fields.get("file", "")
    verification_value = fields.get("verification", "")

    repo_files = _repo_python_files(repo_root)
    symbols = _repo_symbol_index(repo_root)
    aliases = {
        "src/tok/compression.py": "src/tok/compression/__init__.py",
        "src/tok/bridge_memory.py": "src/tok/runtime/memory/bridge_memory.py",
    }

    resolved_file: str | None = None
    if not file_value:
        failures.append("missing_file_field")
    elif _looks_like_placeholder(file_value):
        failures.append("placeholder_file_field")
    else:
        resolved_file, resolve_warnings = _resolve_repo_file(file_value, repo_files=repo_files, aliases=aliases)
        warnings.extend(resolve_warnings)
        if resolved_file is None:
            failures.append("file_not_found")

    if not verification_value:
        failures.append("missing_verification_field")
    elif _looks_like_placeholder(verification_value):
        failures.append("placeholder_verification_field")
    else:
        verification_identifiers = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", verification_value)
        expected_hits = [
            term
            for term in definition.expected_verification_terms
            if term.lower() in verification_value.lower() and term in symbols
        ]
        identifier_hits = [ident for ident in verification_identifiers if ident in symbols]
        if not expected_hits and not identifier_hits:
            failures.append("identifier_not_found")

    if resolved_file and definition.name == "research-loop" and resolved_file.endswith("compression/__init__.py"):
        warnings.append("fixture_reference_stale")

    return not failures, sorted(set(failures)), sorted(set(warnings))


def _message_shape_forensics(messages: list[dict[str, Any]]) -> dict[str, int]:
    shape = {
        "total_messages": 0,
        "content_str_messages": 0,
        "content_list_messages": 0,
        "content_other_messages": 0,
        "tool_use_blocks": 0,
        "tool_result_blocks": 0,
    }
    for message in messages:
        if not isinstance(message, dict):
            continue
        shape["total_messages"] += 1
        content = message.get("content")
        if isinstance(content, str):
            shape["content_str_messages"] += 1
        elif isinstance(content, list):
            shape["content_list_messages"] += 1
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    shape["tool_use_blocks"] += 1
                elif block.get("type") == "tool_result":
                    shape["tool_result_blocks"] += 1
        else:
            shape["content_other_messages"] += 1
    return shape


def _evaluate_task_success(
    definition: BenchmarkDefinition,
    visible_response: str,
    session: RuntimeSession | None = None,
) -> tuple[bool, list[str], list[str]]:
    if session:
        id(session)
        (id(session.bridge_memory.pointers) if session.bridge_memory else "none")
    fields = _extract_labeled_fields(visible_response, session=session)
    # Combine the visible text with resolved values for term matching
    search_space = visible_response.lower()
    for val in fields.values():
        search_space += f" {val.lower()}"

    matched_terms = [term for term in definition.success_terms if _success_term_matches(term, search_space)]
    failures: list[str] = []
    file_value = fields.get("file", "")
    verification_value = fields.get("verification", "")
    require_file_field = bool(definition.require_file_field)
    require_verification_field = bool(definition.require_verification_field)

    if require_file_field:
        if not file_value:
            failures.append("missing_file_field")
        elif _looks_like_placeholder(file_value):
            failures.append("placeholder_file_field")
        elif definition.expected_file_terms:
            # Normalize paths for matching: strip src/ prefix and common prefixes
            normalized_file = file_value.lower()
            for prefix in ("src/tok/", "src/", "tok/"):
                if normalized_file.startswith(prefix):
                    normalized_file = normalized_file[len(prefix) :]
                    break
            if not any(
                term.lower() in normalized_file or term.lower() in file_value.lower()
                for term in definition.expected_file_terms
            ):
                failures.append("unexpected_file_field")

    if require_verification_field:
        if not verification_value:
            failures.append("missing_verification_field")
        elif _looks_like_placeholder(verification_value):
            failures.append("placeholder_verification_field")
        elif definition.expected_verification_terms and not any(
            term.lower() in verification_value.lower() for term in definition.expected_verification_terms
        ):
            failures.append("unexpected_verification_field")

    structured_valid = not failures
    needs_textual_success_terms = not (require_file_field or require_verification_field)
    if (needs_textual_success_terms or not structured_valid) and len(matched_terms) < definition.min_success_terms:
        failures.append("response_missing_success_terms")

    return not failures, matched_terms, failures
