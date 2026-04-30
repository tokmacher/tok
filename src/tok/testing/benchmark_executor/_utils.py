from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ._models import ShellCommandResult

_SKIP_COPY_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    ".ruff_cache",
    ".venv",
    "tmp",
}
_EXECUTION_PATCH_REQUIRED_TOOLS = ("edit_file", "run_tests")
_READ_ONLY_LOOP_TOOLS = frozenset({"list_dir", "view_file", "grep_search"})
_READ_ONLY_LOOP_REPEAT_THRESHOLD = 3
_REDUNDANT_RUN_TESTS_SUPPRESSION_THRESHOLD = 2
_EDIT_SEARCH_MISS_ESCALATION_THRESHOLD = 3
_PYTEST_PREFIX_PATTERNS = (
    re.compile(r"^\s*pytest\b(?P<tail>.*)$"),
    re.compile(r"^\s*python\s+-m\s+pytest\b(?P<tail>.*)$"),
    re.compile(r"^\s*uv\s+run\s+pytest\b(?P<tail>.*)$"),
    re.compile(r"^\s*uv\s+run\s+python\s+-m\s+pytest\b(?P<tail>.*)$"),
)
_PYTEST_CD_WRAPPER_PATTERN = re.compile(r"^\s*cd\s+(.+?)\s*&&\s*(?P<rest>.+)$")
_LOOP_ASYMMETRY_DELTA_THRESHOLD = 2
_LOOP_ASYMMETRY_ABSOLUTE_THRESHOLD = 3
_TOK_CONTROLLED_CONDITION = "tok-controlled"
_TOK_RUNTIME_CONDITIONS = frozenset({"tok-universal", _TOK_CONTROLLED_CONDITION})
_THINKING_TOKEN_RE = re.compile(r"<\|[^|]*\|>")
_EVIDENCE_NOISE_RE = re.compile(r"^\*+\s*$")
_XML_TOOL_PATTERN = re.compile(
    r"<tool_name>\s*(\w+)\s*</tool_name>",
    re.IGNORECASE | re.DOTALL,
)
_XML_PARAM_PATTERN = re.compile(
    r'<parameter\s+name\s*=\s*"([^"]*)">\s*(.*?)\s*</parameter>',
    re.IGNORECASE | re.DOTALL,
)
_TOK_TOOL_PATTERN = re.compile(
    r"@Tool\s+(\w+)\s*\{([^}]*)\}",
    re.IGNORECASE | re.DOTALL,
)
_TEXT_TOOL_USE_PATTERN = re.compile(
    r"Tool\s+use\s*\(\s*(\w+)\s*\)\s*:\s*(\{[^}]*\})",
    re.IGNORECASE | re.DOTALL,
)
_FENCED_JSON_BLOCK_PATTERN = re.compile(
    r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```",
    re.IGNORECASE | re.DOTALL,
)


def _truncate(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]"


def _copytree_ignore(_src: str, names: list[str]) -> set[str]:
    return {name for name in names if name in _SKIP_COPY_NAMES}


def _resolved_path(base: Path, raw_path: str | None) -> Path:
    path_text = str(raw_path or "").strip()
    if not path_text:
        return base
    path = Path(path_text)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _directory_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rel_path = path.relative_to(root).as_posix()
        digest.update(rel_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git(args: list[str], *, cwd: Path, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    )


def _safe_relative_path(workspace_root: Path, candidate: Path) -> str:
    try:
        return str(candidate.relative_to(workspace_root))
    except ValueError:
        return str(candidate)


def _assistant_message_content(blocks: list[dict[str, Any]], fallback_text: str) -> str | list[dict[str, Any]]:
    if not blocks:
        return fallback_text
    if len(blocks) == 1 and blocks[0].get("type") == "text":
        return str(blocks[0].get("text", ""))
    return copy.deepcopy(blocks)


def _count_reacquisition_events(signals: dict[str, int]) -> int:
    event_keys = (
        "answer_anchor_reacquisition_attempt",
        "answer_ready_reacquisition_attempt",
        "repair_phase_reacquisition_attempt",
        "validated_target_reacquisition_attempt",
        "validated_target_exact_reacquisition_attempt",
        "validated_target_reconfirmation_attempt",
    )
    count = sum(int(signals.get(key, 0)) for key in event_keys)
    if count:
        return count
    return 1 if int(signals.get("reacquisition_cost_tokens", 0)) > 0 else 0


def _count_note_prefix(notes: tuple[str, ...], prefix: str) -> int:
    return sum(1 for note in notes if note.startswith(prefix))


def _normalize_pytest_command(command: str) -> str | None:
    raw = command.strip()
    if not raw:
        return None
    cd_match = _PYTEST_CD_WRAPPER_PATTERN.match(raw)
    while cd_match:
        raw = str(cd_match.group("rest") or "").strip()
        cd_match = _PYTEST_CD_WRAPPER_PATTERN.match(raw)
    if not raw:
        return None
    for pattern in _PYTEST_PREFIX_PATTERNS:
        match = pattern.match(raw)
        if not match:
            continue
        tail = str(match.group("tail") or "").strip()
        if tail:
            return f"python -m pytest {tail}"
        return "python -m pytest"
    return None


def _pytest_execution_command(normalized_command: str) -> str:
    """Return an executable pytest command while preserving the public command contract."""
    raw = normalized_command.strip()
    for pattern in _PYTEST_PREFIX_PATTERNS:
        match = pattern.match(raw)
        if not match:
            continue
        tail = str(match.group("tail") or "").strip()
        pytest_bin = shutil.which("pytest")
        if pytest_bin:
            quoted = shlex.quote(pytest_bin)
            return f"{quoted} {tail}".strip()
        return raw
    return raw


def _file_mentioned(path: str, text: str) -> bool:
    if path in text:
        return True
    basename = Path(path).name
    if basename in text and basename != path:
        return True
    return False


def _sentence_count(text: str) -> int:
    body = text.split("Evidence:", 1)[0].strip()
    if not body:
        return 0
    pieces = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", body) if piece.strip()]
    return len(pieces)


def _clean_answer_text(text: str) -> str:
    return _THINKING_TOKEN_RE.sub("", text).strip()


def _evidence_lines(text: str) -> list[str]:
    if "Evidence:" not in text:
        return []
    _, evidence = text.split("Evidence:", 1)
    lines: list[str] = []
    for raw_line in evidence.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _EVIDENCE_NOISE_RE.match(line):
            continue
        if line.startswith("-"):
            lines.append(line.lstrip("-").strip())
        else:
            lines.append(line)
    return lines


def _normalize_pytest_output(text: str) -> str:
    return re.sub(r"in \d+(?:\.\d+)?s", "in <time>s", text)


def _run_zsh_command(
    command: str,
    *,
    cwd: Path,
    timeout_seconds: int,
    env: dict[str, str] | None = None,
) -> ShellCommandResult:
    shell_path = shutil.which("zsh") or shutil.which("bash") or "/bin/sh"
    argv = (shell_path, "-lc", command)
    effective_env = dict(os.environ)
    effective_env["PYTHONDONTWRITEBYTECODE"] = "1"
    existing_pythonpath = effective_env.get("PYTHONPATH", "")
    effective_env["PYTHONPATH"] = str(cwd) if not existing_pythonpath else f"{cwd}{os.pathsep}{existing_pythonpath}"
    if env:
        effective_env.update(env)
    try:
        completed = subprocess.run(
            list(argv),
            cwd=cwd,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=effective_env,
            check=False,
        )
    except Exception as exc:
        return ShellCommandResult(
            command=command,
            argv=argv,
            returncode=127,
            stdout="",
            stderr="",
            execution_error=str(exc),
        )
    return ShellCommandResult(
        command=command,
        argv=argv,
        returncode=int(completed.returncode),
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


def _parse_key_value_object(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for m in re.finditer(r'(?:(\w+)\s*:\s*)?["\']?([^"\',:}\s]+)["\']?', text):
        key = m.group(1)
        value = m.group(2).strip()
        if key:
            try:
                result[key] = int(value)
            except ValueError:
                result[key] = value
    return result


def _extract_text_tool_calls(
    raw_response: str,
    allowed_tools: tuple[str, ...],
) -> list[dict[str, Any]]:
    allowed_lower = {t.lower() for t in allowed_tools}
    blocks: list[dict[str, Any]] = []
    for tool_match in _XML_TOOL_PATTERN.finditer(raw_response):
        tool_name = tool_match.group(1).strip()
        if tool_name.lower() not in allowed_lower:
            continue
        after = raw_response[tool_match.end() :]
        next_tool = _XML_TOOL_PATTERN.search(after)
        search_region = after[: next_tool.start()] if next_tool else after
        tool_input: dict[str, Any] = {}
        for param_match in _XML_PARAM_PATTERN.finditer(search_region):
            key = param_match.group(1).strip()
            value = param_match.group(2).strip()
            tool_input[key] = value
        blocks.append(
            {
                "type": "tool_use",
                "id": f"text_extracted_{len(blocks)}",
                "name": tool_name,
                "input": tool_input,
            }
        )
    for tok_match in _TOK_TOOL_PATTERN.finditer(raw_response):
        tool_name = tok_match.group(1).strip()
        if tool_name.lower() not in allowed_lower:
            continue
        tool_input = _parse_key_value_object(tok_match.group(2))
        blocks.append(
            {
                "type": "tool_use",
                "id": f"tok_extracted_{len(blocks)}",
                "name": tool_name,
                "input": tool_input,
            }
        )
    for text_match in _TEXT_TOOL_USE_PATTERN.finditer(raw_response):
        tool_name = text_match.group(1).strip()
        if tool_name.lower() not in allowed_lower:
            continue
        tool_input = _parse_key_value_object(text_match.group(2))
        blocks.append(
            {
                "type": "tool_use",
                "id": f"text_use_extracted_{len(blocks)}",
                "name": tool_name,
                "input": tool_input,
            }
        )
    for fenced_match in _FENCED_JSON_BLOCK_PATTERN.finditer(raw_response):
        payload_raw = fenced_match.group(1).strip()
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            continue

        payload_items: list[dict[str, Any]]
        if isinstance(payload, dict):
            payload_items = [payload]
        elif isinstance(payload, list):
            payload_items = [item for item in payload if isinstance(item, dict)]
        else:
            payload_items = []

        for item in payload_items:
            tool_name = str(
                item.get("tool") or item.get("tool_name") or item.get("name") or item.get("function") or ""
            ).strip()
            if not tool_name or tool_name.lower() not in allowed_lower:
                continue

            raw_input = item.get("input")
            if not isinstance(raw_input, dict):
                raw_input = item.get("args")
            if not isinstance(raw_input, dict):
                raw_input = item.get("parameters")
            if not isinstance(raw_input, dict):
                raw_input = {
                    key: value
                    for key, value in item.items()
                    if key not in {"tool", "tool_name", "name", "function", "input", "args", "parameters"}
                }

            blocks.append(
                {
                    "type": "tool_use",
                    "id": f"fenced_json_extracted_{len(blocks)}",
                    "name": tool_name,
                    "input": dict(raw_input),
                }
            )
    return blocks
