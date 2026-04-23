from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from tok.testing.stress.executor import ReadOnlyToolExecutor

from ._models import ToolExecutionRecord
from ._utils import (
    _git,
    _normalize_pytest_command,
    _resolved_path,
    _run_zsh_command,
    _safe_relative_path,
    _truncate,
)


class BenchmarkToolExecutor:
    """Execute the benchmark tool allowlist inside a task workspace."""

    def __init__(
        self,
        workspace_root: Path,
        *,
        allowed_tools: tuple[str, ...],
        timeout_seconds: int = 120,
        max_output_chars: int = 12000,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.allowed_tools = tuple(allowed_tools)
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars
        self.read_only = ReadOnlyToolExecutor(self.workspace_root, max_output_chars=max_output_chars)

    def execute_tool(
        self,
        block: dict[str, Any],
        *,
        step_index: int,
    ) -> tuple[dict[str, Any], bool, ToolExecutionRecord]:
        tool_input = block.get("input", {})
        if not isinstance(tool_input, dict):
            tool_input = {}
        canonical_name = self._canonical_tool_name(str(block.get("name", "")).strip())
        if not canonical_name or canonical_name not in self.allowed_tools:
            result = self._tool_result_message(
                block,
                f"ERROR: disallowed tool `{block.get('name', '')}` for this benchmark task",
                is_error=True,
                signal="invalid_tool",
            )
            record = ToolExecutionRecord(
                step_index=step_index,
                tool_name=str(block.get("name", "")).strip(),
                canonical_tool_name=canonical_name or "unsupported",
                tool_input=dict(tool_input),
                invalid=True,
                is_error=True,
                content_preview=_truncate(str(result["content"])),
            )
            return result, True, record

        if canonical_name in {"view_file", "grep_search", "list_dir"}:
            read_only_result, invalid = self.read_only.execute(dict(block))
            signal = str(read_only_result.get("contract_signal", ""))
            invalid = invalid or signal in {"invalid_tool", "bad_tool_args", "mutating_tool", "unsupported_tool"}
            content = str(read_only_result.get("content", ""))
            if (
                canonical_name == "grep_search"
                and content
                and content != "(no matches)"
                and not read_only_result.get("is_error")
                and re.search(r":\d+[:-]", content)
            ):
                content += "\n\nHint: use view_file with start and end parameters to read around these matches."
            result = self._tool_result_message(
                block,
                content,
                is_error=bool(read_only_result.get("is_error")),
                signal=signal,
            )
            record = ToolExecutionRecord(
                step_index=step_index,
                tool_name=str(block.get("name", "")).strip(),
                canonical_tool_name=canonical_name,
                tool_input=dict(tool_input),
                invalid=invalid,
                is_error=bool(read_only_result.get("is_error")),
                content_preview=_truncate(str(read_only_result.get("content", ""))),
            )
            return result, invalid, record

        if canonical_name == "edit_file":
            content, is_error, signal = self._edit_file(tool_input)
        elif canonical_name == "run_tests":
            content, is_error, signal = self._run_tests(tool_input)
        else:
            content, is_error, signal = self._git_diff(tool_input)
        invalid = is_error and signal in {"invalid_tool", "bad_tool_args", "mutating_tool", "unsupported_tool"}
        result = self._tool_result_message(block, content, is_error=is_error, signal=signal)
        record = ToolExecutionRecord(
            step_index=step_index,
            tool_name=str(block.get("name", "")).strip(),
            canonical_tool_name=canonical_name,
            tool_input=dict(tool_input),
            invalid=invalid,
            is_error=is_error,
            content_preview=_truncate(content),
        )
        return result, record.invalid, record

    def modified_files(self) -> tuple[str, ...]:
        completed = _git(["status", "--porcelain", "--untracked-files=no"], cwd=self.workspace_root)
        if completed.returncode != 0:
            return ()
        files: list[str] = []
        for raw_line in completed.stdout.splitlines():
            if not raw_line.strip():
                continue
            path_field = raw_line[3:].strip() if len(raw_line) >= 4 else ""
            if " -> " in path_field:
                path_field = path_field.split(" -> ", 1)[1].strip()
            if path_field:
                files.append(path_field)
        return tuple(sorted(files))

    def _canonical_tool_name(self, name: str) -> str:
        normalized = name.lower().strip()
        if normalized in {"view_file", "read", "read_file"}:
            return "view_file"
        if normalized in {"grep_search", "search", "grep", "rg"}:
            return "grep_search"
        if normalized in {"list_dir", "ls"}:
            return "list_dir"
        if normalized in {"edit_file", "edit", "replace_file_content"}:
            return "edit_file"
        if normalized in {"run_tests", "run", "run_terminal", "bash"}:
            return "run_tests"
        if normalized in {"git_diff", "diff"}:
            return "git_diff"
        return ""

    def _tool_result_message(
        self,
        block: dict[str, Any],
        content: str,
        *,
        is_error: bool = False,
        signal: str = "",
    ) -> dict[str, Any]:
        result_block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": str(block.get("id", "")).strip(),
            "content": _truncate(content, self.max_output_chars),
        }
        if is_error:
            result_block["is_error"] = True
        if signal:
            result_block["contract_signal"] = signal
        return {
            "role": "user",
            "content": [result_block],
        }

    def _safe_path(self, raw_path: str | None) -> Path:
        path = _resolved_path(self.workspace_root, raw_path)
        try:
            path.relative_to(self.workspace_root)
        except ValueError as exc:
            raise RuntimeError(f"path escapes workspace: {raw_path}") from exc
        return path

    def _edit_file(self, tool_input: dict[str, Any]) -> tuple[str, bool, str]:
        raw_path = (
            tool_input.get("path") or tool_input.get("file_path") or tool_input.get("target") or tool_input.get("text")
        )
        if not raw_path:
            return "ERROR: path is required", True, "bad_tool_args"
        path = self._safe_path(str(raw_path))
        old_string = tool_input.get("old_string") or tool_input.get("search") or tool_input.get("before")
        new_string = tool_input.get("new_string") or tool_input.get("replace") or tool_input.get("after")
        content = tool_input.get("content")
        if old_string is None and new_string is None and content is None:
            return "ERROR: edit_file requires old/new strings or full content", True, "bad_tool_args"
        if path.exists():
            original = path.read_text()
        else:
            original = ""
        if old_string is not None and new_string is not None:
            if str(old_string) not in original:
                return f"ERROR: search string not found in {path.name}", True, "bad_tool_args"
            updated = original.replace(str(old_string), str(new_string), 1)
        elif content is not None:
            updated = str(content)
        else:
            return "ERROR: edit_file requires a replacement payload", True, "bad_tool_args"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(updated)
        return f"Edited {_safe_relative_path(self.workspace_root, path)}", False, ""

    def _run_tests(self, tool_input: dict[str, Any]) -> tuple[str, bool, str]:
        command = str(tool_input.get("command") or tool_input.get("cmd") or tool_input.get("text") or "").strip()
        if not command:
            return "ERROR: run_tests requires a pytest command", True, "bad_tool_args"
        normalized_command = _normalize_pytest_command(command)
        if normalized_command is None:
            return "ERROR: only pytest commands are allowed in run_tests", True, "invalid_tool"
        command_result = _run_zsh_command(
            normalized_command,
            cwd=self.workspace_root,
            timeout_seconds=self.timeout_seconds,
        )
        output = command_result.stdout.strip()
        if command_result.stderr.strip():
            output = f"{output}\n{command_result.stderr.strip()}".strip()
        if command_result.execution_error:
            output = f"{output}\nexecution_error: {command_result.execution_error}".strip()
        content = f"$ {normalized_command}\n{output}".strip()
        is_error = command_result.returncode != 0 or bool(command_result.execution_error)
        if command_result.execution_error:
            signal = "command_execution_error"
        elif command_result.returncode != 0:
            signal = "command_failed"
        else:
            signal = ""
        return content, is_error, signal

    def _git_diff(self, tool_input: dict[str, Any]) -> tuple[str, bool, str]:
        paths: list[str] = []
        if tool_input.get("path"):
            paths.append(str(tool_input["path"]))
        if isinstance(tool_input.get("paths"), list):
            paths.extend(str(item) for item in tool_input.get("paths", []))
        args = ["diff", "--unified=3"]
        if paths:
            args.extend(["--", *paths])
        completed = _git(args, cwd=self.workspace_root)
        if completed.returncode not in {0, 1}:
            return f"ERROR: git diff failed: {completed.stderr.strip()}", True, "command_failed"
        return completed.stdout.strip() or "(no diff)", False, ""
