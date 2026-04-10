"""Read-only tool executor for stress harness."""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .models import EXCLUDED_GROUNDED_PATH_FRAGMENTS
from .utils import _strip_answer_labels


class ReadOnlyToolExecutor:
    """Read-only local tool executor for stress loops."""

    def __init__(self, workspace_root: Path, max_output_chars: int = 12000) -> None:
        self.workspace_root = workspace_root
        self.max_output_chars = max_output_chars

    def execute(self, block: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        name = str(block.get("name", "")).strip()
        tool_input = block.get("input", {})
        if not isinstance(tool_input, dict):
            tool_input = {}
        tool_input = {
            key: (_strip_answer_labels(value) if isinstance(value, str) else value) for key, value in tool_input.items()
        }

        # CLI-style Parsing Support (Tok v7 Compatibility for Stress Gauntlet)
        if any(isinstance(v, str) and "--" in v for v in tool_input.values()):
            for key, value in list(tool_input.items()):
                if isinstance(value, str) and "--" in value:
                    try:
                        tokens = shlex.split(value.strip())
                        for i in range(len(tokens) - 1):
                            if tokens[i].startswith("--"):
                                k = tokens[i].lstrip("-").replace("-", "_")
                                v = tokens[i + 1]
                                if k not in tool_input:
                                    tool_input[k] = v
                                    # Clear original field if it matches the CLI string
                                    if tool_input.get(key) == value.strip():
                                        del tool_input[key]
                    except Exception:
                        pass

        normalized = name.lower()
        if normalized in {"view_file", "read"}:
            return self._view_file(block, tool_input), False
        if normalized in {"grep_search", "search", "grep", "rg"}:
            return self._grep_search(block, tool_input), False
        if normalized in {"list_dir", "ls"}:
            return self._list_dir(block, tool_input), False
        if normalized in {"write", "edit", "run", "delta"}:
            return (
                self._blocked_tool(block, "mutating tools are disabled", "mutating_tool"),
                True,
            )
        return (
            self._blocked_tool(
                block,
                "unsupported tool in read-only harness",
                "unsupported_tool",
            ),
            True,
        )

    def _resolved_path(self, raw_path: str | None) -> Path:
        path_text = _strip_answer_labels(str(raw_path or "")).strip()
        if not path_text:
            return self.workspace_root
        path = Path(path_text)
        if not path.is_absolute():
            path = self.workspace_root / path
        return path.resolve()

    def _top_level_entries(self) -> list[str]:
        if not self.workspace_root.exists() or not self.workspace_root.is_dir():
            return []
        names = sorted(item.name for item in self.workspace_root.iterdir())
        return names[:8]

    def _resolve_workspace_path(
        self, raw_path: str | None, *, expect_dir: bool | None = None
    ) -> tuple[Path | None, str]:
        path_text = _strip_answer_labels(str(raw_path or "")).strip()
        if not path_text:
            return self.workspace_root, ""

        direct = self._resolved_path(path_text)
        if direct.exists() and (expect_dir is None or direct.is_dir() == expect_dir):
            return direct, ""

        raw = Path(path_text)
        if raw.is_absolute():
            return None, ""

        candidates: list[Path] = []
        seen: set[Path] = set()
        for prefix in ("src", "tests", "testing"):
            prefixed = self.workspace_root / prefix / raw
            if prefixed.exists() and (expect_dir is None or prefixed.is_dir() == expect_dir):
                resolved = prefixed.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    candidates.append(resolved)

        wanted_suffix = raw.as_posix().lstrip("./")
        if wanted_suffix:
            for candidate in self.workspace_root.rglob(raw.name):
                if not candidate.exists():
                    continue
                if expect_dir is not None and candidate.is_dir() != expect_dir:
                    continue
                rel = candidate.relative_to(self.workspace_root).as_posix()
                if rel == wanted_suffix or rel.endswith("/" + wanted_suffix):
                    resolved = candidate.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        candidates.append(resolved)

        if len(candidates) == 1:
            return candidates[0], ""
        if len(candidates) > 1:
            options = ", ".join(path.relative_to(self.workspace_root).as_posix() for path in candidates[:5])
            return None, f"ambiguous path '{path_text}'. Candidates: {options}"

        roots = ", ".join(self._top_level_entries()) or "."
        return None, f"path '{path_text}' not found. Available top-level entries: {roots}"

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_output_chars:
            return text
        return text[: self.max_output_chars].rstrip() + "\n...[truncated]"

    def _blocked_tool(self, block: dict[str, Any], reason: str, signal: str | None = None) -> dict[str, Any]:
        return {
            "role": "tool_result",
            "tool_use_id": block.get("id", ""),
            "content": f"ERROR: {reason}",
            "is_error": True,
            "contract_signal": signal or "",
        }

    def _view_file(self, block: dict[str, Any], tool_input: dict[str, Any]) -> dict[str, Any]:
        requested_path = tool_input.get("path") or tool_input.get("file_path") or tool_input.get("text")
        if not requested_path:
            return self._blocked_tool(block, "path is required", "bad_tool_args")
        path, hint = self._resolve_workspace_path(requested_path, expect_dir=False)
        start = tool_input.get("start") or tool_input.get("StartLine")
        end = tool_input.get("end") or tool_input.get("EndLine")
        if path is None or not path.exists() or not path.is_file():
            detail = f" ({hint})" if hint else ""
            return {
                "role": "tool_result",
                "tool_use_id": block.get("id", ""),
                "content": f"ERROR: file not found: {requested_path}{detail}",
                "is_error": True,
                "contract_signal": "bad_tool_args",
            }
        try:
            text = path.read_text()
            if start and end:
                start_num = max(1, int(start))
                end_num = max(start_num, int(end))
                lines = text.splitlines()
                selected = "\n".join(lines[start_num - 1 : end_num])
                text = f"Read {path} [L{start_num}-L{end_num}]:\n{selected}"
            else:
                text = f"Read {path}:\n{text}"
        except Exception as exc:
            text = f"ERROR: failed to read {path}: {exc}"
        return {
            "role": "tool_result",
            "tool_use_id": block.get("id", ""),
            "content": self._truncate(text),
        }

    def _grep_search(self, block: dict[str, Any], tool_input: dict[str, Any]) -> dict[str, Any]:
        query = str(
            tool_input.get("query")
            or tool_input.get("pattern")
            or tool_input.get("search")
            or tool_input.get("text")
            or ""
        )
        query = _strip_answer_labels(query).strip()
        raw_search_path = tool_input.get("search_path") or tool_input.get("path") or "."
        search_path, hint = self._resolve_workspace_path(raw_search_path, expect_dir=None)
        if not query:
            return {
                "role": "tool_result",
                "tool_use_id": block.get("id", ""),
                "content": "ERROR: query is required",
                "is_error": True,
                "contract_signal": "bad_tool_args",
            }
        if search_path is None or not search_path.exists():
            detail = f" ({hint})" if hint else ""
            return {
                "role": "tool_result",
                "tool_use_id": block.get("id", ""),
                "content": f"ERROR: search path not found: {raw_search_path}{detail}",
                "is_error": True,
                "contract_signal": "bad_tool_args",
            }
        if shutil.which("rg"):
            cmd = ["rg", "-n", "-C", "2", query, str(search_path)]
        else:
            cmd = ["grep", "-R", "-n", "-C", "2", query, str(search_path)]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                cwd=self.workspace_root,
            )
            content = proc.stdout.strip() or proc.stderr.strip() or "(no matches)"
            if content and content != "(no matches)":
                filtered_lines = [line for line in content.splitlines() if not _is_excluded_grounded_path(line)]
                content = "\n".join(filtered_lines).strip() or "(no matches)"
        except Exception as exc:
            content = f"ERROR: search failed: {exc}"
        return {
            "role": "tool_result",
            "tool_use_id": block.get("id", ""),
            "content": self._truncate(content),
        }

    def _list_dir(self, block: dict[str, Any], tool_input: dict[str, Any]) -> dict[str, Any]:
        raw_path = tool_input.get("path") or tool_input.get("text") or "."
        path, hint = self._resolve_workspace_path(raw_path, expect_dir=True)
        if path is None or not path.exists() or not path.is_dir():
            detail = f" ({hint})" if hint else ""
            return {
                "role": "tool_result",
                "tool_use_id": block.get("id", ""),
                "content": f"ERROR: directory not found: {raw_path}{detail}",
                "is_error": True,
            }
        entries = sorted(item.name for item in path.iterdir())
        content = "\n".join(entries[:200]) or "(empty directory)"
        return {
            "role": "tool_result",
            "tool_use_id": block.get("id", ""),
            "content": self._truncate(content),
        }


def _is_excluded_grounded_path(path: str) -> bool:
    normalized = path.strip()
    return any(fragment in normalized for fragment in EXCLUDED_GROUNDED_PATH_FRAGMENTS)
