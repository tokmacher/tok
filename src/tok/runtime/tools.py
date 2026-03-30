"""Shared runtime tools for Tok - transport-agnostic tool execution."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from datetime import datetime
from typing import Any

from ..protocol.models import Trust, TOOL_SCHEMAS, TokNode, build_tok_traceback
from ..protocol.parser import serialize


# Import NormalizedToolEvent locally to avoid circular imports
def _get_normalized_tool_event() -> Any:
    """Lazy import to avoid circular dependency."""
    from .core import NormalizedToolEvent

    return NormalizedToolEvent


# Optional imports - set to None if not available
TokDelta: type[Any] | None = None
TokDeltaTracker: type[Any] | None = None
apply_delta: Any = None
delta_to_tok: Any = None
diff_tok: Any = None
Sifter: type[Any] | None = None

try:
    from ..utils.delta import (
        TokDelta,
        TokDeltaTracker,
        apply_delta,
        delta_to_tok,
        diff_tok,
    )
    from ..utils.sifter import Sifter
except ImportError:
    TokDelta = None
    apply_delta = None
    diff_tok = None
    delta_to_tok = None
    TokDeltaTracker = None
    Sifter = None


class RuntimeToolExecutor:
    """Transport-agnostic tool execution with security and logging."""

    def __init__(
        self,
        log_path: str = "execution.log",
        workspace_root: str | None = None,
    ):
        self.log_path = log_path
        self.workspace_root = workspace_root or os.getcwd()

        # Delta tracking for semantic diffs
        self.delta_tracker = (
            TokDeltaTracker() if TokDeltaTracker is not None else None
        )
        self._pre_state: dict[str, str] = {}
        self._pending_deltas: list[Any] = []
        self.track_file_changes = True
        self._sifter_cache: dict[str, str] = {}

    def _is_safe_path(self, path: str) -> bool:
        """Restrictions disabled: Returns True for all paths."""
        return True

    def _is_safe_rm(self, cmd: str) -> bool:
        """Restrictions disabled: Returns True for all rm commands."""
        return True

    def _compiler_guard(
        self, tool_name: str, attrs: dict[str, Any], node: Any
    ) -> str | None:
        """Validate a parsed tool node before execution. Returns Tok Traceback or None."""
        node_text = node.text
        from pydantic import ValidationError

        schema_cls = TOOL_SCHEMAS.get(tool_name)
        if schema_cls is None:
            return None
        input_data = attrs

        self._apply_cli_style_attrs(tool_name, node_text, input_data)
        self._fill_missing_attributes(tool_name, input_data, node_text)
        try:
            if tool_name == "edit" and node.children:
                pass
            else:
                schema_cls(**input_data)

            drift_field = self._detect_attr_drift(tool_name, attrs, node_text)
            if drift_field:
                return self._build_drift_error(
                    tool_name, attrs, node_text, drift_field
                )
            return None
        except ValidationError as exc:
            return build_tok_traceback(tool_name, repr(input_data)[:80], exc)

    def _apply_cli_style_attrs(
        self, tool_name: str, node_text: str, input_data: dict[str, Any]
    ) -> None:
        if "--" not in node_text or tool_name not in (
            "read",
            "write",
            "search",
            "grep",
            "grep_search",
        ):
            return
        try:
            tokens = shlex.split(node_text.strip())
            for i in range(len(tokens) - 1):
                key = tokens[i].lstrip("-").replace("-", "_")
                val = tokens[i + 1]
                if key not in input_data:
                    input_data[key] = val
                    if input_data.get("text") == node_text.strip():
                        del input_data["text"]
        except Exception:
            pass

    def _fill_missing_attributes(
        self, tool_name: str, input_data: dict[str, Any], node_text: str
    ) -> None:
        stripped = node_text.strip()
        if tool_name in ("read", "write") and not input_data.get("path"):
            input_data["path"] = stripped.split("\n")[0]
        if tool_name == "run" and not input_data.get("cmd"):
            input_data["cmd"] = stripped
        if (
            tool_name in ("search", "grep", "grep_search")
            and not input_data.get("query")
            and not input_data.get("text")
        ):
            input_data["query"] = stripped

    def _drift_fields(self, tool_name: str) -> list[str]:
        if tool_name == "write":
            return ["text"]
        if tool_name == "edit":
            return ["search", "replace"]
        return []

    def _detect_attr_drift(
        self, tool_name: str, attrs: dict[str, Any], node_text: str
    ) -> str:
        fields = self._drift_fields(tool_name)
        for field_name in fields:
            val = attrs.get(field_name, "")
            if not isinstance(val, str):
                continue
            if (
                "\n" in val
                or "\n" in val
                or val.strip().startswith("|>")
                or val.strip().startswith(">")
                or ("def " in val and ":" in val)
                or ("class " in val and ":" in val)
            ):
                return field_name
        if fields and node_text.strip().startswith("|>"):
            return fields[-1]
        return ""

    def _format_payload(self, payload: str) -> str:
        payload = re.sub(r"^\|>\s*", "", payload, flags=re.MULTILINE)
        payload = re.sub(r"^>\s*", "", payload, flags=re.MULTILINE)
        return (
            payload.replace("\n", "\n").replace("\r", "").replace("\t", "    ")
        )

    def _build_drift_error(
        self,
        tool_name: str,
        attrs: dict[str, Any],
        node_text: str,
        field_name: str,
    ) -> str:
        payload = self._format_payload(
            attrs.get("text") or attrs.get("replace") or node_text
        )
        corrected_node = TokNode(
            type="tool",
            label=tool_name,
            attrs={
                k: v
                for k, v in attrs.items()
                if k not in self._drift_fields(tool_name)
            },
            text=payload,
            trust=Trust.UNTRUSTED,
        )
        corrected_syntax = serialize([corrected_node])
        field_label = field_name or "text"
        return (
            f"@error type:protocol_drift\n"
            f"  msg: Multi_line_payloads_must_use_the_inverted_node_body_syntax.\n"
            f"  fix: REMOVE_the_{field_label}_attribute_entirely_and_place_the_content_below_the_header_starting_with_|>.\n"
            f"  corrected_syntax |>\n{corrected_syntax}"
        )

    def _capture_snapshot(self, path: str) -> str | None:
        """Capture a Tok skeleton snapshot of a file for delta tracking."""
        if (
            not self.track_file_changes
            or not self.delta_tracker
            or Sifter is None
        ):
            return None
        try:
            if path not in self._sifter_cache:
                self._sifter_cache[path] = Sifter.from_file(path)["skeleton"]
            return self._sifter_cache[path]
        except Exception:
            return None

    def _compute_delta(self, path: str, post_state: str) -> list[Any]:
        """Compute delta between pre and post snapshots."""
        if not self.delta_tracker or not post_state or diff_tok is None:
            return []
        pre_state = self._pre_state.get(path)
        if pre_state and pre_state != post_state:
            deltas: list[Any] = diff_tok(pre_state, post_state)
            return deltas
        return []

    def _log_execution(
        self, cmd: str, stdout: str, stderr: str, returncode: int
    ) -> None:
        """Log command execution details to a file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"\n{'=' * 20}\n[{timestamp}] COMMAND: {cmd}\nEXIT CODE: {returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}\n{'=' * 20}\n"

        # Simple rotation: keep the log file under a reasonable size (e.g., 500KB)
        try:
            if (
                os.path.exists(self.log_path)
                and os.path.getsize(self.log_path) > 500 * 1024
            ):
                with open(self.log_path) as f:
                    lines = f.readlines()
                # Keep last 1000 lines
                with open(self.log_path, "w") as f:
                    f.writelines(lines[-1000:])
        except Exception:
            pass

        with open(self.log_path, "a") as f:
            f.write(log_entry)

    def execute_normalized_tool(self, event: Any) -> dict[str, Any]:
        """Execute a normalized tool event and return results."""
        # Lazy import to avoid circular dependency
        NormalizedToolEvent = _get_normalized_tool_event()

        # Ensure we have the right type
        if not isinstance(event, NormalizedToolEvent):
            # Convert dict to NormalizedToolEvent if needed
            if isinstance(event, dict):
                event = NormalizedToolEvent(**event)
            else:
                return {
                    "status": "ERROR",
                    "message": f"Invalid event type: {type(event)}",
                }

        try:
            if event.name == "read":
                return self._execute_read(event)
            elif event.name == "write":
                return self._execute_write(event)
            elif event.name == "edit":
                return self._execute_edit(event)
            elif event.name == "run":
                return self._execute_run(event)
            elif event.name == "delta":
                return self._execute_delta(event)
            else:
                return {
                    "status": "ERROR",
                    "message": f"Unknown tool: {event.name}",
                }
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"Tool execution failed: {str(e)}",
            }

    def _execute_read(self, event: Any) -> dict[str, Any]:
        """Execute read tool with range support."""
        path = event.path
        if not path:
            return {"status": "ERROR", "message": "Missing path for read tool"}

        path = path.strip().rstrip(".*?:;")

        if not self._is_safe_path(path):
            return {
                "status": "ERROR",
                "message": f"Security Violation: {path}",
            }

        if not os.path.exists(path):
            return {"status": "ERROR", "message": f"Not found: {path}"}

        if os.path.isdir(path):
            return {
                "status": "ERROR",
                "message": f"{path} is a directory. Use run tool with 'ls -R {path}' to list contents.",
            }

        try:
            with open(path) as f:
                # Handle range if provided in args
                start = event.args.get("start") or event.args.get("L")
                end = event.args.get("end")

                if start and end:
                    start, end = int(start), int(end)
                    lines = f.readlines()
                    content = "".join(
                        lines[max(0, start - 1) : min(len(lines), end)]
                    )
                    result = f"Read {path} [L{start}-L{end}]:\n{content}"
                else:
                    content = f.read()
                    result = f"Read {path}:\n{content}"

            print(f"[!] Tool Executed: Read {path}")
            return {"status": "SUCCESS", "message": result}
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"Failed to read {path}: {str(e)}",
            }

    def _execute_write(self, event: Any) -> dict[str, Any]:
        """Execute write tool with inversion preference."""
        path = event.path
        if not path:
            return {
                "status": "ERROR",
                "message": "Missing path for write tool",
            }

        # Get content from args or fall back to empty string
        content = event.args.get("content") or event.args.get("text") or ""

        # Validate path: must be alphanumeric, dots, underscores, hyphens, forward
        # slashes
        if not re.match(r"^[\w\.\-/]+$", path):
            return {
                "status": "ERROR",
                "message": f"Invalid filename: {path}. Use alphanumeric, dots, underscores, hyphens only.",
            }

        if not self._is_safe_path(path):
            return {
                "status": "ERROR",
                "message": f"Security Violation: {path}",
            }

        try:
            # SNAPSHOT: Capture state before write
            pre_state = self._capture_snapshot(path)

            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w") as f:
                f.write(content)

            # SNAPSHOT: Capture state after write and compute delta
            if pre_state is not None:
                post_state = self._capture_snapshot(path)
                if post_state:
                    deltas = self._compute_delta(path, post_state)
                    if deltas:
                        self._pending_deltas.extend(deltas)

            print(f"[!] Tool Executed: Wrote to {path}")
            return {"status": "SUCCESS", "message": f"Wrote to {path}"}
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"Failed to write to {path}: {str(e)}",
            }

    def _execute_edit(self, event: Any) -> dict[str, Any]:
        """Execute edit tool with multi-line support."""
        path = event.path
        if not path:
            return {"status": "ERROR", "message": "Missing path for edit tool"}

        search = event.args.get("search") or event.args.get("before")
        replace = event.args.get("replace") or event.args.get("after")

        if search is None or replace is None:
            return {
                "status": "ERROR",
                "message": "Edit tool requires both search and replace",
            }

        if not self._is_safe_path(path):
            return {
                "status": "ERROR",
                "message": f"Security Violation: {path}",
            }

        if not os.path.exists(path):
            return {"status": "ERROR", "message": f"Not found: {path}"}

        try:
            # SNAPSHOT: Capture state before edit
            pre_state = self._capture_snapshot(path)

            with open(path) as f:
                fc = f.read()

            if search not in fc:
                return {
                    "status": "ERROR",
                    "message": f"Search string not found in {path}",
                }

            with open(path, "w") as f:
                f.write(fc.replace(search, replace, 1))

            # SNAPSHOT: Capture state after edit and compute delta
            if pre_state is not None:
                post_state = self._capture_snapshot(path)
                if post_state:
                    deltas = self._compute_delta(path, post_state)
                    if deltas:
                        self._pending_deltas.extend(deltas)

            print(f"[!] Tool Executed: Edited {path}")
            return {"status": "SUCCESS", "message": f"Edited {path}"}
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"Failed to edit {path}: {str(e)}",
            }

    def _execute_run(self, event: Any) -> dict[str, Any]:
        """Execute run tool with logging."""
        cmd = event.command
        if not cmd:
            return {
                "status": "ERROR",
                "message": "Missing command for run tool",
            }

        try:
            # Detect shell metacharacters or builtins that require shell=True
            shell_metachars = [
                "|",
                "&&",
                "||",
                ";",
                ">",
                "<",
                "*",
                "?",
                "[",
                "]",
                "(",
                ")",
                "$",
                "`",
                "\n",
            ]
            shell_builtins = ["cd ", "export ", "source ", "alias "]
            use_shell = any(c in cmd for c in shell_metachars) or any(
                b in cmd for b in shell_builtins
            )

            if use_shell:
                proc = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True
                )
            else:
                args = shlex.split(cmd)
                proc = subprocess.run(args, capture_output=True, text=True)

            self._log_execution(cmd, proc.stdout, proc.stderr, proc.returncode)

            output = proc.stdout if proc.stdout else proc.stderr
            if output:
                lines = output.strip().split("\n")
                snippet = "\n".join(lines[:5])
                print(f"[*] Output Snippet:\n{snippet}")
                if len(lines) > 5:
                    print(
                        f"[...] ({len(lines) - 5} more lines in execution.log)"
                    )

            res_msg = f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            status = "SUCCESS" if proc.returncode == 0 else "FAILURE"
            return {
                "status": status,
                "message": f"Command `{cmd}` exit {proc.returncode}:\n{res_msg}",
                "returncode": proc.returncode,
            }
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"Failed to run command: {str(e)}",
            }

    def _execute_delta(self, event: Any) -> dict[str, Any]:
        """Execute delta tool for structural code changes."""
        path = event.path
        if not path:
            return {
                "status": "ERROR",
                "message": "Missing path for delta tool",
            }

        delta_payload = self._resolve_delta_payload(event)
        if not delta_payload:
            return {"status": "ERROR", "message": "Missing delta payload"}

        if not self._is_safe_path(path):
            return {
                "status": "ERROR",
                "message": f"Security Violation: {path}",
            }

        if TokDelta is None or apply_delta is None:
            return {
                "status": "ERROR",
                "message": "Delta functionality not available",
            }

        try:
            pre_state = self._capture_snapshot(path)
            deltas = self._parse_delta_payload(delta_payload, path)
            if deltas:
                apply_delta(pre_state, deltas)
                self._pending_deltas.extend(deltas)
                return {
                    "status": "SUCCESS",
                    "message": f"Applied {len(deltas)} deltas to {path}",
                }
            return {
                "status": "ERROR",
                "message": "No valid deltas found in payload",
            }
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"Failed to apply delta: {e}",
            }

    def _resolve_delta_payload(self, event: Any) -> str:
        payload = event.args.get("payload") or ""
        if not payload and hasattr(event, "args"):
            payload = event.args.get("text") or event.args.get("content") or ""
        return payload

    def _parse_delta_payload(self, delta_payload: str, path: str) -> list[Any]:
        if TokDelta is None:
            return []
        deltas: list[Any] = []
        current_delta: Any = None
        for line in delta_payload.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            add_match = re.match(r"add\s+@(\w+)\|(\w+)", stripped)
            rem_match = re.match(r"remove\s+@(\w+)\|(\w+)", stripped)
            upd_match = re.match(r"update\s+@(\w+)\|(\w+)", stripped)

            if add_match:
                current_delta = TokDelta(
                    op="add",
                    target_type=add_match.group(1),
                    target_label=add_match.group(2),
                    file=path,
                )
                deltas.append(current_delta)
            elif rem_match:
                current_delta = TokDelta(
                    op="remove",
                    target_type=rem_match.group(1),
                    target_label=rem_match.group(2),
                    file=path,
                )
                deltas.append(current_delta)
            elif upd_match:
                current_delta = TokDelta(
                    op="update",
                    target_type=upd_match.group(1),
                    target_label=upd_match.group(2),
                    file=path,
                )
                deltas.append(current_delta)
            elif (
                stripped.startswith("+") and ":" in stripped and current_delta
            ):
                k, v = stripped[1:].split(":", 1)
                current_delta.new_attrs[k.strip()] = v.strip()
        return deltas

    def get_pending_deltas(self) -> list[Any]:
        """Get pending deltas for processing."""
        return self._pending_deltas

    def clear_pending_deltas(self) -> None:
        """Clear pending deltas after processing."""
        self._pending_deltas.clear()


# Global executor instance for backward compatibility
_default_executor = None


def get_default_executor() -> RuntimeToolExecutor:
    """Get or create the default tool executor."""
    global _default_executor
    if _default_executor is None:
        _default_executor = RuntimeToolExecutor()
    return _default_executor


def execute_normalized_tool(event: Any) -> dict[str, Any]:
    """Convenience function to execute a normalized tool event."""
    executor = get_default_executor()
    return executor.execute_normalized_tool(event)
