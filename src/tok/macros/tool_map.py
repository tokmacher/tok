"""Static op-to-tool mapping for macro expansion.

Maps mined IR op names (view, grep, edit, pytest, ...) to Claude Code tool
schemas with positional arg marshalling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tok.macros.ir import Instruction


@dataclass(frozen=True)
class ToolMapping:
    tool_name: str
    arg_map: dict[int, str] = field(default_factory=dict)
    shell_template: str | None = None


OP_TOOL_MAP: dict[str, ToolMapping] = {
    "view": ToolMapping(tool_name="Read", arg_map={0: "file_path"}),
    "cat": ToolMapping(tool_name="Read", arg_map={0: "file_path"}),
    "read": ToolMapping(tool_name="Read", arg_map={0: "file_path"}),
    "edit": ToolMapping(tool_name="Edit", arg_map={0: "file_path"}),
    "grep": ToolMapping(tool_name="Grep", arg_map={0: "pattern", 1: "path"}),
    "ls": ToolMapping(tool_name="Bash", arg_map={}, shell_template="ls {args}"),
    "pytest": ToolMapping(tool_name="Bash", arg_map={}, shell_template="pytest {args}"),
    "bash": ToolMapping(tool_name="Bash", arg_map={}, shell_template="{args}"),
    "npm": ToolMapping(tool_name="Bash", arg_map={}, shell_template="npm {args}"),
}


def lookup(op_name: str) -> ToolMapping | None:
    return OP_TOOL_MAP.get(op_name)


def resolve_args(ins: Instruction, bindings: dict[str, str]) -> dict[str, Any]:
    mapping = lookup(ins.op)
    if mapping is None:
        msg = f"No tool mapping for op: {ins.op}"
        raise KeyError(msg)

    resolved_positional: list[str] = []
    for arg in ins.args:
        if isinstance(arg, str) and arg.startswith("$"):
            var_name = arg[1:]
            if var_name not in bindings:
                msg = f"Unbound variable ${var_name} in instruction {ins.op}"
                raise KeyError(msg)
            resolved_positional.append(bindings[var_name])
        else:
            resolved_positional.append(str(arg))

    if mapping.shell_template is not None:
        args_str = " ".join(resolved_positional)
        return {"command": mapping.shell_template.format(args=args_str)}

    result: dict[str, Any] = {}
    for idx, param_name in mapping.arg_map.items():
        if idx < len(resolved_positional):
            result[param_name] = resolved_positional[idx]
    return result
