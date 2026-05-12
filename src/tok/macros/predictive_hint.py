"""Predictive macro hints — lightweight in-session result injection.

When a JIT macro pattern match fires, this module builds a hint telling Claude
which files/searches it has already performed, so it can skip redundant reads.
"""

from __future__ import annotations

from tok.macros.ir import Instruction, Macro
from tok.runtime.memory.bridge_memory import MemoryEntry

_READ_OPS = {"view", "cat", "read"}
_SEARCH_OPS = {"grep"}
_EDIT_OPS = {"edit"}
_SHELL_OPS = {"pytest", "bash", "ls", "npm"}


def _extract_args_from_cmd(cmd_str: str) -> tuple[str, tuple[str, ...]]:
    parts = cmd_str.strip().split()
    if not parts:
        return "", ()
    return parts[0], tuple(parts[1:])


def _extract_files_from_args(args: tuple[str, ...]) -> list[str]:
    files: list[str] = []
    for arg in args:
        if "/" in arg and not arg.startswith("-"):
            files.append(arg)
    return files


def _extract_search_term(args: tuple[str, ...]) -> str | None:
    for arg in args:
        if not arg.startswith("-") and not arg.startswith("$") and "/" not in arg:
            return arg
    return None


def build_predictive_hint(
    macro: Macro,
    recent_cmds: list[MemoryEntry],
) -> str | None:
    if not recent_cmds:
        return None

    macro_ops = [ins.op for ins in macro.instructions]
    macro_len = len(macro_ops)

    cmds_as_ops: list[str] = []
    cmds_as_entries: list[MemoryEntry] = []
    for entry in recent_cmds:
        op, _ = _extract_args_from_cmd(entry.value)
        if op:
            cmds_as_ops.append(op)
            cmds_as_entries.append(entry)

    best_offset = -1
    for i in range(len(cmds_as_ops) - macro_len + 1):
        if cmds_as_ops[i : i + macro_len] == macro_ops:
            best_offset = i

    if best_offset < 0:
        return None

    matched_entries = cmds_as_entries[best_offset : best_offset + macro_len]
    matched_instructions = list(macro.instructions)

    for trailing_idx in range(best_offset + macro_len, len(cmds_as_entries)):
        trailing_op = cmds_as_ops[trailing_idx]
        if trailing_op in _READ_OPS or trailing_op in _SEARCH_OPS:
            matched_entries.append(cmds_as_entries[trailing_idx])
            matched_instructions.append(Instruction(op=trailing_op, args=()))
        else:
            break

    search_terms: list[str] = []
    files: list[str] = []
    shell_commands: list[str] = []

    for ins, entry in zip(matched_instructions, matched_entries, strict=False):
        _, args = _extract_args_from_cmd(entry.value)

        if ins.op in _SEARCH_OPS:
            term = _extract_search_term(args)
            if term:
                search_terms.append(term)
            files.extend(_extract_files_from_args(args))

        elif ins.op in _READ_OPS:
            files.extend(_extract_files_from_args(args))

        elif ins.op in _EDIT_OPS:
            files.extend(_extract_files_from_args(args))

        elif ins.op in _SHELL_OPS:
            shell_commands.append(entry.value.strip())

    if not search_terms and not files and not shell_commands:
        return None

    parts: list[str] = []
    if search_terms:
        terms = ", ".join(f"'{t}'" for t in search_terms)
        parts.append(f"searched for {terms}")
    if files:
        unique_files = list(dict.fromkeys(files))
        parts.append(f"read {', '.join(unique_files)}")
    if shell_commands:
        parts.append(f"ran {', '.join(shell_commands)}")

    return f"Previously in this session: {', '.join(parts)}."
