"""Macro JIT execution and self-healing logic."""

import logging
import re
from typing import TYPE_CHECKING, Any

from tok.utils.event_logging import log_macro_used

if TYPE_CHECKING:
    from tok.runtime.core import RuntimeSession
    from tok.runtime.memory.tok_state import BridgeMemoryState

logger = logging.getLogger("tok.runtime")


def _parse_jit_args(args_raw: str) -> dict[str, Any]:
    """Naive parser for JIT macro arguments in 'key=value, ...' format."""
    inputs: dict[str, Any] = {}
    if not args_raw.strip():
        return inputs
    for part in args_raw.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            # Strip whitespace and common quotes
            v = v.strip().strip("'").strip('"')
            inputs[k.strip()] = v
    return inputs


def _attribute_macro_savings(session: "RuntimeSession", wire_state: str) -> None:
    """Credit token savings to macros referenced in cmds field of a wire state."""
    from tok.runtime.pipeline.tool_processing import count_tokens

    registry = session.bridge_memory.macro_registry
    if not registry.macros or not wire_state:
        return

    # Extract cmds field value from >>> t:X|g:Y|cmds:...|... format
    # cmds field uses alias 'c' (TOK_FIELD_ALIAS maps "cmds" → "c")
    cmds_match = re.search(r"(?:cmds:|c:)([^|>\n]+)", wire_state)
    if not cmds_match:
        return

    cmds_text = cmds_match.group(1)
    for ref_match in re.finditer(r"@(\w+)\(([^)]*)\)", cmds_text):
        macro_name = ref_match.group(0)[1 : ref_match.group(0).index("(")]
        macro = registry.get(macro_name)
        if not macro:
            continue

        # Expansion text: full instruction sequence as it would appear un-compressed
        expanded = " | ".join(f"{ins.op}({', '.join(str(a) for a in ins.args)})" for ins in macro.instructions)
        reference = ref_match.group(0)  # e.g. "@auto_macro_0(src/foo.py)"
        savings = count_tokens(expanded) - count_tokens(reference)
        if savings > 0:
            registry.record_savings(macro_name, savings)
            log_macro_used(macro_name, tokens_saved=savings)
            session.pending_behavior_signals["macro_savings_attributed"] = (
                session.pending_behavior_signals.get("macro_savings_attributed", 0) + 1
            )


def _heal_macro_from_repair(macro_name: str, memory_state: "BridgeMemoryState", heal_turn: int = 0) -> None:
    """Detect if the agent diverged from a JIT-offered macro and update it."""
    from tok.macros.ir import Instruction

    macro = memory_state.macro_registry.get(macro_name)
    if not macro:
        return

    # Extract the most recent instructions from rolling_cmds.
    recent_cmds = [entry for entry in memory_state.rolling_cmds if entry.last_seen_turn > heal_turn]

    if not recent_cmds:
        return

    recent_ins: list[Instruction] = []
    for entry in recent_cmds:
        parts = entry.value.strip().split()
        if not parts:
            continue
        op = parts[0]
        # Ignore tool-compatible noise (tok_*) and existing macro calls (@*)
        if op.startswith(("tok_", "@")):
            continue
        recent_ins.append(Instruction(op=op, args=tuple(parts[1:])))

    if not recent_ins:
        return

    if memory_state.macro_registry.update_from_repair(macro_name, tuple(recent_ins)):
        memory_state.macro_registry.save_global()


def _jit_context_matches(macro: Any, session: "RuntimeSession") -> bool:
    """Return True when the macro's context_requirements are satisfied by the session."""
    reqs: dict[str, str] = getattr(macro, "context_requirements", {}) or {}
    if not reqs:
        return True
    required_file = reqs.get("file")
    if required_file:
        top_files = session.bridge_memory.top_hot_files(n=5)
        if required_file not in top_files:
            return False
    required_marker = reqs.get("marker_file")
    if required_marker:
        session_markers: frozenset[str] = getattr(session, "_project_markers", frozenset())
        if required_marker not in session_markers:
            return False
    return True


def execute_jit_macro(session: "RuntimeSession", macro_name: str, args_raw: str) -> str:
    """Symbolically execute a macro in the current session context."""
    macro = session.bridge_memory.macro_registry.get(macro_name)
    if not macro:
        return f"Error: Macro @{macro_name} not found in registry."

    inputs = _parse_jit_args(args_raw)

    try:
        from tok.macros.ir import TokIR, execute_ir

        result = execute_ir(
            TokIR(macro.instructions),
            inputs,
            session.bridge_memory.macro_registry,
        )
        # Record use in registry
        session.bridge_memory.macro_registry.record_use(macro_name)
        log_macro_used(macro_name)
        return str(result)
    except Exception as e:
        logger.exception("JIT execution failure for @%s: %s", macro_name, e)
        return f"Error during JIT execution of @{macro_name}: {e}"
