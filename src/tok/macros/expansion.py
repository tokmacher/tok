"""Macro-to-tool_use expansion engine.

Converts a Macro's IR instructions into Anthropic tool_use content blocks,
resolving $variable placeholders against provided parameter bindings.
"""

from __future__ import annotations

import uuid
from typing import Any

from tok.macros.ir import Macro, MacroRegistry
from tok.macros.tool_map import lookup, resolve_args


def _make_tool_id() -> str:
    return f"toolu_macro_{uuid.uuid4().hex[:12]}"


def expand_macro(macro: Macro, params: dict[str, str]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for ins in macro.instructions:
        mapping = lookup(ins.op)
        if mapping is None:
            continue
        try:
            tool_input = resolve_args(ins, params)
        except KeyError:
            continue
        blocks.append(
            {
                "type": "tool_use",
                "id": _make_tool_id(),
                "name": mapping.tool_name,
                "input": tool_input,
            }
        )
    return blocks


def expand_macro_tool_use_block(
    block: dict[str, Any],
    registry: MacroRegistry,
) -> list[dict[str, Any]]:
    if block.get("type") != "tool_use":
        return [block]

    name = block.get("name", "")
    if not name.startswith("@"):
        return [block]

    macro_name = name[1:]
    macro = registry.get(macro_name)
    if macro is None:
        return [block]

    params: dict[str, str] = {}
    input_data = block.get("input", {})
    for key in macro.inputs:
        if key in input_data:
            params[key] = str(input_data[key])

    expanded = expand_macro(macro, params)
    if not expanded:
        return [block]

    return expanded


def expand_tool_use_blocks(
    blocks: list[dict[str, Any]],
    registry: MacroRegistry,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for block in blocks:
        result.extend(expand_macro_tool_use_block(block, registry))
    return result


def macro_hint_for_session(registry: MacroRegistry) -> str | None:
    macros = list(registry.macros.values())
    if not macros:
        return None

    high_hit = sorted(macros, key=lambda m: -m.hit_count)[:3]
    lines: list[str] = []
    for m in high_hit:
        sig = f"@{m.name}({', '.join(m.inputs)})"
        ops = " -> ".join(ins.op + "(...)" for ins in m.instructions)
        lines.append(f"  {sig} = {ops} (used {m.hit_count}x)")

    return "Registered macros:\n" + "\n".join(lines)
