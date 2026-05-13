"""Macro instruction argument parameterization helpers."""

from __future__ import annotations

from tok.macros.ir import Instruction

from .tool_map import OP_TOOL_MAP


def parameterize_instructions(
    instructions: list[Instruction],
) -> tuple[tuple[Instruction, ...], tuple[str, ...], str | None]:
    """Replace literal args with $pN variables and return (instructions, inputs, context_file).

    Prefer tool_map arg metadata when available; fall back to a simple path-like
    heuristic for unknown ops.
    """
    param_inputs: list[str] = []
    arg_to_param: dict[str, str] = {}
    context_file: str | None = None
    final_instructions: list[Instruction] = []

    for ins in instructions:
        mapping = OP_TOOL_MAP.get(ins.op)
        new_args: list[str] = []
        for idx, arg in enumerate(ins.args):
            if not isinstance(arg, str):
                new_args.append(str(arg))
                continue

            should_parameterize = False
            if mapping is not None and idx in mapping.arg_map:
                slot = mapping.arg_map[idx]
                should_parameterize = slot in {"file_path", "path", "pattern", "command"}
            else:
                should_parameterize = len(arg) > 3 and "/" in arg

            if should_parameterize:
                if context_file is None and "/" in arg:
                    context_file = arg
                if arg not in arg_to_param:
                    p_name = f"p{len(param_inputs)}"
                    arg_to_param[arg] = f"${p_name}"
                    param_inputs.append(p_name)
                new_args.append(arg_to_param[arg])
            else:
                new_args.append(arg)

        final_instructions.append(Instruction(op=ins.op, args=tuple(new_args), target=ins.target))

    return tuple(final_instructions), tuple(param_inputs), context_file
