"""
Native integration bridging the Tok runtime memory with the Pattern Reactor.

This module provides the bridge between Tok's runtime memory (bridge_memory)
and the macro pattern mining system, enabling discovery of repeated command
patterns from bridge history.
"""

from __future__ import annotations

import collections
import datetime
import logging
import shlex
from typing import TYPE_CHECKING

from tok.macros.ir import Instruction, Macro, MacroRegistry, TokIR
from tok.macros.miner import IRPatternMiner

if TYPE_CHECKING:
    from tok.runtime.memory.bridge_memory import BridgeMemoryState, MemoryEntry

logger = logging.getLogger("tok.macros.integration")


def _ops_already_registered(registry: MacroRegistry, op_key: tuple[str, ...]) -> bool:
    for macro in registry.macros.values():
        if tuple(ins.op for ins in macro.instructions) == op_key:
            return True
    return False


def _build_instructions_from_cmds(
    sorted_cmds: list[tuple[int, MemoryEntry]],
) -> list[Instruction]:
    instructions: list[Instruction] = []
    for _, entry in sorted_cmds:
        cmd_str = entry.value.strip()
        if not cmd_str:
            continue
        try:
            parts = shlex.split(cmd_str)
        except ValueError:
            parts = cmd_str.split()
        if not parts:
            continue
        instructions.append(Instruction(op=parts[0], args=tuple(parts[1:]), target=None))
    return instructions


def _build_batch_instructions(
    sorted_cmds: list[tuple[int, MemoryEntry]],
) -> tuple[list[Instruction], dict[int, list[Instruction]]]:
    per_turn: dict[int, list[Instruction]] = {}
    for _, entry in sorted_cmds:
        cmd_str = entry.value.strip()
        if not cmd_str:
            continue
        try:
            parts = shlex.split(cmd_str)
        except ValueError:
            parts = cmd_str.split()
        if not parts:
            continue
        t = entry.last_seen_turn
        per_turn.setdefault(t, []).append(Instruction(op=parts[0], args=tuple(parts[1:]), target=None))

    batch: list[Instruction] = []
    for t in sorted(per_turn.keys()):
        seen_ops: set[str] = set()
        for ins in per_turn[t]:
            if ins.op not in seen_ops:
                batch.append(ins)
                seen_ops.add(ins.op)
    return batch, per_turn


def distill_bridge_history(
    memory_state: BridgeMemoryState,
    miner: IRPatternMiner | None = None,
    project_markers: frozenset[str] | None = None,
) -> list[Macro]:
    """
    Extracts tool-call histories from the active bridge memory, converts them
    to an IR sequence, and mines them for repeated patterns.

    Any discovered macros are automatically registered back to the memory state.

    When *project_markers* is provided (e.g. ``frozenset({'package.json'})``),
    newly discovered macros have their ``context_requirements`` enriched with the
    first matching marker so they surface via speculative injection in future
    sessions that share the same project type (Local Mesh Discovery).
    """
    if miner is None:
        miner = IRPatternMiner(min_frequency=2)

    cmds = memory_state.rolling_cmds
    if not cmds:
        logger.debug("PatternReactor: No rolling command history to distill.")
        return []

    enumerated_cmds = list(enumerate(cmds))
    sorted_cmds = sorted(enumerated_cmds, key=lambda x: (x[1].last_seen_turn, x[0]))

    instructions = _build_instructions_from_cmds(sorted_cmds)
    batch_instructions, per_turn = _build_batch_instructions(sorted_cmds)

    if not instructions:
        return []

    ir = TokIR(instructions=tuple(instructions))

    turn_primary_ops: list[Instruction] = []
    for t in sorted(per_turn.keys()):
        if per_turn[t]:
            turn_primary_ops.append(per_turn[t][0])

    cross_turn_discovered: list[Macro] = []
    if len(turn_primary_ops) >= 2:
        pair_counts: collections.Counter[tuple[str, ...]] = collections.Counter()
        for length in (2, 3):
            if len(turn_primary_ops) >= length:
                pair_counts[tuple(ins.op for ins in turn_primary_ops[:length])] += 1
        for op_pair, count in pair_counts.items():
            if count >= 1 and not _ops_already_registered(memory_state.macro_registry, op_pair):
                cross_ins = turn_primary_ops[: len(op_pair)]
                param_inputs: list[str] = []
                arg_to_param: dict[str, str] = {}
                final_ins: list[Instruction] = []
                for ins in cross_ins:
                    new_args: list[str] = []
                    for arg in ins.args:
                        if isinstance(arg, str) and len(arg) > 3 and "/" in arg:
                            if arg not in arg_to_param:
                                p_name = f"p{len(param_inputs)}"
                                arg_to_param[arg] = f"${p_name}"
                                param_inputs.append(p_name)
                            new_args.append(arg_to_param[arg])
                        else:
                            new_args.append(arg)
                    final_ins.append(Instruction(op=ins.op, args=tuple(new_args), target=None))
                context_file: str | None = None
                for arg in arg_to_param:
                    if "/" in arg:
                        context_file = arg
                        break
                cross_turn_discovered.append(
                    Macro(
                        name=f"cross_turn_{len(op_pair)}",
                        instructions=tuple(final_ins),
                        inputs=tuple(param_inputs),
                        hit_count=3,
                        last_seen=datetime.datetime.now().isoformat(),
                        context_requirements=({"file": context_file} if context_file else {}),
                    )
                )

    batch_discovered: list[Macro] = []
    if len(batch_instructions) >= 2:
        batch_ir = TokIR(instructions=tuple(batch_instructions))
        batch_miner = IRPatternMiner(min_frequency=1)
        batch_discovered = batch_miner.mine([batch_ir], registry=memory_state.macro_registry)
    else:
        batch_discovered = miner.mine([ir], registry=memory_state.macro_registry)

    if not batch_discovered:
        for m in cross_turn_discovered:
            existing = memory_state.macro_registry.find_op_sequence_duplicate(m)
            if not existing:
                batch_discovered.append(m)

    discovered = batch_discovered

    logger.info(
        "PatternReactor: op_seq = %s",
        [ins.op for ins in instructions],
    )
    logger.debug(
        "PatternReactor: Mining %d bridged instructions (batch: %d).",
        len(instructions),
        len(batch_instructions),
    )

    if discovered:
        logger.info(
            "PatternReactor: Distilled %d new macros from bridge history.",
            len(discovered),
        )

        primary_marker = min(project_markers) if project_markers else None

        for macro in discovered:
            macro.last_seen = datetime.datetime.now().isoformat()
            if macro.hit_count < 3:
                macro.hit_count = 3
            if primary_marker and not macro.context_requirements.get("file"):
                macro.context_requirements["marker_file"] = primary_marker
                logger.debug(
                    "PatternReactor: Tagged @%s with marker_file=%s",
                    macro.name,
                    primary_marker,
                )
            memory_state.macro_registry.register(macro)

        memory_state.macro_registry.save_global()

    return discovered
