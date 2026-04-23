"""
Native integration bridging the Tok runtime memory with the Pattern Reactor.

This module provides the bridge between Tok's runtime memory (bridge_memory)
and the macro pattern mining system, enabling discovery of repeated command
patterns from bridge history.
"""

from __future__ import annotations

import datetime
import logging
import shlex
from typing import TYPE_CHECKING

from tok.macros.ir import Instruction, Macro, TokIR
from tok.macros.miner import IRPatternMiner

if TYPE_CHECKING:
    from tok.runtime.memory.bridge_memory import BridgeMemoryState

logger = logging.getLogger("tok.macros.integration")


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
        miner = IRPatternMiner(min_frequency=3)

    cmds = memory_state.rolling_cmds
    if not cmds:
        logger.debug("PatternReactor: No rolling command history to distill.")
        return []

    enumerated_cmds = list(enumerate(cmds))
    sorted_cmds = sorted(enumerated_cmds, key=lambda x: (x[1].last_seen_turn, x[0]))
    instructions = []

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

        op = parts[0]
        args = tuple(parts[1:])
        instructions.append(Instruction(op=op, args=args, target=None))

    if not instructions:
        return []

    ir = TokIR(instructions=tuple(instructions))

    logger.info("PatternReactor: op_seq = %s", [ins.op for ins in instructions])
    logger.debug("PatternReactor: Mining %d bridged instructions.", len(instructions))
    discovered = miner.mine([ir], registry=memory_state.macro_registry)

    if discovered:
        logger.info(
            "PatternReactor: Distilled %d new macros from bridge history.",
            len(discovered),
        )

        primary_marker = min(project_markers) if project_markers else None

        for macro in discovered:
            macro.last_seen = datetime.datetime.now().isoformat()
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
