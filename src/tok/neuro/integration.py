"""Native integration bridging the Tok runtime memory with the Pattern Reactor."""

from __future__ import annotations

import logging
import shlex
from typing import TYPE_CHECKING

from tok.neuro.ir import Instruction, Macro, TokIR
from tok.neuro.miner import IRPatternMiner

if TYPE_CHECKING:
    from tok.runtime.memory.bridge_memory import BridgeMemoryState

logger = logging.getLogger("tok.neuro.integration")


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

    # 1. Extract command history from rolling memory (chronological)
    cmds = memory_state.rolling_cmds
    if not cmds:
        logger.debug("NeuroReactor: No rolling command history to distill.")
        return []

    # Sort by last_seen_turn to roughly approximate sequence
    # Since rolling_cmds is already chronological, we just use enumeration to be extra safe
    enumerated_cmds = list(enumerate(cmds))
    sorted_cmds = sorted(enumerated_cmds, key=lambda x: (x[1].last_seen_turn, x[0]))
    instructions = []

    for _, entry in sorted_cmds:
        cmd_str = entry.value.strip()
        if not cmd_str:
            continue

        # Parse instruction: First word is the operation, rest is args.
        # Use shlex.split to handle quoted arguments; fall back to whitespace split.
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

    # Wrap the entire hot-memory sequence into a single TokIR history blob
    # In a fully accurate system, this would be divided into discrete turn episodes.
    ir = TokIR(instructions=tuple(instructions))

    logger.info("NeuroReactor: op_seq = %s", [ins.op for ins in instructions])
    logger.debug("NeuroReactor: Mining %d bridged instructions.", len(instructions))
    discovered = miner.mine([ir], registry=memory_state.macro_registry)

    if discovered:
        logger.info(
            "NeuroReactor: Distilled %d new macros from bridge history.",
            len(discovered),
        )
        import datetime

        # Pick the canonical project marker (alphabetically first) to tag macros with.
        primary_marker = min(project_markers) if project_markers else None

        for macro in discovered:
            macro.last_seen = datetime.datetime.now().isoformat()
            # Enrich context_requirements with the project marker when available and
            # the macro doesn't already carry a file-specific requirement.
            if primary_marker and not macro.context_requirements.get("file"):
                macro.context_requirements["marker_file"] = primary_marker
                logger.debug(
                    "NeuroReactor: Tagged @%s with marker_file=%s",
                    macro.name,
                    primary_marker,
                )
            memory_state.macro_registry.register(macro)

        memory_state.macro_registry.save_global()

    return discovered
