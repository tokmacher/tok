from __future__ import annotations
import collections
from .ir import Instruction, TokIR, Macro, MacroProvenance, MacroRegistry


class IRPatternMiner:
    def __init__(self, min_frequency: int = 3):
        self.min_frequency = min_frequency

    @staticmethod
    def _get_dependency_fingerprint(
        instructions: list[Instruction],
        start: int,
        length: int,
    ) -> tuple[tuple[int, int], ...]:
        """Return a tuple of (producer_slot, consumer_slot) pairs for the slice.

        A pair (i, j) means the instruction at slice-relative position j uses
        the *target* variable produced by the instruction at position i.  This
        fingerprint distinguishes sequences that look identical at the op level
        but wire data differently (e.g. A→B→C(A.result) vs A→B→C(B.result)).
        """
        slice_ins = instructions[start : start + length]
        # Map target name → slice-relative index of the producer
        target_to_slot: dict[str, int] = {}
        for idx, ins in enumerate(slice_ins):
            if ins.target:
                target_to_slot[ins.target] = idx

        edges: list[tuple[int, int]] = []
        for consumer_idx, ins in enumerate(slice_ins):
            for arg in ins.args:
                if isinstance(arg, str) and arg.startswith("$"):
                    var_name = arg[1:]
                    if var_name in target_to_slot:
                        edges.append((target_to_slot[var_name], consumer_idx))
        return tuple(sorted(edges))

    def mine(
        self,
        histories: list[TokIR],
        registry: MacroRegistry | None = None,
    ) -> list[Macro]:
        """Find frequent instruction sequences across a list of IR histories.

        Pattern keys now include a dependency fingerprint so that two n-gram
        sequences with identical op names but different data-flow edges are
        counted and registered as distinct macros.
        """
        if not histories:
            return []

        # 1. Flatten instruction lists, keeping full Instruction objects for dep analysis
        all_instructions: list[list[Instruction]] = [
            list(ir.instructions) for ir in histories
        ]
        all_ops: list[list[str]] = [
            [ins.op for ins in ins_list] for ins_list in all_instructions
        ]

        # 2. Find frequent (op-sequence, dep-fingerprint) pairs of length 2 or 3
        patterns: collections.Counter[
            tuple[tuple[str, ...], tuple[tuple[int, int], ...]]
        ] = collections.Counter()
        for ins_list, ops in zip(all_instructions, all_ops, strict=True):
            for length in (2, 3):
                for i in range(len(ops) - length + 1):
                    op_key = tuple(ops[i : i + length])
                    dep_fp = self._get_dependency_fingerprint(
                        ins_list, i, length
                    )
                    patterns[(op_key, dep_fp)] += 1

        frequent = [
            (op_key, dep_fp)
            for (op_key, dep_fp), count in patterns.items()
            if count >= self.min_frequency
        ]

        # 3. Filter out sub-sequences by op_key only (dep_fp varies per slice).
        # If (A,B) and (A,B,C) both frequent, keep (A,B,C).
        frequent.sort(key=lambda x: len(x[0]), reverse=True)
        filtered_frequent: list[
            tuple[tuple[str, ...], tuple[tuple[int, int], ...]]
        ] = []
        for op_key, dep_fp in frequent:
            is_subseq = False
            for longer_op_key, _ in filtered_frequent:
                for i in range(len(longer_op_key) - len(op_key) + 1):
                    if longer_op_key[i : i + len(op_key)] == op_key:
                        is_subseq = True
                        break
                if is_subseq:
                    break
            if not is_subseq:
                filtered_frequent.append((op_key, dep_fp))

        macros = []
        next_idx = 0
        if registry:
            # Find the highest auto_macro index to avoid collisions
            for name in registry.macros:
                if name.startswith("auto_macro_"):
                    try:
                        idx = int(name.split("_")[-1])
                        next_idx = max(next_idx, idx + 1)
                    except ValueError:
                        pass

        for i, (ops_seq, dep_fp) in enumerate(filtered_frequent):
            # Find the first concrete example of this (op-sequence, dep-fingerprint) pair
            found_ins: list[Instruction] = []
            for ins_list in all_instructions:
                ir_ops = [ins.op for ins in ins_list]
                for start in range(len(ir_ops) - len(ops_seq) + 1):
                    if tuple(ir_ops[start : start + len(ops_seq)]) == ops_seq:
                        candidate = ins_list[start : start + len(ops_seq)]
                        if (
                            self._get_dependency_fingerprint(
                                ins_list, start, len(ops_seq)
                            )
                            == dep_fp
                        ):
                            found_ins = candidate
                            break
                if found_ins:
                    break

            if not found_ins:
                continue

            # Parameterization: identify path-like arguments and replace with $pN.
            # Also collect context_requirements from the first concrete path arg
            # (the file the pattern was originally mined from).
            param_inputs: list[str] = []
            arg_to_param: dict[str, str] = {}
            context_file: str | None = None
            final_instructions: list[Instruction] = []
            for ins in found_ins:
                new_args: list[str] = []
                for arg in ins.args:
                    if isinstance(arg, str) and len(arg) > 3 and "/" in arg:
                        if context_file is None:
                            context_file = (
                                arg  # record first path as context hint
                            )
                        if arg not in arg_to_param:
                            p_name = f"p{len(param_inputs)}"
                            arg_to_param[arg] = f"${p_name}"
                            param_inputs.append(p_name)
                        new_args.append(arg_to_param[arg])
                    else:
                        new_args.append(arg)
                final_instructions.append(
                    Instruction(
                        op=ins.op, args=tuple(new_args), target=ins.target
                    )
                )

            # Encode dep fingerprint in provenance source_code for traceability
            dep_str = (
                "|".join(f"{p}->{c}" for p, c in dep_fp) if dep_fp else "none"
            )
            new_macro = Macro(
                name=f"auto_macro_{next_idx + i}",
                instructions=tuple(final_instructions),
                inputs=tuple(param_inputs),
                context_requirements=(
                    {"file": context_file} if context_file else {}
                ),
                provenance=MacroProvenance(
                    source_code=(
                        " -> ".join(ins.op for ins in found_ins)
                        + f" [deps:{dep_str}]"
                    ),
                    composed_of=tuple(
                        ins.op[1:]
                        for ins in found_ins
                        if ins.op.startswith("@")
                    ),
                ),
            )

            # Deduplication: same op sequence → skip to prevent cross-session bloat
            if registry and registry.find_op_sequence_duplicate(new_macro):
                continue

            # Skip single-instruction macros that wrap an existing macro call
            if len(found_ins) == 1 and found_ins[0].op.startswith("@"):
                continue

            macros.append(new_macro)

        return macros
