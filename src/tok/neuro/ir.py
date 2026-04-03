from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Iterable

from ..utils.event_logging import log_macro_registered


@dataclass(frozen=True)
class Instruction:
    op: str
    args: tuple[Any, ...]
    target: str | None = (
        None  # None means it's a 'return' or 'check' instruction
    )

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "args": list(self.args), "target": self.target}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Instruction:
        return cls(
            op=data["op"], args=tuple(data["args"]), target=data.get("target")
        )


@dataclass(frozen=True)
class MacroProvenance:
    source_code: str | None = None
    episode_id: str | None = None
    source_file: str | None = None
    composed_of: tuple[
        str, ...
    ] = ()  # Names of lower-level macros this abstracts

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_code": self.source_code,
            "composed_of": list(self.composed_of),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MacroProvenance:
        return cls(
            source_code=data.get("source_code"),
            composed_of=tuple(data.get("composed_of", [])),
        )


@dataclass(kw_only=True)
class Macro:
    name: str
    instructions: tuple[Instruction, ...]
    inputs: tuple[str, ...]
    provenance: MacroProvenance | None = None
    hit_count: int = 1
    last_seen: str | None = None  # ISO format timestamp
    is_durable: bool = False
    lifetime_savings: int = 0  # cumulative tokens saved by this macro
    avg_tokens_per_use: float = (
        0.0  # rolling average tokens saved per invocation
    )
    # Context requirements for JIT filtering, e.g. {"file": "src/tok/cli.py"}.
    # An empty dict means the macro applies in any context.
    context_requirements: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "instructions": [i.to_dict() for i in self.instructions],
            "inputs": list(self.inputs),
            "provenance": (
                self.provenance.to_dict() if self.provenance else None
            ),
            "hit_count": self.hit_count,
            "last_seen": self.last_seen,
            "is_durable": self.is_durable,
            "lifetime_savings": self.lifetime_savings,
            "avg_tokens_per_use": self.avg_tokens_per_use,
            "context_requirements": self.context_requirements,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Macro:
        return cls(
            name=data["name"],
            instructions=tuple(
                Instruction.from_dict(i) for i in data["instructions"]
            ),
            inputs=tuple(data["inputs"]),
            provenance=(
                MacroProvenance.from_dict(data["provenance"])
                if data.get("provenance")
                else None
            ),
            hit_count=data.get("hit_count", 1),
            last_seen=data.get("last_seen"),
            is_durable=data.get("is_durable", False),
            lifetime_savings=data.get("lifetime_savings", 0),
            avg_tokens_per_use=float(data.get("avg_tokens_per_use", 0.0)),
            context_requirements=dict(data.get("context_requirements") or {}),
        )


class MacroRegistry:
    def __init__(self) -> None:
        self.macros: dict[str, Macro] = {}

    def _is_duplicate(self, m1: Macro, m2: Macro) -> bool:
        """Checks if two macros are semantically identical."""
        return m1.instructions == m2.instructions and m1.inputs == m2.inputs

    def find_duplicate(self, macro: Macro) -> str | None:
        """Returns the name of an existing identical macro, if any."""
        for existing in self.macros.values():
            if self._is_duplicate(existing, macro):
                return existing.name
        return None

    def find_op_sequence_duplicate(self, macro: Macro) -> str | None:
        """Returns the name of an existing macro with the same op sequence, if any.

        Two macros are considered op-sequence duplicates if they share the same
        ordered list of operator names, regardless of concrete argument values.
        This prevents session-to-session macro bloat where the same workflow
        pattern is re-registered with different file paths each run.
        """
        new_ops = tuple(ins.op for ins in macro.instructions)
        for existing in self.macros.values():
            if tuple(ins.op for ins in existing.instructions) == new_ops:
                return existing.name
        return None

    def match_recent_sequence(
        self, instructions: list[Instruction]
    ) -> Macro | None:
        """
        Checks if the end of the provided sequence of instructions matches any
        known macro's instruction sequence (by operator names).
        Returns the macro if a high-confidence match is found.
        """
        if not instructions:
            return None

        # Extract op sequence from the input
        input_ops = [ins.op for ins in instructions]

        # Check each macro to see if its op sequence matches the end of input_ops
        for macro in self.macros.values():
            m_ops = [ins.op for ins in macro.instructions]
            if len(m_ops) < 2:  # Only JIT sequences of 2+ ops
                continue

            if input_ops[-len(m_ops) :] == m_ops:
                # High confidence match: the exact pattern was just completed
                return macro

        return None

    def register(self, macro: Macro) -> str:
        """
        Registers a macro. If an identical one or an op-sequence duplicate exists,
        returns the existing name. Otherwise, registers and returns the new name.
        """
        # First check for exact duplicate (same ops + same args)
        duplicate_name = self.find_duplicate(macro)
        if duplicate_name:
            return duplicate_name

        # Then check for op-sequence duplicate (same ops, different args)
        op_seq_duplicate = self.find_op_sequence_duplicate(macro)
        if op_seq_duplicate:
            return op_seq_duplicate

        self.macros[macro.name] = macro
        log_macro_registered(macro.name, source="mined")
        return macro.name

    def get(self, name: str) -> Macro | None:
        return self.macros.get(name)

    def record_use(self, name: str) -> None:
        if macro := self.macros.get(name):
            macro.hit_count += 1
            macro.last_seen = datetime.datetime.now().isoformat()

    # Macros that have saved at least this many tokens are preserved regardless of age/hits.
    ROI_PROTECTION_THRESHOLD: int = 50

    def apply_decay(self, max_age_days: int = 7, min_hits: int = 3) -> None:
        """Prune macros that are old and have low hit counts.

        High-ROI macros (lifetime_savings >= ROI_PROTECTION_THRESHOLD) are kept
        even if they would otherwise fall below the age/hit threshold, since they
        have proven value even with infrequent use.
        """
        now = datetime.datetime.now()
        to_remove = []
        for name, macro in self.macros.items():
            if macro.is_durable:
                continue
            if not macro.last_seen:
                continue
            # ROI protection: preserve macros that have saved significant tokens
            if macro.lifetime_savings >= self.ROI_PROTECTION_THRESHOLD:
                continue
            try:
                last_seen = datetime.datetime.fromisoformat(macro.last_seen)
                age = (now - last_seen).days
                if age > max_age_days and macro.hit_count < min_hits:
                    to_remove.append(name)
            except ValueError:
                pass
        for name in to_remove:
            del self.macros[name]

    def record_savings(self, name: str, tokens: int) -> None:
        """Attribute token savings to a macro and update its rolling average."""
        if tokens <= 0:
            return
        macro = self.macros.get(name)
        if macro is None:
            return
        macro.lifetime_savings += tokens
        uses = max(macro.hit_count, 1)
        macro.avg_tokens_per_use = macro.lifetime_savings / uses

    def update_from_repair(
        self,
        name: str,
        new_instructions: tuple[Instruction, ...],
        new_inputs: tuple[str, ...] | None = None,
    ) -> bool:
        """Replace a macro's instructions with a repaired version.

        Preserves ``hit_count``, ``lifetime_savings``, ``avg_tokens_per_use``,
        ``context_requirements``, and ``is_durable`` from the existing macro.
        Returns ``True`` if the macro was found and updated, ``False`` otherwise.

        Only updates when the new instruction op-sequence differs from the current
        one — a no-op repair is not written.
        """
        macro = self.macros.get(name)
        if macro is None:
            return False
        old_ops = tuple(ins.op for ins in macro.instructions)
        new_ops = tuple(ins.op for ins in new_instructions)
        if old_ops == new_ops and macro.instructions == new_instructions:
            return False  # identical — nothing to heal
        macro.instructions = new_instructions
        if new_inputs is not None:
            macro.inputs = new_inputs
        macro.last_seen = datetime.datetime.now().isoformat()
        return True

    def load_global(self, path: str | None = None) -> None:
        if path is None:
            path = os.path.expanduser("~/.tok/macros.tok")
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
                for m_data in data:
                    macro = Macro.from_dict(m_data)
                    self.register(macro)
            # Apply decay after loading
            self.apply_decay()
            # Prune op-sequence duplicates accumulated across sessions,
            # keeping the representative with the highest hit_count.
            seen_op_seqs: dict[tuple[str, ...], str] = {}
            to_remove = []
            for name, macro in list(self.macros.items()):
                op_seq = tuple(ins.op for ins in macro.instructions)
                if op_seq in seen_op_seqs:
                    existing_name = seen_op_seqs[op_seq]
                    existing = self.macros[existing_name]
                    if macro.hit_count >= existing.hit_count:
                        to_remove.append(existing_name)
                        seen_op_seqs[op_seq] = name
                    else:
                        to_remove.append(name)
                else:
                    seen_op_seqs[op_seq] = name
            for name in to_remove:
                del self.macros[name]
        except Exception as e:
            import logging

            logging.getLogger("tok.neuro").error(
                f"Failed to load global macros from {path}: {e}"
            )

    def save_global(self, path: str | None = None) -> None:
        if path is None:
            path = os.path.expanduser("~/.tok/macros.tok")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, "w") as f:
                json.dump(
                    [m.to_dict() for m in self.macros.values()], f, indent=2
                )
        except Exception as e:
            import logging

            logging.getLogger("tok.neuro").error(
                f"Failed to save global macros to {path}: {e}"
            )


class TokIR:
    def __init__(
        self,
        instructions: Iterable[Instruction] | None = None,
        macros: Iterable[Macro] | None = None,
    ):
        self.instructions: list[Instruction] = (
            list(instructions) if instructions is not None else []
        )
        self.macros: list[Macro] = list(macros) if macros is not None else []

    def add_instruction(self, ins: Instruction) -> None:
        self.instructions.append(ins)

    def add_macro(self, macro: Macro) -> None:
        self.macros.append(macro)

    def get_defined_macros(self) -> list[Macro]:
        return self.macros

    def to_dict(self) -> dict[str, Any]:
        return {
            "instructions": [i.to_dict() for i in self.instructions],
            "macros": [m.to_dict() for m in self.macros],
        }


def execute_ir(
    ir: TokIR, inputs: dict[str, Any], registry: MacroRegistry | None = None
) -> Any:
    """
    Executes a sequence of instructions using a local scope initialized with inputs.
    """
    scope = dict(inputs)
    last_result = None

    for ins in ir.instructions:
        # Check if it's a macro call (op starts with '@')
        if ins.op.startswith("@") and registry:
            macro_name = ins.op[1:]
            if registry:
                registry.record_use(macro_name)
            macro = registry.get(macro_name)
            if macro:
                # Resolve args for macro
                macro_inputs = {}
                for param, arg in zip(macro.inputs, ins.args, strict=True):
                    if isinstance(arg, str) and arg.startswith("$"):
                        macro_inputs[param] = scope[arg[1:]]
                    else:
                        macro_inputs[param] = arg

                # Execute nested IR
                result = execute_ir(
                    TokIR(macro.instructions), macro_inputs, registry
                )
                if ins.target:
                    scope[ins.target] = result
                last_result = result
                continue
        # Resolve arguments (literals or variable references)
        resolved_args = []
        for arg in ins.args:
            if isinstance(arg, str) and arg.startswith("$"):
                var_name = arg[1:]
                if var_name not in scope:
                    raise NameError(f"Undefined variable in IR: {arg}")
                resolved_args.append(scope[var_name])
            else:
                resolved_args.append(arg)

        # Execute operator
        result = _execute_op(ins.op, resolved_args)

        if ins.target:
            scope[ins.target] = result

        last_result = result

    return last_result


def _execute_op(op: str, args: list[Any]) -> Any:
    if op == "add":
        return args[0] + args[1]
    if op == "sub":
        return args[0] - args[1]
    if op == "mul":
        return args[0] * args[1]
    if op == "div":
        return args[0] // args[1]
    if op == "const":
        return args[0]
    if op == "identity":
        return args[0]
    if op == "list":
        return list(args)
    if op == "get":
        # get l idx
        return args[0][args[1]]
    if op == "len":
        return len(args[0])

    # Coding specific ops (simplified versions of apply_skill)
    if op == "reverse":
        return args[0][::-1]
    if op == "filter":
        # filter list predicate
        pred = args[1]
        if pred == "even":
            return [x for x in args[0] if x % 2 == 0]
        if pred == "vowel":
            return "".join(c for c in args[0] if c.lower() in "aeiou")
        return [x for x in args[0] if x == pred]
    if op == "unique":
        return list(dict.fromkeys(args[0]))
    if op == "sort":
        return sorted(args[0])
    if op == "join":
        return "".join(str(x) for x in args[0])

    raise ValueError(f"Unknown IR operator: {op}")
