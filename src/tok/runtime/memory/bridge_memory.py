"""Bridge-native structured memory for the invisible Tok bridge."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from tok.compression import (
    CANONICAL_MEMORY_FIELDS,
    TOK_FIELD_ALIAS,
    TOK_REVERSE_ALIAS,
)
from tok.memory.pointers import PointerRegistry
from tok.neuro.ir import Instruction, Macro, MacroRegistry
from tok.runtime.policy.smart_policy import (
    CANONICAL_WIRE_FIELD_ORDER,
    MemoryProjectionProfile,
)
from tok.utils.event_logging import log_memory_promotion, log_rolling_state

# Field-specific decay rates. 0 = immortal (never decay).
# Hot bucket uses these rates; durable uses half-rates (floored at 1 for non-zero).
DECAY_RATES: dict[str, int] = {
    "constraints": 0,  # user invariants — never decay
    "goal": 0,  # core orientation — never decay
    "facts": 0,  # file/search snapshots — never decay
    "blockers": 1,
    "errs": 1,
    "files": 1,
    "questions": 1,
    "cmds": 2,  # ephemeral — decay faster
    "tests": 2,
    "next": 2,  # highly transient
    "turns": 1,
    "edited": 1,
}

# Per-field promotion thresholds for hot→durable.
PROMOTION_THRESHOLDS: dict[str, int] = {
    "goal": 2,
    "files": 3,
    "edited": 2,  # edited files promote faster than regular files
    "constraints": 2,
    "facts": 3,
    "blockers": 3,
    "errs": 4,
}

MULTI_VALUE_FIELDS = {
    "files",
    "edited",
    "cmds",
    "tests",
    "errs",
    "blockers",
    "constraints",
    "questions",
    "facts",
    "goal",
}
HOT_LIMITS = {
    "turns": 1,
    "goal": 4,
    "files": 4,
    "edited": 4,
    "cmds": 16,
    "tests": 8,
    "errs": 8,
    "blockers": 4,
    "constraints": 4,
    "questions": 4,
    "next": 2,
    "facts": 16,
}
DURABLE_LIMITS = {
    "turns": 1,
    "goal": 4,
    "files": 8,
    "edited": 8,
    "cmds": 8,
    "tests": 8,
    "errs": 8,
    "blockers": 8,
    "constraints": 8,
    "questions": 8,
    "next": 4,
    "facts": 32,
}

# Hard caps on total entries across all fields in each bucket
HOT_TOTAL_CAP = 60
DURABLE_TOTAL_CAP = 200


_SECTION_HEADERS: dict[str, str] = {
    "@h": "h",
    "@d": "d",
    "@rolling_cmds": "rolling_cmds",
    "@macros": "macros",
}


def _dispatch_section_header(
    line: str, section: str | None, current_field: str | None
) -> tuple[str | None, str | None, bool]:
    """
    Check if line is a section header and return updated state.

    Returns (new_section, new_current_field, was_handled).
    """
    new_section = _SECTION_HEADERS.get(line)
    if new_section is not None:
        return new_section, None, True
    return section, current_field, False


def _parse_macro_instructions(body: str) -> list[Instruction]:
    """Parse the instruction body of a macro definition."""
    instructions: list[Instruction] = []
    for cmd in body.split("|"):
        cmd = cmd.strip()
        if not cmd:
            continue
        target = None
        if "=" in cmd:
            eq_pos = cmd.index("=")
            lhs = cmd[:eq_pos].strip()
            if lhs.isidentifier():
                target = lhs
                cmd = cmd[eq_pos + 1 :].strip()
        if "(" in cmd and cmd.endswith(")"):
            op = cmd[: cmd.index("(")]
            cmd_args = tuple(a.strip() for a in cmd[cmd.index("(") + 1 : -1].split(",") if a.strip())
            instructions.append(Instruction(op=op, args=cmd_args, target=target))
    return instructions


def _is_macro_relevant(
    macro: Macro,
    markers: frozenset[str] | None,
    high_hit: list[Macro],
    active_files: set[str],
) -> bool:
    """Check if a macro is relevant for wire_state output."""
    reqs = macro.context_requirements
    if markers is not None and reqs and "marker_file" in reqs:
        if reqs["marker_file"] not in markers:
            return False
    if macro in high_hit and macro.hit_count > 2:
        return True
    if macro.provenance and macro.provenance.source_file in active_files:
        return True
    for ins in macro.instructions:
        if any(isinstance(a, str) and a in active_files for a in ins.args):
            return True
    return False


def _jaccard_tokens(text: str) -> set[str]:
    """Tokenize text for Jaccard similarity."""
    return set(re.sub(r"[^a-z0-9_/]", " ", text.lower()).split())


_JACCARD_THRESHOLD = 0.25


def _match_facts_to_questions(
    questions: list[MemoryEntry],
    all_facts: list[MemoryEntry],
) -> tuple[int, dict[str, str]]:
    """
    Find facts that answer open questions via Jaccard similarity.

    Returns (promotion_count, {question_value: matching_fact_value}).
    """
    promotions = 0
    answered: dict[str, str] = {}
    for question in questions:
        q_tokens = _jaccard_tokens(question.value)
        if not q_tokens:
            continue
        for fact in all_facts:
            f_tokens = _jaccard_tokens(fact.value)
            if not f_tokens:
                continue
            intersection = len(q_tokens & f_tokens)
            union = len(q_tokens | f_tokens)
            if union and intersection / union >= _JACCARD_THRESHOLD:
                answered[question.value] = fact.value
                promotions += 1
                break
    return promotions, answered


@dataclass
class MemoryEntry:
    value: str
    score: int = 1
    last_seen_turn: int = 0


@dataclass
class BridgeMemoryState:
    turn: int = 0
    hot: dict[str, list[MemoryEntry]] = field(default_factory=dict)
    durable: dict[str, list[MemoryEntry]] = field(default_factory=dict)
    rolling_cmds: list[MemoryEntry] = field(default_factory=list)
    pointers: PointerRegistry = field(default_factory=PointerRegistry)
    macro_registry: MacroRegistry = field(default_factory=MacroRegistry)
    _file_heat: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))
    _prev_field_hashes: dict[str, str] = field(default_factory=dict, repr=False)
    load_global_macros: bool = field(default=True, repr=False)

    def __post_init__(self) -> None:
        # Load global macros at the start of a session
        if self.load_global_macros:
            self.macro_registry.load_global()

    def bump_file_heat(self, path: str, weight: float = 1.0) -> None:
        """Increment the heat score for a file path."""
        self._file_heat[path] += weight

    def top_hot_files(self, n: int = 3) -> list[str]:
        """Return the top N files by heat score."""
        return sorted(self._file_heat, key=lambda p: -self._file_heat[p])[:n]

    @staticmethod
    def _parse_macro_line(line: str, registry: MacroRegistry) -> None:
        """Parse a macro definition line and register it."""
        parts = line[3:].split("->")
        if len(parts) != 2:
            return
        sig = parts[0].strip()
        body = parts[1].strip()
        if not (sig.startswith("@") and "(" in sig):
            return
        name = sig[1 : sig.index("(")]
        raw_args = sig[sig.index("(") + 1 : -1].split(",")
        args = tuple(a.strip() for a in raw_args if a.strip())
        instructions = _parse_macro_instructions(body)
        registry.register(
            Macro(
                name=name,
                inputs=args,
                instructions=tuple(instructions),
            )
        )

    @staticmethod
    def _parse_entry_line(
        line: str,
        bucket: dict[str, list[MemoryEntry]],
        current_field: str,
    ) -> None:
        """Parse an entry line (|> value|score:X|last:Y) into the bucket."""
        parts = line[3:].split("|")
        value = parts[0].strip()
        score = 1
        last_seen = 0
        for part in parts[1:]:
            if part.startswith("score:"):
                with contextlib.suppress(ValueError):
                    score = int(part.split(":", 1)[1])
            elif part.startswith("last:"):
                with contextlib.suppress(ValueError):
                    last_seen = int(part.split(":", 1)[1])
        bucket.setdefault(current_field, []).append(MemoryEntry(value=value, score=score, last_seen_turn=last_seen))

    @staticmethod
    def _parse_rolling_cmd_line(
        line: str,
        rolling_cmds: list[MemoryEntry],
    ) -> None:
        """Parse a rolling_cmds entry line."""
        parts = line[3:].split("|")
        value = parts[0].strip()
        last_seen = 0
        for part in parts[1:]:
            if part.startswith("last:"):
                with contextlib.suppress(ValueError):
                    last_seen = int(part.split(":", 1)[1])
        rolling_cmds.append(MemoryEntry(value=value, score=1, last_seen_turn=last_seen))

    @classmethod
    def _parse_mem_header(cls, line: str, state: BridgeMemoryState) -> None:
        """Parse the @mem line for turn counter."""
        for token in line.split():
            if token.startswith("t:"):
                with contextlib.suppress(ValueError):
                    state.turn = int(token.split(":", 1)[1])

    @classmethod
    def from_tok(cls, text: str, *, load_global_macros: bool = True) -> BridgeMemoryState:
        state = cls(load_global_macros=load_global_macros)
        state.pointers = PointerRegistry.from_tok(text)
        section: str | None = None
        current_field: str | None = None

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("@mem"):
                cls._parse_mem_header(line, state)
                continue
            section, current_field, handled = _dispatch_section_header(line, section, current_field)
            if handled:
                continue
            if section == "macros" and line.startswith("|> "):
                cls._parse_macro_line(line, state.macro_registry)
                continue
            if line.startswith("@f ") and section in ("h", "d"):
                current_field = line.split(" ", 1)[1].strip()
                continue
            if line.startswith("|> ") and section and current_field:
                bucket = state.hot if section == "h" else state.durable
                cls._parse_entry_line(line, bucket, current_field)
                continue
            if line.startswith("|> ") and section == "rolling_cmds":
                cls._parse_rolling_cmd_line(line, state.rolling_cmds)
                continue

        return state

    @staticmethod
    def _serialize_bucket(bucket: dict[str, list[MemoryEntry]], lines: list[str]) -> None:
        """Serialize canonical and extra fields from a bucket."""
        for f in CANONICAL_MEMORY_FIELDS:
            entries = bucket.get(f, [])
            if not entries:
                continue
            lines.append(f"@f {f}")
            for entry in sorted(
                entries,
                key=lambda e: (-e.score, -e.last_seen_turn, e.value),
            ):
                lines.append(f"  |> {entry.value}|score:{entry.score}|last:{entry.last_seen_turn}")
        for extra in ("questions", "facts"):
            extra_entries = bucket.get(extra, [])
            if extra_entries:
                lines.append(f"@field {extra}")
                for entry in extra_entries:
                    lines.append(f"  |> {entry.value}|score:{entry.score}|last:{entry.last_seen_turn}")

    @staticmethod
    def _serialize_macro_line(macro: Macro) -> str:
        """Serialize a single macro to its tok representation."""
        sig = f"@{macro.name}({', '.join(macro.inputs)})"
        body_parts = []
        for ins in macro.instructions:
            op_str = f"{ins.op}({', '.join(str(a) for a in ins.args)})"
            if ins.target:
                op_str = f"{ins.target}={op_str}"
            body_parts.append(op_str)
        body = " | ".join(body_parts)
        return f"  |> {sig} -> {body}"

    def to_tok(self) -> str:
        lines = [f"@mem v:b1 t:{self.turn}"]
        lines.append(self.pointers.to_tok().strip())
        for section_name, bucket in (("h", self.hot), ("d", self.durable)):
            lines.append(f"@{section_name}")
            self._serialize_bucket(bucket, lines)

        if self.macro_registry.macros:
            lines.append("@macros")
            for macro in self.macro_registry.macros.values():
                lines.append(self._serialize_macro_line(macro))

        if self.rolling_cmds:
            lines.append("@rolling_cmds")
            for entry in self.rolling_cmds[-50:]:
                lines.append(f"  |> {entry.value}|last:{entry.last_seen_turn}")

        return "\n".join(lines) + "\n"

    def replace_hot_from_wire_state(self, tok_state: str) -> dict[str, int]:
        """Replace hot memory from a wire state string."""
        fields = _parse_wire_state(tok_state)
        if not fields:
            return {}
        turns = fields.get("turns", ["0"])
        self.turn = max(self.turn + 1, _safe_int(turns[0] if turns else "0"))
        previous_hot_values = {(field, entry.value) for field, entries in self.hot.items() for entry in entries}
        new_hot: dict[str, list[MemoryEntry]] = {}
        touched = set()
        for fld_name, values in fields.items():
            if fld_name == "turns":
                new_hot[fld_name] = [MemoryEntry(value=values[0], score=1, last_seen_turn=self.turn)]
                touched.add(fld_name)
                continue
            entries = [
                MemoryEntry(value=value, score=3, last_seen_turn=self.turn)
                for value in values[: HOT_LIMITS.get(fld_name, 2)]
                if value
            ]
            if entries:
                new_hot[fld_name] = entries
                touched.add(fld_name)
                # 3B: Evict contradictions from durable
                for entry in entries:
                    self._drop_conflicts(self.durable, fld_name, entry.value)
        self.hot = new_hot
        self._trim_bucket(self.hot, HOT_LIMITS)
        metrics = self._decay_bucket(self.durable, touched=touched, prefix="durable", half_rate=True)
        metrics = _merge_metrics(metrics, self._promote_hot_to_durable())
        metrics = _merge_metrics(metrics, self._promote_facts_for_questions())
        current_hot_values = {(field, entry.value) for field, entries in self.hot.items() for entry in entries}
        promotions = len(current_hot_values - previous_hot_values)
        demotions = len(previous_hot_values - current_hot_values)
        if promotions:
            metrics["hot_promotions"] = promotions
        if demotions:
            metrics["hot_demotions"] = demotions
        metrics["hot_entries"] = sum(len(entries) for entries in self.hot.values())
        metrics["durable_entries"] = sum(len(entries) for entries in self.durable.values())
        return metrics

    def ingest_wire_state(self, tok_state: str) -> dict[str, int]:
        """Ingest a wire state string into memory."""
        fields = _parse_wire_state(tok_state)
        if not fields:
            return {}
        turns = fields.get("turns", ["0"])
        self.turn = max(self.turn + 1, _safe_int(turns[0] if turns else "0"))
        touched = set()
        for fld_name, values in fields.items():
            for value in values:
                if not value:
                    continue
                self._upsert(self.hot, fld_name, value, score_delta=2)
                if fld_name in {
                    "constraints",
                    "facts",
                    "blockers",
                    "questions",
                    "tests",
                    "errs",
                }:
                    self._upsert(self.durable, fld_name, value, score_delta=1)
                touched.add(fld_name)
        metrics = self._decay_bucket(self.hot, touched=touched, prefix="hot")
        metrics = _merge_metrics(
            metrics,
            self._decay_bucket(self.durable, touched=touched, prefix="durable", half_rate=True),
        )
        metrics = _merge_metrics(metrics, self._promote_hot_to_durable())
        metrics = _merge_metrics(metrics, self._promote_facts_for_questions())
        trim_metrics = self._trim_all()
        metrics = _merge_metrics(metrics, trim_metrics)
        metrics["hot_entries"] = sum(len(entries) for entries in self.hot.values())
        metrics["durable_entries"] = sum(len(entries) for entries in self.durable.values())
        log_rolling_state(self.turn, trim_metrics.get("trimmed", 0))
        return metrics

    def _get_merged_entries(self, field_name: str, limit: int) -> list[MemoryEntry]:
        """Merge hot + durable entries for a field, deduped and sorted."""
        hot_entries = self.hot.get(field_name, [])
        durable_entries = self.durable.get(field_name, [])
        if field_name in {"turns", "next", "goal"}:
            return hot_entries[:1] or durable_entries[:1]
        seen: set[str] = set()
        merged: list[MemoryEntry] = []
        for e in hot_entries + durable_entries:
            if e.value not in seen:
                merged.append(e)
                seen.add(e.value)
        if field_name == "facts":
            merged.sort(
                key=lambda e: (
                    0 if e.value.startswith("answer_") else 1,
                    -e.score,
                    -e.last_seen_turn,
                    e.value,
                )
            )
        else:
            merged.sort(key=lambda e: (-e.score, -e.last_seen_turn, e.value))
        return merged[:limit]

    def _resolve_wire_files(
        self,
        profile: MemoryProjectionProfile | None,
    ) -> list[MemoryEntry]:
        """Resolve and augment file entries for wire_state."""
        file_limit = (
            profile.field_limits.get("files", HOT_LIMITS.get("files", 2)) if profile else HOT_LIMITS.get("files", 2)
        )
        files = self._get_merged_entries("files", file_limit)
        if self._file_heat:
            top_heat_files = self.top_hot_files(file_limit)
            existing = {f.value for f in files}
            for hf in top_heat_files:
                if hf not in existing and len(files) < file_limit:
                    files.append(
                        MemoryEntry(
                            value=hf,
                            score=5,
                            last_seen_turn=self.turn,
                        )
                    )
        return files

    def _resolve_wire_facts(
        self,
        profile: MemoryProjectionProfile | None,
    ) -> tuple[list[MemoryEntry], list[MemoryEntry]]:
        """Return (file_facts, other_facts)."""
        f_limit = profile.fact_limit if profile else HOT_LIMITS.get("facts", 4)
        facts = self._get_merged_entries("facts", f_limit)
        file_facts = [f for f in facts if f.value.startswith("file[") and ":" in f.value]
        other_facts = [f for f in facts if not (f.value.startswith("file[") and ":" in f.value)]
        return file_facts, other_facts

    def _emit_files_field(
        self,
        files: list[MemoryEntry],
        file_facts: list[MemoryEntry],
        state_parts: list[str],
        omit_unchanged: bool,
    ) -> None:
        """Emit the files field into state_parts with freshness indicators."""
        if not files:
            return

        # Build a lookup from file path to freshness data
        freshness_lookup: dict[str, str] = {}
        for fact in file_facts:
            if fact.value.startswith("file[") and ":" in fact.value:
                # Parse: file[path]:LINE_COUNT|digest|~TOKENS
                bracket_end = fact.value.find("]", 5)
                if bracket_end < 0:
                    continue
                path = fact.value[5:bracket_end]
                colon_idx = fact.value.find(":", bracket_end)
                if colon_idx < 0:
                    continue
                rest = fact.value[colon_idx + 1 :]
                # Extract LINE_COUNT|~TOKENS (skip digest which may contain |)
                # Format: LINE_COUNT|digest|~TOKENS where digest may contain |
                if "|" in rest:
                    parts = rest.split("|")
                    if len(parts) >= 3:
                        line_count = parts[0]
                        # tokens is the last part (starts with ~)
                        tokens = parts[-1] if parts[-1].startswith("~") else parts[2]
                        freshness_lookup[path] = f"{line_count}|{tokens}"

        file_values = []
        for e in files:
            val = e.value
            if len(val) > 20:
                val = self.pointers.get_pointer(val)
            # Append freshness data if available
            freshness = freshness_lookup.get(e.value)
            if freshness:
                val = f"{val}:{freshness}"
            file_values.append(val)

        f_key = TOK_FIELD_ALIAS.get("files", "files")
        serialized = f"{f_key}:" + ",".join(file_values)
        self._emit_field("files", serialized, state_parts, omit_unchanged)

    def _emit_facts_section(
        self,
        file_facts: list[MemoryEntry],
        other_facts: list[MemoryEntry],
        listed_files: set[str],
        state_parts: list[str],
    ) -> None:
        """Emit facts directly into state_parts."""
        for fact in file_facts:
            # Extract path from fact value: file[path]:...
            val = fact.value
            if val.startswith("file[") and "]" in val:
                bracket_end = val.index("]")
                path = val[5:bracket_end]
                if path not in listed_files:
                    state_parts.append(val)
            else:
                state_parts.append(val)
        for fact in other_facts:
            state_parts.append(fact.value)

    def _emit_facts_with_pointers(
        self,
        file_facts: list[MemoryEntry],
        other_facts: list[MemoryEntry],
        listed_files: set[str],
        state_parts: list[str],
    ) -> None:
        """Emit facts with pointer compression (fallback path)."""
        for fact in file_facts:
            # Extract path from fact value: file[path]:...
            val = fact.value
            if val.startswith("file[") and "]" in val:
                bracket_end = val.index("]")
                path = val[5:bracket_end]
                if path not in listed_files:
                    state_parts.append(val)
            else:
                state_parts.append(val)
        for fact in other_facts:
            val = fact.value
            if ":" in val:
                fk, fv = val.split(":", 1)
                if fk in {"answer_file", "answer_verification"} and len(fv) > 20:
                    fv = self.pointers.get_pointer(fv)
                    val = f"{fk}:{fv}"
            state_parts.append(val)

    def _emit_generic_field(
        self,
        field: str,
        profile: MemoryProjectionProfile | None,
        state_parts: list[str],
        omit_unchanged: bool,
    ) -> None:
        """Emit a non-files, non-facts field."""
        limit = profile.field_limits.get(field, HOT_LIMITS.get(field, 2)) if profile else HOT_LIMITS.get(field, 2)
        entries = self._get_merged_entries(field, limit)
        if not entries:
            return
        alias = TOK_FIELD_ALIAS.get(field, field)
        values = []
        for e in entries:
            v = e.value
            if field == "facts" and ":" in v:
                fk, fv = v.split(":", 1)
                if fk in {"answer_file", "answer_verification"} and len(fv) > 20:
                    fv = self.pointers.get_pointer(fv)
                    v = f"{fk}:{fv}"
            values.append(v)
        serialized = f"{alias}:" + ",".join(values)
        self._emit_field(field, serialized, state_parts, omit_unchanged)

    def _emit_field(
        self,
        field: str,
        serialized: str,
        state_parts: list[str],
        omit_unchanged: bool,
    ) -> None:
        """Append serialized field respecting omit_unchanged."""
        if omit_unchanged:
            prev_hash = self._prev_field_hashes.get(field)
            cur_hash = hashlib.md5(serialized.encode()).hexdigest()  # nosec B324
            if prev_hash == cur_hash:
                return
            self._prev_field_hashes[field] = cur_hash
        state_parts.append(serialized)

    def _collect_relevant_macros(
        self,
        markers: frozenset[str] | None,
    ) -> list[Macro]:
        """Filter and deduplicate macros for wire_state."""
        active_files = {e.value for e in (self.hot.get("files", []) + self.durable.get("files", []))}
        all_macros = list(self.macro_registry.macros.values())
        high_hit = sorted(all_macros, key=lambda m: -m.hit_count)[:3]

        relevant: list[Macro] = []
        for macro in all_macros:
            if not _is_macro_relevant(macro, markers, high_hit, active_files):
                continue
            relevant.append(macro)

        seen_names: set[str] = set()
        final: list[Macro] = []
        for m in relevant:
            if m.name not in seen_names:
                final.append(m)
                seen_names.add(m.name)
        return sorted(final, key=lambda m: -m.hit_count)[:5]

    def _build_extra_blocks(
        self,
        markers: frozenset[str] | None,
    ) -> list[str]:
        """Build pointer and macro blocks for wire_state."""
        extra_blocks: list[str] = []
        ptr_tok = self.pointers.to_tok().strip()
        if ptr_tok:
            extra_blocks.append(ptr_tok)
        if self.macro_registry.macros and self.load_global_macros:
            relevant = self._collect_relevant_macros(markers)
            if relevant:
                macro_lines = ["@macros"]
                for macro in relevant:
                    macro_lines.append(self._serialize_macro_line(macro))
                extra_blocks.append("\n".join(macro_lines))
        return extra_blocks

    def wire_state(
        self,
        profile: MemoryProjectionProfile | None = None,
        markers: frozenset[str] | None = None,
        omit_unchanged: bool = False,
        is_fallback_context: bool = False,
    ) -> str:
        state_parts: list[str] = []
        field_order = profile.field_order if profile is not None else CANONICAL_WIRE_FIELD_ORDER

        files = self._resolve_wire_files(profile)
        listed_files = {e.value for e in files}
        file_facts, other_facts = self._resolve_wire_facts(profile)

        facts_emitted = False
        for fld_name in field_order:
            if fld_name == "files":
                self._emit_files_field(files, file_facts, state_parts, omit_unchanged)
            elif fld_name == "facts":
                self._emit_facts_section(
                    file_facts,
                    other_facts,
                    listed_files,
                    state_parts,
                )
                facts_emitted = True
            else:
                self._emit_generic_field(
                    fld_name,
                    profile,
                    state_parts,
                    omit_unchanged,
                )

        if not facts_emitted:
            self._emit_facts_with_pointers(
                file_facts,
                other_facts,
                listed_files,
                state_parts,
            )

        state_line = ">>> " + "|".join(state_parts) if state_parts else ""

        extra_blocks = self._build_extra_blocks(markers)

        if (is_fallback_context or state_line) and extra_blocks:
            return state_line + "\n" + "\n".join(extra_blocks)
        return state_line

    @staticmethod
    def _extract_file_digest(text: str, path: str, was_edited: bool = False) -> str:
        """Extract a semantically dense ≤160-char digest from file content."""
        lines = text.splitlines()

        # Handle empty file edge case
        if not lines or not any(line.strip() for line in lines):
            return "(empty file)"

        if was_edited:
            sigs = [line.strip() for line in lines if re.match(r"^\s*(def |class |async def )", line)]
            if sigs:
                return (" ".join(sigs))[:160]

        if path.endswith(".py"):
            top_level = [line.strip() for line in lines if re.match(r"^(def |class |async def |[A-Z_]+ =)", line)]
            if top_level:
                return (" ".join(top_level))[:160]

        meaningful = [
            line.strip()
            for line in lines
            if line.strip() and not line.strip().startswith(("#", "//", "/*", '"""', "'''"))
        ]
        return (" ".join(meaningful))[:160]

    def record_file_snapshot(self, path: str, snippet: str) -> bool:
        """Capture a recently read file so future turns can reuse it without rereading."""
        normalized_path = path.strip()
        if not normalized_path:
            return False
        snippet = snippet.strip()
        if not snippet:
            return False

        heat = self._file_heat.get(normalized_path, 0)
        was_edited = heat >= 2.0

        # Calculate line count for freshness visibility
        line_count = len(snippet.splitlines())
        estimated_tokens = line_count * 4  # Rough estimate: ~4 tokens per line

        digest = self._extract_file_digest(snippet, normalized_path, was_edited=was_edited)
        if not digest:
            digest = " ".join(snippet.split())[:160]

        # Enhanced fact format: includes line count and token savings indicator
        fact_key = f"file[{normalized_path}]"
        # Format: file[path]:LINE_COUNT|digest|~TOKENS_SAVED
        value = f"{fact_key}:{line_count}|{digest}|~{estimated_tokens}t"

        base_score = 2
        heat_bonus = int(heat * 2)

        self._upsert(self.hot, "facts", value, score_delta=base_score + heat_bonus)
        self._upsert(self.hot, "files", normalized_path[:96], score_delta=1 + heat_bonus)
        if was_edited:
            self._upsert(
                self.hot,
                "edited",
                normalized_path[:96],
                score_delta=2 + heat_bonus,
            )
        self._trim_all()
        return True

    def record_search_snapshot(self, query: str, snippet: str) -> bool:
        """Capture search results so redundant queries can be avoided."""
        normalized_query = query.strip()
        if not normalized_query:
            return False
        snippet = snippet.strip()
        if not snippet:
            return False
        snippet = " ".join(snippet.split())[:160]
        fact_key = f"search[{normalized_query}]:"
        value = f"{fact_key}{snippet}"
        self._upsert(self.hot, "facts", value, score_delta=2)
        self._trim_all()
        return True

    def record_history_snapshot(self, path: str, revision: str, snippet: str) -> bool:
        normalized_path = path.strip()
        if not normalized_path:
            return False
        snippet = snippet.strip()
        if not snippet:
            return False
        digest = self._extract_file_digest(snippet, normalized_path)
        if not digest:
            digest = " ".join(snippet.split())[:160]
        fact_key = f"history_file[{normalized_path}@{revision}]"
        value = f"{fact_key}:{digest}"
        self._upsert(self.hot, "facts", value, score_delta=2)
        self._trim_all()
        return True

    def record_metadata_snapshot(self, path: str, subtype: str, snippet: str) -> bool:
        normalized_path = (path or "").strip()
        subtype = subtype.strip()
        if not subtype:
            return False
        snippet = snippet.strip()
        if not snippet:
            return False
        snippet = " ".join(snippet.split())[:160]
        fact_key = f"meta[{normalized_path}:{subtype}]" if normalized_path else f"meta[:{subtype}]"
        value = f"{fact_key}:{snippet}"
        self._upsert(self.hot, "facts", value, score_delta=1)
        self._trim_all()
        return True

    def get_file_fact_digests(self) -> dict[str, str]:
        """
        Extract file digests from facts, handling new format with line counts.
        New format: file[path]:LINE_COUNT|digest|~tokens
        Legacy format: file[path]:digest.
        """
        result: dict[str, str] = {}
        for entry in self.hot.get("facts", []):
            # Defensive validation: ensure entry starts with expected prefix
            if not entry.value.startswith("file["):
                continue
            bracket_end = entry.value.find("]", 5)
            if bracket_end < 0:
                continue
            path = entry.value[5:bracket_end]
            colon_idx = entry.value.find(":", bracket_end)
            if colon_idx < 0:
                continue

            rest = entry.value[colon_idx + 1 :]
            # Handle new format: LINE_COUNT|digest|~tokens
            # Digest may contain |, so we take parts[1] through parts[-2] as digest
            # or just parts[1] if only 3 parts exist
            if "|" in rest:
                parts = rest.split("|")
                if len(parts) >= 2:
                    # digest is between LINE_COUNT and ~tokens
                    # If last part starts with ~, digest is parts[1] (or parts[1:-1] joined)
                    if len(parts) >= 3 and parts[-1].startswith("~"):
                        digest = "|".join(parts[1:-1]) if len(parts) > 3 else parts[1]
                    else:
                        digest = parts[1]
                    if digest:  # Only store non-empty digests
                        result[path] = digest
            else:
                # Legacy format: just digest
                result[path] = rest
        return result

    def record_hypothesis(self, text: str) -> bool:
        """
        Record an open question or hypothesis in the bounded questions queue.

        Questions are subject to the same decay and HOT_LIMITS cap as other
        multi-value fields, so the queue is automatically bounded by the
        existing trim logic.
        """
        text = text.strip()
        if not text:
            return False
        value = text[:120]
        self._upsert(self.hot, "questions", value, score_delta=1)
        self._trim_all()
        return True

    def _upsert(
        self,
        bucket: dict[str, list[MemoryEntry]],
        field: str,
        value: str,
        score_delta: int,
    ) -> None:
        """Upsert a value into a memory bucket."""
        self._drop_conflicts(bucket, field, value)

        # Preservation logic for NeuroReactor: keep chronological log (BEFORE deduplication)
        if field == "cmds":
            self.rolling_cmds.append(MemoryEntry(value=value, score=1, last_seen_turn=self.turn))
            if len(self.rolling_cmds) > 100:
                self.rolling_cmds = self.rolling_cmds[-100:]

        entries = bucket.setdefault(field, [])
        for entry in entries:
            if entry.value == value:
                entry.score += score_delta
                entry.last_seen_turn = self.turn
                return
        entries.append(MemoryEntry(value=value, score=score_delta, last_seen_turn=self.turn))

    def _drop_conflicts(self, bucket: dict[str, list[MemoryEntry]], field: str, value: str) -> None:
        """Drop conflicting entries from a bucket."""
        if field in {"turns", "goal", "next"}:
            bucket[field] = [entry for entry in bucket.get(field, []) if entry.value == value]
            return

        if field != "facts" or ":" not in value:
            return

        key, raw_value = value.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key or not raw_value:
            return

        bucket[field] = [
            entry for entry in bucket.get(field, []) if not (entry.value.startswith(f"{key}:") and entry.value != value)
        ]

    def _decay_bucket(
        self,
        bucket: dict[str, list[MemoryEntry]],
        touched: set[str],
        prefix: str,
        *,
        half_rate: bool = False,
    ) -> dict[str, int]:
        """Apply score decay to a memory bucket."""
        demoted = 0
        for f, entries in list(bucket.items()):
            rate = DECAY_RATES.get(f, 1)
            if rate == 0:
                continue  # immortal field
            if half_rate:
                # Durable bucket decays at half rate (min 1 for non-zero fields)
                rate = max(1, rate // 2) if rate > 1 else 1
            if f not in touched:
                for entry in entries:
                    before = entry.score
                    entry.score = max(0, entry.score - rate)
                    # TTL eviction for hypotheses: drop if older than 5 turns
                    if f == "questions":
                        age = self.turn - entry.last_seen_turn
                        if age > 5:
                            entry.score = 0

                    if entry.score < before:
                        demoted += 1
            bucket[f] = [e for e in entries if e.score > 0]
            if not bucket[f]:
                del bucket[f]
        return {f"{prefix}_decays": demoted} if demoted else {}

    def _promote_hot_to_durable(self) -> dict[str, int]:
        """Promote entries from hot to durable memory based on thresholds."""
        promoted = 0
        for f, entries in list(self.hot.items()):
            threshold = PROMOTION_THRESHOLDS.get(f)
            if threshold is None:
                continue
            for entry in entries:
                if entry.score >= threshold:
                    before = len(self.durable.get(f, []))
                    self._upsert(self.durable, f, entry.value, score_delta=1)
                    after = len(self.durable.get(f, []))
                    if after > before:
                        promoted += 1
                        log_memory_promotion(f, entry.value, bucket="durable")
        return {"durable_promotions": promoted} if promoted else {}

    def _promote_facts_for_questions(self) -> dict[str, int]:
        """
        Promote facts that directly answer open questions to durable memory.

        Uses token-level Jaccard similarity between each fact and each open
        question.  When similarity exceeds the threshold the fact is promoted to
        durable storage and the question is cleared from hot memory so it does
        not accumulate stale context.

        Returns a metrics dict with 'hypothesis_promotions' count.
        """
        questions = self.hot.get("questions", [])
        if not questions:
            return {}
        all_facts = self.hot.get("facts", []) + self.durable.get("facts", [])
        if not all_facts:
            return {}

        promotions, answered = _match_facts_to_questions(questions, all_facts)
        for fact_value in answered.values():
            self._upsert(self.durable, "facts", fact_value, score_delta=2)

        if answered:
            answered_values = set(answered.keys())
            self.hot["questions"] = [q for q in self.hot.get("questions", []) if q.value not in answered_values]
            if not self.hot["questions"]:
                del self.hot["questions"]

        return {"hypothesis_promotions": promotions} if promotions else {}

    def _trim_bucket_to_cap(
        self, bucket: dict[str, list[MemoryEntry]], cap: int
    ) -> tuple[dict[str, list[MemoryEntry]], int]:
        """Trim a bucket to a hard cap across all fields."""
        total = sum(len(entries) for entries in bucket.values())
        if total <= cap:
            return bucket, 0

        all_entries: list[tuple[str, MemoryEntry]] = []
        for fld_name, entries in bucket.items():
            for entry in entries:
                all_entries.append((fld_name, entry))

        all_entries.sort(key=lambda x: (-x[1].score, -x[1].last_seen_turn))
        kept = all_entries[:cap]

        new_bucket: dict[str, list[MemoryEntry]] = {}
        for fld_name, entry in kept:
            if fld_name not in new_bucket:
                new_bucket[fld_name] = []
            new_bucket[fld_name].append(entry)

        return new_bucket, total - cap

    def _trim_all(self) -> dict[str, int]:
        """Trim all memory buckets to their limits."""
        trimmed_hot = 0
        trimmed_durable = 0

        # First apply per-field limits
        for fld_name, limit in HOT_LIMITS.items():
            if fld_name in self.hot:
                before = len(self.hot[fld_name])
                self.hot[fld_name] = sorted(
                    self.hot[fld_name],
                    key=lambda e: (-e.score, -e.last_seen_turn, e.value),
                )[:limit]
                trimmed_hot += max(0, before - len(self.hot[fld_name]))

        for fld_name, limit in DURABLE_LIMITS.items():
            if fld_name in self.durable:
                before = len(self.durable[fld_name])
                self.durable[fld_name] = sorted(
                    self.durable[fld_name],
                    key=lambda e: (-e.score, -e.last_seen_turn, e.value),
                )[:limit]
                trimmed_durable += max(0, before - len(self.durable[fld_name]))

        # Then apply hard caps on total entries
        self.hot, hot_trimmed = self._trim_bucket_to_cap(self.hot, HOT_TOTAL_CAP)
        trimmed_hot += hot_trimmed

        self.durable, durable_trimmed = self._trim_bucket_to_cap(self.durable, DURABLE_TOTAL_CAP)
        trimmed_durable += durable_trimmed

        metrics: dict[str, int] = {}
        if trimmed_hot:
            metrics["hot_trims"] = trimmed_hot
        if trimmed_durable:
            metrics["durable_trims"] = trimmed_durable
        return metrics

    def _trim_bucket(self, bucket: dict[str, list[MemoryEntry]], limits: dict[str, int]) -> None:
        """Trim a bucket to per-field limits."""
        for fld_name, limit in limits.items():
            if fld_name in bucket:
                bucket[fld_name] = sorted(
                    bucket[fld_name],
                    key=lambda e: (-e.score, -e.last_seen_turn, e.value),
                )[:limit]


def _safe_int(value: str) -> int:
    """Parse a string to int, returning 0 on failure."""
    try:
        return int(value)
    except ValueError:
        return 0


def _merge_metrics(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    """Merge two metrics dictionaries."""
    merged = dict(left)
    for key, value in right.items():
        merged[key] = merged.get(key, 0) + value
    return merged


def _parse_wire_state(tok_state: str) -> dict[str, list[str]]:
    """Parse a wire state string into a dictionary."""
    line = tok_state.strip().splitlines()[0] if tok_state.strip() else ""
    if line.startswith(">>>"):
        line = line[3:].strip()
    if not line:
        return {}
    result: dict[str, list[str]] = {}
    for part in line.split("|"):
        if ":" not in part:
            continue
        key, raw_value = part.split(":", 1)
        key = key.strip()
        # Resolve alias if present
        key = TOK_REVERSE_ALIAS.get(key, key)
        raw_value = raw_value.strip()
        if not key or not raw_value or raw_value.lower() == "none":
            continue
        if key in {"turns", "next"}:
            result[key] = [raw_value]
        elif key in CANONICAL_MEMORY_FIELDS or key in {
            "facts",
            "questions",
            "goal",
        }:
            result[key] = [item.strip() for item in raw_value.split(",") if item.strip()]
        else:
            result.setdefault("facts", []).append(f"{key}:{raw_value}")
    return result


def clean_system_context(
    state: BridgeMemoryState, system_prompt: str | list[dict[str, Any]]
) -> str | list[dict[str, Any]]:
    """
    Remove verbose user prompts from active context while preserving task information.

    Extracts core goals and constraints from the bloated prompt, ingests them into state,
    and returns a significantly smaller representation while preserving list-type
    system prompt structure when present.
    """
    from tok.compression import compress_user_prompt

    system_text = ""
    if isinstance(system_prompt, list):
        system_text = "\n".join(
            str(block.get("text", ""))
            for block in system_prompt
            if isinstance(block, dict) and block.get("type") == "text"
        )
    else:
        system_text = str(system_prompt)

    if not system_text:
        return copy.deepcopy(system_prompt) if isinstance(system_prompt, list) else ""

    # Compress the bloated prompt into Tok-style fields
    # Use a larger snippet limit for cleaning than for standard history compression
    compressed = compress_user_prompt(system_text)

    # Re-ingest the compressed representation into bridge memory to ensure
    # the extracted goals and constraints are now part of durations/hot buckets.
    state.ingest_wire_state(f">>> {compressed}")

    optimized_text = f"### Optimized Task Context\n{compressed}"
    if not isinstance(system_prompt, list):
        return optimized_text

    rewritten_blocks = copy.deepcopy(system_prompt)
    last_text_index: int | None = None
    last_cached_text_index: int | None = None
    for index, block in enumerate(rewritten_blocks):
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        last_text_index = index
        if "cache_control" in block:
            last_cached_text_index = index

    target_index = last_cached_text_index if last_cached_text_index is not None else last_text_index
    if target_index is None:
        rewritten_blocks.append({"type": "text", "text": optimized_text})
        return rewritten_blocks

    target_block = rewritten_blocks[target_index]
    if isinstance(target_block, dict):
        target_block["text"] = optimized_text
    return rewritten_blocks
