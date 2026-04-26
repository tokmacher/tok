"""Tok models - Core data structures for Tok Protocol."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

logger = logging.getLogger(__name__)


class Trust(Enum):
    """Trust levels for Tok content."""

    SYSTEM = "system"
    UNTRUSTED = "untrusted"
    EXTERNAL = "external"


@dataclass
class TokNode:
    """A node in the Tok AST."""

    type: str
    label: str = ""
    attrs: dict[str, Any] = field(default_factory=dict)
    children: list[TokNode] = field(default_factory=list)
    text: str = ""
    trust: Trust = Trust.SYSTEM
    cardinality: int | None = None
    headers: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    ref: str | None = None
    _processed_as_attr: bool = False

    def __post_init__(self) -> None:
        pass


class EntropyTracker(BaseModel):
    """Tracks signal frequency and recency for spaced retrieval."""

    heatmap: dict[str, int] = Field(default_factory=dict)
    last_seen: dict[str, int] = Field(default_factory=dict)
    primacy_locked: list[str] = Field(default_factory=list)
    stages: dict[str, int] = Field(default_factory=dict)  # Stage 0-3
    next_refresh: dict[str, int] = Field(default_factory=dict)  # Turn to re-inject

    threshold_permanent: int = 20  # Turn frequency to promote to Stage 3
    threshold_archive: int = 10  # Turns of inactivity to archive (Stage 1+)

    # Spaced Intervals (Turns)
    INTERVALS: ClassVar[dict[int, int]] = {
        0: 1,  # Every turn
        1: 5,  # Every 5 turns
        2: 25,  # Every 25 turns
        3: 0,  # Always (Permanent)
    }

    def cleanup_primacy_locked(self, valid_keys: set[str]) -> None:
        """Remove invalid keys from primacy_locked that don't exist in memory."""
        self.primacy_locked = [k for k in self.primacy_locked if k in valid_keys]
        if len(self.primacy_locked) > 0:
            logger.debug(
                "Primacy lock cleanup: %d keys protected",
                len(self.primacy_locked),
            )


class TokMemory(BaseModel):
    """
    Pydantic model for the Tok Sovereign Memory.
    Provides schema validation for the semi-structured Tok format.
    """

    meta: dict[str, str] = Field(default_factory=dict)
    agent: dict[str, str] = Field(default_factory=dict)
    tools: dict[str, str] = Field(default_factory=dict)
    grammar: dict[str, str] = Field(default_factory=dict)
    state: str = ""
    hot_state: str = ""
    cold_state: str = ""
    permanent_state: str = ""
    scheduled_state: str = ""
    entropy: EntropyTracker = Field(default_factory=EntropyTracker)

    @classmethod
    def from_tok(cls, text: str) -> TokMemory:
        """Parse a Tok string into a Pydantic model."""
        sections: dict[str, str] = {}
        # Simple regex split on @tags
        chunks = re.split(
            r"^@(meta|agent|tools|grammar|state|hot_state|cold_state|permanent_state|scheduled_state|entropy)\s*",
            text,
            flags=re.MULTILINE,
        )

        for i in range(1, len(chunks), 2):
            tag = chunks[i]
            body = chunks[i + 1].strip() if i + 1 < len(chunks) else ""
            sections[tag] = body

        return cls(
            meta=cls._parse_kv(sections.get("meta", "")),
            agent=cls._parse_kv(sections.get("agent", "")),
            tools=cls._parse_multiline_kv(sections.get("tools", ""), separator=" -> "),
            grammar=cls._parse_multiline_kv(sections.get("grammar", ""), separator=": "),
            state=sections.get("state", "").strip(),
            hot_state=sections.get("hot_state", "").strip(),
            cold_state=sections.get("cold_state", "").strip(),
            permanent_state=sections.get("permanent_state", "").strip(),
            scheduled_state=sections.get("scheduled_state", "").strip(),
            entropy=cls._parse_entropy(sections.get("entropy", "")),
        )

    @staticmethod
    def _parse_entropy(text: str) -> EntropyTracker:
        """Parse the entropy heatmap from Tok format."""
        if not text:
            return EntropyTracker()

        heatmap = {}
        last_seen = {}
        primacy = []
        stages = {}
        next_ref = {}

        # Format: key|heat:N|last:M|stage:S|next:T|locked
        lines = text.splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            key = parts[0]
            heat = 0
            last = 0
            stage = 0
            nr = 0
            locked = False
            for p in parts[1:]:
                if p.startswith("heat:"):
                    heat = int(p.split(":")[1])
                if p.startswith("last:"):
                    last = int(p.split(":")[1])
                if p.startswith("stage:"):
                    stage = int(p.split(":")[1])
                if p.startswith("next:"):
                    nr = int(p.split(":")[1])
                if p == "locked":
                    locked = True

            heatmap[key] = heat
            last_seen[key] = last
            stages[key] = stage
            next_ref[key] = nr
            if locked:
                primacy.append(key)

        return EntropyTracker(
            heatmap=heatmap,
            last_seen=last_seen,
            primacy_locked=primacy,
            stages=stages,
            next_refresh=next_ref,
        )

    @staticmethod
    def _parse_kv(text: str) -> dict[str, str]:
        """Parse space-separated or semicolon-separated key:value pairs."""
        results = {}
        # Matches key:value (e.g. v:1.8)
        matches = re.findall(r"(\w+):([^; \n]+)", text)
        for k, v in matches:
            results[k] = v
        return results

    @staticmethod
    def _parse_multiline_kv(text: str, separator: str = ": ") -> dict[str, str]:
        """Parse indented or multi-line key/value pairs with a custom separator."""
        results = {}
        lines = text.splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if separator in line:
                k, v = line.split(separator, 1)
                results[k.strip()] = v.strip()
            else:
                results[line] = ""
        return results

    def _parse_facts(self, state_str: str) -> dict[str, str]:
        """Parse a state string (hot/cold/perm) into key:value facts."""
        facts: dict[str, str] = {}
        if not state_str:
            return facts
        pairs = re.split(r"[;\n]", state_str)
        for p in pairs:
            p = p.strip()
            if ":" in p:
                k, v = p.split(":", 1)
                k = k.strip()
                v = v.strip()
                # Strict Key Validation: Only alphanumeric keys, ignore metadata
                if re.match(r"^[a-zA-Z0-9_\-]+$", k) and k not in [
                    "heat",
                    "last",
                    "stage",
                    "next",
                    "locked",
                ]:
                    facts[k] = v
        return facts

    def rebalance_states(self, turn_count: int) -> None:
        """Physically move facts between tiers based on entropy heatmap."""
        # PHASE 0: INVALIDATION PASS - Detect & demote contradictions
        # (Reactive Demotion: User's new assertion invalidates stale permanent facts)
        hot_facts = self._parse_facts(self.hot_state)
        perm_facts = self._parse_facts(self.permanent_state)

        for key, new_val in hot_facts.items():
            if key in perm_facts and perm_facts[key] != new_val:
                # Contradiction detected!
                if key not in self.entropy.primacy_locked:
                    # Safe to demote (not identity-locked)
                    # HEAT TRANSFER: Transfer heat from old value to new value
                    # This prevents the "Amnesia Trap" for software agents
                    # Stage is recomputed from transferred heat, not directly inherited
                    old_heat = self.entropy.heatmap.get(key, 0)
                    logger.debug("UPDATE: %s:%s → %s", key, perm_facts[key], new_val)
                    logger.debug("HEAT INHERITED: %s (State Continuity)", old_heat)
                    del perm_facts[key]
                    self.entropy.heatmap[key] = old_heat
                    self.entropy.stages[key] = (
                        3 if old_heat >= self.entropy.threshold_permanent else self.entropy.stages.get(key, 0)
                    )
                    self.entropy.next_refresh.pop(key, None)
                else:
                    logger.debug(
                        "BLOCKED: %s is primacy-locked (identity protected)",
                        key,
                    )

        # Rebuild permanent_state without demoted facts
        self.permanent_state = "; ".join(f"{k}:{v}" for k, v in perm_facts.items())

        # Aggregate all facts
        all_facts = {}

        # Merge from all existing tiers (prefer hot over flat state)
        source_states = [
            self.state,
            self.cold_state,
            self.hot_state,
            self.permanent_state,
        ]
        for s in source_states:
            if not s:
                continue
            # Handle both semicolon and newline separation
            pairs = re.split(r"[;\n]", s)
            for p in pairs:
                p = p.strip()
                if ":" in p:
                    k, v = p.split(":", 1)
                    k = k.strip()
                    v = v.strip()
                    # Ensure metadata keys are NOT included in facts partitioning
                    if re.match(r"^[a-zA-Z0-9_\-]+$", k) and k not in [
                        "heat",
                        "last",
                        "stage",
                        "next",
                        "locked",
                    ]:
                        all_facts[k] = v

        hot = []
        cold = []
        perm = []
        scheduled = []

        for k, v in all_facts.items():
            heat = self.entropy.heatmap.get(k, 0)
            last = self.entropy.last_seen.get(k, 0)
            stage = self.entropy.stages.get(k, 0)
            next_ref = self.entropy.next_refresh.get(k, 0)
            age = turn_count - last

            # 1. Update Stages & Scheduling
            _old_stage = stage
            # If it's already in the permanent_state string OR locked OR high heat
            # (Checks source_states specifically for permanent)
            is_currently_perm = k in self._parse_facts(self.permanent_state)

            if k in self.entropy.primacy_locked or heat >= self.entropy.threshold_permanent or is_currently_perm:
                stage = 3
                self.entropy.next_refresh[k] = 0
                logger.debug("Memory Promotion: %s reached Stage 3 (Permanent)", k)
            elif stage == 0 and age >= self.entropy.threshold_archive:
                # First Decay: Move Stage 0 -> 1
                stage = 1
                self.entropy.next_refresh[k] = turn_count + self.entropy.INTERVALS[stage]
            elif stage > 0 and stage < 3:
                # Already cold - Check if we need to refresh
                if turn_count >= next_ref > 0:
                    scheduled.append(f"{k}:{v}")
                    # Push next refresh back
                    self.entropy.next_refresh[k] = turn_count + self.entropy.INTERVALS[stage]
                    # Slow decay: After a refresh, if it's still not mentioned,
                    # eventually move to Stage 2
                    if stage == 1 and age > self.entropy.INTERVALS[2]:
                        stage = 2

            self.entropy.stages[k] = stage

            # 2. Partition
            fact_str = f"{k}:{v}"
            if stage == 3:
                perm.append(fact_str)
            elif stage == 0:
                hot.append(fact_str)
            else:
                cold.append(fact_str)

        self.hot_state = "; ".join(hot)
        self.cold_state = "; ".join(cold)
        self.permanent_state = "; ".join(perm)
        self.scheduled_state = "; ".join(scheduled)
        self.state = ""

    def to_tok(self) -> str:
        """Serialize the model back to Tok protocol format."""
        lines = []
        if self.meta:
            lines.append(f"@meta {' '.join(f'{k}:{v}' for k, v in self.meta.items())}")
        if self.agent:
            lines.append(f"@agent {' '.join(f'{k}:{v}' for k, v in self.agent.items())}")

        if self.tools:
            lines.append("\n@tools")
            for k, v in self.tools.items():
                lines.append(f"  {k} -> {v}" if v else f"  {k}")

        if self.grammar:
            lines.append("\n@grammar")
            for k, v in self.grammar.items():
                lines.append(f"  {k}: {v}" if v else f"  {k}")

        if self.state:
            state_val = self.state.strip()
            if state_val:
                lines.append(f"\n@state\n  {state_val}")

        if self.hot_state:
            lines.append(f"\n@hot_state\n  {self.hot_state.strip()}")

        if self.cold_state:
            lines.append(f"\n@cold_state\n  {self.cold_state.strip()}")

        if self.permanent_state:
            lines.append(f"\n@permanent_state\n  {self.permanent_state.strip()}")

        if self.scheduled_state:
            lines.append(f"\n@scheduled_state\n  {self.scheduled_state.strip()}")

        if self.entropy.heatmap:
            lines.append("\n@entropy")
            for key, heat in self.entropy.heatmap.items():
                last = self.entropy.last_seen.get(key, 0)
                stage = self.entropy.stages.get(key, 0)
                nr = self.entropy.next_refresh.get(key, 0)
                locked = "|locked" if key in self.entropy.primacy_locked else ""
                lines.append(f"  {key}|heat:{heat}|last:{last}|stage:{stage}|next:{nr}{locked}")

        return "\n".join(lines).strip()


class ToolName(str, Enum):
    READ = "read"
    WRITE = "write"
    EDIT = "edit"
    RUN = "run"
    SEARCH = "search"
    DELTA = "delta"


class TokToolCall(BaseModel):
    """Pydantic model for validating tool calls before execution."""

    tool: ToolName
    path: str | None = None
    content: str | None = Field(default=None, alias="content")
    search: str | None = None
    replace: str | None = None
    cmd: str | None = None

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def validate_tool_call(cls, data: dict[str, Any]) -> dict[str, Any]:
        if not data.get("tool"):
            msg = "Missing tool name"
            raise ValueError(msg)
        # Validate path characters
        if data.get("path"):
            if not re.match(r"^[\w\.\-\/]+$", data["path"]):
                msg = f"Invalid path: {data['path']}"
                raise ValueError(msg)
        # Validate content doesn't have literal \n
        if data.get("content") and "\\n" in data["content"]:
            msg = "Content contains literal \\n - use real newlines"
            raise ValueError(msg)
        return data


class ReadToolSchema(BaseModel):
    path: str

    @model_validator(mode="before")
    @classmethod
    def validate_read(cls, data: dict[str, Any]) -> dict[str, Any]:
        path = (data.get("path") or "").strip()
        if not path:
            msg = "path is required"
            raise ValueError(msg)
        return data


class WriteToolSchema(BaseModel):
    path: str
    content: str | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_write(cls, data: dict[str, Any]) -> dict[str, Any]:
        path = (data.get("path") or "").strip()
        if not path:
            msg = "path is required"
            raise ValueError(msg)
        return data


class EditToolSchema(BaseModel):
    path: str
    search: str
    replace: str

    @model_validator(mode="before")
    @classmethod
    def validate_edit(cls, data: dict[str, Any]) -> dict[str, Any]:
        if not (data.get("path") or "").strip():
            msg = "path is required"
            raise ValueError(msg)
        if not (data.get("search") or "").strip():
            msg = "search field is required and must not be empty"
            raise ValueError(msg)
        if "replace" not in data or data["replace"] is None:
            msg = "replace field is required (use empty string for deletion)"
            raise ValueError(msg)
        return data


class RunToolSchema(BaseModel):
    cmd: str

    BLOCKED_PREFIXES: ClassVar[tuple[str, ...]] = ()
    BLOCKED_TOKENS: ClassVar[tuple[str, ...]] = ()

    @model_validator(mode="before")
    @classmethod
    def validate_run(cls, data: dict[str, Any]) -> dict[str, Any]:
        raw_cmd = (data.get("cmd") or "").strip()
        if not raw_cmd:
            msg = "cmd is required"
            raise ValueError(msg)
        return data


class SearchToolSchema(BaseModel):
    query: str

    @model_validator(mode="before")
    @classmethod
    def validate_search(cls, data: dict[str, Any]) -> dict[str, Any]:
        query = (data.get("query") or data.get("path") or "").strip()
        if not query:
            msg = "search query must not be empty"
            raise ValueError(msg)
        return data


class DeltaToolSchema(BaseModel):
    path: str
    delta: str | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_delta(cls, data: dict[str, Any]) -> dict[str, Any]:
        path = (data.get("path") or "").strip()
        if not path:
            msg = "path is required"
            raise ValueError(msg)
        return data


def build_tok_traceback(_tool_name: str, raw_input: str, exc: ValidationError) -> str:
    """Convert a pydantic ValidationError into a dense @error Tok block."""
    first_err = exc.errors()[0]
    loc = ".".join(str(x) for x in first_err["loc"]) if first_err["loc"] else "input"
    msg = first_err["msg"].replace(" ", "_")
    raw_preview = raw_input[:50].replace("\n", "\\n")
    if len(raw_input) > 50:
        raw_preview += "..."
    return (
        f"@error type:validation\n"
        f"  loc: {loc}\n"
        f"  msg: {msg}\n"
        f"  raw: {raw_preview}\n"
        f"  fix: Correct_{loc}_and_retry_strictly_in_TOK"
    )


TOOL_SCHEMAS: dict[str, type] = {
    "read": ReadToolSchema,
    "write": WriteToolSchema,
    "edit": EditToolSchema,
    "run": RunToolSchema,
    "search": SearchToolSchema,
    "delta": DeltaToolSchema,
}
