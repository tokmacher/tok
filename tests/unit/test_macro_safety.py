"""Tests for evidence safety gating during macro execution."""

from __future__ import annotations

from tok.macros.ir import Instruction, Macro
from tok.runtime.core import RuntimeSession
from tok.runtime.policy import macro_handling
from tok.runtime.repeat_targets import evidence_identity_key


def test_edit_macro_blocked_without_exact_evidence() -> None:
    session = RuntimeSession()
    macro = Macro(
        name="m1",
        instructions=(Instruction(op="edit", args=("$p0",), target=None),),
        inputs=("p0",),
    )
    session.bridge_memory.macro_registry.macros["m1"] = macro
    session.bridge_memory.macro_registry.get = session.bridge_memory.macro_registry.macros.get  # type: ignore[attr-defined]
    key = evidence_identity_key("Read", path="src/a.py")
    assert key is not None
    session.record_non_exact_evidence(key, form="summary")
    out = macro_handling.execute_jit_macro(session, "m1", "p0=src/a.py")
    assert "requires exact evidence" in out


def test_edit_macro_allowed_with_exact_evidence() -> None:
    session = RuntimeSession()
    macro = Macro(
        name="m1",
        instructions=(Instruction(op="view", args=("$p0",), target=None),),
        inputs=("p0",),
    )
    session.bridge_memory.macro_registry.macros["m1"] = macro
    session.bridge_memory.macro_registry.get = session.bridge_memory.macro_registry.macros.get  # type: ignore[attr-defined]
    key = evidence_identity_key("Read", path="src/a.py")
    assert key is not None
    session.record_exact_evidence(key)
    out = macro_handling.execute_jit_macro(session, "m1", "p0=src/a.py")
    assert "requires exact evidence" not in out
