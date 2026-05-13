"""Tests for tool_map-driven macro parameterization."""

from __future__ import annotations

from tok.macros.ir import Instruction
from tok.macros.parameterization import parameterize_instructions


def test_parameterize_grep_and_view_uses_tool_map_slots() -> None:
    ins = [
        Instruction(op="grep", args=("TODO", "src/"), target=None),
        Instruction(op="view", args=("src/a.py",), target=None),
    ]
    out, inputs, _context = parameterize_instructions(ins)
    assert inputs == ("p0", "p1", "p2")
    assert out[0].args == ("$p0", "$p1")
    assert out[1].args == ("$p2",)


def test_parameterize_collapses_different_grep_terms() -> None:
    ins1 = [
        Instruction(op="grep", args=("TODO", "src/"), target=None),
        Instruction(op="view", args=("src/a.py",), target=None),
    ]
    ins2 = [
        Instruction(op="grep", args=("FIXME", "src/"), target=None),
        Instruction(op="view", args=("src/b.py",), target=None),
    ]
    out1, inputs1, _ = parameterize_instructions(ins1)
    out2, inputs2, _ = parameterize_instructions(ins2)
    assert inputs1 == inputs2
    assert out1 == out2


def test_parameterize_unknown_op_falls_back_to_path_heuristic() -> None:
    ins = [
        Instruction(op="unknown", args=("src/a.py", "literal"), target=None),
    ]
    out, inputs, _ = parameterize_instructions(ins)
    assert inputs == ("p0",)
    assert out[0].args[0] == "$p0"
    assert out[0].args[1] == "literal"
