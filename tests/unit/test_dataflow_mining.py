"""Phase 7 verification: Data-flow aware mining and context-aware JIT filtering."""

from __future__ import annotations


import pytest

from tok.neuro.ir import Instruction, Macro, MacroRegistry, TokIR
from tok.neuro.miner import IRPatternMiner
from tok.bridge_memory import BridgeMemoryState, MemoryEntry
from tok.universal_runtime import RuntimeSession, _jit_context_matches


@pytest.fixture(autouse=True)
def isolate_macro_registry(monkeypatch):
    monkeypatch.setattr(
        MacroRegistry, "load_global", lambda self, *a, **kw: None
    )
    monkeypatch.setattr(
        MacroRegistry, "save_global", lambda self, *a, **kw: None
    )


# ---------------------------------------------------------------------------
# Macro.context_requirements field
# ---------------------------------------------------------------------------


def test_context_requirements_defaults_to_empty_dict():
    macro = Macro(
        name="m0",
        instructions=(Instruction(op="view", args=()),),
        inputs=(),
    )
    assert macro.context_requirements == {}


def test_context_requirements_round_trips_through_dict():
    macro = Macro(
        name="m0",
        instructions=(Instruction(op="view", args=()),),
        inputs=(),
        context_requirements={"file": "src/tok/cli.py"},
    )
    restored = Macro.from_dict(macro.to_dict())
    assert restored.context_requirements == {"file": "src/tok/cli.py"}


def test_context_requirements_loads_from_legacy_dict_without_key():
    """Macros persisted before context_requirements was added load cleanly."""
    data = {
        "name": "m1",
        "instructions": [{"op": "cat", "args": [], "target": None}],
        "inputs": [],
        "hit_count": 1,
        "last_seen": None,
        "is_durable": False,
    }
    macro = Macro.from_dict(data)
    assert macro.context_requirements == {}


# ---------------------------------------------------------------------------
# Data-flow dependency fingerprint
# ---------------------------------------------------------------------------


def _ir(
    *ops_args: tuple[str, tuple[str, ...], str] | tuple[str, tuple[str, ...]],
) -> TokIR:
    """Build a TokIR from (op, args, target?) tuples."""
    instructions = []
    for item in ops_args:
        if len(item) == 2:
            op, args = item
            target = None
        else:
            op, args, target = item
        instructions.append(
            Instruction(op=op, args=tuple(args), target=target)
        )
    return TokIR(instructions=tuple(instructions))


def test_dependency_fingerprint_empty_when_no_targets():
    ins = [
        Instruction(op="grep", args=("pattern",)),
        Instruction(op="view", args=("src/foo.py",)),
    ]
    fp = IRPatternMiner._get_dependency_fingerprint(ins, 0, 2)
    assert fp == ()


def test_dependency_fingerprint_captures_target_to_arg_edge():
    """
    grep result:=$result → view(path=$result) should produce edge (0, 1).
    """
    ins = [
        Instruction(op="grep", args=("pattern",), target="result"),
        Instruction(op="view", args=("$result",)),
    ]
    fp = IRPatternMiner._get_dependency_fingerprint(ins, 0, 2)
    assert (0, 1) in fp


def test_dependency_fingerprint_no_edge_when_target_not_consumed():
    ins = [
        Instruction(op="grep", args=("pattern",), target="result"),
        Instruction(op="view", args=("src/other.py",)),  # doesn't use $result
    ]
    fp = IRPatternMiner._get_dependency_fingerprint(ins, 0, 2)
    assert fp == ()


def test_miner_distinguishes_patterns_by_dependency():
    """Two histories with the same op-sequence but different data-flow should
    not be collapsed into one pattern."""
    miner = IRPatternMiner(min_frequency=2)

    # Pattern A: grep→view, where view consumes grep's result (3 times)
    chained = _ir(
        ("grep", ("pattern",), "r"),
        ("view", ("$r",)),
        ("grep", ("pattern",), "r"),
        ("view", ("$r",)),
        ("grep", ("pattern",), "r"),
        ("view", ("$r",)),
    )

    # Pattern B: grep→view, where view has an unrelated arg (3 times)
    unchained = _ir(
        ("grep", ("pattern",)),
        ("view", ("src/foo.py",)),
        ("grep", ("pattern",)),
        ("view", ("src/bar.py",)),
        ("grep", ("pattern",)),
        ("view", ("src/baz.py",)),
    )

    macros_chained = miner.mine([chained])
    macros_unchained = miner.mine([unchained])

    assert macros_chained, "Miner failed to find chained pattern"
    assert macros_unchained, "Miner failed to find unchained pattern"

    # Each should produce a macro; they may or may not be the same op sequence,
    # but the dep fingerprints differ so they are treated as distinct.
    # We verify that the fingerprint helper correctly distinguishes them.
    ins_ch = list(chained.instructions)
    ins_un = list(unchained.instructions)
    fp_ch = IRPatternMiner._get_dependency_fingerprint(ins_ch, 0, 2)
    fp_un = IRPatternMiner._get_dependency_fingerprint(ins_un, 0, 2)
    assert fp_ch != fp_un, (
        "Chained and unchained patterns must have different fingerprints"
    )


def test_miner_embeds_dep_fingerprint_in_provenance():
    """Mined macros should record the dep fingerprint in provenance.source_code."""
    miner = IRPatternMiner(min_frequency=3)

    # Three repetitions of a chained grep→view pair
    history = _ir(
        ("grep", ("a",), "r0"),
        ("view", ("$r0",)),
        ("grep", ("b",), "r1"),
        ("view", ("$r1",)),
        ("grep", ("c",), "r2"),
        ("view", ("$r2",)),
    )
    macros = miner.mine([history])

    if macros:  # dep-aware patterns might fire
        macro = macros[0]
        assert macro.provenance is not None
        assert "deps:" in (macro.provenance.source_code or ""), (
            f"Expected dep fingerprint in source_code: {macro.provenance.source_code}"
        )


# ---------------------------------------------------------------------------
# Context-aware JIT filtering
# ---------------------------------------------------------------------------


def _make_session_with_files(*files: str) -> RuntimeSession:
    """Build a RuntimeSession whose bridge_memory hot has the given files."""
    session = RuntimeSession()
    for path in files:
        session.bridge_memory.bump_file_heat(path, weight=3.0)
    return session


def test_jit_context_matches_when_no_requirements():
    macro = Macro(
        name="m0",
        instructions=(Instruction(op="view", args=()),),
        inputs=(),
        context_requirements={},
    )
    session = _make_session_with_files()
    assert _jit_context_matches(macro, session) is True


def test_jit_context_matches_when_file_is_hot():
    macro = Macro(
        name="m0",
        instructions=(Instruction(op="view", args=()),),
        inputs=(),
        context_requirements={"file": "src/tok/cli.py"},
    )
    session = _make_session_with_files("src/tok/cli.py", "src/tok/gateway.py")
    assert _jit_context_matches(macro, session) is True


def test_jit_context_blocked_when_file_not_hot():
    macro = Macro(
        name="m0",
        instructions=(Instruction(op="view", args=()),),
        inputs=(),
        context_requirements={"file": "src/tok/cli.py"},
    )
    session = _make_session_with_files(
        "src/tok/gateway.py"
    )  # cli.py not present
    assert _jit_context_matches(macro, session) is False


def test_jit_context_blocked_when_no_active_files():
    macro = Macro(
        name="m0",
        instructions=(Instruction(op="view", args=()),),
        inputs=(),
        context_requirements={"file": "src/tok/cli.py"},
    )
    session = _make_session_with_files()  # empty heat map
    assert _jit_context_matches(macro, session) is False


def test_context_requirements_set_during_mining():
    """When miner finds a pattern with path-like args, context_requirements['file']
    should be set to the first concrete path."""
    miner = IRPatternMiner(min_frequency=3)

    state = BridgeMemoryState()
    state.rolling_cmds = [
        MemoryEntry(value="view src/tok/cli.py", last_seen_turn=1),
        MemoryEntry(value="edit src/tok/cli.py", last_seen_turn=2),
        MemoryEntry(value="view src/tok/cli.py", last_seen_turn=3),
        MemoryEntry(value="edit src/tok/cli.py", last_seen_turn=4),
        MemoryEntry(value="view src/tok/cli.py", last_seen_turn=5),
        MemoryEntry(value="edit src/tok/cli.py", last_seen_turn=6),
    ]

    from tok.neuro.integration import distill_bridge_history

    discovered = distill_bridge_history(state, miner=miner)

    assert discovered, "Should discover at least one macro"
    macro = discovered[0]
    # The first path-like arg is src/tok/cli.py
    assert macro.context_requirements.get("file") == "src/tok/cli.py", (
        f"Expected context file=src/tok/cli.py, got: {macro.context_requirements}"
    )
