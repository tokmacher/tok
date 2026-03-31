"""Phase 5 verification: IRPatternMiner produces $pN placeholders for path arguments."""

from __future__ import annotations

import pytest

from tok.bridge_memory import BridgeMemoryState, MemoryEntry
from tok.neuro.integration import distill_bridge_history
from tok.neuro.ir import MacroRegistry


@pytest.fixture(autouse=True)
def isolate_macro_registry(monkeypatch):
    monkeypatch.setattr(
        MacroRegistry, "load_global", lambda self, *a, **kw: None
    )
    monkeypatch.setattr(
        MacroRegistry, "save_global", lambda self, *a, **kw: None
    )


def test_parameterized_macro_uses_placeholder_for_path_args():
    """Miner should replace path-like args (containing '/') with $p0, $p1, etc."""
    state = BridgeMemoryState()

    # Three repetitions of: view src/foo.py → edit src/foo.py
    state.rolling_cmds = [
        MemoryEntry(value="view src/alpha.py", last_seen_turn=1),
        MemoryEntry(value="edit src/alpha.py", last_seen_turn=2),
        MemoryEntry(value="view src/beta.py", last_seen_turn=3),
        MemoryEntry(value="edit src/beta.py", last_seen_turn=4),
        MemoryEntry(value="view src/gamma.py", last_seen_turn=5),
        MemoryEntry(value="edit src/gamma.py", last_seen_turn=6),
    ]

    discovered = distill_bridge_history(state)

    assert len(discovered) >= 1, "Miner should discover at least one macro"
    macro = discovered[0]

    ops = [ins.op for ins in macro.instructions]
    assert ops == ["view", "edit"], f"Expected [view, edit], got {ops}"

    # At least one arg should be a $pN placeholder
    all_args = [arg for ins in macro.instructions for arg in ins.args]
    placeholders = [
        a for a in all_args if isinstance(a, str) and a.startswith("$p")
    ]
    assert placeholders, f"Expected $pN placeholders in args, got: {all_args}"

    # Inputs tuple should be populated (not empty as with the old hardcoded logic)
    assert len(macro.inputs) > 0, (
        f"Expected non-empty inputs, got: {macro.inputs}"
    )


def test_non_path_args_are_not_parameterized():
    """Short args and args without '/' should stay literal (not replaced with $pN)."""
    state = BridgeMemoryState()

    # Three repetitions of: pytest -v → cat file.txt (no paths with /)
    state.rolling_cmds = [
        MemoryEntry(value="pytest -v", last_seen_turn=1),
        MemoryEntry(value="cat foo", last_seen_turn=2),
        MemoryEntry(value="pytest -v", last_seen_turn=3),
        MemoryEntry(value="cat bar", last_seen_turn=4),
        MemoryEntry(value="pytest -v", last_seen_turn=5),
        MemoryEntry(value="cat baz", last_seen_turn=6),
    ]

    discovered = distill_bridge_history(state)

    if discovered:
        macro = discovered[0]
        all_args = [arg for ins in macro.instructions for arg in ins.args]
        placeholders = [
            a for a in all_args if isinstance(a, str) and a.startswith("$p")
        ]
        # No path-like args → no placeholders expected
        assert not placeholders, (
            f"Unexpected $pN placeholders for non-path args: {all_args}"
        )
