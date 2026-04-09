from __future__ import annotations

import pytest

from tok.neuro.integration import distill_bridge_history
from tok.neuro.ir import MacroRegistry
from tok.runtime.memory.bridge_memory import BridgeMemoryState, MemoryEntry


@pytest.fixture(autouse=True)
def isolate_macro_registry(monkeypatch) -> None:
    """Prevent tests from reading or writing the global macro store."""
    monkeypatch.setattr(MacroRegistry, "load_global", lambda self, *a, **_: None)
    monkeypatch.setattr(MacroRegistry, "save_global", lambda self, *a, **_: None)


def test_neuro_reactor_discovers_simple_pattern() -> None:
    # Setup a state with a repeating sequence of commands
    state = BridgeMemoryState()

    # Sequence 1: pytest -> cat -> edit
    state.rolling_cmds = [
        MemoryEntry(value="pytest -v test_foo.py", last_seen_turn=1),
        MemoryEntry(value="cat test_foo.py", last_seen_turn=2),
        MemoryEntry(value="edit test_foo.py", last_seen_turn=3),
        # Sequence 2: pytest -> cat -> edit
        MemoryEntry(value="pytest -v test_bar.py", last_seen_turn=4),
        MemoryEntry(value="cat test_bar.py", last_seen_turn=5),
        MemoryEntry(value="edit test_bar.py", last_seen_turn=6),
        # Sequence 3: pytest -> cat -> edit (Trigger threshold)
        MemoryEntry(value="pytest -v test_baz.py", last_seen_turn=7),
        MemoryEntry(value="cat test_baz.py", last_seen_turn=8),
        MemoryEntry(value="edit test_baz.py", last_seen_turn=9),
    ]

    # Run the distillation hook
    discovered = distill_bridge_history(state)

    # The simple miner should find the overlapping sequence of 3 operations
    assert len(discovered) >= 1
    macro = discovered[0]

    # The macro should reflect the sequence of operations (pytest, cat, edit)
    ops = [ins.op for ins in macro.instructions]
    assert ops == ["pytest", "cat", "edit"]

    # verify it registered to memory payload
    assert state.macro_registry.get(macro.name) is not None


def test_neuro_reactor_serializes_macro() -> None:
    state = BridgeMemoryState()

    state.rolling_cmds = [
        MemoryEntry(value="ls -la", last_seen_turn=1),
        MemoryEntry(value="cat file.txt", last_seen_turn=2),
        MemoryEntry(value="ls -la", last_seen_turn=3),
        MemoryEntry(value="cat file2.txt", last_seen_turn=4),
        MemoryEntry(value="ls -la", last_seen_turn=5),
        MemoryEntry(value="cat file3.txt", last_seen_turn=6),
    ]

    distill_bridge_history(state)

    # Render the tok file representation
    tok_output = state.to_tok()

    assert "@macros" in tok_output
    assert "auto_macro_0" in tok_output
    assert "|> @auto_macro_0(" in tok_output
    assert "ls(-la)" in tok_output


def test_neuro_reactor_deserializes_macro() -> None:
    raw_tok = "@memory version:bridge-v1 turn:10\n@hot\n@macros\n  |> @auto_macro_0(p0, p1) -> ls(-la) | cat($p1)\n"

    state = BridgeMemoryState.from_tok(raw_tok)

    macro = state.macro_registry.get("auto_macro_0")
    assert macro is not None
    assert len(macro.inputs) == 2
    assert macro.inputs[0] == "p0"

    ops = [ins.op for ins in macro.instructions]
    assert ops == ["ls", "cat"]
