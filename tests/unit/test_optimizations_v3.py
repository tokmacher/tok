"""Tests for Local Mesh Discovery and Macro Healing."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

from tok.neuro.ir import Instruction, Macro, MacroRegistry
from tok.runtime.memory.bridge_memory import BridgeMemoryState
from tok.universal_runtime import (
    RuntimeRequest,
    RuntimeSession,
    UniversalTokRuntime,
)

# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_tok_storage():
    with (
        patch.object(MacroRegistry, "load_global"),
        patch.object(MacroRegistry, "save_global"),
        patch.object(
            RuntimeSession,
            "_load_bridge_memory",
            side_effect=BridgeMemoryState,
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(
    messages: list[dict[str, Any]] | None = None,
) -> RuntimeRequest:
    return RuntimeRequest(
        model="claude-sonnet-4-6",
        messages=messages or [{"role": "user", "content": "hello"}],
    )


# ---------------------------------------------------------------------------
# Local Mesh Discovery
# ---------------------------------------------------------------------------


class TestLocalMeshDiscovery:
    @patch("tok.universal_runtime._discover_project_markers")
    def test_speculative_hint_injected_via_marker(self, mock_discover) -> None:
        """Speculative injection should fire if a project marker exists in the CWD."""
        mock_discover.return_value = frozenset({"package.json"})

        session = RuntimeSession()
        session._project_markers = frozenset({"package.json"})
        # Macro requires package.json
        macro = Macro(
            name="npm_install",
            instructions=(Instruction(op="bash", args=("npm install",)),),
            inputs=(),
            hit_count=5,
            context_requirements={"marker_file": "package.json"},
        )
        session.bridge_memory.macro_registry.macros["npm_install"] = macro

        runtime = UniversalTokRuntime()
        prepared = runtime.prepare_request(_make_request(), session)

        system = prepared.body.get("system", "")
        assert "@npm_install" in system
        assert "Available macros" in system

    @patch("tok.universal_runtime._discover_project_markers")
    def test_no_hint_when_marker_missing(self, mock_discover) -> None:
        """Macro requiring a marker should NOT be injected if marker is missing."""
        session = RuntimeSession()
        session._project_markers = frozenset({"requirements.txt"})  # mismatch

        macro = Macro(
            name="npm_install",
            instructions=(Instruction(op="bash", args=("npm install",)),),
            inputs=(),
            hit_count=5,
            context_requirements={"marker_file": "package.json"},
        )
        session.bridge_memory.macro_registry.macros["npm_install"] = macro

        runtime = UniversalTokRuntime()
        prepared = runtime.prepare_request(_make_request(), session)

        system = prepared.body.get("system", "")
        assert "@npm_install" not in system


# ---------------------------------------------------------------------------
# Macro Healing
# ---------------------------------------------------------------------------


class TestMacroHealing:
    def test_macro_heals_when_divergence_detected(self) -> None:
        """Macro should update its instructions if the agent performs a different successful sequence."""
        session = RuntimeSession()

        # Original macro: just op1
        macro = Macro(
            name="repaired_op",
            instructions=(Instruction(op="op1", args=()),),
            inputs=(),
            hit_count=10,
        )
        session.bridge_memory.macro_registry.macros["repaired_op"] = macro

        # 1. Simulate JIT offer
        runtime = UniversalTokRuntime()
        # Mock match_recent_sequence to return our macro
        with patch.object(
            session.bridge_memory.macro_registry,
            "match_recent_sequence",
            return_value=macro,
        ):
            # We need some rolling cmds so JIT logic runs
            session.write_memory(">>> cmds:setup_cmd")
            runtime.prepare_request(_make_request(), session)

        assert session._pending_macro_heal == "repaired_op"

        # 2. Simulate the agent doing something else successfully
        # Use turn 1 to ensure they are "recent" relative to JIT offer turn 0
        with patch.dict(os.environ, {"TOK_MACRO_HEAL": "1"}):
            session.write_memory(">>> t:1|cmds:op1,op2|facts:done")

        # 3. Verify the macro was updated
        updated = session.bridge_memory.macro_registry.get("repaired_op")
        assert updated is not None
        ops = [ins.op for ins in updated.instructions]
        assert ops == ["op1", "op2"]
        assert session._pending_macro_heal == ""

    def test_no_healing_when_no_offer(self) -> None:
        """Macro should NOT update if it wasn't offered (no _pending_macro_heal)."""
        session = RuntimeSession()
        macro = Macro(
            name="steady_op",
            instructions=(Instruction(op="op1", args=()),),
            inputs=(),
            hit_count=10,
        )
        session.bridge_memory.macro_registry.macros["steady_op"] = macro

        session.write_memory(">>> cmds:op1,op3|facts:done")

        updated = session.bridge_memory.macro_registry.get("steady_op")
        assert updated is not None, "Macro should exist"
        assert len(updated.instructions) == 1
        assert updated.instructions[0].op == "op1"

    def test_no_healing_on_identical_sequence(self) -> None:
        """Macro should NOT update if the agent's sequence was identical to the macro."""
        session = RuntimeSession()
        macro = Macro(
            name="perfect_op",
            instructions=(Instruction(op="op1", args=()),),
            inputs=(),
            hit_count=10,
        )
        session.bridge_memory.macro_registry.macros["perfect_op"] = macro
        session._pending_macro_heal = "perfect_op"
        session._pending_macro_heal_turn = 0

        session.write_memory(">>> t:1|cmds:op1|facts:done")

        with patch.object(
            session.bridge_memory.macro_registry,
            "update_from_repair",
            return_value=False,
        ):
            session.write_memory(">>> t:2|facts:done")
            # update_from_repair returns False if ops are identical

        assert session._pending_macro_heal == ""
