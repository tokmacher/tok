import os
import sys
from unittest.mock import patch

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pytest

from tok.neuro.ir import Instruction, Macro, MacroRegistry
from tok.runtime.core import RuntimeSession, UniversalTokRuntime
from tok.runtime.memory.bridge_memory import MemoryEntry
from tok.runtime.policy.macro_handling import execute_jit_macro
from tok.runtime.types import RuntimeRequest


@pytest.fixture
def jit_execution_setup():
    """Setup for JIT execution tests."""
    registry = MacroRegistry()
    # Define a test macro: grep -> view
    macro = Macro(
        name="grep_view",
        instructions=(
            Instruction(op="grep", args=("$pattern",)),
            Instruction(op="view", args=("$file",)),
        ),
        inputs=("pattern", "file"),
        hit_count=3,  # Ensure it passes threshold by default for matching tests
    )
    registry.register(macro)

    session = RuntimeSession()
    session.bridge_memory.macro_registry = registry
    # Simulate some recent commands that match the macro
    session.bridge_memory.rolling_cmds = [
        MemoryEntry(value="grep reactor", score=1, last_seen_turn=1),
        MemoryEntry(value="view src/tok/neuro/ir.py", score=1, last_seen_turn=2),
    ]

    yield registry, session, macro


def test_jit_matching_in_prepare_request(jit_execution_setup):
    """Test that JIT matching signals are present in prepared request."""
    registry, session, macro = jit_execution_setup

    # Create a dummy request
    request = RuntimeRequest(
        model="claude-3-opus-20240229",
        messages=[{"role": "user", "content": "What was the result of the grep?"}],
        tool_compatible=True,
    )

    runtime = UniversalTokRuntime()
    prepared = runtime.prepare_request(request, session)

    # Check if jit signals are present in the returned prepared request
    assert prepared.behavior_signals.get("jit_offer_available") == 1
    assert prepared.behavior_signals.get("jit_offer_grep_view") == 1


def test_jit_matching_threshold_enforced(jit_execution_setup):
    """Test that JIT threshold is enforced."""
    registry, session, macro = jit_execution_setup

    # Set hit_count to 1 (below default threshold of 3)
    macro.hit_count = 1
    request = RuntimeRequest(
        model="claude-3-opus-20240229",
        messages=[{"role": "user", "content": "test"}],
        tool_compatible=True,
    )
    runtime = UniversalTokRuntime()
    prepared = runtime.prepare_request(request, session)

    # Should NOT have jit signals
    assert prepared.behavior_signals.get("jit_offer_available") is None


def test_jit_execution_runner(jit_execution_setup):
    """Test the symbolic runner directly."""
    registry, session, macro = jit_execution_setup

    # Test the symbolic runner directly
    with patch("tok.neuro.ir.execute_ir") as mock_exec:
        mock_exec.return_value = "Found 3 matches."

        result = execute_jit_macro(
            session,
            "grep_view",
            "pattern='reactor', file='src/tok/neuro/ir.py'",
        )

        assert result == "Found 3 matches."
        # Verify macro use was recorded (3 initial + 1)
        assert macro.hit_count == 4


def test_jit_arg_parsing():
    """Test JIT argument parsing."""
    from tok.runtime.policy.macro_handling import _parse_jit_args

    args = _parse_jit_args("pattern='test', file=\"path/to/file.py\", count=5")
    assert args["pattern"] == "test"
    assert args["file"] == "path/to/file.py"
    assert args["count"] == "5"
