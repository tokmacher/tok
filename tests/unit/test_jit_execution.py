import os
import sys
from unittest.mock import patch

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pytest

from tok.macros.ir import Instruction, Macro, MacroRegistry
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
    with patch("tok.macros.ir.execute_ir") as mock_exec:
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


# ---------------------------------------------------------------------------
# RED tests: process_response_impl must execute EXECUTE_JIT tokens in text
# ---------------------------------------------------------------------------


def test_process_response_executes_jit_token_in_text(jit_execution_setup):
    """EXECUTE_JIT(@name(args)) in model response must be replaced with execution result."""
    from unittest.mock import patch as _patch

    from tok.runtime._runtime_orchestration import process_response_impl

    registry, session, macro = jit_execution_setup
    macro.hit_count = 5

    raw_response = ">>> t:1|g:find pattern\nEXECUTE_JIT(@grep_view(pattern='reactor', file='src/tok/neuro/ir.py'))\n"

    with _patch("tok.macros.ir.execute_ir", return_value="Found 3 matches in ir.py"):
        runtime = UniversalTokRuntime()
        result = process_response_impl(
            runtime,
            raw_response,
            model="claude-3-5-sonnet-20241022",
            session=session,
            jit_executor=execute_jit_macro,
        )

    # The EXECUTE_JIT token must not appear in the returned content
    full_text = " ".join(str(b.get("text", "")) for b in result.content_blocks if b.get("type") == "text")
    assert "EXECUTE_JIT(" not in full_text, f"Raw JIT token leaked into output: {full_text!r}"
    # The execution result must appear somewhere (inline or via signal)
    assert result.behavior_signals.get("jit_executed", 0) == 1, (
        f"jit_executed signal missing; signals={result.behavior_signals}"
    )


def test_process_response_jit_signal_absent_without_token(jit_execution_setup):
    """Responses without EXECUTE_JIT must not set jit_executed."""
    from tok.runtime._runtime_orchestration import process_response_impl

    _registry, session, _macro = jit_execution_setup
    raw_response = ">>> t:1|g:regular response\n"

    runtime = UniversalTokRuntime()
    result = process_response_impl(
        runtime,
        raw_response,
        model="claude-3-5-sonnet-20241022",
        session=session,
        jit_executor=execute_jit_macro,
    )

    assert result.behavior_signals.get("jit_executed", 0) == 0


def test_process_response_jit_no_executor_emits_not_executed_signal(jit_execution_setup, monkeypatch):
    """Without a jit_executor, EXECUTE_JIT in text must set jit_detected_not_executed."""
    from tok.runtime._runtime_orchestration import process_response_impl

    _registry, session, _macro = jit_execution_setup
    monkeypatch.setenv("TOK_NEURO_REACTOR", "1")

    raw_response = ">>> t:1|g:find\nEXECUTE_JIT(@grep_view(pattern='x', file='y.py'))\n"

    runtime = UniversalTokRuntime()
    result = process_response_impl(
        runtime,
        raw_response,
        model="claude-3-5-sonnet-20241022",
        session=session,
        jit_executor=None,
    )

    assert result.behavior_signals.get("jit_detected_not_executed", 0) == 1


def test_process_response_jit_no_executor_and_env_disabled_leaves_marker_unsignaled(jit_execution_setup, monkeypatch):
    """Without an executor and without the reactor flag, JIT markers are inert text."""
    from tok.runtime._runtime_orchestration import process_response_impl

    _registry, session, _macro = jit_execution_setup
    monkeypatch.setenv("TOK_NEURO_REACTOR", "0")

    raw_response = ">>> t:1|g:find\nEXECUTE_JIT(@grep_view(pattern='x', file='y.py'))\n"

    runtime = UniversalTokRuntime()
    result = process_response_impl(
        runtime,
        raw_response,
        model="claude-3-5-sonnet-20241022",
        session=session,
        jit_executor=None,
    )

    full_text = " ".join(str(b.get("text", "")) for b in result.content_blocks if b.get("type") == "text")
    assert "EXECUTE_JIT(" in full_text
    assert result.behavior_signals.get("jit_executed") is None
    assert result.behavior_signals.get("jit_detected_not_executed") is None


def test_process_response_does_not_execute_jit_marker_inside_fenced_code(jit_execution_setup):
    """A documented JIT marker in a code fence must not run as a command."""
    from tok.runtime._runtime_orchestration import process_response_impl

    _registry, session, _macro = jit_execution_setup
    raw_response = (
        "Here is the literal syntax:\n"
        "```text\n"
        "EXECUTE_JIT(@grep_view(pattern='reactor', file='src/tok/neuro/ir.py'))\n"
        "```\n"
    )

    runtime = UniversalTokRuntime()
    calls: list[tuple[str, str]] = []

    def fail_if_called(_session: RuntimeSession, name: str, args: str) -> str:
        calls.append((name, args))
        raise AssertionError("JIT executor must not run for fenced examples")

    result = process_response_impl(
        runtime,
        raw_response,
        model="claude-3-5-sonnet-20241022",
        session=session,
        jit_executor=fail_if_called,
    )

    assert calls == []
    assert result.behavior_signals.get("jit_executed") is None
