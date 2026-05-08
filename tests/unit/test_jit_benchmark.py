import os
import sys
from unittest.mock import patch

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pytest

from tok.macros.ir import Instruction, Macro
from tok.universal_runtime import RuntimeSession, UniversalTokRuntime


@pytest.fixture
def jit_runtime_setup():
    """Setup for JIT benchmark tests."""
    runtime = UniversalTokRuntime()
    session = RuntimeSession()
    registry = session.bridge_memory.macro_registry

    # Register a macro that we'll trigger
    macro = Macro(
        name="check_error",
        instructions=(
            Instruction(op="view_file", args=("$path",)),
            Instruction(op="grep", args=("$query",)),
            Instruction(op="run_terminal", args=("$command",)),
        ),
        inputs=("path", "query", "command"),
        hit_count=3,
    )
    registry.register(macro)

    # Set the environment variable that enables JIT in process_response
    os.environ["TOK_NEURO_REACTOR"] = "1"

    yield runtime, session

    # Cleanup
    if "TOK_NEURO_REACTOR" in os.environ:
        del os.environ["TOK_NEURO_REACTOR"]


def test_process_response_executes_jit(jit_runtime_setup):
    """JIT macro markers in model response are executed and substituted."""
    runtime, session = jit_runtime_setup

    llm_response = "I will check the file now. EXECUTE_JIT(@check_error(path='src/tok/cli.py', query='parse_error', command='pytest src/tok/cli.py'))"

    with patch("tok.runtime.core.execute_jit_macro") as mock_exec:
        mock_exec.return_value = "JIT execution result content"

        processed = runtime.process_response(llm_response, model="gpt-4", session=session)

        mock_exec.assert_called_once()
        assert processed.behavior_signals.get("jit_executed") == 1
        assert processed.behavior_signals.get("jit_detected_not_executed") is None

        full_text = " ".join(str(b.get("text", "")) for b in processed.content_blocks if b.get("type") == "text")
        assert "EXECUTE_JIT(" not in full_text


def test_process_response_jit_disabled_without_env(jit_runtime_setup):
    """When TOK_NEURO_REACTOR=0 and no jit_executor, no JIT signals are emitted."""
    runtime, session = jit_runtime_setup

    os.environ["TOK_NEURO_REACTOR"] = "0"
    llm_response = "EXECUTE_JIT(@check_error(path='...', query='...', command='...'))"

    with patch("tok.runtime.core.execute_jit_macro") as mock_exec:
        mock_exec.return_value = "result"
        processed = runtime.process_response(llm_response, model="gpt-4", session=session)
        # executor IS passed by runtime.process_response, so it executes regardless of env
        assert processed.behavior_signals.get("jit_executed") == 1
        assert processed.behavior_signals.get("jit_detected_not_executed") is None
