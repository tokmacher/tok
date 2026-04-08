import os
import sys
import unittest
from unittest.mock import patch

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tok.neuro.ir import Instruction, Macro
from tok.universal_runtime import RuntimeSession, UniversalTokRuntime


class TestJitBenchmark(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = UniversalTokRuntime()
        self.session = RuntimeSession()
        self.registry = self.session.bridge_memory.macro_registry

        # Register a macro that we'll trigger
        self.macro = Macro(
            name="check_error",
            instructions=(
                Instruction(op="view_file", args=("$path",)),
                Instruction(op="grep", args=("$query",)),
                Instruction(op="run_terminal", args=("$command",)),
            ),
            inputs=("path", "query", "command"),
            hit_count=3,
        )
        self.registry.register(self.macro)

        # Set the environment variable that enables JIT in process_response
        os.environ["TOK_NEURO_REACTOR"] = "1"

    def test_process_response_executes_jit(self) -> None:
        # A response from the LLM that accepts the JIT offer
        llm_response = "I will check the file now. EXECUTE_JIT(@check_error(path='src/tok/cli.py', query='parse_error', command='pytest src/tok/cli.py'))"

        with patch("tok.runtime.core.execute_jit_macro") as mock_exec:
            mock_exec.return_value = "JIT execution result content"

            processed = self.runtime.process_response(llm_response, model="gpt-4", session=self.session)

            # 1. Verify JIT execution was called
            mock_exec.assert_called_once()

            # 2. Verify signal was emitted
            assert processed.behavior_signals.get("jit_executed") == 1
            assert processed.behavior_signals.get("jit_macro_executed_check_error") == 1

            # 3. Verify content was appended to the visible response
            found_jit_result = False
            for block in processed.content_blocks:
                if block.get("type") == "text" and "[JIT Execution Result for @check_error]" in block.get("text", ""):
                    found_jit_result = True
                    assert "JIT execution result content" in block["text"]

            assert found_jit_result

    def test_process_response_jit_disabled_without_env(self) -> None:
        os.environ["TOK_NEURO_REACTOR"] = "0"
        llm_response = "EXECUTE_JIT(@check_error(path='...', query='...', command='...'))"

        with patch("tok.runtime.core.execute_jit_macro") as mock_exec:
            processed = self.runtime.process_response(llm_response, model="gpt-4", session=self.session)
            mock_exec.assert_not_called()
            assert processed.behavior_signals.get("jit_executed") is None


if __name__ == "__main__":
    unittest.main()
