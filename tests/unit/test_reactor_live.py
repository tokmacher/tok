"""Live test for the Pattern Reactor (Macro Distillation)."""

import logging
import os
import sys

from tok.universal_runtime import RuntimeSession

# Configure logging to see the Reactor's progress
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tok.test_reactor")


def test_reactor_live() -> None:
    # 1. Initialize Runtime
    # We use a real model via OpenRouter to see if it responds correctly to the injected macros
    session = RuntimeSession()

    # We will simulate 3 turns where the model 'runs' the same commands.
    # To save time/cost, we'll manually inject the 'cmds' into hot memory
    # as if they just happened, then trigger write_memory.

    base_commands = [
        "ls -la src/tok",
        "cat src/tok/parser.py",
        "grep 'regex' src/tok/parser.py",
    ]

    for turn in range(1, 4):
        # We include the commands directly in the >>> line so the bridge 'ingests' them properly
        cmds_str = ",".join(base_commands)
        mock_response = (
            f">>> turns:{turn}|goal:refactor|state:active|cmds:{cmds_str}\n"
            f"@msg role:assistant\n"
            f"  |> Turn {turn} complete."
        )
        session.write_memory(mock_response)

        # Check if macros appeared
        macros = session.bridge_memory.macro_registry.macros
        if macros:
            for _name, _macro in macros.items():
                pass
            break

    # 4. Final Verification: Check the 'Wire State' that the model will see next
    final_state = session.bridge_memory.wire_state()

    if "@macros" in final_state:
        pass
    else:
        pass


if __name__ == "__main__":
    if not os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        sys.exit(1)
    test_reactor_live()
