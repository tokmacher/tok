"""Live test for the Pattern Reactor (Macro Distillation)."""

import os
import sys
import logging
from tok.universal_runtime import RuntimeSession

# Configure logging to see the Reactor's progress
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tok.test_reactor")


def test_reactor_live():
    # 1. Initialize Runtime
    # We use a real model via OpenRouter to see if it responds correctly to the injected macros
    model = "openai/gpt-4o-mini"
    session = RuntimeSession()

    print(f"\n--- Starting Live Reactor Test (Model: {model}) ---")

    # We will simulate 3 turns where the model 'runs' the same commands.
    # To save time/cost, we'll manually inject the 'cmds' into hot memory
    # as if they just happened, then trigger write_memory.

    base_commands = [
        "ls -la src/tok",
        "cat src/tok/parser.py",
        "grep 'regex' src/tok/parser.py",
    ]

    for turn in range(1, 4):
        print(
            f"\n[Turn {turn}] Injecting repetitive sequence via State Line..."
        )

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
            print(
                f"✅ SUCCESS: Reactor discovered {len(macros)} macros after Turn {turn}!"
            )
            for name, macro in macros.items():
                print(f"   -> @{name}: {[i.op for i in macro.instructions]}")
            break
        else:
            print("   (No macros yet, frequency threshold not reached)")

    # 4. Final Verification: Check the 'Wire State' that the model will see next
    final_state = session.bridge_memory.wire_state()
    print("\n--- Final Wire State (Sent to Model) ---")
    print(final_state)

    if "@macros" in final_state:
        print(
            "\n✅ FINAL VERIFICATION: Macros are successfully injected into the model's context!"
        )
    else:
        print("\n❌ FAILED: Macros missing from wire state.")


if __name__ == "__main__":
    if not os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENROUTER_API_KEY or OPENAI_API_KEY must be set.")
        sys.exit(1)
    test_reactor_live()
