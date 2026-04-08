import pytest

from tok.compression import compress_user_prompt
from tok.runtime.memory.bridge_memory import (
    BridgeMemoryState,
    clean_system_context,
)
from tok.universal_runtime import detect_prompt_bloat


@pytest.fixture
def memory_state():
    return BridgeMemoryState()


def test_detect_prompt_bloat_size_threshold() -> None:
    """Verify that prompts over the character threshold are detected."""
    small = "A short system prompt."
    large = "x" * 2001
    assert not detect_prompt_bloat(small)
    assert detect_prompt_bloat(large)


def test_detect_prompt_bloat_leakage() -> None:
    """Verify that user prompt leakage into system context is detected."""
    user_prompt = "Implement a calculator with these exact specs:" + (" y" * 150)
    system_prompt = f"Previous instructions: {user_prompt}\nNow do it."

    # Correctly identifies leakage when significant part of user prompt is in system
    assert detect_prompt_bloat(system_prompt, user_prompt)

    # Doesn't trigger for small overlapping words
    assert not detect_prompt_bloat("Instructions: do it.", "Implement it.")


def test_compress_user_prompt_complex_markdown() -> None:
    """Test compression with nested markdown, lists, and multiple goals."""
    verbose = """
    # High Level Goal
    We need to implement Phase 7b of the project.

    ## Requirements
    1. Speed must be under 100ms.
    2. Security should be the top priority.
       * Use OAuth2 for all connections.
       * Ensure data is encrypted at rest.

    ## Constraints
    - Never use external libraries for encryption.
    - Avoid using global state.

    ## Files
    Modify: `src/tok/core.py`, `tests/main.py`.
    Also check `config.yaml`.
    """
    compressed = compress_user_prompt(verbose)

    # Verify goals are extracted (heuristically)
    assert "Phase 7b" in compressed
    assert "Speed must" in compressed or "under 100ms" in compressed

    # Verify files are extracted
    assert "src/tok/core.py" in compressed
    assert "config.yaml" in compressed

    # Verify constraints are extracted
    assert "Never use" in compressed or "external libraries" in compressed
    assert "Avoid using" in compressed


def test_compress_user_prompt_no_matches_fallback() -> None:
    """Verify fallback behavior when no heuristics match."""
    verbose = "This is a very long string that doesn't look like instructions but is very long " * 10
    compressed = compress_user_prompt(verbose)
    assert "goal:" in compressed
    assert len(compressed) < 150


def test_clean_system_context_multiple_calls() -> None:
    """Ensure cleaning doesn't fail or double-compress weirdly on repeat calls."""
    state = BridgeMemoryState()
    verbose = "Goal: Task A\n" + ("x" * 3000)

    # First clean
    cleaned_1 = clean_system_context(state, verbose)
    assert "Task A" in cleaned_1

    # Second clean with already cleaned prompt should still work and be stable
    cleaned_2 = clean_system_context(state, cleaned_1)
    assert "Task A" in cleaned_2
    assert len(cleaned_2) <= len(cleaned_1)


def test_compress_user_prompt_file_extensions() -> None:
    """Verify various file extensions are captured."""
    text = "Check main.rs, helper.go, and script.sh"
    compressed = compress_user_prompt(text)
    assert "main.rs" in compressed
    assert "helper.go" in compressed
    assert "script.sh" in compressed


def test_clean_system_context_memory_integration() -> None:
    """Verify extracted info is actually in the memory state after cleaning."""
    state = BridgeMemoryState()
    verbose = "Implement Phase 8 requirements pronto."

    clean_system_context(state, verbose)

    wire = state.wire_state()
    assert "Phase 8" in wire
    # Should be in goal or facts
    assert "g:Implement Phase 8" in wire or "x:Implement Phase 8" in wire


def test_clean_system_context_preserves_list_shape_without_cached_text_block(
    monkeypatch,
) -> None:
    from tok import compression

    state = BridgeMemoryState()
    monkeypatch.setattr(
        compression,
        "compress_user_prompt",
        lambda prompt: "g:phase_9|constraints:preserve_shape",
    )

    cleaned = clean_system_context(
        state,
        [
            {"type": "text", "text": "Prelude"},
            {"type": "text", "text": "Goal: Phase 9\n" + ("noise " * 600)},
            {
                "type": "tool_use",
                "id": "sys_tool",
                "name": "noop",
                "input": {},
            },
        ],
    )

    assert isinstance(cleaned, list)
    assert cleaned[0]["text"] == "Prelude"
    assert cleaned[1]["text"] == "### Optimized Task Context\ng:phase_9|constraints:preserve_shape"
    assert cleaned[2]["type"] == "tool_use"


def test_compress_user_prompt_megaprompt() -> None:
    """Simulate a massive 30K+ char prompt with noise and a single clear goal."""
    noise = "This is a random log line from a previous session.\n" * 500
    goal_line = "Goal: Ensure Phase 7b is verified robustly."
    more_noise = "Another repetitive line.\n" * 500

    verbose = noise + goal_line + "\n" + more_noise
    assert len(verbose) > 25000

    compressed = compress_user_prompt(verbose)

    assert "Phase 7b" in compressed
    assert len(compressed) < 2000


def test_incremental_stacking() -> None:
    """Verify that as prompts 'stack up', the bridge consolidates to the latest goal."""
    from typing import cast

    state = BridgeMemoryState()

    sys1 = "Goal: Phase A"
    clean1 = clean_system_context(state, sys1)
    state.turn += 1

    sys2 = cast("str", clean1) + "\nGoal: Phase B"
    clean_system_context(state, sys2)
    state.turn += 1

    state.wire_state()
    hot_goals = [e.value for e in state.hot.get("goal", [])]
    durable_goals = [e.value for e in state.durable.get("goal", [])]
    all_goals = hot_goals + durable_goals

    assert any("Phase B" in g for g in all_goals)


if __name__ == "__main__":
    # Manual run if not using pytest
    pytest.main([__file__])
