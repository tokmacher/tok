"""
Unit test for memory extraction - tests the extraction logic directly.

NOTE: TokOrchestrator.__init__ calls OpenAI() which requires OPENAI_API_KEY
or OPENROUTER_API_KEY. These tests are skipped unless the key is present.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "src")))

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("OPENAI_API_KEY"),
    reason="requires OPENROUTER_API_KEY or OPENAI_API_KEY",
)

from tok.adapters.orchestrator import TokOrchestrator


def test_extract_compression() -> None:
    """Test >>> line extraction."""
    tok = TokOrchestrator()

    # Test 1: >>> line present
    response = """Some text
>>> DELTA: identity=Alice project=chimera
More text
"""
    result = tok._extract_compression(response)
    assert result == ">>> DELTA: identity=Alice project=chimera", f"Expected >>> line, got: {result}"

    # Test 2: No >>> line
    response = "Just some plain text without protocol"
    result = tok._extract_compression(response)
    assert result is None, f"Expected None, got: {result}"

    # Test 3: @state block present (should fallback)
    response = """@state
identity: Alice
project: chimera
secret: X99
"""
    result = tok._extract_compression(response)
    # Should extract @state content
    assert result is not None and "identity" in result, f"Expected state content, got: {result}"


def test_extract_state() -> None:
    """Test @state block extraction."""
    tok = TokOrchestrator()

    # Test 1: @state block with content
    response = """Some text
@state
identity: Alice
project: chimera
secret: X99

More text
"""
    result = tok._extract_state_from_response(response)
    expected = "identity: Alice\nproject: chimera\nsecret: X99"
    assert result == expected, f"Expected:\n{expected}\n\nGot:\n{result}"

    # Test 2: No @state block
    response = "Just plain text"
    result = tok._extract_state_from_response(response)
    assert result is None, f"Expected None, got: {result}"

    # Test 3: @state in code block (common model output)
    response = """Understood. I have recorded:
```
@state
name: Bob
code: Z-777
```
"""
    result = tok._extract_state_from_response(response)
    # This might not work because of the code block - let's see


def test_memory_persistence_unit() -> None:
    """Test that memory is persisted and reloaded via the session bridge_memory."""
    from tok.universal_runtime import RuntimeSession

    tok = TokOrchestrator()

    # Inject a wire state directly into the session bridge_memory
    wire_state = ">>> t:1|g:test_goal|f:identity.py"
    tok.adapter.session.bridge_memory.ingest_wire_state(wire_state)
    tok.adapter.session._save_bridge_memory()

    # Reload via a new session pointing at the same memory directory
    new_session = RuntimeSession(memory_dir=tok.adapter.session.memory_dir)
    reloaded_state = new_session.load_memory()

    assert reloaded_state, "Expected non-empty state after persistence"
    assert "test_goal" in reloaded_state


if __name__ == "__main__":
    test_extract_compression()
    test_extract_state()
    test_memory_persistence_unit()
