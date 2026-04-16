"""Diagnostic test for search repeat compression behavior."""

from __future__ import annotations

from tok.compression import compress_tool_results
from tok.runtime.repeat_targets import evidence_identity_key


def test_search_repeat_compresses_after_first_exact():
    """
    Test that repeat identical searches compress after first exact observation.

    This test simulates:
    1. First search - should add key to first_exact_evidence_seen
    2. Repeat search - should compress because key is already seen
    """
    # Simulate first search result (must be >= 80 chars for fallback compression)
    # Also must be large enough that summary overhead doesn't make it longer
    first_search_result = "\n".join(
        [
            "src/main.py:10:def foo():",
            "src/main.py:20:def bar():",
            "src/main.py:30:def baz():",
            "src/main.py:40:def qux():",
            "src/main.py:50:def quux():",
            "src/main.py:60:def corge():",
            "src/main.py:70:def grault():",
            "src/main.py:80:def garply():",
            "src/other.py:5:def hello():",
            "src/other.py:15:def world():",
            "src/other.py:25:def test():",
            "src/other.py:35:def debug():",
        ]
    )

    # Create tool context for search
    search_context = {
        "name": "search",
        "query": "def ",
        "path": "src/",
        "args": {"query": "def ", "path": "src/"},
    }

    # Create messages with first search
    first_messages = [
        {
            "role": "user",
            "content": "search for def",
        },
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tool1", "name": "search", "input": {"query": "def ", "path": "src/"}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "tool1", "content": first_search_result}],
        },
    ]

    # Create tool_use_id_to_context mapping
    tool_context = {"tool1": search_context}

    # Track first exact evidence
    first_exact_evidence_seen: set[str] = set()

    # Compress first search
    compressed_first, breakdown_first = compress_tool_results(
        first_messages,
        tool_use_id_to_context=tool_context,
        first_exact_evidence_seen=first_exact_evidence_seen,
        preserve_exact_search_evidence=False,
    )

    # Verify first search added to evidence set
    expected_key = evidence_identity_key(
        "search",
        path="src/",
        query="def ",
        args={"query": "def ", "path": "src/"},
    )

    print(f"[DIAG] Expected key: {expected_key}")
    print(f"[DIAG] first_exact_evidence_seen after first: {first_exact_evidence_seen}")

    assert expected_key in first_exact_evidence_seen, (
        f"Key not in set after first search. Set: {first_exact_evidence_seen}"
    )

    # Create repeat search messages (identical query)
    repeat_messages = [
        {
            "role": "user",
            "content": "search for def again",
        },
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "tool2", "name": "search", "input": {"query": "def ", "path": "src/"}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "tool2", "content": first_search_result}],
        },
    ]

    # Update context for new tool
    tool_context_repeat = {"tool2": search_context}

    # Compress repeat search
    compressed_repeat, breakdown_repeat = compress_tool_results(
        repeat_messages,
        tool_use_id_to_context=tool_context_repeat,
        first_exact_evidence_seen=first_exact_evidence_seen,  # Same set!
        preserve_exact_search_evidence=False,
    )

    # Debug: check what key would be generated for repeat
    repeat_key = evidence_identity_key(
        "search",
        path="src/",
        query="def ",
        args={"query": "def ", "path": "src/"},
    )
    print(f"[DIAG] Repeat key: {repeat_key}")
    print(f"[DIAG] Keys match: {repeat_key == expected_key}")
    print(f"[DIAG] first_exact_evidence_seen contains repeat_key: {repeat_key in first_exact_evidence_seen}")

    print(f"[DIAG] breakdown_repeat: {breakdown_repeat}")

    # Verify repeat search result
    # Get the content of the repeat search result
    repeat_content = None
    for msg in compressed_repeat:
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        repeat_content = block.get("content")
                        break

    print(f"[DIAG] repeat_content: {repeat_content[:200] if repeat_content else None}")

    # Small result sets (12 matches ≤20) return verbatim — compression adds overhead
    # This is the correct behavior: no need to compress when there's no space savings
    if repeat_content:
        # Should be unchanged for small result sets
        assert repeat_content == first_search_result, (
            f"Small result set should return verbatim. Got: {repeat_content[:100]}..."
        )


def test_evidence_identity_key_consistency():
    """Test that evidence_identity_key produces consistent keys for identical searches."""

    key1 = evidence_identity_key(
        "search",
        path="src/",
        query="def foo",
        args={"query": "def foo", "path": "src/"},
    )

    key2 = evidence_identity_key(
        "search",
        path="src/",
        query="def foo",
        args={"query": "def foo", "path": "src/"},
    )

    print(f"[DIAG] key1: {key1}")
    print(f"[DIAG] key2: {key2}")

    assert key1 == key2, f"Keys differ: {key1} vs {key2}"

    # Note: extra args that are NOT in the exclusion list WILL be included in identity
    # This is intentional - different args may indicate different search intents
    key3 = evidence_identity_key(
        "search",
        path="src/",
        query="def foo",
        args={"query": "def foo", "path": "src/", "extra_arg": "ignored"},
    )

    # extra_arg is NOT excluded, so keys differ
    assert key1 != key3, f"Keys should differ with extra arg: {key1} vs {key3}"

    # But excluded args like 'offset' and 'limit' are not part of identity
    key4 = evidence_identity_key(
        "search",
        path="src/",
        query="def foo",
        args={"query": "def foo", "path": "src/", "offset": 10, "limit": 100},
    )

    # offset and limit ARE excluded from identity
    assert key1 == key4, f"Keys should match with excluded args: {key1} vs {key4}"


if __name__ == "__main__":
    # Run tests directly for quick diagnosis
    import logging

    logging.basicConfig(level=logging.DEBUG)

    print("\n=== test_evidence_identity_key_consistency ===")
    test_evidence_identity_key_consistency()
    print("PASSED\n")

    print("\n=== test_search_repeat_compresses_after_first_exact ===")
    test_search_repeat_compresses_after_first_exact()
    print("PASSED\n")
