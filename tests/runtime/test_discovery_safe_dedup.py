"""Tests for discovery-safe deduplication - first exact observation contract.

This module tests the core contract that:
1. First exact observations MUST be preserved exactly (not summarized)
2. Repeated observations MAY use semantic deduplication

See: docs/internal/discovery-safe-dedup-benchmark.md
"""

from __future__ import annotations

import hashlib
from typing import Any

from tok.compression._history_pipeline import compress_tool_results_impl
from tok.runtime.repeat_targets import evidence_identity_key
from tok.runtime.pipeline._tool_context import build_tool_use_id_to_context


def _tool_use(tool_id: str, tool_name: str, **input_kw: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": tool_id,
                "name": tool_name,
                "input": input_kw,
            }
        ],
    }


def _tool_result(tool_id: str, content: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": content,
            }
        ],
    }


def _make_cache_key(tool_name: str, context: dict[str, Any]) -> str:
    """Create a cache key for result cache testing."""
    from tok.runtime.pipeline._tool_repeat_detection import (
        _make_cache_key as make_key,
    )

    return make_key(tool_name, context)


class TestFirstExactObservationContract:
    """Test Rule 1: First exact observation must not be replaced by summary-only."""

    def test_first_file_read_is_exact_not_summary(self):
        """First file observation must be exact, not substituted with summary."""
        file_content = "def foo():\n    pass\n\ndef bar():\n    return 42\n"
        messages = [
            _tool_use("t1", "read_file", file_path="/tmp/foo.py"),
            _tool_result("t1", file_content),
        ]
        tool_use_id_to_context = build_tool_use_id_to_context(messages)
        first_exact_evidence_seen: set[str] = set()

        result_cache: dict[
            str, tuple[str, str, float] | tuple[str, str] | tuple[str]
        ] = {}
        cache_key = _make_cache_key("read_file", tool_use_id_to_context["t1"])
        digest = hashlib.sha256(file_content.encode()).hexdigest()[:8]
        result_cache[cache_key] = (digest, file_content)

        compressed, breakdown = compress_tool_results_impl(
            messages,
            result_cache=result_cache,
            tool_use_id_to_context=tool_use_id_to_context,
            compression_level="balanced",
            first_exact_evidence_seen=first_exact_evidence_seen,
        )

        # First read should be preserved exactly
        result_msg = compressed[1]
        assert result_msg["role"] == "user"
        content_blocks = result_msg["content"]
        assert len(content_blocks) == 1
        assert content_blocks[0]["content"] == file_content
        assert "tok_compressed" not in file_content

        # Verify the evidence was tracked
        evidence_key = evidence_identity_key(
            "read_file",
            path="/tmp/foo.py",
            args={"file_path": "/tmp/foo.py"},
        )
        assert evidence_key in first_exact_evidence_seen

    def test_repeated_file_read_can_dedup_after_first_exact(self):
        """After first exact observation, repeated reads may dedup."""
        file_content = "def foo():\n    pass\n\ndef bar():\n    return 42\n"
        messages = [
            _tool_use("t1", "read_file", file_path="/tmp/foo.py"),
            _tool_result("t1", file_content),
            _tool_use("t2", "read_file", file_path="/tmp/foo.py"),
            _tool_result("t2", file_content),
        ]
        tool_use_id_to_context = build_tool_use_id_to_context(messages)
        first_exact_evidence_seen: set[str] = set()

        # Pre-populate result cache for second read
        result_cache: dict[
            str, tuple[str, str, float] | tuple[str, str] | tuple[str]
        ] = {}
        cache_key = _make_cache_key("read_file", tool_use_id_to_context["t1"])
        digest = hashlib.sha256(file_content.encode()).hexdigest()[:8]
        result_cache[cache_key] = (digest, file_content)

        compressed, breakdown = compress_tool_results_impl(
            messages,
            result_cache=result_cache,
            tool_use_id_to_context=tool_use_id_to_context,
            compression_level="balanced",
            first_exact_evidence_seen=first_exact_evidence_seen,
        )

        # First read should be exact
        first_result = compressed[1]["content"][0]["content"]
        assert first_result == file_content

        # Second read may be deduplicated (check if it was processed via cache)
        second_result = compressed[3]["content"][0]["content"]
        # Either exact or compressed/deduplicated is acceptable for repeat
        assert (
            second_result == file_content
            or "tok_compressed" in second_result
            or "|unchanged|" in second_result
        )

        # Verify evidence key was tracked only once
        evidence_key = evidence_identity_key(
            "read_file",
            path="/tmp/foo.py",
            args={"file_path": "/tmp/foo.py"},
        )
        assert evidence_key in first_exact_evidence_seen


class TestFirstGrepSearchObservationContract:
    """Test that first grep/search observations are exact."""

    def test_first_grep_search_is_exact_not_summary(self):
        """First grep/search result set must be exact, not pre-filtered."""
        grep_results = (
            "src/main.py:10:def main():\n"
            "src/main.py:15:    print('hello')\n"
            "src/utils.py:5:def helper():\n"
        )
        messages = [
            _tool_use("t1", "grep_search", query="def ", path="src/"),
            _tool_result("t1", grep_results),
        ]
        tool_use_id_to_context = build_tool_use_id_to_context(messages)
        first_exact_evidence_seen: set[str] = set()

        result_cache: dict[
            str, tuple[str, str, float] | tuple[str, str] | tuple[str]
        ] = {}
        cache_key = _make_cache_key(
            "grep_search", tool_use_id_to_context["t1"]
        )
        digest = hashlib.sha256(grep_results.encode()).hexdigest()[:8]
        result_cache[cache_key] = (digest, grep_results)

        compressed, breakdown = compress_tool_results_impl(
            messages,
            result_cache=result_cache,
            tool_use_id_to_context=tool_use_id_to_context,
            compression_level="balanced",
            first_exact_evidence_seen=first_exact_evidence_seen,
        )

        # First search should be preserved exactly
        result_msg = compressed[1]
        content_blocks = result_msg["content"]
        assert len(content_blocks) == 1
        # Content should be the original grep results (exact)
        assert (
            grep_results in content_blocks[0]["content"]
            or content_blocks[0]["content"] == grep_results
        )

        # Verify the evidence was tracked with proper identity
        evidence_key = evidence_identity_key(
            "grep_search",
            path="src/",
            query="def ",
            args={"query": "def ", "path": "src/"},
        )
        assert evidence_key in first_exact_evidence_seen

    def test_repeated_grep_search_can_dedup_after_first_exact(self):
        """After first exact search, repeated searches may dedup."""
        grep_results = (
            "src/main.py:10:def main():\nsrc/main.py:15:    print('hello')\n"
        )
        messages = [
            _tool_use("t1", "grep_search", query="def ", path="src/"),
            _tool_result("t1", grep_results),
            _tool_use("t2", "grep_search", query="def ", path="src/"),
            _tool_result("t2", grep_results),
        ]
        tool_use_id_to_context = build_tool_use_id_to_context(messages)
        first_exact_evidence_seen: set[str] = set()

        result_cache: dict[
            str, tuple[str, str, float] | tuple[str, str] | tuple[str]
        ] = {}
        cache_key = _make_cache_key(
            "grep_search", tool_use_id_to_context["t1"]
        )
        digest = hashlib.sha256(grep_results.encode()).hexdigest()[:8]
        result_cache[cache_key] = (digest, grep_results)

        compressed, breakdown = compress_tool_results_impl(
            messages,
            result_cache=result_cache,
            tool_use_id_to_context=tool_use_id_to_context,
            compression_level="balanced",
            first_exact_evidence_seen=first_exact_evidence_seen,
        )

        # First search should be exact
        first_result = compressed[1]["content"][0]["content"]
        assert grep_results in first_result or first_result == grep_results

        # Evidence key should be tracked
        evidence_key = evidence_identity_key(
            "grep_search",
            path="src/",
            query="def ",
            args={"query": "def ", "path": "src/"},
        )
        assert evidence_key in first_exact_evidence_seen


class TestEvidenceIdentityKey:
    """Test evidence identity key generation for different evidence types."""

    def test_file_read_identity_key_format(self):
        """File reads keyed by canonical path."""
        key = evidence_identity_key(
            "read_file",
            path="/tmp/foo.py",
            args={"file_path": "/tmp/foo.py"},
        )
        assert key is not None
        assert key.startswith("file_read|")
        assert "/tmp/foo.py" in key

    def test_search_identity_key_includes_query_and_scope(self):
        """Search identity includes normalized query, scope, and flags."""
        key1 = evidence_identity_key(
            "grep_search",
            path="src/",
            query="def ",
            args={"query": "def ", "path": "src/", "case_sensitive": True},
        )
        key2 = evidence_identity_key(
            "grep_search",
            path="src/",
            query="def ",
            args={"query": "def ", "path": "src/", "case_sensitive": True},
        )
        # Same query, scope, and flags should produce same key
        assert key1 == key2
        assert key1 is not None
        assert key1.startswith("search|")
        assert "def" in key1
        assert "src" in key1

    def test_search_identity_key_differs_by_flags(self):
        """Search keys differ when flags differ."""
        key1 = evidence_identity_key(
            "grep_search",
            path="src/",
            query="def ",
            args={"query": "def ", "path": "src/", "case_sensitive": True},
        )
        key2 = evidence_identity_key(
            "grep_search",
            path="src/",
            query="def ",
            args={"query": "def ", "path": "src/", "case_sensitive": False},
        )
        # Different flags should produce different keys
        assert key1 != key2

    def test_listing_identity_key_format(self):
        """Directory listings keyed by path + mode."""
        key = evidence_identity_key(
            "list_dir",
            path="/tmp",
            args={"path": "/tmp"},
        )
        assert key is not None
        assert key.startswith("listing|")
        assert "/tmp" in key


class TestDirectoryListingObservationContract:
    """Test first directory listing observations are exact."""

    def test_first_directory_listing_is_exact(self):
        """First directory listing must be exact, not curated."""
        dir_listing = "file1.py\nfile2.py\nsubdir/\nREADME.md\n"
        messages = [
            _tool_use("t1", "list_dir", path="/tmp"),
            _tool_result("t1", dir_listing),
        ]
        tool_use_id_to_context = build_tool_use_id_to_context(messages)
        first_exact_evidence_seen: set[str] = set()

        compressed, breakdown = compress_tool_results_impl(
            messages,
            result_cache={},
            tool_use_id_to_context=tool_use_id_to_context,
            compression_level="balanced",
            first_exact_evidence_seen=first_exact_evidence_seen,
        )

        # First listing should be preserved exactly
        result_msg = compressed[1]
        content_blocks = result_msg["content"]
        assert len(content_blocks) == 1
        assert content_blocks[0]["content"] == dir_listing

        # Verify the evidence was tracked
        evidence_key = evidence_identity_key(
            "list_dir",
            path="/tmp",
            args={"path": "/tmp"},
        )
        assert evidence_key in first_exact_evidence_seen
