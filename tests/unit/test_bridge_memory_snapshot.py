"""Tests for file snapshot format with freshness signaling in bridge_memory."""

from __future__ import annotations

import pytest

from tok.runtime.memory.bridge_memory import BridgeMemoryState


class TestRecordFileSnapshotFormat:
    """Test suite for record_file_snapshot enhanced format."""

    def test_new_format_includes_line_count(self):
        """Verify new format includes line count in fact value."""
        state = BridgeMemoryState()
        content = "line1\nline2\nline3\nline4\nline5"

        state.record_file_snapshot("/test/file.py", content)

        facts = state.hot.get("facts", [])
        assert len(facts) == 1
        # Format: file[path]:LINE_COUNT|digest|~TOKENS
        assert "5|" in facts[0].value  # 5 lines
        assert "|~20t" in facts[0].value  # ~20 tokens (5 lines * 4)

    def test_large_file_shows_token_savings(self):
        """Verify large files show significant token savings."""
        state = BridgeMemoryState()
        # Simulate 1000 line file
        content = "\n".join([f"def func_{i}(): pass" for i in range(1000)])

        state.record_file_snapshot("/test/large.py", content)

        facts = state.hot.get("facts", [])
        assert len(facts) == 1
        # Should show 1000 lines and ~4000 tokens
        assert "1000|" in facts[0].value
        assert "|~4000t" in facts[0].value

    def test_extracts_digest_from_content(self):
        """Verify semantic digest is extracted from content."""
        state = BridgeMemoryState()
        content = "def foo():\n    pass\n\nclass Bar:\n    pass"

        state.record_file_snapshot("/test/file.py", content)

        facts = state.hot.get("facts", [])
        assert len(facts) == 1
        # Should contain function/class definitions in digest
        assert "def " in facts[0].value or "class " in facts[0].value

    def test_creates_files_entry(self):
        """Verify files field is also populated."""
        state = BridgeMemoryState()
        content = "line1\nline2\nline3"

        state.record_file_snapshot("/test/file.py", content)

        files = state.hot.get("files", [])
        assert len(files) == 1
        assert files[0].value == "/test/file.py"


class TestGetFileFactDigests:
    """Test suite for get_file_fact_digests with new format."""

    def test_parses_new_format_correctly(self):
        """Verify parsing of new format: LINE_COUNT|digest|~tokens."""
        state = BridgeMemoryState()
        content = "def foo():\n    pass\n"

        state.record_file_snapshot("/test/file.py", content)

        digests = state.get_file_fact_digests()

        assert "/test/file.py" in digests
        # Should extract digest (not the LINE_COUNT| prefix or |~tokens suffix)
        digest = digests["/test/file.py"]
        assert "|" not in digest  # Should be clean digest
        assert "~" not in digest

    def test_handles_legacy_format(self):
        """Verify backward compatibility with legacy format (no line count)."""
        state = BridgeMemoryState()
        # Manually insert legacy format fact
        state._upsert(
            state.hot,
            "facts",
            "file[/test/legacy.py]:def foo() pass",
            score_delta=1,
        )

        digests = state.get_file_fact_digests()

        assert "/test/legacy.py" in digests
        assert digests["/test/legacy.py"] == "def foo() pass"

    def test_empty_content_returns_empty_digest(self):
        """Verify empty content handling."""
        state = BridgeMemoryState()

        result = state.record_file_snapshot("/test/empty.py", "   ")

        assert result is False

    def test_multiple_files_parsed_correctly(self):
        """Verify multiple files with new format are all parsed."""
        state = BridgeMemoryState()

        state.record_file_snapshot("/test/a.py", "def a():\n    pass\n")
        state.record_file_snapshot(
            "/test/b.py", "def b():\n    pass\n\nclass B:\n    pass\n"
        )

        digests = state.get_file_fact_digests()

        assert "/test/a.py" in digests
        assert "/test/b.py" in digests
        # Both should have clean digests without line count tokens
        for path, digest in digests.items():
            assert "|" not in digest
            assert "~" not in digest


class TestFileSnapshotWithHeat:
    """Test suite for file snapshots with heat tracking."""

    def test_edited_files_get_higher_score(self):
        """Verify files with heat >= 2.0 get edited flag."""
        state = BridgeMemoryState()
        # Simulate edited file by bumping heat
        state.bump_file_heat("/test/edited.py", weight=3.0)

        content = "def foo(): pass"
        state.record_file_snapshot("/test/edited.py", content)

        # Check edited field
        edited = state.hot.get("edited", [])
        assert len(edited) == 1
        assert edited[0].value == "/test/edited.py"

    def test_heat_bonus_increases_score(self):
        """Verify heat bonus increases fact score."""
        state = BridgeMemoryState()
        state.bump_file_heat("/test/hot.py", weight=5.0)

        content = "def foo(): pass"
        state.record_file_snapshot("/test/hot.py", content)

        facts = state.hot.get("facts", [])
        # Score should be base (2) + heat bonus (5*2=10) = 12
        assert facts[0].score >= 12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
