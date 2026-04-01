"""Tests for explorer.py module - file exploration utilities."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from tok.explorer import (
    get_file_overview,
    explore_file,
    explore_module,
    list_large_files,
)


class TestGetFileOverview:
    """Test suite for get_file_overview function."""

    def test_returns_error_for_missing_file(self):
        """Verify error handling for non-existent file."""
        result = get_file_overview("/nonexistent/path.py")

        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_returns_error_for_directory(self):
        """Verify error handling for directory path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_file_overview(tmpdir)

            assert "error" in result
            assert "not a file" in result["error"].lower()

    def test_returns_error_for_non_python_file(self):
        """Verify error handling for non-Python files."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"some text")
            f.flush()

            try:
                result = get_file_overview(f.name)
                assert "error" in result
                assert "not a python" in result["error"].lower()
            finally:
                os.unlink(f.name)

    def test_parses_python_file_correctly(self):
        """Verify successful parsing of Python file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("""def foo():
    pass

class Bar:
    def method(self):
        pass
""")
            f.flush()

            try:
                result = get_file_overview(f.name)

                assert "error" not in result
                assert result["line_count"] == 6
                assert result["is_large"] is False
                assert len(result["classes"]) == 1
                assert result["classes"][0]["name"] == "Bar"
                assert len(result["functions"]) == 1
                assert result["functions"][0]["name"] == "foo"
            finally:
                os.unlink(f.name)

    def test_detects_large_file(self):
        """Verify large file detection (>500 lines)."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            # Write 600 lines
            for i in range(600):
                f.write(f"line_{i}\n")
            f.flush()

            try:
                result = get_file_overview(f.name)
                assert result["is_large"] is True
                assert result["line_count"] == 600
            finally:
                os.unlink(f.name)

    def test_handles_syntax_error(self):
        """Verify graceful handling of syntax errors."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("def broken(  # missing closing\n")
            f.flush()

            try:
                result = get_file_overview(f.name)
                assert "error" in result
                assert "syntax" in result["error"].lower()
            finally:
                os.unlink(f.name)


class TestExploreFile:
    """Test suite for explore_file function."""

    def test_overview_mode(self):
        """Verify overview mode output format."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("def foo():\n    pass\n")
            f.flush()

            try:
                result = explore_file(f.name, mode="overview")

                assert result.startswith("@file")
                assert "lines:" in result
                assert "large:" in result
            finally:
                os.unlink(f.name)

    def test_error_for_missing_file(self):
        """Verify error handling for missing file."""
        result = explore_file("/nonexistent.py")

        assert result.startswith("@error")


class TestListLargeFiles:
    """Test suite for list_large_files function."""

    def test_finds_large_python_files(self):
        """Verify detection of large Python files in directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create small file (<500 lines)
            small = Path(tmpdir) / "small.py"
            small.write_text("def foo(): pass\n")

            # Create large file (>500 lines)
            large = Path(tmpdir) / "large.py"
            large.write_text(
                "\n".join([f"line_{i}" for i in range(600)]) + "\n"
            )

            result = list_large_files(tmpdir)

            # Should only find large.py
            assert len(result) == 1
            assert result[0]["path"] == str(large)
            assert result[0]["line_count"] == 600

    def test_skips_non_python_files(self):
        """Verify non-Python files are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_file = Path(tmpdir) / "large.txt"
            txt_file.write_text(
                "\n".join([f"line_{i}" for i in range(600)]) + "\n"
            )

            result = list_large_files(tmpdir)

            assert len(result) == 0

    def test_skips_ignored_directories(self):
        """Verify __pycache__ and similar are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pycache = Path(tmpdir) / "__pycache__"
            pycache.mkdir()
            large = pycache / "large.py"
            large.write_text(
                "\n".join([f"line_{i}" for i in range(600)]) + "\n"
            )

            result = list_large_files(tmpdir)

            # Should skip __pycache__
            assert len(result) == 0

    def test_returns_sorted_by_size(self):
        """Verify results are sorted by line count descending."""
        with tempfile.TemporaryDirectory() as tmpdir:
            medium = Path(tmpdir) / "medium.py"
            medium.write_text(
                "\n".join([f"line_{i}" for i in range(600)]) + "\n"
            )

            large = Path(tmpdir) / "large.py"
            large.write_text(
                "\n".join([f"line_{i}" for i in range(1000)]) + "\n"
            )

            result = list_large_files(tmpdir)

            assert len(result) == 2
            assert result[0]["line_count"] == 1000
            assert result[1]["line_count"] == 600


class TestExploreModule:
    """Test suite for explore_module function."""

    def test_explores_single_file(self):
        """Verify module exploration of single file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("def foo(): pass\n")
            f.flush()

            try:
                result = explore_module(f.name, mode="overview")

                assert result.startswith("@file")
            finally:
                os.unlink(f.name)

    def test_explores_directory(self):
        """Verify module exploration of directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = Path(tmpdir) / "__init__.py"
            init_file.write_text("# init\n")

            result = explore_module(tmpdir, mode="overview")

            assert result.startswith("@module")

    def test_error_for_empty_directory(self):
        """Verify error for empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = explore_module(tmpdir)

            assert result.startswith("@error")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
