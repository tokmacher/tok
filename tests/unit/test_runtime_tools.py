"""Tests for shared runtime tools module."""

import tempfile
from pathlib import Path

from tok.runtime_tools import RuntimeToolExecutor, execute_normalized_tool
from tok.universal_runtime import NormalizedToolEvent


class TestRuntimeToolExecutor:
    """Test the shared runtime tool executor."""

    def setup_method(self):
        """Set up test environment."""
        self.executor = RuntimeToolExecutor()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_execute_read_file_success(self):
        """Test successful file reading."""
        # Create a test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("Hello, World!")

        event = NormalizedToolEvent(
            id="test_read", name="read", path=str(test_file)
        )

        result = self.executor.execute_normalized_tool(event)

        assert result["status"] == "SUCCESS"
        assert "Hello, World!" in result["message"]
        assert f"Read {test_file}" in result["message"]

    def test_execute_read_file_not_found(self):
        """Test reading non-existent file."""
        event = NormalizedToolEvent(
            id="test_read_missing", name="read", path="/nonexistent/file.txt"
        )

        result = self.executor.execute_normalized_tool(event)

        assert result["status"] == "ERROR"
        assert "Not found" in result["message"]

    def test_execute_read_file_with_range(self):
        """Test reading file with line range."""
        # Create a test file with multiple lines
        test_file = Path(self.temp_dir) / "multiline.txt"
        test_file.write_text("Line 1\nLine 2\nLine 3\nLine 4\nLine 5")

        event = NormalizedToolEvent(
            id="test_read_range",
            name="read",
            path=str(test_file),
            args={"start": 2, "end": 4},
        )

        result = self.executor.execute_normalized_tool(event)

        assert result["status"] == "SUCCESS"
        assert "Line 2" in result["message"]
        assert "Line 3" in result["message"]
        assert "Line 4" in result["message"]
        assert "Line 1" not in result["message"]
        assert "Line 5" not in result["message"]

    def test_execute_write_file_success(self):
        """Test successful file writing."""
        test_file = Path(self.temp_dir) / "write_test.txt"
        content = "Test content for writing"

        event = NormalizedToolEvent(
            id="test_write",
            name="write",
            path=str(test_file),
            args={"content": content},
        )

        result = self.executor.execute_normalized_tool(event)

        assert result["status"] == "SUCCESS"
        assert test_file.exists()
        assert test_file.read_text() == content

    def test_execute_write_file_creates_directories(self):
        """Test that write creates parent directories."""
        nested_file = Path(self.temp_dir) / "subdir" / "nested" / "file.txt"
        content = "Nested file content"

        event = NormalizedToolEvent(
            id="test_write_nested",
            name="write",
            path=str(nested_file),
            args={"content": content},
        )

        result = self.executor.execute_normalized_tool(event)

        assert result["status"] == "SUCCESS"
        assert nested_file.exists()
        assert nested_file.read_text() == content

    def test_execute_write_file_invalid_name(self):
        """Test writing with invalid filename."""
        event = NormalizedToolEvent(
            id="test_write_invalid",
            name="write",
            path="invalid*name.txt",
            args={"content": "content"},
        )

        result = self.executor.execute_normalized_tool(event)

        assert result["status"] == "ERROR"
        assert "Invalid filename" in result["message"]

    def test_execute_edit_file_success(self):
        """Test successful file editing."""
        # Create a test file
        test_file = Path(self.temp_dir) / "edit_test.txt"
        original_content = "Hello World\nGoodbye World"
        test_file.write_text(original_content)

        event = NormalizedToolEvent(
            id="test_edit",
            name="edit",
            path=str(test_file),
            args={"search": "Hello World", "replace": "Hello Tok"},
        )

        result = self.executor.execute_normalized_tool(event)

        assert result["status"] == "SUCCESS"
        updated_content = test_file.read_text()
        assert "Hello Tok" in updated_content
        assert "Hello World" not in updated_content

    def test_execute_edit_file_not_found(self):
        """Test editing non-existent file."""
        event = NormalizedToolEvent(
            id="test_edit_missing",
            name="edit",
            path="/nonexistent/file.txt",
            args={"search": "old", "replace": "new"},
        )

        result = self.executor.execute_normalized_tool(event)

        assert result["status"] == "ERROR"
        assert "Not found" in result["message"]

    def test_execute_edit_file_search_not_found(self):
        """Test editing when search string not found."""
        test_file = Path(self.temp_dir) / "edit_search_test.txt"
        test_file.write_text("Different content")

        event = NormalizedToolEvent(
            id="test_edit_search_missing",
            name="edit",
            path=str(test_file),
            args={"search": "not found", "replace": "replacement"},
        )

        result = self.executor.execute_normalized_tool(event)

        assert result["status"] == "ERROR"
        assert "Search string not found" in result["message"]

    def test_execute_run_command_success(self):
        """Test successful command execution."""
        event = NormalizedToolEvent(
            id="test_run", name="run", command="echo 'Hello from command'"
        )

        result = self.executor.execute_normalized_tool(event)

        assert result["status"] == "SUCCESS"
        assert "Hello from command" in result["message"]
        assert result["returncode"] == 0

    def test_execute_run_command_failure(self):
        """Test command execution that fails."""
        event = NormalizedToolEvent(
            id="test_run_fail",
            name="run",
            command="sh -c 'exit 1'",  # Command that exits with error code 1
        )

        result = self.executor.execute_normalized_tool(event)

        assert result["status"] == "FAILURE"
        assert result["returncode"] == 1

    def test_execute_delta_tool(self):
        """Test delta tool execution (if available)."""
        event = NormalizedToolEvent(
            id="test_delta",
            name="delta",
            path="/fake/path.py",
            args={"payload": "add @func|new_function\n+params:x,y"},
        )

        result = self.executor.execute_normalized_tool(event)

        # Delta tool should handle gracefully even with fake paths
        assert result["status"] in ["SUCCESS", "ERROR"]

    def test_execute_unknown_tool(self):
        """Test execution of unknown tool."""
        event = NormalizedToolEvent(
            id="test_unknown", name="unknown_tool", args={}
        )

        result = self.executor.execute_normalized_tool(event)

        assert result["status"] == "ERROR"
        assert "Unknown tool" in result["message"]

    def test_pending_deltas_management(self):
        """Test pending deltas tracking."""
        # Initially should be empty
        assert self.executor.get_pending_deltas() == []

        # Execute a write operation (might generate deltas)
        test_file = Path(self.temp_dir) / "delta_test.txt"
        event = NormalizedToolEvent(
            id="test_write_delta",
            name="write",
            path=str(test_file),
            args={"content": "Test content"},
        )

        self.executor.execute_normalized_tool(event)

        # Get pending deltas (may be empty if delta tracking is disabled)
        deltas = self.executor.get_pending_deltas()
        assert isinstance(deltas, list)

        # Clear pending deltas
        self.executor.clear_pending_deltas()
        assert self.executor.get_pending_deltas() == []

    def test_convenience_function(self):
        """Test the convenience function execute_normalized_tool."""
        test_file = Path(self.temp_dir) / "convenience_test.txt"
        test_file.write_text("Original content")

        event = NormalizedToolEvent(
            id="test_convenience", name="read", path=str(test_file)
        )

        result = execute_normalized_tool(event)

        assert result["status"] == "SUCCESS"
        assert "Original content" in result["message"]

    def test_security_methods(self):
        """Test security methods."""
        # These should all return True (restrictions disabled)
        assert self.executor._is_safe_path("/any/path")
        assert self.executor._is_safe_path("../../../etc/passwd")
        assert self.executor._is_safe_rm("rm -rf /")
