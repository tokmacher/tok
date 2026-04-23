"""Additional tests for runtime tools module to improve coverage."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tok.runtime.tools import RuntimeToolExecutor
from tok.universal_runtime import NormalizedToolEvent


class TestIsSafePath:
    """Tests for _is_safe_path method."""

    def setup_method(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.executor = RuntimeToolExecutor(workspace_root=self.temp_dir)

    def teardown_method(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_empty_path(self) -> None:
        assert self.executor._is_safe_path("") is False

    def test_whitespace_only_path(self) -> None:
        assert self.executor._is_safe_path("   ") is False

    def test_oserror_in_realpath(self) -> None:
        with patch("os.path.realpath", side_effect=OSError("mocked")):
            assert self.executor._is_safe_path("/some/path") is False

    def test_valueerror_in_realpath(self) -> None:
        with patch("os.path.realpath", side_effect=ValueError("mocked")):
            assert self.executor._is_safe_path("/some/path") is False

    def test_path_outside_workspace(self) -> None:
        assert self.executor._is_safe_path("/etc/passwd") is False

    def test_symlink_inside_workspace(self) -> None:
        target = Path(self.temp_dir) / "target.txt"
        target.write_text("hello")
        link = Path(self.temp_dir) / "link.txt"
        link.symlink_to(target)
        assert self.executor._is_safe_path(str(link)) is True

    def test_symlink_outside_workspace(self) -> None:
        target = Path("/etc/passwd")
        if target.exists():
            link = Path(self.temp_dir) / "link"
            try:
                link.symlink_to(target)
                result = self.executor._is_safe_path(str(link))
                assert result is False
            finally:
                if link.is_symlink():
                    link.unlink()

    def test_path_with_dotdot(self) -> None:
        assert self.executor._is_safe_path(f"{self.temp_dir}/../etc") is False


class TestIsSafeRm:
    """Tests for _is_safe_rm method."""

    def setup_method(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.executor = RuntimeToolExecutor(workspace_root=self.temp_dir)

    def teardown_method(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_empty_rm_command(self) -> None:
        assert self.executor._is_safe_rm("rm ") is False

    def test_rm_workspace_root(self) -> None:
        assert self.executor._is_safe_rm(f"rm -rf {self.temp_dir}") is False

    def test_rm_protected_child_git(self) -> None:
        git_dir = Path(self.temp_dir) / ".git"
        git_dir.mkdir()
        assert self.executor._is_safe_rm(f"rm -rf {git_dir}") is False

    def test_rm_protected_child_inside_git(self) -> None:
        git_dir = Path(self.temp_dir) / ".git"
        git_dir.mkdir()
        nested = git_dir / "objects"
        nested.mkdir()
        assert self.executor._is_safe_rm(f"rm -rf {nested}") is False

    def test_rm_blocked_root_bin(self) -> None:
        assert self.executor._is_safe_rm("rm -rf /bin") is False

    def test_rm_blocked_root_sbin(self) -> None:
        assert self.executor._is_safe_rm("rm -rf /sbin") is False

    def test_rm_blocked_root_system(self) -> None:
        assert self.executor._is_safe_rm("rm -rf /System") is False

    def test_rm_blocked_root_library(self) -> None:
        assert self.executor._is_safe_rm("rm -rf /Library") is False

    def test_rm_blocked_root_usr(self) -> None:
        assert self.executor._is_safe_rm("rm -rf /usr") is False

    def test_oserror_in_realpath(self) -> None:
        with patch("os.path.realpath", side_effect=OSError("mocked")):
            assert self.executor._is_safe_rm("rm /some/path") is False

    def test_valueerror_in_realpath(self) -> None:
        with patch("os.path.realpath", side_effect=ValueError("mocked")):
            assert self.executor._is_safe_rm("rm /some/path") is False

    def test_safe_rm_inside_workspace(self) -> None:
        safe_path = Path(self.temp_dir) / "scratch"
        result = self.executor._is_safe_rm(f"rm -rf {safe_path}")
        assert result is True


class TestCompilerGuard:
    """Tests for _compiler_guard method."""

    def setup_method(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.executor = RuntimeToolExecutor(workspace_root=self.temp_dir)

    def teardown_method(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_unknown_tool_returns_none(self) -> None:
        from tok.protocol.models import TokNode

        node = TokNode(type="tool", label="unknown_tool", attrs={}, text="unknown_tool", trust=None)
        result = self.executor._compiler_guard("unknown_tool", {}, node)
        assert result is None

    def test_cli_style_edit_with_drift(self) -> None:
        from tok.protocol.models import TokNode

        node = TokNode(
            type="tool",
            label="edit",
            attrs={"path": "/tmp/f", "search": "old", "replace": "new"},
            text="edit --path /tmp/f\n|> search\nold\nreplace\nnew",
            trust=None,
        )
        result = self.executor._compiler_guard("edit", node.attrs, node)
        assert result is None or "error" in result


class TestApplyCliStyleAttrs:
    """Tests for _apply_cli_style_attrs method."""

    def setup_method(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.executor = RuntimeToolExecutor(workspace_root=self.temp_dir)

    def teardown_method(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_no_dashes_returns_unchanged(self) -> None:
        input_data = {"path": "/tmp/f"}
        self.executor._apply_cli_style_attrs("read", "read /tmp/f", input_data)
        assert input_data == {"path": "/tmp/f"}

    def test_unsupported_tool_returns_unchanged(self) -> None:
        input_data = {"cmd": "echo hello"}
        self.executor._apply_cli_style_attrs("run", "run echo hello", input_data)
        assert input_data == {"cmd": "echo hello"}

    def test_cli_style_read(self) -> None:
        input_data: dict[str, str] = {}
        self.executor._apply_cli_style_attrs(
            "read",
            "read --path /tmp/f --offset 10",
            input_data,
        )
        assert "path" in input_data
        assert "offset" in input_data

    def test_cli_style_write(self) -> None:
        input_data: dict[str, str] = {}
        self.executor._apply_cli_style_attrs(
            "write",
            "write --path /tmp/f --content hello",
            input_data,
        )
        assert "path" in input_data

    def test_shlex_split_error(self) -> None:
        input_data: dict[str, str] = {}
        with patch("tok.runtime.tools.shlex.split", side_effect=ValueError("mocked")):
            self.executor._apply_cli_style_attrs("read", "read --x y", input_data)
        assert input_data == {}

    def test_cli_style_removes_duplicate_text(self) -> None:
        input_data = {"text": "read --path /tmp/f"}
        self.executor._apply_cli_style_attrs(
            "read",
            "read --path /tmp/f",
            input_data,
        )
        assert "text" not in input_data


class TestFillMissingAttributes:
    """Tests for _fill_missing_attributes method."""

    def setup_method(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.executor = RuntimeToolExecutor(workspace_root=self.temp_dir)

    def teardown_method(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_read_with_path_already_set(self) -> None:
        input_data = {"path": "/tmp/f"}
        self.executor._fill_missing_attributes("read", input_data, "/tmp/f\nsecond line")
        assert input_data["path"] == "/tmp/f"

    def test_read_without_path(self) -> None:
        input_data: dict[str, str] = {}
        self.executor._fill_missing_attributes("read", input_data, "/tmp/f\nsecond line")
        assert input_data.get("path") == "/tmp/f"

    def test_write_with_path_already_set(self) -> None:
        input_data = {"path": "/tmp/f"}
        self.executor._fill_missing_attributes("write", input_data, "/tmp/f\ncontent")
        assert input_data["path"] == "/tmp/f"

    def test_write_without_path(self) -> None:
        input_data: dict[str, str] = {}
        self.executor._fill_missing_attributes("write", input_data, "/tmp/f\ncontent")
        assert input_data.get("path") == "/tmp/f"

    def test_run_with_cmd_already_set(self) -> None:
        input_data = {"cmd": "echo hello"}
        self.executor._fill_missing_attributes("run", input_data, "echo hello")
        assert input_data["cmd"] == "echo hello"

    def test_run_without_cmd(self) -> None:
        input_data: dict[str, str] = {}
        self.executor._fill_missing_attributes("run", input_data, "echo hello")
        assert input_data.get("cmd") == "echo hello"

    def test_search_with_query(self) -> None:
        input_data = {"query": "existing"}
        self.executor._fill_missing_attributes("search", input_data, "new search")
        assert input_data["query"] == "existing"

    def test_search_without_query(self) -> None:
        input_data: dict[str, str] = {}
        self.executor._fill_missing_attributes("search", input_data, "new search")
        assert input_data.get("query") == "new search"

    def test_grep_without_query_or_text(self) -> None:
        input_data: dict[str, str] = {}
        self.executor._fill_missing_attributes("grep", input_data, "pattern")
        assert input_data.get("query") == "pattern"


class TestDriftDetection:
    """Tests for _detect_attr_drift and related methods."""

    def setup_method(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.executor = RuntimeToolExecutor(workspace_root=self.temp_dir)

    def teardown_method(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_drift_fields_write(self) -> None:
        fields = self.executor._drift_fields("write")
        assert "text" in fields

    def test_drift_fields_edit(self) -> None:
        fields = self.executor._drift_fields("edit")
        assert "search" in fields
        assert "replace" in fields

    def test_drift_fields_other(self) -> None:
        fields = self.executor._drift_fields("read")
        assert fields == []

    def test_detect_attr_drift_with_newline(self) -> None:
        result = self.executor._detect_attr_drift("write", {"text": "line1\nline2"}, "")
        assert result == "text"

    def test_detect_attr_drift_with_pipe(self) -> None:
        result = self.executor._detect_attr_drift("write", {"text": "|> structure"}, "")
        assert result == "text"

    def test_detect_attr_drift_with_gt(self) -> None:
        result = self.executor._detect_attr_drift("write", {"text": "> structure"}, "")
        assert result == "text"

    def test_detect_attr_drift_with_def(self) -> None:
        result = self.executor._detect_attr_drift("write", {"text": "def foo():\n    pass"}, "")
        assert result == "text"

    def test_detect_attr_drift_with_class(self) -> None:
        result = self.executor._detect_attr_drift("write", {"text": "class Foo:\n    pass"}, "")
        assert result == "text"

    def test_detect_attr_drift_with_node_text_pipe(self) -> None:
        result = self.executor._detect_attr_drift("write", {}, "|> content")
        assert result == "text"

    def test_detect_attr_drift_no_drift(self) -> None:
        result = self.executor._detect_attr_drift("write", {"text": "simple text"}, "")
        assert result == ""

    def test_detect_attr_drift_non_string_value(self) -> None:
        result = self.executor._detect_attr_drift("write", {"text": 123}, "")
        assert result == ""


class TestFormatPayload:
    """Tests for _format_payload method."""

    def setup_method(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.executor = RuntimeToolExecutor(workspace_root=self.temp_dir)

    def teardown_method(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_strips_pipe_gt(self) -> None:
        result = self.executor._format_payload("|> content\n> more")
        assert "|> " not in result
        assert "> " not in result

    def test_strips_carriage_return(self) -> None:
        result = self.executor._format_payload("line1\rline2")
        assert "\r" not in result

    def test_converts_tabs_to_spaces(self) -> None:
        result = self.executor._format_payload("line1\tline2")
        assert "\t" not in result
        assert "    " in result


class TestBuildDriftError:
    """Tests for _build_drift_error method."""

    def setup_method(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.executor = RuntimeToolExecutor(workspace_root=self.temp_dir)

    def teardown_method(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_build_drift_error_write(self) -> None:
        result = self.executor._build_drift_error(
            "write",
            {"text": "multi\nline\ncontent"},
            "edit --path /tmp/f\n|> text\nmulti\nline\ncontent",
            "text",
        )
        assert "error" in result.lower()
        assert "protocol_drift" in result

    def test_build_drift_error_edit(self) -> None:
        result = self.executor._build_drift_error(
            "edit",
            {"search": "old", "replace": "new\nmulti\nline"},
            "edit --path /tmp/f\n|> search\nold\nreplace\nnew\nmulti\nline",
            "replace",
        )
        assert "error" in result.lower()


class TestCaptureSnapshot:
    """Tests for _capture_snapshot method."""

    def setup_method(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.executor = RuntimeToolExecutor(workspace_root=self.temp_dir)

    def teardown_method(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_track_file_changes_disabled(self) -> None:
        self.executor.track_file_changes = False
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("hello")
        result = self.executor._capture_snapshot(str(test_file))
        assert result is None

    def test_no_delta_tracker(self) -> None:
        self.executor.delta_tracker = None
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("hello")
        result = self.executor._capture_snapshot(str(test_file))
        assert result is None

    def test_no_sifter(self) -> None:
        import tok.runtime.tools as rt

        original_sifter = rt.Sifter
        rt.Sifter = None
        try:
            self.executor.delta_tracker = MagicMock()
            test_file = Path(self.temp_dir) / "test.txt"
            test_file.write_text("hello")
            result = self.executor._capture_snapshot(str(test_file))
            assert result is None
        finally:
            rt.Sifter = original_sifter

    def test_sifter_exception(self) -> None:
        self.executor.delta_tracker = MagicMock()
        with patch("tok.runtime.tools.Sifter") as mock_sifter:
            mock_sifter.from_file.side_effect = Exception("mocked")
            test_file = Path(self.temp_dir) / "test.txt"
            test_file.write_text("hello")
            result = self.executor._capture_snapshot(str(test_file))
            assert result is None

    def test_caches_skeleton(self) -> None:
        self.executor.delta_tracker = MagicMock()
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("hello")
        with patch("tok.runtime.tools.Sifter") as mock_sifter:
            mock_sifter.from_file.return_value = {"skeleton": "mocked skeleton"}
            result1 = self.executor._capture_snapshot(str(test_file))
            self.executor._capture_snapshot(str(test_file))
            assert mock_sifter.from_file.call_count == 1
            assert result1 == "mocked skeleton"


class TestComputeDelta:
    """Tests for _compute_delta method."""

    def setup_method(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.executor = RuntimeToolExecutor(workspace_root=self.temp_dir)

    def teardown_method(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_no_delta_tracker(self) -> None:
        self.executor.delta_tracker = None
        result = self.executor._compute_delta("/tmp/f", "post state")
        assert result == []

    def test_no_post_state(self) -> None:
        self.executor.delta_tracker = MagicMock()
        result = self.executor._compute_delta("/tmp/f", "")
        assert result == []

    def test_no_diff_tok(self) -> None:
        import tok.runtime.tools as rt

        original_diff = rt.diff_tok
        rt.diff_tok = None
        try:
            self.executor.delta_tracker = MagicMock()
            result = self.executor._compute_delta("/tmp/f", "post state")
            assert result == []
        finally:
            rt.diff_tok = original_diff

    def test_no_pre_state(self) -> None:
        self.executor.delta_tracker = MagicMock()
        result = self.executor._compute_delta("/tmp/f", "post state")
        assert result == []

    def test_unchanged_state(self) -> None:
        self.executor.delta_tracker = MagicMock()
        self.executor._pre_state["/tmp/f"] = "same"
        import tok.runtime.tools as rt

        original = rt.diff_tok
        rt.diff_tok = MagicMock(return_value=[])
        try:
            result = self.executor._compute_delta("/tmp/f", "same")
            assert result == []
        finally:
            rt.diff_tok = original

    def test_changed_state(self) -> None:
        self.executor.delta_tracker = MagicMock()
        self.executor._pre_state["/tmp/f"] = "old"
        import tok.runtime.tools as rt

        mock_deltas = [MagicMock(), MagicMock()]
        rt.diff_tok = MagicMock(return_value=mock_deltas)
        try:
            result = self.executor._compute_delta("/tmp/f", "new")
            assert result == mock_deltas
        finally:
            rt.diff_tok = None


class TestLogExecution:
    """Tests for _log_execution method."""

    def setup_method(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.log_path = os.path.join(self.temp_dir, "execution.log")
        self.executor = RuntimeToolExecutor(workspace_root=self.temp_dir, log_path=self.log_path)

    def teardown_method(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_creates_log_directory(self) -> None:
        nested_log = os.path.join(self.temp_dir, "nested", "dir", "execution.log")
        executor = RuntimeToolExecutor(log_path=nested_log)
        executor._log_execution("echo test", "output", "", 0)
        assert os.path.exists(nested_log)

    def test_log_rotation(self) -> None:
        big_content = ("x" * 100 + "\n") * 6000
        with open(self.log_path, "w") as f:
            f.write(big_content)
        self.executor._log_execution("echo test", "output", "", 0)
        with open(self.log_path) as f:
            content = f.read()
        assert len(content) < len(big_content)


class TestExecuteNormalizedToolEdgeCases:
    """Edge case tests for execute_normalized_tool."""

    def setup_method(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.executor = RuntimeToolExecutor(workspace_root=self.temp_dir)

    def teardown_method(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dict_event_converted(self) -> None:
        event = {"id": "t1", "name": "read", "path": "/tmp/f", "args": {}}
        result = self.executor.execute_normalized_tool(event)
        assert result["status"] in ["ERROR", "SUCCESS"]

    def test_invalid_event_type(self) -> None:
        result = self.executor.execute_normalized_tool("not a dict or NormalizedToolEvent")
        assert result["status"] == "ERROR"
        assert "Invalid event type" in result["message"]


class TestExecuteReadEdgeCases:
    """Edge case tests for _execute_read."""

    def setup_method(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.executor = RuntimeToolExecutor(workspace_root=self.temp_dir)

    def teardown_method(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_read_directory(self) -> None:
        result = self.executor._execute_read(NormalizedToolEvent(id="t1", name="read", path=self.temp_dir))
        assert result["status"] == "ERROR"
        assert "directory" in result["message"]

    def test_read_file_exception(self) -> None:
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("hello")
        with patch("builtins.open", side_effect=OSError("mocked")):
            result = self.executor._execute_read(NormalizedToolEvent(id="t1", name="read", path=str(test_file)))
        assert result["status"] == "ERROR"
        assert "Failed to read" in result["message"]


class TestExecuteWriteEdgeCases:
    """Edge case tests for _execute_write."""

    def setup_method(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.executor = RuntimeToolExecutor(workspace_root=self.temp_dir)

    def teardown_method(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_write_invalid_path_chars(self) -> None:
        event = NormalizedToolEvent(id="t1", name="write", path="invalid*path", args={"content": "test"})
        result = self.executor._execute_write(event)
        assert result["status"] == "ERROR"
        assert "Invalid filename" in result["message"]

    def test_write_security_violation(self) -> None:
        event = NormalizedToolEvent(id="t1", name="write", path="/etc/passwd", args={"content": "test"})
        result = self.executor._execute_write(event)
        assert result["status"] == "ERROR"
        assert "Security Violation" in result["message"]


class TestExecuteEditEdgeCases:
    """Edge case tests for _execute_edit."""

    def setup_method(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.executor = RuntimeToolExecutor(workspace_root=self.temp_dir)

    def teardown_method(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_edit_missing_path(self) -> None:
        event = NormalizedToolEvent(id="t1", name="edit", path="", args={"search": "old", "replace": "new"})
        result = self.executor._execute_edit(event)
        assert result["status"] == "ERROR"

    def test_edit_missing_search_replace(self) -> None:
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("content")
        event = NormalizedToolEvent(id="t1", name="edit", path=str(test_file), args={"search": None})
        result = self.executor._execute_edit(event)
        assert result["status"] == "ERROR"
        assert "requires both" in result["message"]

    def test_edit_security_violation(self) -> None:
        event = NormalizedToolEvent(id="t1", name="edit", path="/etc/passwd", args={"search": "old", "replace": "new"})
        result = self.executor._execute_edit(event)
        assert result["status"] == "ERROR"
        assert "Security Violation" in result["message"]


class TestExecuteDelta:
    """Tests for _execute_delta."""

    def setup_method(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.executor = RuntimeToolExecutor(workspace_root=self.temp_dir)

    def teardown_method(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_delta_missing_path(self) -> None:
        event = NormalizedToolEvent(id="t1", name="delta", path="", args={"payload": "add @func|foo"})
        result = self.executor._execute_delta(event)
        assert result["status"] == "ERROR"

    def test_delta_missing_payload(self) -> None:
        event = NormalizedToolEvent(id="t1", name="delta", path="/tmp/f", args={})
        result = self.executor._execute_delta(event)
        assert result["status"] == "ERROR"

    def test_delta_security_violation(self) -> None:
        event = NormalizedToolEvent(id="t1", name="delta", path="/etc/passwd", args={"payload": "add @func|foo"})
        result = self.executor._execute_delta(event)
        assert result["status"] == "ERROR"
        assert "Security Violation" in result["message"]

    def test_delta_tok_not_available(self) -> None:
        import tok.runtime.tools as rt

        original_tok_delta = rt.TokDelta
        original_apply = rt.apply_delta
        rt.TokDelta = None
        rt.apply_delta = None
        try:
            event = NormalizedToolEvent(id="t1", name="delta", path="/tmp/f", args={"payload": "add @func|foo"})
            result = self.executor._execute_delta(event)
            assert result["status"] == "ERROR"
        finally:
            rt.TokDelta = original_tok_delta
            rt.apply_delta = original_apply


class TestResolveDeltaPayload:
    """Tests for _resolve_delta_payload."""

    def setup_method(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.executor = RuntimeToolExecutor(workspace_root=self.temp_dir)

    def teardown_method(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_payload_from_args(self) -> None:
        event = NormalizedToolEvent(id="t1", name="delta", path="/tmp/f", args={"payload": "add @func|foo"})
        result = self.executor._resolve_delta_payload(event)
        assert result == "add @func|foo"

    def test_payload_from_text(self) -> None:
        event = NormalizedToolEvent(id="t1", name="delta", path="/tmp/f", args={"text": "add @func|foo"})
        result = self.executor._resolve_delta_payload(event)
        assert result == "add @func|foo"

    def test_payload_from_content(self) -> None:
        event = NormalizedToolEvent(id="t1", name="delta", path="/tmp/f", args={"content": "add @func|foo"})
        result = self.executor._resolve_delta_payload(event)
        assert result == "add @func|foo"


class TestParseDeltaPayload:
    """Tests for _parse_delta_payload."""

    def setup_method(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.executor = RuntimeToolExecutor(workspace_root=self.temp_dir)

    def teardown_method(self) -> None:
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_parse_add_delta(self) -> None:
        import tok.runtime.tools as rt

        if rt.TokDelta is None:
            pytest.skip("TokDelta not available")
        result = self.executor._parse_delta_payload("add @func|new_func", "/tmp/f")
        assert len(result) == 1
        assert result[0].op == "add"
        assert result[0].target_type == "func"
        assert result[0].target_label == "new_func"

    def test_parse_remove_delta(self) -> None:
        import tok.runtime.tools as rt

        if rt.TokDelta is None:
            pytest.skip("TokDelta not available")
        result = self.executor._parse_delta_payload("remove @class|OldClass", "/tmp/f")
        assert len(result) == 1
        assert result[0].op == "remove"
        assert result[0].target_type == "class"
        assert result[0].target_label == "OldClass"

    def test_parse_update_delta(self) -> None:
        import tok.runtime.tools as rt

        if rt.TokDelta is None:
            pytest.skip("TokDelta not available")
        result = self.executor._parse_delta_payload("update @method|foo", "/tmp/f")
        assert len(result) == 1
        assert result[0].op == "update"
        assert result[0].target_type == "method"
        assert result[0].target_label == "foo"

    def test_parse_delta_with_attrs(self) -> None:
        import tok.runtime.tools as rt

        if rt.TokDelta is None:
            pytest.skip("TokDelta not available")
        result = self.executor._parse_delta_payload(
            "add @func|new_func\n+params:x,y\n+returns:int",
            "/tmp/f",
        )
        assert len(result) == 1
        assert result[0].op == "add"

    def test_parse_empty_payload(self) -> None:
        result = self.executor._parse_delta_payload("", "/tmp/f")
        assert result == []

    def test_parse_no_matching_lines(self) -> None:
        result = self.executor._parse_delta_payload("not a delta command", "/tmp/f")
        assert result == []


class TestDefaultExecutor:
    """Tests for get_default_executor."""

    def teardown_method(self) -> None:
        import tok.runtime.tools as rt

        rt._default_executor = None

    def test_singleton_pattern(self) -> None:
        import tok.runtime.tools as rt

        rt._default_executor = None
        executor1 = rt.get_default_executor()
        executor2 = rt.get_default_executor()
        assert executor1 is executor2

    def test_creates_with_defaults(self) -> None:
        import tok.runtime.tools as rt

        rt._default_executor = None
        executor = rt.get_default_executor()
        assert executor is not None
        assert executor.log_path is not None
