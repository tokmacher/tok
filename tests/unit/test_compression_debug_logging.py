"""Tests for Improvement 0.1.5-4: per-compression-decision DEBUG logging."""

from __future__ import annotations

import logging

import pytest


class TestCompressionDebugLogging:
    """Verify that compression decisions emit structured DEBUG log records."""

    @pytest.fixture()
    def debug_caplog(self, caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:  # type: ignore[no-untyped-def]
        caplog.set_level(logging.DEBUG, logger="tok.compression")
        return caplog

    def test_debug_log_on_tool_result_compress(self, debug_caplog: pytest.LogCaptureFixture) -> None:
        from tok.compression import tok_tool_result

        large_ls = "\n".join(f"-rw-r--r--  1 user  staff  {i} file_{i}.py" for i in range(30))
        tok_tool_result(large_ls)
        records = [r for r in debug_caplog.records if "decision=" in r.message]
        assert len(records) >= 1, f"Expected at least 1 decision log, got {debug_caplog.messages}"

    def test_debug_log_on_tool_result_preserve(self, debug_caplog: pytest.LogCaptureFixture) -> None:
        from tok.compression import tok_tool_result

        short = "ok"
        tok_tool_result(short)
        records = [
            r for r in debug_caplog.records if "decision=preserved" in r.message or "decision=bypassed" in r.message
        ]
        assert len(records) >= 1, f"Expected a decision log, got {debug_caplog.messages}"

    def test_debug_log_includes_kind_field(self, debug_caplog: pytest.LogCaptureFixture) -> None:
        from tok.compression import tok_tool_result

        large_ls = "\n".join(f"-rw-r--r--  1 user  staff  {i} file_{i}.py" for i in range(30))
        tok_tool_result(large_ls)
        records = [r for r in debug_caplog.records if "kind=" in r.message]
        assert len(records) >= 1, f"Expected kind= field in log, got {debug_caplog.messages}"

    def test_debug_log_not_emitted_at_info_level(self, caplog: pytest.LogCaptureFixture) -> None:  # type: ignore[no-untyped-def]
        caplog.set_level(logging.INFO, logger="tok.compression")
        from tok.compression import tok_tool_result

        large_ls = "\n".join(f"-rw-r--r--  1 user  staff  {i} file_{i}.py" for i in range(30))
        tok_tool_result(large_ls)
        decision_records = [r for r in caplog.records if "decision=" in r.message]
        assert len(decision_records) == 0, f"DEBUG decision logs should not appear at INFO level, got {caplog.messages}"
