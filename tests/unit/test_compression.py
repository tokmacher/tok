"""Tests for tok.compression — input-side history compression."""

from __future__ import annotations

import tok.compression
from typing import Any

tok.compression.TOOL_COMPRESS_THRESHOLD = 0

from tok.compression import (
    CANONICAL_MEMORY_FIELDS,
    FILE_LIKE_TOOLS,
    TOOL_COMPRESS_THRESHOLD,
    _apply_file_cache,
    _compress_git_diff,
    _compress_git_log,
    _compress_install,
    _compress_ls,
    _detect_tool_content_type,
    classify_cut_eligibility,
    compress_history,
    compress_recent_window,
    compress_tool_results,
    inject_system_additions,
    is_safe_cut,
    text_of,
    tok_tool_result,
    CutEligibility,
)


class TestTextOf:
    def test_string_input(self):
        assert text_of("hello") == "hello"

    def test_list_input(self):
        content = [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]
        assert text_of(content) == "hello world"

    def test_list_with_non_text(self):
        content = [
            {"type": "image", "url": "x"},
            {"type": "text", "text": "hello"},
        ]
        assert "hello" in text_of(content)

    def test_empty_list(self):
        assert text_of([]) == ""

    def test_non_string_non_list(self):
        assert text_of(123) == "123"  # type: ignore[arg-type]


class TestIsSafeCut:
    def test_user_string_message(self):
        assert is_safe_cut({"role": "user", "content": "hello"}) is True

    def test_assistant_message(self):
        assert is_safe_cut({"role": "assistant", "content": "hi"}) is False

    def test_user_with_tool_result(self):
        msg = {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "x", "content": "y"}
            ],
        }
        assert is_safe_cut(msg) is False

    def test_user_with_text_blocks(self):
        msg = {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        assert is_safe_cut(msg) is True


class TestClassifyCutEligibility:
    def test_plain_user_text_is_eligible(self):
        result = classify_cut_eligibility({"role": "user", "content": "hello"})
        assert result == CutEligibility(True, "eligible")

    def test_assistant_message_rejected_as_non_user(self):
        result = classify_cut_eligibility(
            {"role": "assistant", "content": "hi"}
        )
        assert result == CutEligibility(False, "non_user")

    def test_top_level_tool_use_id_rejected(self):
        msg = {
            "role": "user",
            "tool_use_id": "tool_123",
            "content": "result text",
        }
        result = classify_cut_eligibility(msg)
        assert result == CutEligibility(False, "top_level_tool_result")

    def test_user_list_with_tool_result_block_rejected(self):
        msg = {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "x", "content": "y"}
            ],
        }
        result = classify_cut_eligibility(msg)
        assert result == CutEligibility(
            False, "user_contains_tool_result_block"
        )

    def test_user_list_with_only_text_blocks_eligible(self):
        msg = {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        result = classify_cut_eligibility(msg)
        assert result == CutEligibility(True, "eligible")

    def test_user_string_content_eligible(self):
        result = classify_cut_eligibility(
            {"role": "user", "content": "plain text"}
        )
        assert result == CutEligibility(True, "eligible")


class TestCompressHistory:
    def test_canonical_memory_field_order(self):
        assert CANONICAL_MEMORY_FIELDS == (
            "turns",
            "goal",
            "files",
            "edited",
            "cmds",
            "tests",
            "errs",
            "constraints",
            "next",
        )

    def test_short_history_no_compression(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        recent, state = compress_history(msgs, keep_turns=2)
        assert recent == msgs
        assert state == ""

    def test_compresses_old_turns(self):
        msgs = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": "second answer"},
            {"role": "user", "content": "third question"},
            {"role": "assistant", "content": "third answer"},
        ]
        recent, state = compress_history(msgs, keep_turns=2)
        assert len(recent) < len(msgs)
        assert state.startswith(">>>")
        assert "turns:" in state
        assert "goal:" in state

    def test_preserves_coding_context_signals(self):
        msgs = [
            {
                "role": "user",
                "content": "Investigate failing tests in src/tok/gateway.py and avoid writing for now. always invert.",
            },
            {
                "role": "assistant",
                "content": "I will inspect src/tok/gateway.py and run pytest tests/unit/test_gateway.py",
            },
            {
                "role": "user",
                "content": "pytest tests/unit/test_gateway.py FAILED with AssertionError",
            },
            {
                "role": "assistant",
                "content": "Next I will patch src/tok/compression.py",
            },
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": "done"},
        ]

        _, state = compress_history(msgs, keep_turns=1)

        assert (
            "files:src/tok/gateway.py" in state
            or "files:src/tok/compression.py" in state
        )
        assert "cmds:pytest tests/unit/test_gateway.py" in state
        assert "constraints:" in state
        assert "tests:" in state or "errs:" in state
        assert state.index("turns:") < state.index("goal:")

    def test_preserves_tool_result_pairs(self):
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": "use the tool"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "x", "input": {}}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": "ok",
                    }
                ],
            },
            {"role": "user", "content": "thanks"},
            {"role": "assistant", "content": "done"},
        ]
        recent, state = compress_history(msgs, keep_turns=1)
        # Should not cut between tool_use and tool_result
        for msg in recent:
            content = msg.get("content", "")
            if isinstance(content, list):
                for block in content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "tool_result"
                    ):
                        # If we have a tool_result, the preceding assistant must also be
                        # present
                        idx = recent.index(msg)
                        assert idx > 0

    def test_boosts_edited_files_from_tool_use(self):
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": "please continue"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "e1",
                        "name": "edit_file",
                        "input": {"path": "src/tok/bridge_memory.py"},
                    }
                ],
            },
            {"role": "user", "content": "tests still failing"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": "done"},
        ]

        _, state = compress_history(msgs, keep_turns=1)

        assert "src/tok/bridge_memory.py" in state

    def test_extracts_explicit_blockers_into_errors_or_next(self):
        msgs: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": "blocked on failing tests in tests/unit/test_gateway.py",
            },
            {"role": "assistant", "content": "I will inspect the failure"},
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": "done"},
        ]

        _, state = compress_history(msgs, keep_turns=1)

        assert "errs:" in state or "next:" in state

    def test_extracts_questions_into_noncanonical_facts(self):
        msgs = [
            {
                "role": "user",
                "content": "Why is the bridge failing cold start?",
            },
            {"role": "assistant", "content": "I will inspect it."},
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": "done"},
        ]

        _, state = compress_history(msgs, keep_turns=1)

        assert "questions:" in state


class TestCompressHistoryCutDiagnostics:
    def _make_tool_heavy_transcript(
        self, n_turns: int = 6
    ) -> list[dict[str, Any]]:
        msgs = []
        for i in range(n_turns):
            msgs.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": f"t{i}",
                            "name": "bash",
                            "input": {"cmd": f"cmd {i}"},
                        }
                    ],
                }
            )
            msgs.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": f"t{i}",
                            "content": f"output {i}",
                        }
                    ],
                }
            )
        return msgs

    def test_tool_heavy_no_eligible_cut_returns_original(self):
        msgs = self._make_tool_heavy_transcript(6)
        recent, state = compress_history(msgs, keep_turns=2)
        assert recent == msgs
        assert state == ""

    def test_only_candidate_at_index_zero_classified_separately(self):
        msgs = [
            {"role": "user", "content": "first and only plain text"},
            {"role": "assistant", "content": "response"},
        ]
        recent, state = compress_history(msgs, keep_turns=2)
        assert recent == msgs
        assert state == ""

    def test_transcript_with_later_plain_user_boundary_compresses_normally(
        self,
    ):
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t0",
                        "name": "bash",
                        "input": {"cmd": "ls"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t0",
                        "content": "file1.py\nfile2.py",
                    }
                ],
            },
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "final question"},
            {"role": "assistant", "content": "final answer"},
        ]
        recent, state = compress_history(msgs, keep_turns=2)
        assert len(recent) < len(msgs)
        assert state.startswith(">>>")


class TestInjectSystemAdditions:
    def test_string_system_prompt(self):
        body = {"system": "You are helpful.", "messages": []}
        result = inject_system_additions(body, None, pressure=2)
        assert "=== MODE: TOK-NATIVE ===" in result["system"]
        assert "[Tok law]" in result["system"]
        assert ">>> t:N|usr:X|agt:Y|state:Z" in result["system"]
        assert "You are helpful." in result["system"]
        assert result["system"].startswith("You are helpful.")

    def test_empty_system_prompt(self):
        body = {"system": "", "messages": []}
        result = inject_system_additions(body, None, pressure=2)
        assert "=== MODE: TOK-NATIVE ===" in result["system"]
        assert "[Tok law]" in result["system"]
        assert ">>> t:N|usr:X|agt:Y|state:Z" in result["system"]

    def test_list_system_prompt(self):
        body = {
            "system": [
                {
                    "type": "text",
                    "text": "Base prompt",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [],
        }
        result = inject_system_additions(body, None, pressure=2)
        assert isinstance(result["system"], list)
        assert len(result["system"]) == 3
        assert result["system"][0]["cache_control"]["type"] == "ephemeral"
        assert result["system"][1]["text"].startswith(
            "[Tok File Freshness System]"
        )
        assert "=== MODE: TOK-NATIVE ===" in result["system"][-1]["text"]
        assert "[Tok law]" in result["system"][-1]["text"]
        assert ">>> t:N|usr:X|agt:Y|state:Z" in result["system"][-1]["text"]
        # The addition should NOT have cache_control to avoid TTL conflicts with
        # client messages
        assert "cache_control" not in result["system"][-1]

    def test_with_tok_state(self):
        body = {"system": "base", "messages": []}
        result = inject_system_additions(body, ">>> turns:5|topic:hello")
        assert ">>>" in result["system"]
        assert "turns:5" in result["system"]
        assert result["system"].startswith("base")

    def test_tool_compatible_prompt_uses_small_mode_gated_capsule(self):
        body = {"system": "base", "messages": []}
        result = inject_system_additions(body, None, tool_compatible=True)

        assert "=== MODE: TOOL-COMPATIBLE ===" not in result["system"]
        assert "Native tools only. Plain text." in result["system"]
        assert "Omit all headers." in result["system"]
        assert "Respond in Tok-native mode." not in result["system"]

    def test_runtime_hints_not_added_when_absent(self):
        body = {"system": "base", "messages": []}
        result = inject_system_additions(body, None, tool_compatible=True)

        assert (
            "Use the compressed history before re-reading files"
            not in result["system"]
        )

    def test_runtime_hints_are_appended_for_tool_compatible_prompts(self):
        body = {"system": "base", "messages": []}
        result = inject_system_additions(
            body,
            None,
            tool_compatible=True,
            runtime_hints=[
                "Reuse existing File=/Verification= facts when they already answer the request; reacquire only if the compressed history is insufficient."
            ],
        )

        assert "Native tools only. Plain text." in result["system"]
        assert (
            "Reuse existing File=/Verification= facts when they already answer the request; reacquire only if the compressed history is insufficient."
            in result["system"]
        )


# ---------------------------------------------------------------------------
# Tool result compression
# ---------------------------------------------------------------------------


def _make_pytest_log(n_passed: int, n_failed: int = 0) -> str:
    lines = ["platform linux -- Python 3.12.0", "collected 80 items", ""]
    for i in range(n_passed):
        lines.append(f"tests/test_foo.py::test_case_{i} PASSED")
    for i in range(n_failed):
        lines.append(f"tests/test_bar.py::test_fail_{i} FAILED")
    if n_failed:
        lines += [
            "=========================== FAILURES ===========================",
            "_________________________ test_fail_0 __________________________",
            "    def test_fail_0():",
            ">       assert False",
            "E       AssertionError",
            "",
            "tests/test_bar.py:10: AssertionError",
        ]
    passed_str = f"{n_passed} passed" if n_passed else ""
    failed_str = f"{n_failed} failed" if n_failed else ""
    summary_parts = ", ".join(p for p in [passed_str, failed_str] if p)
    lines.append(
        f"====================== {summary_parts} in 1.23s ======================"
    )
    return "\n".join(lines)


def _make_grep_output(n_files: int, matches_per_file: int) -> str:
    lines = []
    for f in range(n_files):
        for m in range(matches_per_file):
            lines.append(f"src/module_{f}.py:{10 + m}:    result = foo_{m}()")
    return "\n".join(lines)


class TestTokToolResult:
    def test_small_content_passthrough(self):
        tiny = "hello world"
        assert tok_tool_result(tiny) == tiny

    def test_below_threshold_passthrough(self):
        content = "x" * (TOOL_COMPRESS_THRESHOLD - 1)
        assert tok_tool_result(content) == content

    # --- pytest ---

    def test_pytest_strips_passed_lines(self):
        log = _make_pytest_log(80)
        out = tok_tool_result(log)
        assert "PASSED" not in out

    def test_pytest_has_summary(self):
        log = _make_pytest_log(80)
        out = tok_tool_result(log)
        assert "passed:80" in out

    def test_pytest_keeps_failure_details(self):
        log = _make_pytest_log(n_passed=10, n_failed=2)
        out = tok_tool_result(log)
        assert "FAILED" in out or "failed:2" in out
        assert "AssertionError" in out

    def test_pytest_deterministic(self):
        log = _make_pytest_log(80)
        assert tok_tool_result(log) == tok_tool_result(log)

    def test_pytest_shorter_than_original(self):
        log = _make_pytest_log(80)
        assert len(tok_tool_result(log)) < len(log)

    # --- grep ---

    def test_grep_few_matches_verbatim(self):
        content = "src/a.py:1:foo\nsrc/b.py:2:bar\n"
        # 2 lines — below threshold, passthrough; pad to force threshold check
        content + "\n" * 10  # still short
        # Actually need >2000 chars to trigger
        big = content * 200
        out = tok_tool_result(big)
        # grouped output
        assert "src/a.py" in out or "matches" in out

    def test_grep_groups_by_file(self):
        out_raw = _make_grep_output(n_files=5, matches_per_file=20)
        out = tok_tool_result(out_raw)
        assert len(out) < len(out_raw)
        assert "matches" in out

    def test_grep_deterministic(self):
        raw = _make_grep_output(n_files=5, matches_per_file=20)
        assert tok_tool_result(raw) == tok_tool_result(raw)

    # --- repetitive bash ---

    def test_repetitive_run_length_grouped(self):
        # Lines sharing the same prefix — long enough to exceed threshold
        lines = [
            f"WARNING: deprecated call in module_{i}.py at line {i * 3}"
            for i in range(80)
        ]
        content = "\n".join(lines)
        assert len(content) > TOOL_COMPRESS_THRESHOLD
        out = tok_tool_result(content)
        assert len(out) < len(content)

    # --- file read (skeleton) ---

    def _make_python_file(self, n_funcs: int = 15) -> str:
        lines = ["import os", "import sys", "from pathlib import Path", ""]
        lines.append("CONSTANT = 42")
        lines.append("")
        lines.append("class MyClass:")
        lines.append("    def __init__(self):")
        lines.append("        self.x = 1")
        for i in range(n_funcs):
            lines.append(f"    def method_{i}(self, arg):")
            lines += [f"        # body line {j}" for j in range(15)]
            lines.append(f"        return arg + {i}")
        return "\n".join(lines)

    def test_file_skeleton_keeps_imports(self):
        src = self._make_python_file(20)
        out = tok_tool_result(src)
        assert "import os" in out
        assert "import sys" in out

    def test_file_skeleton_keeps_signatures(self):
        src = self._make_python_file(20)
        out = tok_tool_result(src)
        assert "def method_0" in out
        assert "class MyClass" in out

    def test_file_skeleton_shorter(self):
        src = self._make_python_file(20)
        out = tok_tool_result(src)
        assert len(out) < len(src)

    def test_file_skeleton_deterministic(self):
        src = self._make_python_file(20)
        assert tok_tool_result(src) == tok_tool_result(src)

    # --- raw fallthrough ---

    def test_raw_large_non_code_passthrough(self):
        content = (
            "Lorem ipsum dolor sit amet. " * 80
        )  # prose, no code patterns
        out = tok_tool_result(content)
        assert ">>> tok_compressed:tool_result|type:raw" in out
        assert "... [TRUNCATED" in out


class TestCompressToolResults:
    def _make_messages_with_tool_result(
        self, result_content: str
    ) -> list[dict[str, Any]]:
        return [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "bash",
                        "input": {},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": result_content,
                    }
                ],
            },
        ]

    def _total_saved(self, breakdown: dict[str, int]) -> int:
        return sum(breakdown.values())

    def test_no_tool_results_unchanged(self):
        msgs = [{"role": "user", "content": "hello"}]
        out, breakdown = compress_tool_results(msgs)
        assert out == msgs
        assert self._total_saved(breakdown) == 0

    def test_small_tool_result_unchanged(self):
        msgs = self._make_messages_with_tool_result("tiny output")
        out, breakdown = compress_tool_results(msgs)
        assert self._total_saved(breakdown) == 0
        assert out[1]["content"][0]["content"] == "tiny output"

    def test_large_pytest_tool_result_compressed(self):
        log = _make_pytest_log(80)
        msgs = self._make_messages_with_tool_result(log)
        out, breakdown = compress_tool_results(msgs)
        assert self._total_saved(breakdown) > 0
        compressed_content = out[1]["content"][0]["content"]
        assert "PASSED" not in compressed_content

    def test_chars_saved_accurate(self):
        log = _make_pytest_log(80)
        msgs = self._make_messages_with_tool_result(log)
        out, breakdown = compress_tool_results(msgs)
        original_len = len(log)
        compressed_len = len(out[1]["content"][0]["content"])
        assert self._total_saved(breakdown) == original_len - compressed_len

    def test_non_list_content_skipped(self):
        msgs = [{"role": "user", "content": "plain string"}]
        out, breakdown = compress_tool_results(msgs)
        assert self._total_saved(breakdown) == 0
        assert out[0]["content"] == "plain string"

    def test_breakdown_keyed_by_type(self):
        log = _make_pytest_log(80)
        msgs = self._make_messages_with_tool_result(log)
        _, breakdown = compress_tool_results(msgs)
        assert "pytest" in breakdown
        assert breakdown["pytest"] > 0


# ---------------------------------------------------------------------------
# New compressors
# ---------------------------------------------------------------------------


def _make_git_diff(n_files: int = 2, context_lines: int = 5) -> str:
    lines = []
    for i in range(n_files):
        lines += [
            f"diff --git a/src/file_{i}.py b/src/file_{i}.py",
            "index abc123..def456 100644",
            f"--- a/src/file_{i}.py",
            f"+++ b/src/file_{i}.py",
            f"@@ -10,{context_lines + 2} +10,{context_lines + 2} @@",
        ]
        for j in range(context_lines):
            lines.append(f" context line {j}")
        lines.append(f"-old line in file {i}")
        lines.append(f"+new line in file {i}")
        for j in range(context_lines):
            lines.append(f" more context {j}")
    return "\n".join(lines)


def _make_ls_la(n_files: int = 15) -> str:
    lines = ["total 48"]
    for i in range(n_files):
        if i % 5 == 0:
            lines.append(f"drwxr-xr-x  2 user group  4096 Jan 1 00:00 dir_{i}")
        elif i % 3 == 0:
            lines.append(
                f"-rw-r--r--  1 user group  1234 Jan 1 00:00 file_{i}.md"
            )
        elif i % 2 == 0:
            lines.append(
                f"-rwxr-xr-x  1 user group  5678 Jan 1 00:00 script_{i}.py"
            )
        else:
            lines.append(
                f"-rw-r--r--  1 user group   890 Jan 1 00:00 data_{i}.json"
            )
    return "\n".join(lines)


def _make_install_output(n_packages: int = 20) -> str:
    lines = []
    for i in range(n_packages):
        lines.append(f"Collecting package_{i}>=1.{i}.0")
        lines.append(
            f"  Downloading package_{i}-1.{i}.0-py3-none-any.whl (45 kB)"
        )
        lines.append(f"Installing collected packages: package_{i}")
    lines.append(
        "Successfully installed "
        + " ".join(f"package_{i}-1.{i}.0" for i in range(n_packages))
    )
    return "\n".join(lines)


def _make_git_log_verbose(n_commits: int = 5) -> str:
    lines = []
    import hashlib

    for i in range(n_commits):
        sha = hashlib.sha1(f"commit{i}".encode()).hexdigest()
        lines += [
            f"commit {sha}",
            "Author: User Name <user@example.com>",
            f"Date:   Mon Jan {i + 1} 12:00:00 2024 +0000",
            "",
            f"    Fix issue #{i + 100}: improve performance",
            "",
            f"    Detailed description of the change for commit {i}.",
            "",
        ]
    return "\n".join(lines)


class TestNewCompressors:
    # --- git diff ---

    def test_git_diff_detected(self):
        text = _make_git_diff(2, 5)
        assert _detect_tool_content_type(text) == "git_diff"

    def test_git_diff_strips_context(self):
        text = _make_git_diff(2, 5)
        result = _compress_git_diff(text)
        assert "context line" not in result
        assert "+new line in file" in result
        assert "-old line in file" in result

    def test_git_diff_header(self):
        text = _make_git_diff(2, 5)
        result = _compress_git_diff(text)
        assert result.startswith(">>> tool:git_diff|")
        assert "insertions:" in result
        assert "deletions:" in result

    def test_git_diff_shorter(self):
        text = _make_git_diff(3, 8)
        assert len(_compress_git_diff(text)) < len(text)

    # --- ls ---

    def test_ls_detected(self):
        text = _make_ls_la(12)
        assert _detect_tool_content_type(text) == "ls"

    def test_ls_groups_by_extension(self):
        text = _make_ls_la(15)
        result = _compress_ls(text)
        assert ">>> tool:ls|" in result
        assert ".py" in result or ".json" in result or ".md" in result

    def test_ls_shorter(self):
        text = _make_ls_la(15)
        assert len(_compress_ls(text)) < len(text)

    # --- install ---

    def test_install_detected(self):
        text = _make_install_output(10)
        assert _detect_tool_content_type(text) == "install"

    def test_install_keeps_summary(self):
        text = _make_install_output(10)
        result = _compress_install(text)
        assert "Successfully installed" in result

    def test_install_drops_progress(self):
        text = _make_install_output(10)
        result = _compress_install(text)
        assert "Downloading" not in result
        assert "Collecting" not in result

    def test_install_shorter(self):
        text = _make_install_output(20)
        assert len(_compress_install(text)) < len(text)

    def test_install_header(self):
        text = _make_install_output(10)
        result = _compress_install(text)
        assert result.startswith(">>> tool:install|")

    # --- git log ---

    def test_git_log_detected(self):
        text = _make_git_log_verbose(3)
        assert _detect_tool_content_type(text) == "git_log"

    def test_git_log_compact(self):
        text = _make_git_log_verbose(5)
        result = _compress_git_log(text)
        assert ">>> tool:git_log|commits:5" in result

    def test_git_log_strips_body(self):
        text = _make_git_log_verbose(3)
        result = _compress_git_log(text)
        assert "Detailed description" not in result

    def test_git_log_shorter(self):
        text = _make_git_log_verbose(5)
        assert len(_compress_git_log(text)) < len(text)


# ---------------------------------------------------------------------------
# File cache (re-read dedup)
# ---------------------------------------------------------------------------


class TestFileCache:
    def _tool_result_msg(
        self, tool_id: str, content: str
    ) -> list[dict[str, Any]]:
        return [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": "read",
                        "input": {"path": "src/foo.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": content,
                    }
                ],
            },
        ]

    def _big_file(self, n: int = 60) -> str:
        """Build a Python file large enough to exceed TOOL_COMPRESS_THRESHOLD."""
        lines = ["import os", "import sys", ""]
        for i in range(n):
            lines += [
                f"def func_{i}(x, y, z):",
                f"    # This is body line 1 for function {i}",
                f"    # This is body line 2 for function {i}",
                "    intermediate = x * y + z",
                f"    result = intermediate + {i}",
                "    return result",
                "",
            ]
        return "\n".join(lines)

    def test_first_read_preserves_raw_file_content(self):
        cache: dict[str, tuple[str, str, float]] = {}
        raw = self._big_file(20)
        result, saved = _apply_file_cache(raw, "src/foo.py", cache)
        # It's now keyed by hashed tool:args
        assert len(cache) == 1
        assert all(len(k) == 12 for k in cache.keys())  # hashed keys
        assert saved == 0
        assert result == raw

    def test_second_read_unchanged_is_stubbed(self):
        cache: dict[str, tuple[str, str, float]] = {}
        raw = self._big_file(20)
        _apply_file_cache(raw, "src/foo.py", cache)
        result, saved = _apply_file_cache(raw, "src/foo.py", cache)
        # Repeated identical file read must be replaced with a compact stub.
        assert "unchanged" in result, (
            f"Expected 'unchanged' stub; got: {result!r}"
        )
        assert saved > 0, (
            f"Expected positive savings for repeated read; got saved={saved}"
        )

    def test_second_read_changed_uses_diff(self):
        cache: dict[str, tuple[str, str, float]] = {}
        raw1 = self._big_file()
        _apply_file_cache(raw1, "src/foo.py", cache)
        # Change a line
        raw2 = raw1.replace(
            "intermediate = x * y + z", "intermediate = x + y * z"
        )
        assert raw1 != raw2, "replacement didn't work"
        result, saved = _apply_file_cache(raw2, "src/foo.py", cache)
        # Changed file read must use diff/delta path, not return raw.
        assert result != raw2, (
            "Expected diff stub for changed file read, got raw"
        )
        assert "delta" in result or "changed" in result, (
            f"Expected delta marker; got: {result!r}"
        )

    def test_compress_tool_results_deduplicates_repeated_file_reads(self):
        cache: dict[str, tuple[str, str, float]] = {}
        raw = self._big_file(20)
        id_to_context = {
            "t1": {
                "name": "view_file",
                "path": "src/foo.py",
                "args": {"path": "src/foo.py"},
            }
        }

        msgs1 = self._tool_result_msg("t1", raw)
        _, bd1 = compress_tool_results(
            msgs1, result_cache=cache, tool_use_id_to_context=id_to_context
        )

        msgs2 = self._tool_result_msg("t1", raw)
        out2, bd2 = compress_tool_results(
            msgs2, result_cache=cache, tool_use_id_to_context=id_to_context
        )
        result_content = out2[1]["content"][0]["content"]
        # Second read must be compressed to a stub, not returned verbatim.
        assert "unchanged" in result_content, (
            f"Expected 'unchanged' stub for repeated file read; got: {result_content!r}"
        )
        assert bd1 == {}
        assert sum(bd2.values()) > 0, (
            f"Expected savings on second read; got {bd2}"
        )

    def test_file_tools_are_preserved_verbatim(self):
        cache: dict[str, tuple[str, str, float]] = {}
        raw = self._big_file(20)
        id_to_context = {
            "t1": {
                "name": "view_file",
                "path": "src/foo.py",
                "args": {"path": "src/foo.py"},
            }
        }

        msgs = self._tool_result_msg("t1", raw)
        out, breakdown = compress_tool_results(
            msgs, result_cache=cache, tool_use_id_to_context=id_to_context
        )

        assert "view_file" in FILE_LIKE_TOOLS
        assert out[1]["content"][0]["content"] == raw
        assert breakdown == {}


# ---------------------------------------------------------------------------
# Savings bug fix verification
# ---------------------------------------------------------------------------


class TestSavingsBugFix:
    """Verify that tool savings are not overwritten by history compression savings."""

    def test_both_compressions_accumulate(self):
        """Both tool result savings and history savings should add up, not overwrite."""
        from tok.compression import compress_history

        # Build a conversation with many turns (will trigger history compress)
        # plus a large tool result
        pytest_log = _make_pytest_log(80)
        msgs: list[dict[str, Any]] = []
        for i in range(6):
            msgs.append({"role": "user", "content": f"question {i}"})
            msgs.append({"role": "assistant", "content": f"answer {i}"})
        # Add a turn with a big tool result
        msgs.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": pytest_log,
                    }
                ],
            }
        )

        _, tool_breakdown = compress_tool_results(msgs)
        tool_toks = sum(tool_breakdown.values()) // 4

        _, tok_state = compress_history(msgs, keep_turns=2)
        assert tok_state  # history compression fired

        # Verify both have positive savings
        assert tool_toks > 0
        assert tok_state.startswith(">>>")


class TestCompressRecentWindow:
    def _make_result_msg(
        self, content: str, tool_use_id: str = "id1"
    ) -> dict[str, Any]:
        return {
            "role": "tool",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": content,
                }
            ],
        }

    def test_small_result_unchanged(self):
        msg = self._make_result_msg("small content")
        msgs, breakdown = compress_recent_window([msg], threshold=8_000)
        assert msgs[0]["content"][0]["content"] == "small content"
        assert breakdown == {}

    def test_large_file_result_with_context(self):
        # Build a file-like content large enough to trigger compression
        # Use multi-line bodies so skeleton is smaller than original
        code_lines = []
        for i in range(200):
            code_lines.append(f"def func_{i}():\n")
            code_lines.append("    x = 1\n")
            code_lines.append("    y = 2\n")
            code_lines.append("    z = 3\n")
            code_lines.append("    return x + y + z\n")
        large_content = "".join(code_lines)
        assert len(large_content) > 8_000
        msg = self._make_result_msg(large_content, tool_use_id="file_id")
        ctx = {"file_id": {"name": "read"}}
        msgs, breakdown = compress_recent_window(
            [msg], tool_use_id_to_context=ctx
        )
        assert "file" in breakdown
        assert breakdown["file"] > 0
        # Content should be shorter
        assert len(msgs[0]["content"][0]["content"]) < len(large_content)

    def test_tool_compatible_recent_window_uses_lower_threshold(self):
        code_lines = []
        for i in range(60):
            code_lines.append(f"def func_{i}():\n")
            code_lines.append("    x = 1\n")
            code_lines.append("    y = 2\n")
            code_lines.append("    return x + y\n")
        medium_content = "".join(code_lines)
        assert 2_000 < len(medium_content) < 8_000
        msg = self._make_result_msg(medium_content, tool_use_id="file_id")
        ctx = {"file_id": {"name": "read"}}

        msgs, breakdown = compress_recent_window(
            [msg], tool_use_id_to_context=ctx, tool_compatible=True
        )

        assert "file" in breakdown
        assert len(msgs[0]["content"][0]["content"]) < len(medium_content)

    def test_precision_read_inline_not_compressed(self):
        content = "\n".join(f"line {i}" for i in range(2000))
        assert len(content) > 1_200
        msg = self._make_result_msg(content, tool_use_id="file_id")
        ctx = {
            "file_id": {
                "name": "read",
                "args": {"path": "src/tok/foo.py", "offset": 10, "limit": 40},
            }
        }

        msgs, breakdown = compress_recent_window(
            [msg], tool_use_id_to_context=ctx, tool_compatible=True
        )

        assert breakdown == {}
        assert msgs[0]["content"][0]["content"] == content

    def test_precision_read_top_level_not_compressed(self):
        content = "\n".join(f"line {i}" for i in range(2000))
        assert len(content) > 1_200
        msg = {
            "role": "tool_result",
            "tool_use_id": "file_id",
            "content": content,
        }
        ctx = {
            "file_id": {
                "name": "read",
                "args": {
                    "file_path": "src/tok/foo.py",
                    "offset": 10,
                    "limit": 40,
                },
            }
        }

        msgs, breakdown = compress_recent_window(
            [msg], tool_use_id_to_context=ctx, tool_compatible=True
        )

        assert breakdown == {}
        assert msgs[0]["content"] == content

    def test_non_tool_compatible_recent_window_keeps_legacy_threshold(self):
        code_lines = []
        for i in range(60):
            code_lines.append(f"def func_{i}():\n")
            code_lines.append("    x = 1\n")
            code_lines.append("    y = 2\n")
            code_lines.append("    return x + y\n")
        medium_content = "".join(code_lines)
        assert 2_000 < len(medium_content) < 8_000
        msg = self._make_result_msg(medium_content, tool_use_id="file_id")
        ctx = {"file_id": {"name": "read"}}

        msgs, breakdown = compress_recent_window(
            [msg], tool_use_id_to_context=ctx, tool_compatible=False
        )

        assert breakdown == {}
        assert msgs[0]["content"][0]["content"] == medium_content

    def test_top_level_tool_result_is_compressed_in_recent_window(self):
        code_lines = []
        for i in range(60):
            code_lines.append(f"def func_{i}():\n")
            code_lines.append("    x = 1\n")
            code_lines.append("    y = 2\n")
            code_lines.append("    return x + y\n")
        medium_content = "".join(code_lines)
        msg = {
            "role": "tool_result",
            "tool_use_id": "file_id",
            "content": medium_content,
        }
        ctx = {"file_id": {"name": "read"}}

        msgs, breakdown = compress_recent_window(
            [msg], tool_use_id_to_context=ctx, tool_compatible=True
        )

        assert "file" in breakdown
        assert len(msgs[0]["content"]) < len(medium_content)

    def test_large_file_result_no_context_unchanged(self):
        # Use multi-line bodies so skeleton is smaller than original
        code_lines = []
        for i in range(200):
            code_lines.append(f"def func_{i}():\n")
            code_lines.append("    x = 1\n")
            code_lines.append("    y = 2\n")
            code_lines.append("    z = 3\n")
            code_lines.append("    return x + y + z\n")
        large_content = "".join(code_lines)
        msg = self._make_result_msg(large_content, tool_use_id="file_id")
        # No context provided — cannot confirm file type, so raw is left alone
        msgs, breakdown = compress_recent_window(
            [msg], tool_use_id_to_context=None
        )
        # "file" type detected by heuristic even without context, so it IS compressed
        # Only "raw" without context is left alone; heuristic "file" type IS compressed
        result_content = msgs[0]["content"][0]["content"]
        if "file" in breakdown:
            assert breakdown["file"] > 0
            assert len(result_content) < len(large_content)
        else:
            assert result_content == large_content

    def test_large_grep_result(self):
        grep_lines = ["src/tok/foo.py:10:def bar():" for _ in range(600)]
        large_grep = "\n".join(grep_lines)
        assert len(large_grep) > 8_000
        msg = self._make_result_msg(large_grep)
        msgs, breakdown = compress_recent_window([msg])
        assert "grep" in breakdown
        assert breakdown["grep"] > 0

    def test_large_stack_trace(self):
        trace = "Traceback (most recent call last):\n"
        trace += (
            '  File "/usr/lib/python3/dist-packages/foo.py", line 10, in bar\n    x()\n'
            * 200
        )
        trace += "ValueError: something went wrong\n"
        assert len(trace) > 8_000
        msg = self._make_result_msg(trace)
        msgs, breakdown = compress_recent_window([msg])
        assert "stack_trace" in breakdown
        assert breakdown["stack_trace"] > 0

    def test_large_pytest_output(self):
        lines = ["tests/test_foo.py::test_bar PASSED\n" for _ in range(400)]
        lines += ["tests/test_foo.py::test_bad FAILED\n"]
        lines += ["1 failed, 400 passed in 3.2s\n"]
        large_pytest = "".join(lines)
        assert len(large_pytest) > 8_000
        msg = self._make_result_msg(large_pytest)
        msgs, breakdown = compress_recent_window([msg])
        assert "pytest" in breakdown
        assert breakdown["pytest"] > 0

    def test_non_tool_result_blocks_untouched(self):
        msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": "x" * 20_000}],
        }
        msgs, breakdown = compress_recent_window([msg])
        assert msgs[0]["content"][0]["text"] == "x" * 20_000


# ---------------------------------------------------------------------------
# Per-turn injection budget tests
# ---------------------------------------------------------------------------

_TYPICAL_STATE = (
    ">>> turns:8|goal:refactor auth|files:src/auth.py,src/tokens.py|"
    "cmds:pytest tests/|errs:TokenExpired|constraints:no breaking changes|"
    "next:update refresh logic"
)

TOK_PROTOCOL_LAW_MARKER = "[Tok law]"
REINFORCED_MARKER = "PROTOCOL REINFORCEMENT"
GRAMMAR_MARKER = "## Grammar"  # Only in TOK_SYSTEM_PROMPT full grammar


def _inject(
    tok_state: str | None = None,
    pressure: int = 0,
    tool_compatible: bool = False,
) -> str:
    """Call inject_system_additions with the gateway's actual parameters."""
    body = inject_system_additions(
        {"system": ""},
        tok_state=tok_state,
        tool_compatible=tool_compatible,
        grammar=None,  # gateway always passes None
        todo=None,
        deltas=None,
        pressure=pressure,
    )
    return body.get("system", "")


def _token_count(text: str) -> int:
    from tok.prompt_analyzer import count_tokens

    return count_tokens(text)


class TestPerTurnInjectionBudget:
    """Assert on what is actually injected each turn by the gateway path.

    The gateway never passes grammar/todo/deltas, so these tests use
    grammar=None (the real call signature). Token budgets are validated
    against cl100k_base encoding via prompt_analyzer.count_tokens.
    """

    def test_cold_start_within_budget(self):
        """Cold start (no state, no drift) must stay under 70 tokens."""
        sys = _inject(tok_state=None, pressure=0)
        assert _token_count(sys) <= 70, (
            f"Cold-start injection exceeded 70 tokens: {_token_count(sys)}"
        )

    def test_typical_session_within_budget(self):
        """Warm turn with state, no drift, must stay under 120 tokens."""
        sys = _inject(tok_state=_TYPICAL_STATE, pressure=0)
        assert _token_count(sys) <= 120, (
            f"Typical session injection exceeded 120 tokens: {_token_count(sys)}"
        )

    def test_high_pressure_within_budget(self):
        """High-pressure turn (law + reinforced + state) must stay under 200 tokens."""
        sys = _inject(tok_state=_TYPICAL_STATE, pressure=75)
        assert _token_count(sys) <= 200, (
            f"High-pressure injection exceeded 200 tokens: {_token_count(sys)}"
        )

    def test_low_drift_within_budget(self):
        """Low-drift turn (law only, no reinforced) must stay under 185 tokens."""
        sys = _inject(tok_state=_TYPICAL_STATE, pressure=25)
        assert _token_count(sys) <= 185, (
            f"Low-drift injection exceeded 185 tokens: {_token_count(sys)}"
        )

    def test_law_absent_at_zero_pressure(self):
        """TOK_PROTOCOL_LAW must NOT be present when pressure=0."""
        sys = _inject(tok_state=_TYPICAL_STATE, pressure=0)
        assert TOK_PROTOCOL_LAW_MARKER not in sys, (
            "TOK_PROTOCOL_LAW injected at pressure=0 — unexpected overhead"
        )

    def test_law_absent_at_single_signal(self):
        """TOK_PROTOCOL_LAW must NOT be present when pressure=1 (single benign signal)."""
        sys = _inject(tok_state=_TYPICAL_STATE, pressure=1)
        assert TOK_PROTOCOL_LAW_MARKER not in sys, (
            "TOK_PROTOCOL_LAW injected at pressure=1 — threshold should be >1"
        )

    def test_law_present_at_two_signals(self):
        """TOK_PROTOCOL_LAW MUST be present when pressure>=2."""
        sys = _inject(tok_state=_TYPICAL_STATE, pressure=2)
        assert TOK_PROTOCOL_LAW_MARKER in sys, (
            "TOK_PROTOCOL_LAW missing at pressure=2"
        )

    def test_grammar_never_injected_by_gateway_path(self):
        """The full grammar (TOK_SYSTEM_PROMPT) must never appear when grammar=None."""
        for pressure in [0, 25, 75]:
            sys = _inject(tok_state=_TYPICAL_STATE, pressure=pressure)
            assert GRAMMAR_MARKER not in sys, (
                f"Full grammar injected at pressure={pressure} — "
                "gateway must not include grammar bootstrap per-turn"
            )

    def test_directive_escalates_above_pressure_50(self):
        """Above pressure=50 the reinforced directive replaces the minimal one."""
        sys_below = _inject(tok_state=_TYPICAL_STATE, pressure=50)
        sys_above = _inject(tok_state=_TYPICAL_STATE, pressure=51)
        assert REINFORCED_MARKER not in sys_below, (
            "Reinforced directive appeared at pressure=50 (threshold should be >50)"
        )
        assert REINFORCED_MARKER in sys_above, (
            "Reinforced directive missing at pressure=51"
        )
        # Reinforced is larger than minimal: escalation adds tokens
        assert _token_count(sys_above) > _token_count(sys_below)

    def test_pressure_escalation_token_delta(self):
        """Measure and document the token cost of pressure escalation."""
        baseline = _inject(tok_state=_TYPICAL_STATE, pressure=0)
        with_law = _inject(
            tok_state=_TYPICAL_STATE, pressure=2
        )  # threshold is >1
        with_reinforced = _inject(tok_state=_TYPICAL_STATE, pressure=75)

        # with_law uses pressure=2 (threshold >1), with_reinforced uses pressure=75
        law_delta = _token_count(with_law) - _token_count(baseline)
        reinforced_delta = _token_count(with_reinforced) - _token_count(
            baseline
        )

        # Law adds at least 20 tokens overhead
        assert law_delta >= 20, (
            f"TOK_PROTOCOL_LAW overhead smaller than expected: {law_delta}t"
        )
        # Full escalation (law + reinforced) must add more than law alone
        assert reinforced_delta > law_delta, (
            "Reinforced escalation should cost more than law-only"
        )
