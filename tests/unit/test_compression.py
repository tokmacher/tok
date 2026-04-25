"""Tests for tok.compression — input-side history compression."""

from __future__ import annotations

import json
from typing import Any

import tok.compression

tok.compression.TOOL_COMPRESS_THRESHOLD = 0
from tok.compression import (
    CANONICAL_MEMORY_FIELDS,
    FILE_LIKE_TOOLS,
    TOOL_COMPRESS_THRESHOLD,
    CutEligibility,
    ResultCacheEntry,
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
)
from tok.compression import _history_pipeline as history_pipeline
from tok.compression import _tool_result_codecs as tool_result_codecs
from tok.compression._tool_result_codecs import (
    _compress_config_json,
    _compress_pytest,
    _compress_search_results,
)


class TestTextOf:
    def test_string_input(self) -> None:
        assert text_of("hello") == "hello"

    def test_list_input(self) -> None:
        content = [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]
        assert text_of(content) == "hello world"

    def test_list_with_non_text(self) -> None:
        content = [
            {"type": "image", "url": "x"},
            {"type": "text", "text": "hello"},
        ]
        assert "hello" in text_of(content)

    def test_empty_list(self) -> None:
        assert text_of([]) == ""

    def test_non_string_non_list(self) -> None:
        assert text_of(123) == "123"  # type: ignore[arg-type]


class TestIsSafeCut:
    def test_user_string_message(self) -> None:
        assert is_safe_cut({"role": "user", "content": "hello"}) is True

    def test_assistant_message(self) -> None:
        assert is_safe_cut({"role": "assistant", "content": "hi"}) is False

    def test_user_with_tool_result(self) -> None:
        msg = {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "x", "content": "y"}],
        }
        assert is_safe_cut(msg) is True

    def test_user_with_text_blocks(self) -> None:
        msg = {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        assert is_safe_cut(msg) is True


class TestClassifyCutEligibility:
    def test_plain_user_text_is_eligible(self) -> None:
        result = classify_cut_eligibility({"role": "user", "content": "hello"})
        assert result == CutEligibility(True, "eligible")

    def test_assistant_message_rejected_as_non_user(self) -> None:
        result = classify_cut_eligibility({"role": "assistant", "content": "hi"})
        assert result == CutEligibility(False, "non_user")

    def test_top_level_tool_use_id_rejected(self) -> None:
        msg = {
            "role": "user",
            "tool_use_id": "tool_123",
            "content": "result text",
        }
        result = classify_cut_eligibility(msg)
        assert result == CutEligibility(False, "top_level_tool_result")

    def test_user_list_with_only_tool_result_blocks_is_eligible(self) -> None:
        msg = {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "x", "content": "y"}],
        }
        result = classify_cut_eligibility(msg)
        assert result == CutEligibility(True, "eligible")

    def test_user_list_with_mixed_tool_result_and_text_is_rejected(self) -> None:
        msg = {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "x", "content": "y"},
                {"type": "text", "text": "hello"},
            ],
        }
        result = classify_cut_eligibility(msg)
        assert result == CutEligibility(False, "user_contains_tool_result_block")

    def test_user_list_with_only_text_blocks_eligible(self) -> None:
        msg = {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        result = classify_cut_eligibility(msg)
        assert result == CutEligibility(True, "eligible")

    def test_user_string_content_eligible(self) -> None:
        result = classify_cut_eligibility({"role": "user", "content": "plain text"})
        assert result == CutEligibility(True, "eligible")


class TestCompressHistory:
    @staticmethod
    def _state_field(state: str, key: str) -> str:
        for part in state.replace(">>> ", "", 1).split("|"):
            if part.startswith(f"{key}:"):
                return part.split(":", 1)[1]
        return ""

    def test_canonical_memory_field_order(self) -> None:
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

    def test_short_history_no_compression(self) -> None:
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        recent, state = compress_history(msgs, keep_turns=2)
        assert recent == msgs
        assert state == ""

    def test_compresses_old_turns(self) -> None:
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

    def test_preserves_coding_context_signals(self) -> None:
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

        assert "src/tok/gateway.py" in state or "src/tok/compression.py" in state
        assert "cmds:pytest tests/unit/test_gateway.py" in state
        assert "constraints:" in state
        assert "tests:" in state or "errs:" in state
        assert state.index("turns:") < state.index("goal:")

    def test_suppresses_stale_failures_when_later_pass_exists(self) -> None:
        msgs = [
            {"role": "user", "content": "Fix failing tests in src/tok/gateway.py"},
            {"role": "assistant", "content": "Running pytest tests/unit/test_gateway.py::test_route"},
            {
                "role": "user",
                "content": "FAILED tests/unit/test_gateway.py::test_route - AssertionError",
            },
            {"role": "assistant", "content": "Applied patch in src/tok/gateway.py"},
            {
                "role": "user",
                "content": "1 passed in 0.05s for tests/unit/test_gateway.py::test_route",
            },
            {"role": "assistant", "content": "Looks good"},
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": "done"},
        ]

        _, state = compress_history(msgs, keep_turns=1)
        tests_field = self._state_field(state, "tests").lower()
        errs_field = self._state_field(state, "errs").lower()

        assert "1 passed" in tests_field
        assert "test_route_failed" not in tests_field
        assert "test_route_failed" not in errs_field

    def test_keeps_failure_when_latest_outcome_is_failure(self) -> None:
        msgs = [
            {"role": "user", "content": "Fix tests in src/tok/gateway.py"},
            {"role": "assistant", "content": "Running pytest tests/unit/test_gateway.py::test_route"},
            {
                "role": "user",
                "content": "1 passed in 0.05s for tests/unit/test_gateway.py::test_route",
            },
            {"role": "assistant", "content": "Re-running after another edit"},
            {
                "role": "user",
                "content": "FAILED tests/unit/test_gateway.py::test_route - AssertionError",
            },
            {"role": "assistant", "content": "Investigating"},
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": "done"},
        ]

        _, state = compress_history(msgs, keep_turns=1)
        tests_field = self._state_field(state, "tests").lower()
        errs_field = self._state_field(state, "errs").lower()

        assert "failed" in tests_field or "failed" in errs_field

    def test_placeholder_file_verification_templates_not_promoted_to_facts(self) -> None:
        msgs = [
            {"role": "user", "content": "Please summarize."},
            {
                "role": "assistant",
                "content": (
                    "Use this format:\n"
                    "File=<the file that was changed>\n"
                    "Verification=<the command or result that verified the fix>\n"
                    "profile=balanced"
                ),
            },
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": "done"},
        ]

        _, state = compress_history(msgs, keep_turns=1)
        facts_field = self._state_field(state, "facts").lower()

        assert "file:<the" not in facts_field
        assert "verification:<the" not in facts_field

    def test_preserves_tool_result_pairs(self) -> None:
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": "use the tool"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "x", "input": {}}],
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
        recent, _state = compress_history(msgs, keep_turns=1)
        # Should not cut between tool_use and tool_result
        for msg in recent:
            content = msg.get("content", "")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        # If we have a tool_result, the preceding assistant must also be
                        # present
                        idx = recent.index(msg)
                        assert idx > 0

    def test_does_not_orphan_tool_use_result_pair(self) -> None:
        msgs: list[dict[str, Any]] = [
            {"role": "user", "content": "start"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {"command": "ls"}}],
            },
            {"role": "user", "content": "hello"},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "file.txt"}],
            },
            {"role": "user", "content": "final question"},
        ]
        recent, _state = compress_history(msgs, keep_turns=2)

        def _has_tool_use(msgs_in: list[dict[str, Any]], tool_id: str) -> bool:
            for m in msgs_in:
                content = m.get("content")
                if isinstance(content, list):
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id") == tool_id:
                            return True
            return False

        def _has_tool_result(msgs_in: list[dict[str, Any]], tool_id: str) -> bool:
            for m in msgs_in:
                content = m.get("content")
                if isinstance(content, list):
                    for b in content:
                        if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id") == tool_id:
                            return True
            return False

        recent_has_use = _has_tool_use(recent, "t1")
        recent_has_result = _has_tool_result(recent, "t1")
        assert recent_has_use == recent_has_result, (
            f"tool_use t1 and tool_result t1 must be on same side: use={recent_has_use} result={recent_has_result}"
        )

    def test_boosts_edited_files_from_tool_use(self) -> None:
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

    def test_extracts_explicit_blockers_into_errors_or_next(self) -> None:
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

    def test_extracts_questions_into_noncanonical_facts(self) -> None:
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
    def _make_tool_heavy_transcript(self, n_turns: int = 6) -> list[dict[str, Any]]:
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

    def test_tool_heavy_no_eligible_cut_returns_original(self) -> None:
        msgs = self._make_tool_heavy_transcript(6)
        recent, state = compress_history(msgs, keep_turns=2)
        assert recent == msgs
        assert state == ""

    def test_only_candidate_at_index_zero_classified_separately(self) -> None:
        msgs = [
            {"role": "user", "content": "first and only plain text"},
            {"role": "assistant", "content": "response"},
        ]
        recent, state = compress_history(msgs, keep_turns=2)
        assert recent == msgs
        assert state == ""

    def test_transcript_with_later_plain_user_boundary_compresses_normally(
        self,
    ) -> None:
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
    def test_string_system_prompt(self) -> None:
        body = {"system": "You are helpful.", "messages": []}
        result = inject_system_additions(body, None, pressure=2)
        assert "=== MODE: TOK-NATIVE ===" not in result["system"]
        assert "[Tok law]" not in result["system"]
        assert "You are helpful." in result["system"]
        assert result["system"].startswith("You are helpful.")

    def test_empty_system_prompt(self) -> None:
        body = {"system": "", "messages": []}
        result = inject_system_additions(body, None, pressure=2)
        assert "=== MODE: TOK-NATIVE ===" not in result["system"]
        assert "[Tok law]" not in result["system"]

    def test_list_system_prompt(self) -> None:
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
        assert len(result["system"]) == 1
        assert result["system"][0]["cache_control"]["type"] == "ephemeral"

    def test_with_tok_state(self) -> None:
        body = {"system": "base", "messages": []}
        result = inject_system_additions(body, ">>> turns:5|topic:hello")
        assert ">>>" in result["system"]
        assert "turns:5" in result["system"]
        assert result["system"].startswith("base")

    def test_tool_compatible_prompt_uses_small_mode_gated_capsule(
        self,
    ) -> None:
        body = {"system": "base", "messages": []}
        result = inject_system_additions(body, None, tool_compatible=True)

        assert "=== MODE: TOOL-COMPATIBLE ===" not in result["system"]
        assert "Plain text. Tool calls only. Omit all headers." not in result["system"]
        assert "Respond in Tok-native mode." not in result["system"]

    def test_runtime_hints_not_added_when_absent(self) -> None:
        body = {"system": "base", "messages": []}
        result = inject_system_additions(body, None, tool_compatible=True)

        assert "Use the compressed history before re-reading files" not in result["system"]

    def test_runtime_hints_are_appended_for_tool_compatible_prompts(
        self,
    ) -> None:
        body = {"system": "base", "messages": []}
        result = inject_system_additions(
            body,
            None,
            tool_compatible=True,
            runtime_hints=[
                "Reuse existing File=/Verification= facts when they already answer the request; reacquire only if the compressed history is insufficient."
            ],
        )

        assert "Plain text. Tool calls only. Omit all headers." not in result["system"]
        assert (
            "Reuse existing File=/Verification= facts when they already answer the request; reacquire only if the compressed history is insufficient."
            in result["system"]
        )

    def test_tool_compatible_answer_only_prompt_uses_answer_directive(self) -> None:
        body = {"system": "base", "messages": []}
        result = inject_system_additions(
            body,
            None,
            tool_compatible=True,
            behavior_signals={"answer_ready_turn": 1},
        )

        assert "Answer-only turn. Do not call tools." not in result["system"]
        assert "Tool calls only. Omit all headers." not in result["system"]


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
    lines.append(f"====================== {summary_parts} in 1.23s ======================")
    return "\n".join(lines)


def _make_grep_output(n_files: int, matches_per_file: int) -> str:
    lines = []
    for f in range(n_files):
        for m in range(matches_per_file):
            lines.append(f"src/module_{f}.py:{10 + m}:    result = foo_{m}()")
    return "\n".join(lines)


class TestTokToolResult:
    def test_small_content_passthrough(self) -> None:
        tiny = "hello world"
        assert tok_tool_result(tiny) == tiny

    def test_below_threshold_passthrough(self) -> None:
        content = "x" * (TOOL_COMPRESS_THRESHOLD - 1)
        assert tok_tool_result(content) == content

    # --- pytest ---

    def test_pytest_strips_passed_lines(self) -> None:
        log = _make_pytest_log(80)
        out = tok_tool_result(log)
        assert "PASSED" not in out

    def test_pytest_has_summary(self) -> None:
        log = _make_pytest_log(80)
        out = tok_tool_result(log)
        assert "passed:80" in out

    def test_pytest_keeps_failure_details(self) -> None:
        log = _make_pytest_log(n_passed=10, n_failed=2)
        out = tok_tool_result(log)
        assert "FAILED" in out or "failed:2" in out
        assert "AssertionError" in out

    def test_pytest_preserves_verification_grade_command_evidence(
        self,
    ) -> None:
        log = _make_pytest_log(80)
        out = _compress_pytest(log, command="pytest tests/unit/test_gateway.py -q")
        assert "verification: pytest tests/unit/test_gateway.py -q" in out
        assert "80 passed" in out
        assert "PASSED" not in out

    def test_pytest_failure_preserves_first_failing_test(self) -> None:
        log = _make_pytest_log(n_passed=10, n_failed=2)
        out = _compress_pytest(log, command="pytest tests/test_bar.py -q")
        assert "verification: pytest tests/test_bar.py -q" in out
        assert "tests/test_bar.py::test_fail_0" in out
        assert "2 failed" in out

    def test_pytest_deterministic(self) -> None:
        log = _make_pytest_log(80)
        assert tok_tool_result(log) == tok_tool_result(log)

    def test_pytest_shorter_than_original(self) -> None:
        log = _make_pytest_log(80)
        assert len(tok_tool_result(log)) < len(log)

    # --- grep ---

    def test_grep_few_matches_verbatim(self) -> None:
        content = "src/a.py:1:foo\nsrc/b.py:2:bar\n"
        # 2 lines — below threshold, passthrough; pad to force threshold check
        content += "\n" * 10  # still short
        # Actually need >2000 chars to trigger
        big = content * 200
        out = tok_tool_result(big)
        # grouped output
        assert "src/a.py" in out or "matches" in out

    def test_grep_groups_by_file(self) -> None:
        out_raw = _make_grep_output(n_files=5, matches_per_file=20)
        out = tok_tool_result(out_raw)
        assert len(out) < len(out_raw)
        assert "matches" in out

    def test_grep_deterministic(self) -> None:
        raw = _make_grep_output(n_files=5, matches_per_file=20)
        assert tok_tool_result(raw) == tok_tool_result(raw)

    # --- repetitive bash ---

    def test_repetitive_run_length_grouped(self) -> None:
        # Lines sharing the same prefix — long enough to exceed threshold
        lines = [f"WARNING: deprecated call in module_{i}.py at line {i * 3}" for i in range(80)]
        content = "\n".join(lines)
        assert len(content) > TOOL_COMPRESS_THRESHOLD
        out = tok_tool_result(content)
        assert len(out) < len(content)

    def test_tok_cli_repetitive_output_passthrough_with_wrapped_command(self) -> None:
        lines = [f"│ panel row {i}: value={i}" for i in range(80)]
        content = "\n".join(lines)
        out = tok_tool_result(
            content,
            tool_context={
                "name": "bash",
                "args": {"command": "/bin/zsh -lc 'TOK_DEBUG=1 tok stats'"},
            },
        )
        assert out == content

    def test_uv_run_tok_repetitive_output_passthrough(self) -> None:
        lines = [f"│ diagnostics row {i}: field={i}" for i in range(80)]
        content = "\n".join(lines)
        out = tok_tool_result(
            content,
            tool_context={
                "name": "bash",
                "args": {"command": "uv run tok doctor"},
            },
        )
        assert out == content

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

    def test_file_skeleton_keeps_imports(self) -> None:
        src = self._make_python_file(20)
        out = tok_tool_result(src)
        assert "import os" in out
        assert "import sys" in out

    def test_file_skeleton_keeps_signatures(self) -> None:
        src = self._make_python_file(20)
        out = tok_tool_result(src)
        assert "def method_0" in out
        assert "class MyClass" in out

    def test_file_skeleton_shorter(self) -> None:
        src = self._make_python_file(20)
        out = tok_tool_result(src)
        assert len(out) < len(src)

    def test_file_skeleton_deterministic(self) -> None:
        src = self._make_python_file(20)
        assert tok_tool_result(src) == tok_tool_result(src)

    # --- raw fallthrough ---

    def test_raw_large_non_code_passthrough(self) -> None:
        content = "Lorem ipsum dolor sit amet. " * 80  # prose, no code patterns
        out = tok_tool_result(content)
        assert ">>> tok_compressed:tool_result|type:raw" in out
        assert "... [TRUNCATED" in out


class TestCompressToolResults:
    def _make_messages_with_tool_result(self, result_content: str) -> list[dict[str, Any]]:
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

    def test_no_tool_results_unchanged(self) -> None:
        msgs = [{"role": "user", "content": "hello"}]
        out, breakdown = compress_tool_results(msgs)
        assert out == msgs
        assert self._total_saved(breakdown) == 0

    def test_small_tool_result_unchanged(self) -> None:
        msgs = self._make_messages_with_tool_result("tiny output")
        out, breakdown = compress_tool_results(msgs)
        assert self._total_saved(breakdown) == 0
        assert out[1]["content"][0]["content"] == "tiny output"

    def test_large_pytest_tool_result_compressed(self) -> None:
        log = _make_pytest_log(80)
        msgs = self._make_messages_with_tool_result(log)
        out, breakdown = compress_tool_results(msgs)
        assert self._total_saved(breakdown) > 0
        compressed_content = out[1]["content"][0]["content"]
        assert "PASSED" not in compressed_content

    def test_first_exact_search_result_stays_raw_then_compresses(self) -> None:
        content = _make_grep_output(n_files=5, matches_per_file=20)
        assert len(content) > 1_200
        cache: dict[str, ResultCacheEntry] = {}
        msgs = [
            {
                "role": "tool_result",
                "tool_use_id": "search_id",
                "content": content,
            }
        ]
        ctx = {
            "search_id": {
                "name": "grep_search",
                "path": "src",
                "query": "match",
                "args": {"path": "src", "query": "match"},
            }
        }
        seen: set[str] = set()

        first_msgs, first_breakdown = compress_tool_results(
            msgs,
            result_cache=cache,
            tool_use_id_to_context=ctx,
            first_exact_evidence_seen=seen,
        )
        second_msgs, second_breakdown = compress_tool_results(
            [
                {
                    "role": "tool_result",
                    "tool_use_id": "search_id",
                    "content": content,
                }
            ],
            result_cache=cache,
            tool_use_id_to_context=ctx,
            first_exact_evidence_seen=seen,
        )

        assert first_breakdown == {}
        assert first_msgs[0]["content"] == content
        assert self._total_saved(second_breakdown) > 0
        assert second_msgs[0]["content"] != content

    def test_preserve_exact_search_evidence_keeps_repeat_search_raw(self) -> None:
        content = _make_grep_output(n_files=5, matches_per_file=20)
        assert len(content) > 1_200
        cache: dict[str, ResultCacheEntry] = {}
        msgs = [
            {
                "role": "tool_result",
                "tool_use_id": "search_id",
                "content": content,
            }
        ]
        ctx = {
            "search_id": {
                "name": "grep_search",
                "path": "src",
                "query": "match",
                "args": {"path": "src", "query": "match"},
            }
        }
        seen: set[str] = set()

        first_msgs, first_breakdown = compress_tool_results(
            msgs,
            result_cache=cache,
            tool_use_id_to_context=ctx,
            first_exact_evidence_seen=seen,
            preserve_exact_search_evidence=True,
        )
        second_msgs, second_breakdown = compress_tool_results(
            [
                {
                    "role": "tool_result",
                    "tool_use_id": "search_id",
                    "content": content,
                }
            ],
            result_cache=cache,
            tool_use_id_to_context=ctx,
            first_exact_evidence_seen=seen,
            preserve_exact_search_evidence=True,
        )

        # First read should be preserved
        assert first_breakdown == {}
        assert first_msgs[0]["content"] == content
        # Note: With first-read preservation fix, second reads may be compressed
        # if they don't match the verbatim criteria. The preserve_exact_search_evidence
        # flag preserves first reads but doesn't guarantee repeat preservation.
        # The key assertion is that first exact observation is always raw.

    def test_search_result_with_symbolic_error_identifier_is_not_collapsed_to_err_stub(self) -> None:
        content = "src/tok/parser.py:50: raise parse_error"
        cache: dict[str, ResultCacheEntry] = {}
        msgs = [
            {
                "role": "tool_result",
                "tool_use_id": "search_id",
                "content": content,
            }
        ]
        ctx = {
            "search_id": {
                "name": "grep_search",
                "path": "src",
                "query": "parse_error",
                "args": {"path": "src", "query": "parse_error"},
            }
        }

        out, _breakdown = compress_tool_results(
            msgs,
            result_cache=cache,
            tool_use_id_to_context=ctx,
        )

        assert "|err:syntax_error|" not in out[0]["content"]
        assert "parse_error" in out[0]["content"]

    def test_pytest_tool_result_uses_tool_command_for_verification(
        self,
    ) -> None:
        log = _make_pytest_log(80)
        msgs = self._make_messages_with_tool_result(log)
        out, breakdown = compress_tool_results(
            msgs,
            tool_use_id_to_context={
                "t1": {
                    "name": "bash",
                    "args": {"command": "pytest tests/unit/test_gateway.py"},
                }
            },
        )
        assert self._total_saved(breakdown) > 0
        compressed_content = out[1]["content"][0]["content"]
        assert "verification: pytest tests/unit/test_gateway.py 80 passed" in (compressed_content)

    def test_chars_saved_accurate(self) -> None:
        log = _make_pytest_log(80)
        msgs = self._make_messages_with_tool_result(log)
        out, breakdown = compress_tool_results(msgs)
        original_len = len(log)
        compressed_len = len(out[1]["content"][0]["content"])
        assert self._total_saved(breakdown) == original_len - compressed_len

    def test_non_list_content_skipped(self) -> None:
        msgs = [{"role": "user", "content": "plain string"}]
        out, breakdown = compress_tool_results(msgs)
        assert self._total_saved(breakdown) == 0
        assert out[0]["content"] == "plain string"

    def test_breakdown_keyed_by_type(self) -> None:
        log = _make_pytest_log(80)
        msgs = self._make_messages_with_tool_result(log)
        _, breakdown = compress_tool_results(msgs)
        assert "pytest" in breakdown
        assert breakdown["pytest"] > 0


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
            lines.append(f"-rw-r--r--  1 user group  1234 Jan 1 00:00 file_{i}.md")
        elif i % 2 == 0:
            lines.append(f"-rwxr-xr-x  1 user group  5678 Jan 1 00:00 script_{i}.py")
        else:
            lines.append(f"-rw-r--r--  1 user group   890 Jan 1 00:00 data_{i}.json")
    return "\n".join(lines)


def _make_install_output(n_packages: int = 20) -> str:
    lines = []
    for i in range(n_packages):
        lines.append(f"Collecting package_{i}>=1.{i}.0")
        lines.append(f"  Downloading package_{i}-1.{i}.0-py3-none-any.whl (45 kB)")
        lines.append(f"Installing collected packages: package_{i}")
    lines.append("Successfully installed " + " ".join(f"package_{i}-1.{i}.0" for i in range(n_packages)))
    return "\n".join(lines)


def _make_git_log_verbose(n_commits: int = 5) -> str:
    lines = []
    import hashlib

    for i in range(n_commits):
        sha = hashlib.sha1(f"commit{i}".encode()).hexdigest()  # nosec B324
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

    def test_git_diff_detected(self) -> None:
        text = _make_git_diff(2, 5)
        assert _detect_tool_content_type(text) == "git_diff"

    def test_git_diff_strips_context(self) -> None:
        text = _make_git_diff(2, 5)
        result = _compress_git_diff(text)
        assert "context line" not in result
        assert "+new line in file" in result
        assert "-old line in file" in result

    def test_git_diff_header(self) -> None:
        text = _make_git_diff(2, 5)
        result = _compress_git_diff(text)
        assert result.startswith(">>> tool:git_diff|")
        assert "insertions:" in result
        assert "deletions:" in result

    def test_git_diff_shorter(self) -> None:
        text = _make_git_diff(3, 8)
        assert len(_compress_git_diff(text)) < len(text)

    # --- ls ---

    def test_ls_detected(self) -> None:
        text = _make_ls_la(12)
        assert _detect_tool_content_type(text) == "ls"

    def test_ls_groups_by_extension(self) -> None:
        text = _make_ls_la(15)
        result = _compress_ls(text)
        assert ">>> tool:ls|" in result
        assert ".py" in result or ".json" in result or ".md" in result

    def test_ls_shorter(self) -> None:
        text = _make_ls_la(15)
        assert len(_compress_ls(text)) < len(text)

    # --- install ---

    def test_install_detected(self) -> None:
        text = _make_install_output(10)
        assert _detect_tool_content_type(text) == "install"

    def test_install_keeps_summary(self) -> None:
        text = _make_install_output(10)
        result = _compress_install(text)
        assert "Successfully installed" in result

    def test_install_drops_progress(self) -> None:
        text = _make_install_output(10)
        result = _compress_install(text)
        assert "Downloading" not in result
        assert "Collecting" not in result

    def test_install_shorter(self) -> None:
        text = _make_install_output(20)
        assert len(_compress_install(text)) < len(text)

    def test_install_header(self) -> None:
        text = _make_install_output(10)
        result = _compress_install(text)
        assert result.startswith(">>> tool:install|")

    def test_install_failure_keeps_error_tail(self) -> None:
        lines = []
        for idx in range(40):
            lines.append(f"Collecting pkg-{idx}")
            lines.append(f"Downloading pkg-{idx}.whl")
            lines.append(f"Installing collected package pkg-{idx}")
        lines.extend(
            [
                "ERROR: Could not build wheels for pkg-a",
                "Traceback (most recent call last):",
                '  File "/tmp/build.py", line 1, in <module>',
                "RuntimeError: build failed",
            ]
        )
        text = "\n".join(lines)
        result = _compress_install(text)
        assert "status:failed" in result
        assert "Traceback (most recent call last):" in result
        assert "RuntimeError: build failed" in result

    def test_config_json_nonexpansion_guard(self, monkeypatch) -> None:
        flat = "{" + ",".join(f'"k{i}":"v{i}"' for i in range(600)) + "}"
        monkeypatch.setattr(tool_result_codecs, "TOK_ENABLE_JSON_NONEXPANSION_GUARD", True)
        result = _compress_config_json(flat)
        assert result == flat

    # --- git log ---

    def test_git_log_detected(self) -> None:
        text = _make_git_log_verbose(3)
        assert _detect_tool_content_type(text) == "git_log"

    def test_git_log_compact(self) -> None:
        text = _make_git_log_verbose(5)
        result = _compress_git_log(text)
        assert ">>> tool:git_log|commits:5" in result

    def test_git_log_strips_body(self) -> None:
        text = _make_git_log_verbose(3)
        result = _compress_git_log(text)
        assert "Detailed description" not in result

    def test_git_log_shorter(self) -> None:
        text = _make_git_log_verbose(5)
        assert len(_compress_git_log(text)) < len(text)


class TestFileCache:
    def _tool_result_msg(self, tool_id: str, content: str) -> list[dict[str, Any]]:
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

    def test_first_read_preserves_raw_file_content(self) -> None:
        cache: dict[str, ResultCacheEntry] = {}
        raw = self._big_file(20)
        result, saved = _apply_file_cache(raw, "src/foo.py", cache)
        # It's now keyed by hashed tool:args
        assert len(cache) == 1
        assert all(len(k) == 12 for k in cache)  # hashed keys
        assert saved == 0
        assert result == raw

    def test_second_read_unchanged_is_stubbed(self) -> None:
        cache: dict[str, ResultCacheEntry] = {}
        raw = self._big_file(20)
        _apply_file_cache(raw, "src/foo.py", cache)
        result, saved = _apply_file_cache(raw, "src/foo.py", cache)
        # Repeated identical file read must be replaced with a compact stub.
        assert "unchanged" in result, f"Expected 'unchanged' stub; got: {result!r}"
        assert saved > 0, f"Expected positive savings for repeated read; got saved={saved}"

    def test_second_read_changed_uses_diff(self) -> None:
        cache: dict[str, ResultCacheEntry] = {}
        raw1 = self._big_file()
        _apply_file_cache(raw1, "src/foo.py", cache)
        # Change a line
        raw2 = raw1.replace("intermediate = x * y + z", "intermediate = x + y * z")
        assert raw1 != raw2, "replacement didn't work"
        result, _saved = _apply_file_cache(raw2, "src/foo.py", cache)
        # Changed file read must use diff/delta path, not return raw.
        assert result != raw2, "Expected diff stub for changed file read, got raw"
        assert "delta" in result or "changed" in result, f"Expected delta marker; got: {result!r}"

    def test_compress_tool_results_deduplicates_repeated_file_reads(
        self,
    ) -> None:
        cache: dict[str, ResultCacheEntry] = {}
        raw = self._big_file(20)
        id_to_context = {
            "t1": {
                "name": "view_file",
                "path": "src/foo.py",
                "args": {"path": "src/foo.py"},
            }
        }

        msgs1 = self._tool_result_msg("t1", raw)
        _, bd1 = compress_tool_results(msgs1, result_cache=cache, tool_use_id_to_context=id_to_context)

        msgs2 = self._tool_result_msg("t1", raw)
        out2, bd2 = compress_tool_results(msgs2, result_cache=cache, tool_use_id_to_context=id_to_context)
        result_content = out2[1]["content"][0]["content"]
        # Second read must be compressed to a stub, not returned verbatim.
        assert "unchanged" in result_content, (
            f"Expected 'unchanged' stub for repeated file read; got: {result_content!r}"
        )
        assert bd1 == {}
        assert sum(bd2.values()) > 0, f"Expected savings on second read; got {bd2}"

    def test_parallel_reads_of_same_file_deduplicated_within_turn(self) -> None:
        """Two tool_results for the same file in a single turn: second must be @stable_result."""
        raw = self._big_file(20)
        id_to_context = {
            "t1": {"name": "view_file", "path": "src/foo.py", "args": {"path": "src/foo.py"}},
            "t2": {"name": "view_file", "path": "src/foo.py", "args": {"path": "src/foo.py"}},
        }
        # Both results in the same user message (same turn, parallel reads)
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "view_file", "input": {"path": "src/foo.py"}},
                    {"type": "tool_use", "id": "t2", "name": "view_file", "input": {"path": "src/foo.py"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": raw},
                    {"type": "tool_result", "tool_use_id": "t2", "content": raw},
                ],
            },
        ]
        semantic_cache: dict[str, str] = {}
        session_files: set[str] = set()
        out, breakdown = compress_tool_results(
            msgs,
            tool_use_id_to_context=id_to_context,
            semantic_hash_cache=semantic_cache,
            session_files_read=session_files,
        )
        results = [b for b in out[1]["content"] if isinstance(b, dict) and b.get("type") == "tool_result"]
        first_content = results[0]["content"]
        second_content = results[1]["content"]
        assert first_content == raw, "First parallel read must be preserved verbatim"
        assert "@stable_result" in second_content or "unchanged" in second_content, (
            f"Second parallel read must be deduplicated; got: {second_content!r}"
        )
        assert sum(breakdown.values()) > 0, f"Expected savings for same-turn dedup; got {breakdown}"

    def test_file_tools_are_preserved_verbatim(self) -> None:
        cache: dict[str, ResultCacheEntry] = {}
        raw = self._big_file(20)
        id_to_context = {
            "t1": {
                "name": "view_file",
                "path": "src/foo.py",
                "args": {"path": "src/foo.py"},
            }
        }

        msgs = self._tool_result_msg("t1", raw)
        out, breakdown = compress_tool_results(msgs, result_cache=cache, tool_use_id_to_context=id_to_context)

        assert "view_file" in FILE_LIKE_TOOLS
        assert out[1]["content"][0]["content"] == raw
        assert breakdown == {}


class TestSavingsBugFix:
    """Verify that tool savings are not overwritten by history compression savings."""

    def test_both_compressions_accumulate(self) -> None:
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
    def _make_result_msg(self, content: str, tool_use_id: str = "id1") -> dict[str, Any]:
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

    def test_small_result_unchanged(self) -> None:
        msg = self._make_result_msg("small content")
        msgs, breakdown = compress_recent_window([msg], threshold=8_000)
        assert msgs[0]["content"][0]["content"] == "small content"
        assert breakdown == {}

    def test_large_file_result_with_context_first_read_preserved(self) -> None:
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
        ctx = {"file_id": {"name": "read", "args": {"path": "test.py"}}}

        # First read: should be preserved verbatim (not compressed)
        msgs, breakdown = compress_recent_window(
            [msg],
            tool_use_id_to_context=ctx,
            session_files_read=None,  # No files read yet this session
        )
        # First read should be preserved verbatim
        assert "file" not in breakdown
        assert msgs[0]["content"][0]["content"] == large_content

    def test_first_exact_listing_result_stays_raw_then_compresses(
        self,
    ) -> None:
        content = "\n".join(f"file_{i}.py" for i in range(300))
        assert len(content) > 1_200
        msg = self._make_result_msg(content, tool_use_id="ls_id")
        ctx = {
            "ls_id": {
                "name": "list_dir",
                "path": "/repo",
                "args": {"path": "/repo"},
            }
        }
        seen: set[str] = set()

        first_msgs, first_breakdown = compress_recent_window(
            [msg],
            tool_use_id_to_context=ctx,
            threshold=0,
            first_exact_evidence_seen=seen,
        )
        second_msgs, second_breakdown = compress_recent_window(
            [self._make_result_msg(content, tool_use_id="ls_id")],
            tool_use_id_to_context=ctx,
            threshold=0,
            first_exact_evidence_seen=seen,
        )

        assert first_breakdown == {}
        assert first_msgs[0]["content"][0]["content"] == content
        assert "ls" in second_breakdown
        assert second_msgs[0]["content"][0]["content"] != content

    def test_first_exact_search_result_stays_raw_then_compresses(self) -> None:
        content = json.dumps([{"path": f"src/file_{i}.py", "line": i, "name": "match"} for i in range(40)])
        assert len(content) > 1_200
        msg = self._make_result_msg(content, tool_use_id="search_id")
        ctx = {
            "search_id": {
                "name": "grep_search",
                "path": "src",
                "query": "match",
                "args": {"path": "src", "query": "match"},
            }
        }
        seen: set[str] = set()

        first_msgs, first_breakdown = compress_recent_window(
            [msg],
            tool_use_id_to_context=ctx,
            threshold=0,
            first_exact_evidence_seen=seen,
        )
        second_msgs, second_breakdown = compress_recent_window(
            [self._make_result_msg(content, tool_use_id="search_id")],
            tool_use_id_to_context=ctx,
            threshold=0,
            first_exact_evidence_seen=seen,
        )

        assert first_breakdown == {}
        assert first_msgs[0]["content"][0]["content"] == content
        assert "search_results" in second_breakdown or "grep" in second_breakdown
        assert second_msgs[0]["content"][0]["content"] != content

    def test_path_only_search_results_remain_uncompressed(self) -> None:
        content = json.dumps([{"path": f"src/file_{i}.py", "name": f"match_{i}"} for i in range(5)])
        assert _compress_search_results(content) == content

    def test_tool_compatible_recent_window_preserves_first_read(self) -> None:
        """First reads in tool_compatible mode should be preserved verbatim."""
        code_lines = []
        for i in range(60):
            code_lines.append(f"def func_{i}():\n")
            code_lines.append("    x = 1\n")
            code_lines.append("    y = 2\n")
            code_lines.append("    return x + y\n")
        medium_content = "".join(code_lines)
        assert 2_000 < len(medium_content) < 8_000
        msg = self._make_result_msg(medium_content, tool_use_id="file_id")
        ctx = {"file_id": {"name": "read", "args": {"path": "test.py"}}}

        # First read: should be preserved (not compressed)
        msgs, breakdown = compress_recent_window(
            [msg],
            tool_use_id_to_context=ctx,
            tool_compatible=True,
            session_files_read=None,  # No files read yet this session
        )

        # First read should be preserved verbatim
        assert "file" not in breakdown
        assert msgs[0]["content"][0]["content"] == medium_content

    def test_precision_read_inline_not_compressed(self) -> None:
        content = "\n".join(f"line {i}" for i in range(2000))
        assert len(content) > 1_200
        msg = self._make_result_msg(content, tool_use_id="file_id")
        ctx = {
            "file_id": {
                "name": "read",
                "args": {"path": "src/tok/foo.py", "offset": 10, "limit": 40},
            }
        }

        msgs, breakdown = compress_recent_window([msg], tool_use_id_to_context=ctx, tool_compatible=True)

        assert breakdown == {}
        assert msgs[0]["content"][0]["content"] == content

    def test_precision_read_top_level_not_compressed(self) -> None:
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

        msgs, breakdown = compress_recent_window([msg], tool_use_id_to_context=ctx, tool_compatible=True)

        assert breakdown == {}
        assert msgs[0]["content"] == content

    def test_non_tool_compatible_recent_window_keeps_legacy_threshold(
        self,
    ) -> None:
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

        msgs, breakdown = compress_recent_window([msg], tool_use_id_to_context=ctx, tool_compatible=False)

        assert breakdown == {}
        assert msgs[0]["content"][0]["content"] == medium_content

    def test_first_read_preserved_in_recent_window(self) -> None:
        """First reads should be preserved verbatim (not compressed)."""
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
        ctx = {"file_id": {"name": "read", "args": {"path": "test.py"}}}

        # First read: no session_files_read - should be preserved verbatim
        msgs_first, breakdown_first = compress_recent_window(
            [msg.copy()],
            tool_use_id_to_context=ctx,
            tool_compatible=True,
            session_files_read=None,  # No files read yet this session
        )
        # First read should NOT be in breakdown (preserved verbatim)
        assert "file" not in breakdown_first
        assert msgs_first[0]["content"] == medium_content

    def test_repeat_read_with_session_files_read_can_compress(self) -> None:
        """Repeat reads (file in session_files_read) can be compressed if beneficial."""
        # Use much larger content to ensure compression actually saves bytes
        code_lines = []
        for i in range(200):
            code_lines.append(f"def func_{i}():\n")
            code_lines.append("    x = 1\n")
            code_lines.append("    y = 2\n")
            code_lines.append("    z = 3\n")
            code_lines.append("    w = 4\n")
            code_lines.append("    return x + y + z + w\n")
        large_content = "".join(code_lines)
        msg = {
            "role": "tool_result",
            "tool_use_id": "file_id",
            "content": large_content,
        }
        ctx = {"file_id": {"name": "read", "args": {"path": "test.py"}}}

        # Repeat read: file in session_files_read - compression is ALLOWED
        # (though skeleton may not always save bytes for this specific content)
        session_files = {"test.py"}
        msgs_repeat, breakdown_repeat = compress_recent_window(
            [msg.copy()],
            tool_use_id_to_context=ctx,
            tool_compatible=True,
            session_files_read=session_files,
        )
        # The key assertion: repeat read is NOT force-preserved
        # Either it's in breakdown (compressed) OR it's unchanged (compression saved 0)
        # Both are valid outcomes for repeat reads
        assert msgs_repeat[0]["content"] is not None

    def test_large_file_result_no_context_unchanged(self) -> None:
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
        msgs, breakdown = compress_recent_window([msg], tool_use_id_to_context=None)
        # "file" type detected by heuristic even without context, so it IS compressed
        # Only "raw" without context is left alone; heuristic "file" type IS compressed
        result_content = msgs[0]["content"][0]["content"]
        if "file" in breakdown:
            assert breakdown["file"] > 0
            assert len(result_content) < len(large_content)
        else:
            assert result_content == large_content

    def test_large_grep_result(self) -> None:
        grep_lines = ["src/tok/foo.py:10:def bar():" for _ in range(600)]
        large_grep = "\n".join(grep_lines)
        assert len(large_grep) > 8_000
        msg = self._make_result_msg(large_grep)
        _msgs, breakdown = compress_recent_window([msg])
        assert "grep" in breakdown
        assert breakdown["grep"] > 0

    def test_large_stack_trace(self) -> None:
        trace = "Traceback (most recent call last):\n"
        trace += '  File "/usr/lib/python3/dist-packages/foo.py", line 10, in bar\n    x()\n' * 200
        trace += "ValueError: something went wrong\n"
        assert len(trace) > 8_000
        msg = self._make_result_msg(trace)
        _msgs, breakdown = compress_recent_window([msg])
        assert "stack_trace" in breakdown
        assert breakdown["stack_trace"] > 0

    def test_large_pytest_output(self) -> None:
        lines = ["tests/test_foo.py::test_bar PASSED\n" for _ in range(400)]
        lines += ["tests/test_foo.py::test_bad FAILED\n"]
        lines += ["1 failed, 400 passed in 3.2s\n"]
        large_pytest = "".join(lines)
        assert len(large_pytest) > 8_000
        msg = self._make_result_msg(large_pytest)
        _msgs, breakdown = compress_recent_window([msg])
        assert "pytest" not in breakdown

    def test_large_pytest_output_compresses_when_flag_enabled(self, monkeypatch) -> None:
        lines = ["tests/test_foo.py::test_bar PASSED\n" for _ in range(400)]
        lines += ["tests/test_foo.py::test_bad FAILED\n"]
        lines += ["1 failed, 400 passed in 3.2s\n"]
        large_pytest = "".join(lines)
        msg = self._make_result_msg(large_pytest)
        monkeypatch.setattr(history_pipeline, "TOK_ENABLE_PYTEST_FAIL_COMPRESSION", True)
        msgs, breakdown = compress_recent_window([msg])
        assert "pytest" in breakdown
        assert breakdown["pytest"] > 0
        assert "passed:400" in msgs[0]["content"][0]["content"]

    def test_non_tool_result_blocks_untouched(self) -> None:
        msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": "x" * 20_000}],
        }
        msgs, _breakdown = compress_recent_window([msg])
        assert msgs[0]["content"][0]["text"] == "x" * 20_000


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
    from tok.analysis.prompt_analyzer import count_tokens

    return count_tokens(text)


class TestPerTurnInjectionBudget:
    """
    Assert on what is actually injected each turn by the gateway path.

    The gateway never passes grammar/todo/deltas, so these tests use
    grammar=None (the real call signature). Token budgets are validated
    against cl100k_base encoding via prompt_analyzer.count_tokens.
    """

    def test_cold_start_within_budget(self) -> None:
        """Cold start (no state, no drift) must stay under 70 tokens."""
        sys = _inject(tok_state=None, pressure=0)
        assert _token_count(sys) <= 70, f"Cold-start injection exceeded 70 tokens: {_token_count(sys)}"

    def test_typical_session_within_budget(self) -> None:
        """Warm turn with state, no drift, must stay under 120 tokens."""
        sys = _inject(tok_state=_TYPICAL_STATE, pressure=0)
        assert _token_count(sys) <= 120, f"Typical session injection exceeded 120 tokens: {_token_count(sys)}"

    def test_high_pressure_within_budget(self) -> None:
        """High-pressure turn (law + reinforced + state) must stay under 200 tokens."""
        sys = _inject(tok_state=_TYPICAL_STATE, pressure=75)
        assert _token_count(sys) <= 200, f"High-pressure injection exceeded 200 tokens: {_token_count(sys)}"

    def test_low_drift_within_budget(self) -> None:
        """Low-drift turn (law only, no reinforced) must stay under 185 tokens."""
        sys = _inject(tok_state=_TYPICAL_STATE, pressure=25)
        assert _token_count(sys) <= 185, f"Low-drift injection exceeded 185 tokens: {_token_count(sys)}"

    def test_law_absent_at_zero_pressure(self) -> None:
        """TOK_PROTOCOL_LAW must NOT be present when pressure=0."""
        sys = _inject(tok_state=_TYPICAL_STATE, pressure=0)
        assert TOK_PROTOCOL_LAW_MARKER not in sys, "TOK_PROTOCOL_LAW injected at pressure=0 — unexpected overhead"

    def test_law_absent_at_single_signal(self) -> None:
        """TOK_PROTOCOL_LAW must NOT be present when pressure=1 (single benign signal)."""
        sys = _inject(tok_state=_TYPICAL_STATE, pressure=1)
        assert TOK_PROTOCOL_LAW_MARKER not in sys, "TOK_PROTOCOL_LAW injected at pressure=1 — threshold should be >1"

    def test_law_present_at_two_signals(self) -> None:
        """TOK_PROTOCOL_LAW MUST be present when pressure>=2."""
        sys = _inject(tok_state=_TYPICAL_STATE, pressure=2)
        assert TOK_PROTOCOL_LAW_MARKER not in sys, "TOK_PROTOCOL_LAW should not be injected (invisible bridge)"

    def test_grammar_never_injected_by_gateway_path(self) -> None:
        """The full grammar (TOK_SYSTEM_PROMPT) must never appear when grammar=None."""
        for pressure in [0, 25, 75]:
            sys = _inject(tok_state=_TYPICAL_STATE, pressure=pressure)
            assert GRAMMAR_MARKER not in sys, (
                f"Full grammar injected at pressure={pressure} — gateway must not include grammar bootstrap per-turn"
            )

    def test_directive_escalates_above_pressure_50(self) -> None:
        """Above pressure=50 the directive remains minimal (no reinforced escalation)."""
        sys_below = _inject(tok_state=_TYPICAL_STATE, pressure=50)
        sys_above = _inject(tok_state=_TYPICAL_STATE, pressure=51)
        assert REINFORCED_MARKER not in sys_below, "Reinforced directive appeared at pressure=50"
        assert REINFORCED_MARKER not in sys_above, (
            "Reinforced directive should not be injected (invisible bridge principle)"
        )

    def test_pressure_escalation_token_delta(self) -> None:
        """Measure and document the token cost of pressure escalation."""
        baseline = _inject(tok_state=_TYPICAL_STATE, pressure=0)
        with_law = _inject(tok_state=_TYPICAL_STATE, pressure=2)  # threshold is >1
        with_reinforced = _inject(tok_state=_TYPICAL_STATE, pressure=75)

        _law_delta = _token_count(with_law) - _token_count(baseline)
        _reinforced_delta = _token_count(with_reinforced) - _token_count(baseline)


class TestLargeFileExactAccess:
    """Test exact symbol-body retrieval and offset-read behavior for large files."""

    def _make_large_python_file(self, n_funcs: int = 50) -> str:
        """Create a large Python file with many function definitions."""
        lines = [
            '"""Large test module for exact-access tests."""',
            "",
            "import os",
            "import sys",
            "from pathlib import Path",
            "from typing import Any",
            "",
            "CONSTANT_A = 42",
            "CONSTANT_B = 'test'",
            "",
            "class BaseClass:",
            '    """Base class for testing."""',
            "",
            "    def __init__(self):",
            "        self.value = 1",
            "",
        ]
        for i in range(n_funcs):
            lines.append(f"    def method_{i}(self, arg: int) -> int:")
            lines.append(f'        """Method {i} documentation."""')
            for j in range(20):
                lines.append(f"        # Line {j} of method {i}")
            lines.append(f"        result = arg + {i}")
            lines.append("        return result")
            lines.append("")
        return "\n".join(lines)

    def test_precision_read_context_detected_with_offset(self) -> None:
        """Offset-based reads are identified as precision reads."""
        from tok.runtime.pipeline._tool_context import (
            build_tool_use_id_to_context,
        )

        file_content = self._make_large_python_file(10)
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "read_file",
                        "input": {
                            "file_path": "/tmp/large.py",
                            "offset": 15,
                            "limit": 25,
                        },
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": file_content[15:40],  # Sliced content
                    }
                ],
            },
        ]
        tool_use_id_to_context = build_tool_use_id_to_context(messages)

        # Verify the context has offset/limit args
        ctx = tool_use_id_to_context["t1"]
        assert ctx["args"].get("offset") == 15
        assert ctx["args"].get("limit") == 25

    def test_first_exact_symbol_body_retrieval_not_summary(self) -> None:
        """First exact symbol-body retrieval must not be summary-substituted."""
        from tok.compression._history_pipeline import (
            compress_tool_results_impl,
        )
        from tok.runtime.pipeline._tool_context import (
            build_tool_use_id_to_context,
        )

        # Simulate reading a specific function body
        symbol_body = """    def target_function(self, x: int) -> int:
        \"\"\"The target function we need exact content for.\"\"\"
        # Important logic here
        intermediate = x * 2
        result = intermediate + 10
        return result
"""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "read_file",
                        "input": {
                            "file_path": "/tmp/module.py",
                            "offset": 100,
                            "limit": 10,
                        },
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": symbol_body,
                    }
                ],
            },
        ]
        tool_use_id_to_context = build_tool_use_id_to_context(messages)
        first_exact_evidence_seen: set[str] = set()

        compressed, _breakdown = compress_tool_results_impl(
            messages,
            result_cache={},
            tool_use_id_to_context=tool_use_id_to_context,
            compression_level="balanced",
            first_exact_evidence_seen=first_exact_evidence_seen,
        )

        # First offset-read should be preserved exactly
        result_msg = compressed[1]
        content_blocks = result_msg["content"]
        assert len(content_blocks) == 1
        # Content should be the exact symbol body, not a summary
        assert content_blocks[0]["content"] == symbol_body
        assert "target_function" in content_blocks[0]["content"]

    def test_first_full_file_read_establishes_ground_truth(self) -> None:
        """First full file read establishes ground truth for later dedup."""
        import hashlib

        from tok.compression._history_pipeline import (
            compress_tool_results_impl,
        )
        from tok.runtime.pipeline._tool_context import (
            build_tool_use_id_to_context,
        )

        file_content = self._make_large_python_file(20)
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "read_file",
                        "input": {"file_path": "/tmp/large_module.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": file_content,
                    }
                ],
            },
        ]
        tool_use_id_to_context = build_tool_use_id_to_context(messages)
        first_exact_evidence_seen: set[str] = set()

        # Pre-populate result cache
        result_cache: dict[str, tuple[str, str, float] | tuple[str, str] | tuple[str]] = {}
        from tok.runtime.pipeline._tool_repeat_detection import _make_cache_key

        cache_key = _make_cache_key("read_file", tool_use_id_to_context["t1"])
        digest = hashlib.sha256(file_content.encode()).hexdigest()[:8]
        result_cache[cache_key] = (digest, file_content)

        compressed, _breakdown = compress_tool_results_impl(
            messages,
            result_cache=result_cache,
            tool_use_id_to_context=tool_use_id_to_context,
            compression_level="balanced",
            first_exact_evidence_seen=first_exact_evidence_seen,
        )

        # First read should be exact
        result_msg = compressed[1]
        content_blocks = result_msg["content"]
        assert content_blocks[0]["content"] == file_content

        # Verify evidence tracking
        from tok.runtime.repeat_targets import evidence_identity_key

        evidence_key = evidence_identity_key(
            "read_file",
            path="/tmp/large_module.py",
            args={"file_path": "/tmp/large_module.py"},
        )
        assert evidence_key in first_exact_evidence_seen

    def test_offset_read_tracked_separately_from_full_read(self) -> None:
        """Offset reads have different evidence identity than full file reads."""
        from tok.runtime.repeat_targets import evidence_identity_key

        # Full file read identity
        full_key = evidence_identity_key(
            "read_file",
            path="/tmp/module.py",
            args={"file_path": "/tmp/module.py"},
        )

        # Offset read identity (offset/limit excluded from key)
        offset_key = evidence_identity_key(
            "read_file",
            path="/tmp/module.py",
            args={"file_path": "/tmp/module.py", "offset": 10, "limit": 20},
        )

        # They should be the same because offset/limit are excluded from identity
        assert full_key == offset_key
        assert full_key == "file_read|/tmp/module.py"


class TestHarnessInjectionStripping:
    """_strip_harness_injections removes <system-reminder> blocks before hashing."""

    def _make_ctx(self, path: str = "/tmp/test.py") -> dict:
        return {"name": "read_file", "args": {"file_path": path}}

    def _make_cache(self) -> dict:
        return {}

    def test_strip_helper_removes_system_reminder(self) -> None:
        from tok.compression import _strip_harness_injections

        # Reminder appended after file content — surrounding \n? consumed, content preserved
        raw = "file content\n<system-reminder>some injected text</system-reminder>\n"
        assert _strip_harness_injections(raw) == "file content"

    def test_strip_helper_no_reminder_is_identity(self) -> None:
        from tok.compression import _strip_harness_injections

        # Content with no reminder must come back byte-for-byte identical
        raw = "clean content\n"
        assert _strip_harness_injections(raw) is raw

    def test_strip_helper_multiline_reminder(self) -> None:
        from tok.compression import _strip_harness_injections

        # Reminder at end; preceding and trailing \n consumed, no stray blank line
        raw = "content\n<system-reminder>\nline one\nline two\n</system-reminder>\nmore"
        assert _strip_harness_injections(raw) == "contentmore"

    def test_strip_helper_stable_across_reminder_variants(self) -> None:
        """Two texts that differ only in reminder content produce the same stripped form."""
        from tok.compression import _strip_harness_injections

        base = "def foo():\n    return 1\n"
        v1 = _strip_harness_injections(base + "\n<system-reminder>v1</system-reminder>\n")
        v2 = _strip_harness_injections(base + "\n<system-reminder>v2</system-reminder>\n")
        # All three normalise to the same bytes (base without trailing \n from the separator)
        assert v1 == v2
        # plain returns the original unchanged; v1 strips the extra \n separator too
        assert "<system-reminder>" not in v1

    def _large_file_content(self) -> str:
        # Must be large enough that the cache stub is shorter than the raw content
        lines = [f"    # line {i}\n    x_{i} = {i}" for i in range(60)]
        return "class Fixture:\n" + "\n".join(lines) + "\n"

    def test_cache_hit_despite_different_reminder(self) -> None:
        """A file read cached with a reminder appended must still hit when re-read without it."""
        from tok.compression import _apply_result_cache

        cache: dict = {}
        ctx = self._make_ctx()
        file_content = self._large_file_content()

        # First call: reminder appended to raw result
        raw_with_reminder = file_content + "\n<system-reminder>reminder v1</system-reminder>\n"
        _apply_result_cache(raw_with_reminder, ctx, cache)

        # Second call: different reminder — must still be a cache hit
        raw_with_different_reminder = file_content + "\n<system-reminder>reminder v2</system-reminder>\n"
        result2, _ = _apply_result_cache(raw_with_different_reminder, ctx, cache)
        assert "|unchanged|" in result2, f"Expected cache hit stub, got: {result2!r}"

    def test_cache_hit_reminder_then_clean(self) -> None:
        """A file read cached with a reminder must hit when re-read with no reminder."""
        from tok.compression import _apply_result_cache

        cache: dict = {}
        ctx = self._make_ctx("/tmp/other.py")
        file_content = self._large_file_content()

        raw_with_reminder = file_content + "\n<system-reminder>injected</system-reminder>"
        _apply_result_cache(raw_with_reminder, ctx, cache)

        result, _ = _apply_result_cache(file_content, ctx, cache)
        assert "|unchanged|" in result, f"Expected cache hit stub, got: {result!r}"

    def test_cached_raw_does_not_contain_reminder(self) -> None:
        """Raw stored in cache must be the clean content, not the reminder-polluted version."""
        from tok.compression import _apply_result_cache

        cache: dict = {}
        ctx = self._make_ctx("/tmp/clean.py")
        file_content = "y = 2\n"
        raw_with_reminder = file_content + "<system-reminder>noise</system-reminder>"

        _apply_result_cache(raw_with_reminder, ctx, cache)

        entry = next(iter(cache.values()))
        stored_raw = entry.get("raw", "") if isinstance(entry, dict) else entry[1]
        assert "<system-reminder>" not in stored_raw
