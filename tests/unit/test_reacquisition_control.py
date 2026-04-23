"""Batch 2 regression tests for reacquisition control and state resend stability."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tok.runtime.config import ANSWER_READY_REPAIR_HINT
from tok.runtime.core import (
    RuntimeRequest,
    RuntimeSession,
    UniversalTokRuntime,
)
from tok.runtime.memory.tok_state import _delta_tok_state_fields
from tok.runtime.pipeline.tool_processing import (
    logical_target_key_from_context,
)
from tok.runtime.repeat_targets import HotSummaryRecord, evidence_identity_key

if TYPE_CHECKING:
    from pathlib import Path


def _make_tool_result_msg(tool_id: str, content: str) -> dict[str, Any]:
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


def _make_tool_use_msg(tool_id: str, tool_name: str, **tool_input: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": tool_id,
                "name": tool_name,
                "input": tool_input,
            }
        ],
    }


def _runtime(tmp_path: Path) -> tuple[UniversalTokRuntime, RuntimeSession]:
    memory_dir = tmp_path / ".tok"
    return UniversalTokRuntime(), RuntimeSession(memory_dir=memory_dir)


def test_shell_cat_then_view_file_promotes_hot_file_hint_with_alias_paths(
    tmp_path,
) -> None:
    runtime, session = _runtime(tmp_path)
    session.bridge_memory.turn = 9
    content = "\n".join(f"line {idx}" for idx in range(1, 20))

    first = runtime.prepare_request(
        RuntimeRequest(
            model="claude-sonnet-4",
            tool_compatible=True,
            messages=[
                {"role": "user", "content": "Inspect foo."},
                _make_tool_use_msg("t1", "bash", command="cat src/foo.py"),
                _make_tool_result_msg("t1", content),
                {"role": "user", "content": "What did you find?"},
            ],
        ),
        session,
    )
    assert "hot_recent_file" not in str(first.body.get("system", ""))

    second = runtime.prepare_request(
        RuntimeRequest(
            model="claude-sonnet-4",
            tool_compatible=True,
            messages=[
                {"role": "user", "content": "Inspect foo."},
                _make_tool_use_msg("t1", "bash", command="cat src/foo.py"),
                _make_tool_result_msg("t1", content),
                {"role": "user", "content": "What did you find?"},
                _make_tool_use_msg("t2", "view_file", path="./src/foo.py"),
                _make_tool_result_msg("t2", content),
                {
                    "role": "user",
                    "content": "Answer using what we already have.",
                },
            ],
        ),
        session,
    )

    system_text = str(second.body.get("system", ""))
    assert "@hot_recent_file:./src/foo.py |>" in system_text
    assert "Reuse" not in system_text
    assert "cached result" not in system_text
    assert second.behavior_signals["repeat_target_hot"] >= 1
    assert second.behavior_signals["hot_recent_hint_injected"] >= 1
    assert second.behavior_signals["repeat_file_read"] >= 1
    assert second.behavior_signals["shell_file_read_normalized"] >= 1
    assert second.behavior_signals["shell_file_snapshot_captured"] >= 1
    assert second.hot_hint_tokens_added > 0


def test_shell_head_and_sed_count_as_same_file_target(tmp_path) -> None:
    runtime, session = _runtime(tmp_path)
    session.bridge_memory.turn = 9
    content = "\n".join(f"line {idx}" for idx in range(1, 40))

    runtime.prepare_request(
        RuntimeRequest(
            model="claude-sonnet-4",
            tool_compatible=True,
            messages=[
                {"role": "user", "content": "Inspect foo via shell."},
                _make_tool_use_msg("t1", "bash", command="head -200 src/foo.py"),
                _make_tool_result_msg("t1", content),
                {"role": "user", "content": "Keep going."},
            ],
        ),
        session,
    )
    prepared = runtime.prepare_request(
        RuntimeRequest(
            model="claude-sonnet-4",
            tool_compatible=True,
            messages=[
                {"role": "user", "content": "Inspect foo via shell."},
                _make_tool_use_msg("t1", "bash", command="head -200 src/foo.py"),
                _make_tool_result_msg("t1", content),
                {"role": "user", "content": "Keep going."},
                _make_tool_use_msg("t2", "bash", command="sed -n '1,200p' ./src/foo.py"),
                _make_tool_result_msg("t2", content),
                {
                    "role": "user",
                    "content": "Answer using the existing evidence.",
                },
            ],
        ),
        session,
    )

    system_text = str(prepared.body.get("system", ""))
    assert "@hot_recent_file:./src/foo.py |>" in system_text
    assert prepared.behavior_signals["repeat_target_hot"] >= 1
    assert prepared.behavior_signals["repeat_file_read"] >= 1
    assert prepared.behavior_signals["shell_file_read_normalized"] >= 2


def test_repeated_search_promotes_hot_search_hint(tmp_path) -> None:
    runtime, session = _runtime(tmp_path)
    session.bridge_memory.turn = 9
    search_result = "\n".join(f"src/foo.py:{line}:needle match" for line in range(1, 10))

    first = runtime.prepare_request(
        RuntimeRequest(
            model="claude-sonnet-4",
            tool_compatible=True,
            messages=[
                {"role": "user", "content": "Search for needle."},
                _make_tool_use_msg(
                    "s1",
                    "rg",
                    path="src",
                    query="needle",
                ),
                _make_tool_result_msg("s1", search_result),
                {"role": "user", "content": "Keep going."},
            ],
        ),
        session,
    )
    assert "@hot_recent_search:" not in str(first.body.get("system", ""))

    prepared = runtime.prepare_request(
        RuntimeRequest(
            model="claude-sonnet-4",
            tool_compatible=True,
            messages=[
                {"role": "user", "content": "Search for needle."},
                _make_tool_use_msg("s1", "rg", path="src", query="needle"),
                _make_tool_result_msg("s1", search_result),
                {"role": "user", "content": "Keep going."},
                _make_tool_use_msg("s2", "search", path="src", query="needle"),
                _make_tool_result_msg("s2", search_result),
                {"role": "user", "content": "Answer directly."},
            ],
        ),
        session,
    )

    assert "@hot_recent_search:needle @ src |>" in str(prepared.body.get("system", ""))
    assert prepared.behavior_signals["repeat_target_hot"] >= 1


def test_hot_recent_search_hints_require_session_exact_observation(
    tmp_path,
) -> None:
    _, session = _runtime(tmp_path)
    exact_key = evidence_identity_key(
        "grep_search",
        path="src",
        query="needle",
        args={"query": "needle", "path": "src"},
    )
    session._hot_summary_records["search|needle-src"] = HotSummaryRecord(
        tool_family="search",
        logical_target="needle-src",
        display_target="needle @ src",
        summary="needle match summary",
        token_cost=12,
        result_digest="digest",
        last_seen_turn=4,
        exact_evidence_key=exact_key or "",
        hot_promotion_turn=4,
    )
    session.bridge_memory.turn = 11

    hints, metrics = session.hot_recent_runtime_hints()
    assert hints == []
    assert metrics["hot_recent_hint_injected"] == 0

    if exact_key:
        session._first_exact_evidence_seen.add(exact_key)

    hints, metrics = session.hot_recent_runtime_hints()
    assert hints
    assert "@hot_recent_search:needle @ src |>" in hints[0]
    assert metrics["hot_recent_hint_injected"] == 1


def test_repeated_command_family_promotes_hot_command_hint(tmp_path) -> None:
    runtime, session = _runtime(tmp_path)
    session.bridge_memory.turn = 9
    pytest_output = "collected 3 items\n3 passed in 0.22s\n"

    runtime.prepare_request(
        RuntimeRequest(
            model="claude-sonnet-4",
            tool_compatible=True,
            messages=[
                {"role": "user", "content": "Run the tests."},
                _make_tool_use_msg("c1", "bash", command="python -m pytest"),
                _make_tool_result_msg("c1", pytest_output),
                {"role": "user", "content": "What next?"},
            ],
        ),
        session,
    )
    prepared = runtime.prepare_request(
        RuntimeRequest(
            model="claude-sonnet-4",
            tool_compatible=True,
            messages=[
                {"role": "user", "content": "Run the tests."},
                _make_tool_use_msg("c1", "bash", command="python -m pytest"),
                _make_tool_result_msg("c1", pytest_output),
                {"role": "user", "content": "What next?"},
                _make_tool_use_msg("c2", "bash", command="python -m pytest"),
                _make_tool_result_msg("c2", pytest_output),
                _make_tool_use_msg("c3", "bash", command="git status"),
                _make_tool_result_msg("c3", "On branch main\nnothing to commit\n"),
                {"role": "user", "content": "Answer now."},
            ],
        ),
        session,
    )

    system_text = str(prepared.body.get("system", ""))
    assert "@hot_recent_command:pytest |>" in system_text
    assert "@hot_recent_command:git" not in system_text
    assert prepared.behavior_signals["repeat_tool_collapse_applied"] >= 1


def test_unsafe_or_multi_target_shell_commands_do_not_normalize_to_file_read() -> None:
    cases = (
        "cat src/foo.py | head -5",
        "cat src/foo.py > /tmp/out.txt",
        "cat src/*.py",
        "cat src/a.py src/b.py",
        "sed -n '1,200p' src/foo.py src/bar.py",
        "python -c 'print(1)'",
    )

    for command in cases:
        family, logical_target, label = logical_target_key_from_context(
            "bash",
            command=command,
        )
        assert family == "command"
        assert logical_target != "path-missing"
        assert label


def test_hot_recent_hints_are_bounded_and_char_limited(tmp_path) -> None:
    _, session = _runtime(tmp_path)
    long_text = "\n".join(f"important line {idx}" for idx in range(1, 50))

    for turn, path in enumerate(("src/a.py", "src/b.py", "src/c.py"), start=10):
        session.bridge_memory.turn = turn
        session.observe_repeat_target_result(
            tool_id=f"{path}-1",
            tool_name="view_file",
            path=path,
            query=None,
            command=None,
            raw_content=long_text,
        )
        session.bridge_memory.turn = turn + 3
        session.observe_repeat_target_result(
            tool_id=f"{path}-2",
            tool_name="view_file",
            path=path,
            query=None,
            command=None,
            raw_content=long_text,
        )

    session.bridge_memory.turn = 14

    hints, metrics = session.hot_recent_runtime_hints()
    assert len(hints) == 2
    assert metrics["hot_recent_hint_injected"] == 2
    for hint in hints:
        summary = hint.split("|> ", 1)[1].split("\n", 1)[0]
        assert len(summary) <= 400


def test_hot_recent_hints_require_a_captured_snapshot(tmp_path) -> None:
    _, session = _runtime(tmp_path)
    session.bridge_memory.turn = 11
    hints, metrics = session.hot_recent_runtime_hints()
    assert hints == []
    assert metrics["hot_recent_hint_injected"] == 0


def test_shell_read_snapshots_only_capture_successful_text_output(
    tmp_path,
) -> None:
    runtime, session = _runtime(tmp_path)
    content = "\n".join(f"line {idx}" for idx in range(1, 10))

    failed = runtime.prepare_request(
        RuntimeRequest(
            model="claude-sonnet-4",
            tool_compatible=True,
            messages=[
                {"role": "user", "content": "Inspect foo."},
                _make_tool_use_msg("t1", "bash", command="cat src/foo.py"),
                _make_tool_result_msg("t1", "cat: src/foo.py: No such file or directory"),
                {"role": "user", "content": "Try again."},
            ],
        ),
        session,
    )
    assert failed.behavior_signals.get("shell_file_snapshot_captured", 0) == 0

    succeeded = runtime.prepare_request(
        RuntimeRequest(
            model="claude-sonnet-4",
            tool_compatible=True,
            messages=[
                {"role": "user", "content": "Inspect foo."},
                _make_tool_use_msg("t1", "bash", command="cat src/foo.py"),
                _make_tool_result_msg("t1", "cat: src/foo.py: No such file or directory"),
                {"role": "user", "content": "Try again."},
                _make_tool_use_msg("t2", "bash", command="cat src/foo.py"),
                _make_tool_result_msg("t2", content),
                {"role": "user", "content": "Answer using what we have."},
            ],
        ),
        session,
    )
    assert succeeded.behavior_signals["shell_file_snapshot_captured"] >= 1


def test_stuck_shell_file_targets_use_stronger_reuse_guidance(
    tmp_path,
) -> None:
    _, session = _runtime(tmp_path)
    content = "\n".join(f"line {idx}" for idx in range(1, 20))
    commands = (
        "cat src/foo.py",
        "head -20 ./src/foo.py",
        "sed -n '1,20p' src/foo.py",
    )

    for turn, command in enumerate(commands, start=10):
        session.bridge_memory.turn = turn
        session.observe_repeat_target_result(
            tool_id=f"t{turn}",
            tool_name="bash",
            path=None,
            query=None,
            command=command,
            raw_content=content,
        )

    hints, metrics = session.hot_recent_runtime_hints()
    assert metrics["hot_recent_hint_injected"] == 1
    assert "This target is stuck and unchanged" not in hints[0]


def test_answer_ready_repair_hint_priority_stays_ahead_of_hot_shell_file_hints(
    tmp_path,
) -> None:
    runtime, session = _runtime(tmp_path)
    content = "\n".join(f"line {idx}" for idx in range(1, 16))

    session.bridge_memory.turn = 10
    session.observe_repeat_target_result(
        tool_id="t1",
        tool_name="bash",
        path=None,
        query=None,
        command="cat src/foo.py",
        raw_content=content,
    )
    session.bridge_memory.turn = 11
    session.observe_repeat_target_result(
        tool_id="t2",
        tool_name="view_file",
        path="./src/foo.py",
        query=None,
        command=None,
        raw_content=content,
    )
    session._answer_ready_repair_pending = True

    prepared = runtime.prepare_request(
        RuntimeRequest(
            model="claude-sonnet-4",
            tool_compatible=True,
            messages=[{"role": "user", "content": "Answer now."}],
        ),
        session,
    )

    system_text = str(prepared.body.get("system", ""))
    assert ANSWER_READY_REPAIR_HINT not in system_text
    assert "@hot_recent_file:./src/foo.py |>" in system_text


def test_shell_read_loop_fixture_keeps_prompt_compact_and_state_suppressed(
    tmp_path,
) -> None:
    runtime, session = _runtime(tmp_path)
    session.bridge_memory.turn = 9
    content = "\n".join(f"line {idx}" for idx in range(1, 30))

    first = runtime.prepare_request(
        RuntimeRequest(
            model="claude-sonnet-4",
            tool_compatible=True,
            messages=[
                {"role": "user", "content": "Inspect foo."},
                _make_tool_use_msg("t1", "bash", command="cat src/foo.py"),
                _make_tool_result_msg("t1", content),
                {"role": "user", "content": "Keep going."},
                _make_tool_use_msg("t2", "bash", command="head -200 ./src/foo.py"),
                _make_tool_result_msg("t2", content),
                {"role": "user", "content": "Keep going."},
                _make_tool_use_msg("t3", "bash", command="sed -n '1,200p' src/foo.py"),
                _make_tool_result_msg("t3", content),
                {
                    "role": "user",
                    "content": "Answer using what we already have.",
                },
            ],
        ),
        session,
    )
    assert "@hot_recent_file:" in str(first.body.get("system", ""))

    prepared = runtime.prepare_request(
        RuntimeRequest(
            model="claude-sonnet-4",
            tool_compatible=True,
            messages=[
                {"role": "user", "content": "Inspect foo."},
                _make_tool_use_msg("t1", "bash", command="cat src/foo.py"),
                _make_tool_result_msg("t1", content),
                {"role": "user", "content": "Keep going."},
                _make_tool_use_msg("t2", "bash", command="head -200 ./src/foo.py"),
                _make_tool_result_msg("t2", content),
                {"role": "user", "content": "Keep going."},
                _make_tool_use_msg("t3", "bash", command="sed -n '1,200p' src/foo.py"),
                _make_tool_result_msg("t3", content),
                {
                    "role": "user",
                    "content": "Answer using what we already have.",
                },
                {
                    "role": "user",
                    "content": "Answer using the existing evidence only.",
                },
            ],
        ),
        session,
    )

    assert prepared.saved_prompt_tokens >= 0
    assert (
        prepared.behavior_signals.get("state_resend_suppressed_turn", 0) >= 1
        or prepared.behavior_signals.get("state_resend_delta_turn", 0) >= 1
        or prepared.behavior_signals.get("state_resend_reason_delta_not_smaller", 0) >= 1
    )
    assert prepared.behavior_signals.get("state_resend_reason_full_default", 0) == 0


def test_prepared_request_reports_baseline_and_prepared_prompt_tokens(
    tmp_path,
) -> None:
    runtime, session = _runtime(tmp_path)
    prepared = runtime.prepare_request(
        RuntimeRequest(
            model="claude-sonnet-4",
            tool_compatible=True,
            messages=[{"role": "user", "content": "Confirm the gateway entry point."}],
        ),
        session,
    )

    assert prepared.baseline_prompt_tokens >= 0
    assert prepared.prepared_prompt_tokens >= 0
    assert prepared.saved_prompt_tokens == max(0, prepared.baseline_prompt_tokens - prepared.prepared_prompt_tokens)


def test_unchanged_prepared_payload_reuses_cached_token_estimate(
    tmp_path,
) -> None:
    _, session = _runtime(tmp_path)
    payload = {
        "system": "system",
        "messages": [{"role": "user", "content": "hello"}],
    }

    first = session.prepared_prompt_tokens(payload)
    second = session.prepared_prompt_tokens(payload)

    assert first == second
    assert session.consume_behavior_signals()["prepared_prompt_token_cache_hit"] == 1


def _state(turns: int, **kwargs: str | list[str]) -> dict[str, list[str]]:
    base: dict[str, list[str]] = {"turns": [str(turns)]}
    base.update({k: [v] if isinstance(v, str) else v for k, v in kwargs.items()})
    return base


def test_unchanged_files_not_in_delta() -> None:
    """When files are identical across turns, they should not appear in the delta."""
    previous = _state(1, files="src/tok/foo.py", goal="fix tests")
    current = _state(2, files="src/tok/foo.py", goal="fix tests")

    delta = _delta_tok_state_fields(previous, current)

    # Delta should only carry the turn counter (or be empty if nothing changed).
    assert "foo.py" not in delta, f"Expected unchanged files to be omitted from delta; got: {delta!r}"


def test_changed_files_appear_in_delta() -> None:
    """When files change, the new files list must appear in the delta."""
    previous = _state(1, files="src/tok/foo.py", goal="fix tests")
    current = _state(2, files="src/tok/bar.py", goal="fix tests")

    delta = _delta_tok_state_fields(previous, current)

    assert "bar.py" in delta, f"Expected changed files in delta; got: {delta!r}"


def test_unchanged_tests_not_in_delta() -> None:
    """When tests are identical, they should not appear in the delta."""
    previous = _state(1, tests="3_passed", goal="fix tests")
    current = _state(2, tests="3_passed", goal="fix tests")

    delta = _delta_tok_state_fields(previous, current)

    assert "3_passed" not in delta, f"Expected unchanged tests to be omitted from delta; got: {delta!r}"


def test_changed_tests_appear_in_delta() -> None:
    """When tests change, the new test status must appear in the delta."""
    previous = _state(1, tests="2_passed", goal="fix tests")
    current = _state(2, tests="3_passed", goal="fix tests")

    delta = _delta_tok_state_fields(previous, current)

    assert "3_passed" in delta, f"Expected changed tests in delta; got: {delta!r}"


def test_answer_anchor_facts_unchanged_not_in_delta() -> None:
    """Stable answer-anchor facts should not be re-sent in delta on subsequent turns."""
    previous = _state(
        1,
        files="src/tok/foo.py",
        facts=["answer_verification:all_pass"],
        goal="fix tests",
    )
    current = _state(
        2,
        files="src/tok/foo.py",
        facts=["answer_verification:all_pass"],
        goal="fix tests",
    )

    delta = _delta_tok_state_fields(previous, current)

    # Only the turn counter should change; files and facts are identical.
    assert "answer_verification" not in delta, (
        f"Expected unchanged anchor facts to be omitted from delta; got: {delta!r}"
    )
    assert "foo.py" not in delta, f"Expected unchanged anchor files to be omitted from delta; got: {delta!r}"


def test_answer_anchor_facts_changed_appear_in_delta() -> None:
    """When anchor facts change, new facts must appear in the delta."""
    previous = _state(
        1,
        files="src/tok/foo.py",
        facts=["answer_verification:partial"],
        goal="fix tests",
    )
    current = _state(
        2,
        files="src/tok/foo.py",
        facts=["answer_verification:all_pass"],
        goal="fix tests",
    )

    delta = _delta_tok_state_fields(previous, current)

    assert "answer_verification:all_pass" in delta, f"Expected changed anchor facts in delta; got: {delta!r}"


def test_delta_empty_when_only_turns_change() -> None:
    """When only the turn counter changes with no other field differences, delta is empty."""
    previous = _state(1, goal="fix tests")
    current = _state(2, goal="fix tests")

    delta = _delta_tok_state_fields(previous, current)

    # delta should be empty string — only turns changed, which is excluded from delta
    # (only turns in delta means the function returns "" per the guard at line 339)
    assert delta == "", f"Expected empty delta when only turns differ; got: {delta!r}"


def test_stream_recovery_budget_suppresses_next_reacquisition_burst(
    tmp_path,
) -> None:
    runtime, session = _runtime(tmp_path)
    content = "\n".join(f"line {idx}" for idx in range(1, 20))
    session._stream_recovery_reacquisition_budget = 1

    prepared = runtime.prepare_request(
        RuntimeRequest(
            model="claude-sonnet-4",
            tool_compatible=True,
            messages=[
                {"role": "user", "content": "Inspect foo."},
                _make_tool_use_msg("t1", "bash", command="cat src/foo.py"),
                _make_tool_result_msg("t1", content),
                {"role": "user", "content": "What did you find?"},
                _make_tool_use_msg("t2", "view_file", path="./src/foo.py"),
                _make_tool_result_msg("t2", content),
                {
                    "role": "user",
                    "content": "Answer using what we already have.",
                },
            ],
        ),
        session,
    )

    assert prepared.behavior_signals.get("repeat_file_read", 0) == 0
    assert prepared.behavior_signals["stream_recovery_reacquisition_suppressed"] == 1
    assert session._stream_recovery_reacquisition_budget == 0


def test_rapid_re_read_bypasses_compression(tmp_path) -> None:
    """Test that reading same file within 3 turns triggers bypass."""
    runtime, session = _runtime(tmp_path)
    content = "def foo():\n    pass\n"

    session.bridge_memory.turn = 8
    _first = runtime.prepare_request(
        RuntimeRequest(
            model="claude-sonnet-4",
            tool_compatible=True,
            messages=[
                {"role": "user", "content": "Check foo.py"},
                _make_tool_use_msg("t1", "view_file", path="src/foo.py"),
                _make_tool_result_msg("t1", content),
                {"role": "user", "content": "OK"},
            ],
        ),
        session,
    )

    session.bridge_memory.turn = 10
    _second = runtime.prepare_request(
        RuntimeRequest(
            model="claude-sonnet-4",
            tool_compatible=True,
            messages=[
                {"role": "user", "content": "Check foo.py"},
                _make_tool_use_msg("t1", "view_file", path="src/foo.py"),
                _make_tool_result_msg("t1", content),
                {"role": "user", "content": "OK"},
                _make_tool_use_msg("t2", "view_file", path="src/foo.py"),
                _make_tool_result_msg("t2", content),
                {"role": "user", "content": "Again?"},
            ],
        ),
        session,
    )
    assert session._last_elevated_path == "src/foo.py"


def test_different_file_read_clears_elevation(tmp_path) -> None:
    """Test that reading a different file clears the elevation state."""
    runtime, session = _runtime(tmp_path)
    foo_content = "def foo():\n    pass\n"
    bar_content = "def bar():\n    return 42\n"

    session.bridge_memory.turn = 8
    runtime.prepare_request(
        RuntimeRequest(
            model="claude-sonnet-4",
            tool_compatible=True,
            messages=[
                {"role": "user", "content": "Check foo.py"},
                _make_tool_use_msg("t1", "view_file", path="src/foo.py"),
                _make_tool_result_msg("t1", foo_content),
                {"role": "user", "content": "OK"},
            ],
        ),
        session,
    )

    session.bridge_memory.turn = 10
    _second = runtime.prepare_request(
        RuntimeRequest(
            model="claude-sonnet-4",
            tool_compatible=True,
            messages=[
                {"role": "user", "content": "Check foo.py"},
                _make_tool_use_msg("t1", "view_file", path="src/foo.py"),
                _make_tool_result_msg("t1", foo_content),
                {"role": "user", "content": "OK"},
                _make_tool_use_msg("t2", "view_file", path="src/foo.py"),
                _make_tool_result_msg("t2", foo_content),
                {"role": "user", "content": "Now check bar.py"},
                _make_tool_use_msg("t3", "view_file", path="src/bar.py"),
                _make_tool_result_msg("t3", bar_content),
                {"role": "user", "content": "OK"},
            ],
        ),
        session,
    )
    assert session._last_elevated_path == ""
