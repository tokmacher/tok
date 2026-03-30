"""Tests for the evidence-intent layer in repeat_targets.py and core.py."""

from __future__ import annotations

import json


from tok.runtime.repeat_targets import (
    extract_git_history_path,
    extract_metadata_probe,
    extract_shell_search_params,
    resolve_evidence_intent,
)


class TestExtractGitHistoryPath:
    def test_git_show_head(self):
        path, rev = extract_git_history_path("git show HEAD:src/foo.py")
        assert path == "src/foo.py"
        assert rev == "HEAD"

    def test_git_show_head_tilde_n(self):
        path, rev = extract_git_history_path("git show HEAD~3:src/foo.py")
        assert path == "src/foo.py"
        assert rev == "HEAD~N"

    def test_git_show_sha(self):
        path, rev = extract_git_history_path("git show abc1234:src/foo.py")
        assert path == "src/foo.py"
        assert rev == "sha"

    def test_git_show_ref(self):
        path, rev = extract_git_history_path("git show @~1:src/bar.py")
        assert path == "src/bar.py"
        assert rev == "ref"

    def test_git_diff_with_path(self):
        path, rev = extract_git_history_path("git diff HEAD -- src/foo.py")
        assert path == "src/foo.py"
        assert rev == "HEAD"

    def test_not_git_command(self):
        path, rev = extract_git_history_path("cat src/foo.py")
        assert path is None
        assert rev == ""

    def test_empty(self):
        path, rev = extract_git_history_path("")
        assert path is None
        assert rev == ""


class TestExtractShellSearchParams:
    def test_grep_with_pattern_and_scope(self):
        query, scope = extract_shell_search_params("grep -r 'pattern' src/")
        assert query == "pattern"
        assert scope == "src"

    def test_rg_with_pattern_only(self):
        query, scope = extract_shell_search_params("rg 'some_func'")
        assert query == "some_func"
        assert scope is None

    def test_grep_no_args(self):
        query, scope = extract_shell_search_params("grep")
        assert query is None

    def test_not_search_command(self):
        query, scope = extract_shell_search_params("cat src/foo.py")
        assert query is None

    def test_empty(self):
        query, scope = extract_shell_search_params("")
        assert query is None


class TestExtractMetadataProbe:
    def test_git_log(self):
        result = extract_metadata_probe("git log --oneline")
        assert result == "git_log"

    def test_git_status(self):
        result = extract_metadata_probe("git status")
        assert result == "git_status"

    def test_stat_command(self):
        result = extract_metadata_probe("stat src/foo.py")
        assert result == "stat"

    def test_unsafe_marker_rejected(self):
        result = extract_metadata_probe("git log && echo done")
        assert result is None

    def test_not_metadata(self):
        result = extract_metadata_probe("python -m pytest")
        assert result is None


class TestResolveEvidenceIntent:
    def test_native_file_read(self):
        intent = resolve_evidence_intent("read", path="src/foo.py")
        assert intent is not None
        assert intent.domain == "file_current"
        assert intent.anchor == "src/foo.py"
        assert intent.variant == "full"
        assert intent.source_kind == "native_tool"

    def test_git_show_command(self):
        intent = resolve_evidence_intent(
            "bash", command="git show HEAD:src/foo.py"
        )
        assert intent is not None
        assert intent.domain == "file_history"
        assert intent.anchor == "src/foo.py"
        assert intent.variant == "diff"
        assert intent.novelty_key == "HEAD"
        assert intent.source_kind == "git_history"

    def test_shell_grep_command(self):
        intent = resolve_evidence_intent(
            "bash", command="grep -r 'pattern' src/"
        )
        assert intent is not None
        assert intent.domain == "search"
        assert intent.source_kind == "shell_search"
        expected_anchor = json.dumps(
            {"query": "pattern", "scope": "src"},
            sort_keys=True,
            separators=(",", ":"),
        )
        assert intent.anchor == expected_anchor

    def test_native_search(self):
        intent = resolve_evidence_intent("grep", query="pattern", path="src/")
        assert intent is not None
        assert intent.domain == "search"
        assert intent.source_kind == "native_tool"

    def test_metadata_probe(self):
        intent = resolve_evidence_intent("bash", command="git log --oneline")
        assert intent is not None
        assert intent.domain == "file_metadata"
        assert intent.source_kind == "metadata_probe"
        assert intent.novelty_key == "git_log"

    def test_temp_copy_detection(self):
        intent = resolve_evidence_intent(
            "bash", command="cat /tmp/foo_copy.py"
        )
        assert intent is not None
        assert intent.domain == "file_current"
        assert intent.variant == "copy"
        assert intent.source_kind == "temp_copy"

    def test_shell_read_not_temp(self):
        intent = resolve_evidence_intent("bash", command="cat src/foo.py")
        assert intent is not None
        assert intent.domain == "file_current"
        assert intent.variant == "full"
        assert intent.source_kind == "shell_read"

    def test_unknown_tool_returns_none(self):
        intent = resolve_evidence_intent("unknown_tool")
        assert intent is None

    def test_native_file_read_and_git_show_separate_domains(self):
        native = resolve_evidence_intent("read", path="src/foo.py")
        git_show = resolve_evidence_intent(
            "bash", command="git show HEAD:src/foo.py"
        )
        assert native.domain == "file_current"
        assert git_show.domain == "file_history"
        assert native.anchor == git_show.anchor
        assert native.domain != git_show.domain

    def test_shell_grep_merges_with_native_search_anchor(self):
        shell_intent = resolve_evidence_intent(
            "bash", command="grep -r 'pattern' src/"
        )
        native_intent = resolve_evidence_intent(
            "grep", query="pattern", path="src/"
        )
        assert shell_intent.domain == native_intent.domain
        assert shell_intent.anchor == native_intent.anchor


class TestSessionEvidenceTracking:
    def test_repeated_git_show_promotes_history_anchor(self):
        from tok.runtime.core import RuntimeSession

        session = RuntimeSession()
        signals1 = session.observe_repeat_target_result(
            tool_id="t1",
            tool_name="bash",
            path=None,
            query=None,
            command="git show HEAD:src/foo.py",
            raw_content="line1\nline2\nline3\n",
        )
        signals2 = session.observe_repeat_target_result(
            tool_id="t2",
            tool_name="bash",
            path=None,
            query=None,
            command="git show HEAD:src/foo.py",
            raw_content="line1\nline2\nline3\n",
        )
        assert signals1.get("evidence_anchor_hot", 0) == 0
        assert signals2.get("evidence_anchor_hot", 0) == 1

    def test_same_anchor_no_novelty_triggers_missing(self):
        from tok.runtime.core import RuntimeSession

        session = RuntimeSession()
        session.observe_repeat_target_result(
            tool_id="t1",
            tool_name="bash",
            path=None,
            query=None,
            command="git show HEAD:src/foo.py",
            raw_content="line1\nline2\nline3\n",
        )
        session.observe_repeat_target_result(
            tool_id="t2",
            tool_name="bash",
            path=None,
            query=None,
            command="git show HEAD:src/foo.py",
            raw_content="line1\nline2\nline3\n",
        )
        signals3 = session.observe_repeat_target_result(
            tool_id="t3",
            tool_name="bash",
            path=None,
            query=None,
            command="git show HEAD:src/foo.py",
            raw_content="line1\nline2\nline3\n",
        )
        assert signals3.get("evidence_novelty_missing", 0) == 1

    def test_new_novelty_does_not_trigger_missing(self):
        from tok.runtime.core import RuntimeSession

        session = RuntimeSession()
        session.observe_repeat_target_result(
            tool_id="t1",
            tool_name="bash",
            path=None,
            query=None,
            command="git show HEAD:src/foo.py",
            raw_content="line1\nline2\nline3\n",
        )
        session.observe_repeat_target_result(
            tool_id="t2",
            tool_name="bash",
            path=None,
            query=None,
            command="git show HEAD:src/foo.py",
            raw_content="line1\nline2\nline3\n",
        )
        signals3 = session.observe_repeat_target_result(
            tool_id="t3",
            tool_name="bash",
            path=None,
            query=None,
            command="git show HEAD~3:src/foo.py",
            raw_content="old_line1\nold_line2\n",
        )
        assert signals3.get("evidence_novelty_missing", 0) == 0

    def test_neighborhood_thrash_detected(self):
        from tok.runtime.core import RuntimeSession

        session = RuntimeSession()
        paths = [
            "src/tok/core.py",
            "src/tok/adapters.py",
            "src/tok/gateway.py",
        ]
        for i, p in enumerate(paths):
            session.observe_repeat_target_result(
                tool_id=f"t{i}",
                tool_name="read",
                path=p,
                query=None,
                command=None,
                raw_content=f"content of {p}\n" * 10,
            )
        signals = session.observe_repeat_target_result(
            tool_id="t3b",
            tool_name="read",
            path="src/tok/core.py",
            query=None,
            command=None,
            raw_content="content of src/tok/core.py\n" * 10,
        )
        assert signals.get("evidence_neighborhood_hot", 0) == 1

    def test_evidence_intent_advisories_novelty(self):
        from tok.runtime.core import RuntimeSession

        session = RuntimeSession()
        for i in range(3):
            session.observe_repeat_target_result(
                tool_id=f"t{i}",
                tool_name="bash",
                path=None,
                query=None,
                command="git show HEAD:src/foo.py",
                raw_content="line1\nline2\nline3\n",
            )
        hints = session.evidence_intent_advisories()
        assert len(hints) == 1
        assert "already have evidence" in hints[0]

    def test_evidence_intent_advisories_neighborhood(self):
        from tok.runtime.core import RuntimeSession

        session = RuntimeSession()
        paths = [
            "src/tok/a.py",
            "src/tok/b.py",
            "src/tok/c.py",
        ]
        for i, p in enumerate(paths):
            session.observe_repeat_target_result(
                tool_id=f"t{i}",
                tool_name="read",
                path=p,
                query=None,
                command=None,
                raw_content="content\n" * 20,
            )
        session.observe_repeat_target_result(
            tool_id="t3b",
            tool_name="read",
            path="src/tok/a.py",
            query=None,
            command=None,
            raw_content="content\n" * 20,
        )
        hints = session.evidence_intent_advisories()
        assert len(hints) >= 1
        assert any("explored multiple files" in h for h in hints)


class TestBridgeMemoryDomainSeparation:
    def test_history_snapshot_separate_from_current(self):
        from tok.runtime.memory.bridge_memory import BridgeMemoryState

        mem = BridgeMemoryState()
        mem.turn = 1
        mem.record_file_snapshot("src/foo.py", "line1\ndef foo():\n  pass\n")
        mem.record_history_snapshot(
            "src/foo.py", "HEAD", "old_line1\ndef old_foo():\n  pass\n"
        )
        facts = [e.value for e in mem.hot.get("facts", [])]
        file_facts = [f for f in facts if f.startswith("file[")]
        history_facts = [f for f in facts if f.startswith("history_file[")]
        assert len(file_facts) == 1
        assert len(history_facts) == 1
        assert file_facts[0] != history_facts[0]

    def test_metadata_snapshot_separate(self):
        from tok.runtime.memory.bridge_memory import BridgeMemoryState

        mem = BridgeMemoryState()
        mem.turn = 1
        mem.record_file_snapshot("src/foo.py", "line1\ndef foo():\n  pass\n")
        mem.record_metadata_snapshot(
            "src/foo.py", "git_log", "abc123 commit msg"
        )
        facts = [e.value for e in mem.hot.get("facts", [])]
        file_facts = [f for f in facts if f.startswith("file[")]
        meta_facts = [f for f in facts if f.startswith("meta[")]
        assert len(file_facts) == 1
        assert len(meta_facts) == 1

    def test_get_file_fact_digests(self):
        from tok.runtime.memory.bridge_memory import BridgeMemoryState

        mem = BridgeMemoryState()
        mem.turn = 1
        mem.record_file_snapshot("src/foo.py", "line1\ndef foo():\n  pass\n")
        digests = mem.get_file_fact_digests()
        assert "src/foo.py" in digests
        assert digests["src/foo.py"] != ""


class TestTempCopyAliasing:
    def test_matching_digest_aliases(self):
        from tok.runtime.core import RuntimeSession

        session = RuntimeSession()
        content = "line1\ndef foo():\n    pass\nline4\n"
        session.record_file_snapshot("src/foo.py", content)
        alias = session.check_temp_copy_alias("/tmp/foo_copy.py", content)
        assert alias == "src/foo.py"
        assert "/tmp/foo_copy.py" in session._evidence_alias_map

    def test_different_digest_does_not_alias(self):
        from tok.runtime.core import RuntimeSession

        session = RuntimeSession()
        session.record_file_snapshot("src/foo.py", "original content here")
        alias = session.check_temp_copy_alias(
            "/tmp/unrelated.py", "completely different content"
        )
        assert alias is None
        assert "/tmp/unrelated.py" not in session._evidence_alias_map

    def test_non_temp_path_skipped(self):
        from tok.runtime.core import RuntimeSession

        session = RuntimeSession()
        session.record_file_snapshot("src/foo.py", "content")
        alias = session.check_temp_copy_alias("src/bar.py", "content")
        assert alias is None


class TestAnswerReadyPriorityOverEvidence:
    def test_evidence_advisories_not_returned_when_answer_ready(self):
        from tok.runtime.pipeline.request_preparation import (
            _runtime_hints_for_turn,
        )

        runtime_hints = _runtime_hints_for_turn(
            answer_ready=True,
            answer_ready_repair_active=False,
            late_answer_followthrough_active=False,
            late_answer_assembly_repair_mode="",
        )
        assert len(runtime_hints) > 0
        assert all("already have evidence" not in h for h in runtime_hints)

    def test_evidence_advisories_not_returned_when_repair_active(self):
        from tok.runtime.pipeline.request_preparation import (
            _runtime_hints_for_turn,
        )

        runtime_hints = _runtime_hints_for_turn(
            answer_ready=False,
            answer_ready_repair_active=True,
            late_answer_followthrough_active=False,
            late_answer_assembly_repair_mode="",
        )
        assert len(runtime_hints) > 0
        assert all("already have evidence" not in h for h in runtime_hints)
