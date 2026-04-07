from __future__ import annotations

import hashlib
from typing import Any

from tok.compression._history_pipeline import compress_tool_results_impl
from tok.runtime.core import RuntimeSession
from tok.runtime.pipeline._tool_context import build_tool_use_id_to_context
from tok.runtime.pipeline._tool_repeat_detection import _make_cache_key
from tok.runtime.repeat_targets import HotSummaryRecord, evidence_identity_key
from tok.testing.stress.executor import ReadOnlyToolExecutor


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


def _search_output(lines: int = 48) -> str:
    return "\n".join(
        f"src/example.py:{idx}:needle match {idx}"
        for idx in range(1, lines + 1)
    )


class TestHotSearchFirstExactProtection:
    def test_hot_search_cache_hit_does_not_replace_first_exact_observation(
        self,
    ):
        search_output = _search_output()
        messages = [
            _tool_use("t1", "grep_search", query="needle", path="src"),
            _tool_result("t1", search_output),
        ]
        tool_use_id_to_context = build_tool_use_id_to_context(messages)
        first_exact_evidence_seen: set[str] = set()
        result_cache: dict[
            str, tuple[str, str, float] | tuple[str, str] | tuple[str]
        ] = {}
        cache_key = _make_cache_key(
            "grep_search", tool_use_id_to_context["t1"]
        )
        result_cache[cache_key] = (
            hashlib.sha256(search_output.encode()).hexdigest()[:8],
            search_output,
        )

        compressed, breakdown = compress_tool_results_impl(
            messages,
            result_cache=result_cache,
            tool_use_id_to_context=tool_use_id_to_context,
            compression_level="balanced",
            first_exact_evidence_seen=first_exact_evidence_seen,
        )

        first_result = compressed[1]["content"][0]["content"]
        assert first_result == search_output
        assert breakdown == {}
        assert (
            evidence_identity_key(
                "grep_search",
                path="src",
                query="needle",
                args={"query": "needle", "path": "src"},
            )
            in first_exact_evidence_seen
        )

    def test_second_identical_hot_search_result_may_compress_after_exact_seen(
        self,
    ):
        search_output = _search_output()
        messages = [
            _tool_use("t1", "grep_search", query="needle", path="src"),
            _tool_result("t1", search_output),
            _tool_use("t2", "grep_search", query="needle", path="src"),
            _tool_result("t2", search_output),
        ]
        tool_use_id_to_context = build_tool_use_id_to_context(messages)
        first_exact_evidence_seen: set[str] = set()
        result_cache: dict[
            str, tuple[str, str, float] | tuple[str, str] | tuple[str]
        ] = {}
        cache_key = _make_cache_key(
            "grep_search", tool_use_id_to_context["t1"]
        )
        result_cache[cache_key] = (
            hashlib.sha256(search_output.encode()).hexdigest()[:8],
            search_output,
        )

        compressed, breakdown = compress_tool_results_impl(
            messages,
            result_cache=result_cache,
            tool_use_id_to_context=tool_use_id_to_context,
            compression_level="balanced",
            first_exact_evidence_seen=first_exact_evidence_seen,
        )

        first_result = compressed[1]["content"][0]["content"]
        second_result = compressed[3]["content"][0]["content"]

        assert first_result == search_output
        assert second_result != search_output
        assert any(
            marker in second_result
            for marker in (
                ">>> tool:grep|matches:",
                ">>> tool:grep_search|unchanged|cached",
                "@stable_result(",
            )
        )
        assert sum(breakdown.values()) > 0

    def test_hot_recent_search_hints_require_session_exact_observation(
        self, tmp_path
    ):
        session = RuntimeSession(memory_dir=tmp_path / ".tok-hot-search")
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
        session.bridge_memory.turn = 5

        hints, metrics = session.hot_recent_runtime_hints()
        assert hints == []
        assert metrics["hot_recent_hint_injected"] == 0

        if exact_key:
            session._first_exact_evidence_seen.add(exact_key)

        hints, metrics = session.hot_recent_runtime_hints()
        assert hints
        assert "@hot_recent_search:needle @ src |>" in hints[0]
        assert metrics["hot_recent_hint_injected"] == 1


class TestFileTargetedGrepReliability:
    def test_file_targeted_grep_on_existing_file_returns_exact_matches(
        self, tmp_path
    ):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        target = workspace / "src" / "example.py"
        target.parent.mkdir(parents=True)
        target.write_text(
            "alpha\nneedle first match\nomega\nneedle second match\n",
            encoding="utf-8",
        )
        executor = ReadOnlyToolExecutor(
            workspace_root=workspace, max_output_chars=5000
        )

        result, blocked = executor.execute(
            {
                "id": "g1",
                "name": "grep_search",
                "input": {"path": "src/example.py", "query": "needle"},
            }
        )

        assert blocked is False
        assert result.get("is_error") is not True
        assert "ERROR:" not in result["content"]
        assert len(result["content"].splitlines()) == 2
        assert "needle first match" in result["content"]
        assert "needle second match" in result["content"]

    def test_file_targeted_grep_with_no_match_returns_clean_result(
        self, tmp_path
    ):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        target = workspace / "src" / "example.py"
        target.parent.mkdir(parents=True)
        target.write_text("alpha\nomega\n", encoding="utf-8")
        executor = ReadOnlyToolExecutor(
            workspace_root=workspace, max_output_chars=5000
        )

        result, blocked = executor.execute(
            {
                "id": "g2",
                "name": "grep_search",
                "input": {"path": "src/example.py", "query": "needle"},
            }
        )

        assert blocked is False
        assert result["content"] == "(no matches)"
        assert result.get("is_error") is not True
        assert not str(result["content"]).startswith("ERROR:")

    def test_file_targeted_grep_does_not_emit_false_error_stub_for_valid_query(
        self, tmp_path
    ):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        target = workspace / "src" / "example.py"
        target.parent.mkdir(parents=True)
        target.write_text("alpha\nneedle\nomega\n", encoding="utf-8")
        executor = ReadOnlyToolExecutor(
            workspace_root=workspace, max_output_chars=5000
        )

        result, blocked = executor.execute(
            {
                "id": "g3",
                "name": "grep_search",
                "input": {"path": "src/example.py", "query": "needle"},
            }
        )

        assert blocked is False
        assert result.get("is_error") is not True
        assert result.get("contract_signal", "") == ""
        assert not str(result["content"]).startswith("ERROR:")
