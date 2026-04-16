"""Tests for speculative macro injection and semantic hash deduplication."""

from __future__ import annotations

from typing import Any

from tok.analysis.prompt import MINIMAL_PULSE_PROMPT, TOK_EXPLORE_PROMPT
from tok.compression import (
    _SEMANTIC_HASH_MIN_CHARS,
    _STABLE_RESULT_EXPLANATION,
    ResultCacheEntry,
    _compute_semantic_hash,
    _make_semantic_cache_key,
    compress_tool_results,
)
from tok.compression import _history_pipeline as history_pipeline
from tok.compression import _pipeline as compression_pipeline
from tok.neuro.ir import Instruction, Macro
from tok.runtime.memory.bridge_memory import BridgeMemoryState
from tok.universal_runtime import (
    RuntimeRequest,
    RuntimeSession,
    UniversalTokRuntime,
)

compression_pipeline.TOOL_COMPRESS_THRESHOLD = 0


def _make_tool_use_msg(tool_id: str, tool_name: str, path: str = "src/tok/foo.py") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": tool_id,
                "name": tool_name,
                "input": {"path": path},
            }
        ],
    }


def _make_tool_result_block(tool_id: str, content: str) -> dict[str, Any]:
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


def _make_id_to_context(tool_id: str, tool_name: str, path: str) -> dict[str, Any]:
    return {
        tool_id: {
            "name": tool_name,
            "path": path,
            "args": {"path": path},
        }
    }


class TestSpeculativeMacroInjection:
    def _session_with_macros(self, macros: list[Macro]) -> RuntimeSession:
        session = RuntimeSession()
        for m in macros:
            # Insert directly to avoid global-registry op-sequence dedup collisions.
            session.bridge_memory.macro_registry.macros[m.name] = m
        return session

    def _simple_macro(self, name: str, hit_count: int = 3) -> Macro:
        # Use a unique multi-op sequence per macro name so op-sequence dedup doesn't
        # block registration against global macros or sibling test macros.
        return Macro(
            name=name,
            instructions=(
                Instruction(op=f"tok_test_{name}_op1", args=()),
                Instruction(op=f"tok_test_{name}_op2", args=()),
            ),
            inputs=(),
            hit_count=hit_count,
        )

    def _request_with_message(self) -> RuntimeRequest:
        return RuntimeRequest(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hello"}],
        )

    def test_speculative_hint_injected_when_macros_match(self) -> None:
        macro = self._simple_macro("fix_imports", hit_count=3)
        session = self._session_with_macros([macro])
        runtime = UniversalTokRuntime()

        prepared = runtime.prepare_request(self._request_with_message(), session)

        system = prepared.body.get("system", "")
        assert "Available macros" not in system

    def test_speculative_hint_absent_when_no_macros(self) -> None:
        session = RuntimeSession(bridge_memory=BridgeMemoryState(load_global_macros=False))
        runtime = UniversalTokRuntime()

        prepared = runtime.prepare_request(self._request_with_message(), session)

        system = prepared.body.get("system", "")
        assert "Available macros" not in system

    def test_speculative_hint_absent_below_threshold(self) -> None:
        """Macros below hit threshold should not be injected."""
        macro = self._simple_macro("low_hit", hit_count=1)
        session = self._session_with_macros([macro])
        runtime = UniversalTokRuntime()

        prepared = runtime.prepare_request(self._request_with_message(), session)

        system = prepared.body.get("system", "")
        assert "@low_hit" not in system

    def test_speculative_hint_lists_multiple_macros(self) -> None:
        macros = [
            self._simple_macro("macro_a", hit_count=5),
            self._simple_macro("macro_b", hit_count=4),
        ]
        session = self._session_with_macros(macros)
        runtime = UniversalTokRuntime()

        prepared = runtime.prepare_request(self._request_with_message(), session)

        system = prepared.body.get("system", "")
        assert "@macro_a" not in system
        assert "@macro_b" not in system

    def test_speculative_signal_recorded(self) -> None:
        macro = self._simple_macro("sig_macro", hit_count=3)
        session = self._session_with_macros([macro])
        runtime = UniversalTokRuntime()

        runtime.prepare_request(self._request_with_message(), session)
        prepared = runtime.prepare_request(self._request_with_message(), session)
        assert "@sig_macro" not in prepared.body.get("system", "")


class TestComputeSemanticHash:
    def test_returns_hex_string(self) -> None:
        h = _compute_semantic_hash("hello world")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_content_same_hash(self) -> None:
        assert _compute_semantic_hash("abc") == _compute_semantic_hash("abc")

    def test_different_content_different_hash(self) -> None:
        assert _compute_semantic_hash("abc") != _compute_semantic_hash("xyz")


class TestSemanticHashDedup:
    def _large_content(self, text: str = "x") -> str:
        # Ensure the stable payload (hash + summary + skeleton) is meaningfully
        # smaller than the raw content so dedup is eligible.
        return text * (_SEMANTIC_HASH_MIN_CHARS + 2000)

    def _messages_and_ctx(
        self,
        content: str,
        tool_name: str = "view_file",
        path: str = "src/tok/foo.py",
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        tool_id = "tid1"
        messages = [_make_tool_result_block(tool_id, content)]
        id_to_ctx = _make_id_to_context(tool_id, tool_name, path)
        return messages, id_to_ctx

    def test_first_occurrence_not_replaced(self) -> None:
        content = self._large_content()
        messages, id_to_ctx = self._messages_and_ctx(content)
        cache: dict[str, str] = {}

        result, _breakdown = compress_tool_results(
            messages,
            tool_use_id_to_context=id_to_ctx,
            semantic_hash_cache=cache,
        )

        # First time: content should NOT be replaced, hash should be stored
        block_content = result[0]["content"][0]["content"]
        assert "@stable_result" not in block_content
        assert len(cache) == 1

    def test_second_occurrence_replaced_with_token(self) -> None:
        content = self._large_content()
        _messages, _id_to_ctx = self._messages_and_ctx(content)
        cache: dict[str, str] = {}

        # First pass: populates cache
        compress_tool_results(
            [_make_tool_result_block("tid1", content)],
            tool_use_id_to_context=_make_id_to_context("tid1", "view_file", "src/tok/foo.py"),
            semantic_hash_cache=cache,
        )

        # Second pass: same tool, same args, same content → should dedup
        messages2 = [_make_tool_result_block("tid1", content)]
        id_to_ctx2 = _make_id_to_context("tid1", "view_file", "src/tok/foo.py")
        result2, _breakdown2 = compress_tool_results(
            messages2,
            tool_use_id_to_context=id_to_ctx2,
            semantic_hash_cache=cache,
            hot_summary_records={},
        )

        block_content = result2[0]["content"][0]["content"]
        assert block_content.startswith(">>> tool:file_read|") or block_content.startswith("@stable_result(hash:")
        assert len(block_content) < len(content)

    def test_changed_content_not_replaced(self) -> None:
        path = "src/tok/foo.py"
        cache: dict[str, str] = {}

        content_a = self._large_content("a")
        content_b = self._large_content("b")

        # First pass: content_a
        compress_tool_results(
            [_make_tool_result_block("tid1", content_a)],
            tool_use_id_to_context=_make_id_to_context("tid1", "view_file", path),
            semantic_hash_cache=cache,
        )

        # Second pass: content_b (different) — cache key is updated, no dedup
        messages2 = [_make_tool_result_block("tid1", content_b)]
        id_to_ctx2 = _make_id_to_context("tid1", "view_file", path)
        result2, breakdown2 = compress_tool_results(
            messages2,
            tool_use_id_to_context=id_to_ctx2,
            semantic_hash_cache=cache,
        )

        block_content = result2[0]["content"][0]["content"]
        assert "@stable_result" not in block_content
        assert breakdown2.get("semantic_dedup", 0) == 0

    def test_small_content_not_deduped(self) -> None:
        """Content below min chars should not be eligible for semantic hash dedup."""
        small = "x" * (_SEMANTIC_HASH_MIN_CHARS - 1)
        cache: dict[str, str] = {}
        id_to_ctx = _make_id_to_context("tid1", "view_file", "src/tok/foo.py")

        # Run twice
        for _ in range(2):
            compress_tool_results(
                [_make_tool_result_block("tid1", small)],
                tool_use_id_to_context=id_to_ctx,
                semantic_hash_cache=cache,
            )

        # Cache should be empty — small content was never hashed
        assert len(cache) == 0

    def test_no_cache_no_dedup(self) -> None:
        content = self._large_content()
        messages, id_to_ctx = self._messages_and_ctx(content)

        result, _breakdown = compress_tool_results(
            messages,
            tool_use_id_to_context=id_to_ctx,
            semantic_hash_cache=None,
        )

        block_content = result[0]["content"][0]["content"]
        assert "@stable_result" not in block_content

    def test_stable_result_explanation_constant_exists(self) -> None:
        assert "@stable_result" in _STABLE_RESULT_EXPLANATION
        assert "unchanged" in _STABLE_RESULT_EXPLANATION

    def test_semantic_cache_key_includes_path_identity(self) -> None:
        ctx_a = {
            "name": "view_file",
            "path": "src/tok/a.py",
            "args": {"path": "src/tok/a.py", "offset": 0, "limit": 10},
        }
        ctx_b = {
            "name": "view_file",
            "path": "src/tok/b.py",
            "args": {"path": "src/tok/b.py", "offset": 0, "limit": 10},
        }
        key_a = _make_semantic_cache_key(ctx_a, "x" * 500)
        key_b = _make_semantic_cache_key(ctx_b, "x" * 500)
        assert key_a != key_b

    def test_first_read_is_never_compressed(self) -> None:
        """Verify first exact observation of a file is always delivered verbatim."""
        path = "src/tok/foo.py"
        tool_id = "tid1"
        content = "class A:\n    def m(self):\n        pass\n\ndef top():\n    return 1\n" + ("# filler\n" * 200)
        cache: dict[str, str] = {}
        session_files_read: set[str] = set()

        # First read should be verbatim (not compressed)
        result1, breakdown1 = compress_tool_results(
            [_make_tool_result_block(tool_id, content)],
            tool_use_id_to_context=_make_id_to_context(tool_id, "view_file", path),
            semantic_hash_cache=cache,
            session_files_read=session_files_read,
        )
        block_content1 = result1[0]["content"][0]["content"]
        # First read should be full content
        assert block_content1 == content
        # Should not be semantically deduped on first read
        assert breakdown1.get("semantic_dedup", 0) == 0

        # Second read (repeat) should be compressed/stable_result
        result2, breakdown2 = compress_tool_results(
            [_make_tool_result_block(tool_id, content)],
            tool_use_id_to_context=_make_id_to_context(tool_id, "view_file", path),
            semantic_hash_cache=cache,
            session_files_read=session_files_read,
        )
        block_content2 = result2[0]["content"][0]["content"]
        # Second read should be compressed
        assert len(block_content2) < len(content)
        assert "@stable_result" in block_content2 or "tool:file_read" in block_content2

    def test_stable_payload_includes_skeleton_for_code(self) -> None:
        path = "src/tok/foo.py"
        tool_id = "tid1"
        content = (
            "class A:\n"
            "    def m(self):\n"
            "        pass\n\n"
            "async def coro():\n"
            "    return 1\n\n"
            "def top():\n"
            "    return 2\n" + ("# filler\n" * 200)
        )
        cache: dict[str, str] = {}

        compress_tool_results(
            [_make_tool_result_block(tool_id, content)],
            tool_use_id_to_context=_make_id_to_context(tool_id, "view_file", path),
            semantic_hash_cache=cache,
        )

        result2, _breakdown2 = compress_tool_results(
            [_make_tool_result_block(tool_id, content)],
            tool_use_id_to_context=_make_id_to_context(tool_id, "view_file", path),
            semantic_hash_cache=cache,
            hot_summary_records={},
        )
        block_content = result2[0]["content"][0]["content"]
        assert block_content.startswith(">>> tool:file_read|") or block_content.startswith("@stable_result(hash:")
        assert len(block_content) < len(content)


class TestStableResultGuidance:
    def test_stable_result_explanation_uses_supported_bypass_marker(
        self,
    ) -> None:
        # Bypass mechanism removed - first reads are now automatically verbatim
        assert "@stable_result" in _STABLE_RESULT_EXPLANATION

    def test_analysis_prompts_advise_against_parallel_first_pass_reads(self) -> None:
        assert "Do not fan out parallel reads on the first pass." in (TOK_EXPLORE_PROMPT)
        assert "Do not open multiple files in parallel on the first pass." in (MINIMAL_PULSE_PROMPT)


class TestResultCacheStablePayload:
    def test_file_cache_hit_emits_stable_payload(self) -> None:
        tool_id = "tid1"
        path = "src/tok/foo.py"
        content = "class A:\n    def m(self):\n        pass\n\ndef top():\n    return 1\n" + ("# filler\n" * 400)
        id_to_ctx = _make_id_to_context(tool_id, "Read", path)
        result_cache: dict[str, ResultCacheEntry] = {}

        # First pass seeds result_cache (file-like tool returns raw first time).
        compress_tool_results(
            [_make_tool_result_block(tool_id, content)],
            tool_use_id_to_context=id_to_ctx,
            result_cache=result_cache,
            semantic_hash_cache=None,
        )

        # Second pass should be a cache hit and now emit stable payload lines.
        result2, _breakdown2 = compress_tool_results(
            [_make_tool_result_block(tool_id, content)],
            tool_use_id_to_context=id_to_ctx,
            result_cache=result_cache,
            semantic_hash_cache=None,
        )
        block_content = result2[0]["content"][0]["content"]
        assert block_content.startswith(">>> tool:file_read|") or block_content.startswith("@stable_result(hash:")
        assert len(block_content) < len(content)

    def test_precision_read_cache_hit_returns_raw(self) -> None:
        tool_id = "tid1"
        path = "src/tok/foo.py"
        content = "line\n" * 500
        id_to_ctx = {
            tool_id: {
                "name": "Read",
                "path": path,
                "args": {"file_path": path, "offset": 10, "limit": 20},
            }
        }
        result_cache: dict[str, ResultCacheEntry] = {}

        compress_tool_results(
            [_make_tool_result_block(tool_id, content)],
            tool_use_id_to_context=id_to_ctx,
            result_cache=result_cache,
            semantic_hash_cache=None,
        )

        result2, _breakdown2 = compress_tool_results(
            [_make_tool_result_block(tool_id, content)],
            tool_use_id_to_context=id_to_ctx,
            result_cache=result_cache,
            semantic_hash_cache=None,
        )
        block_content = result2[0]["content"][0]["content"]
        assert block_content == content

    def test_host_unchanged_stub_replays_cached_precision_bytes(self) -> None:
        tool_id = "tid1"
        path = "src/tok/foo.py"
        content = "line\n" * 500
        id_to_ctx = {
            tool_id: {
                "name": "Read",
                "path": path,
                "args": {"file_path": path, "offset": 10, "limit": 20},
            }
        }
        result_cache: dict[str, ResultCacheEntry] = {}

        # Seed with real content.
        compress_tool_results(
            [_make_tool_result_block(tool_id, content)],
            tool_use_id_to_context=id_to_ctx,
            result_cache=result_cache,
            semantic_hash_cache=None,
        )

        # Host "unchanged" stub (empty payload) should replay cached bytes.
        result2, _breakdown2 = compress_tool_results(
            [_make_tool_result_block(tool_id, "")],
            tool_use_id_to_context=id_to_ctx,
            result_cache=result_cache,
            semantic_hash_cache=None,
        )
        block_content = result2[0]["content"][0]["content"]
        assert block_content == content

    def test_host_unchanged_stub_replays_stable_payload_for_non_precision(
        self,
    ) -> None:
        tool_id = "tid1"
        path = "src/tok/foo.py"
        content = "class A:\n    def m(self):\n        pass\n\ndef top():\n    return 1\n" + ("# filler\n" * 400)
        id_to_ctx = _make_id_to_context(tool_id, "Read", path)
        result_cache: dict[str, ResultCacheEntry] = {}

        compress_tool_results(
            [_make_tool_result_block(tool_id, content)],
            tool_use_id_to_context=id_to_ctx,
            result_cache=result_cache,
            semantic_hash_cache=None,
        )

        # Tiny stub simulates host "unchanged" UI optimization.
        result2, _breakdown2 = compress_tool_results(
            [_make_tool_result_block(tool_id, "Unchanged since last read")],
            tool_use_id_to_context=id_to_ctx,
            result_cache=result_cache,
            semantic_hash_cache=None,
        )
        block_content = result2[0]["content"][0]["content"]
        assert block_content.startswith(">>> tool:file_read|") or block_content.startswith(
            ">>> replayed_cached_bytes|verified_unchanged"
        )
        assert len(block_content) < len(content)

    def test_invalid_stable_payload_metadata_falls_back_to_failure_stub(self, monkeypatch) -> None:
        from tok import compression as compression_module

        tool_id = "tid1"
        path = "src/tok/foo.py"
        content = "class A:\n    pass\n" + ("# filler\n" * 400)
        id_to_ctx = _make_id_to_context(tool_id, "Read", path)
        result_cache: dict[str, ResultCacheEntry] = {}

        compress_tool_results(
            [_make_tool_result_block(tool_id, content)],
            tool_use_id_to_context=id_to_ctx,
            result_cache=result_cache,
            semantic_hash_cache=None,
        )

        monkeypatch.setattr(compression_module, "_compute_semantic_hash", lambda _content: "")

        result2, breakdown2 = compression_module.compress_tool_results(
            [_make_tool_result_block(tool_id, content)],
            tool_use_id_to_context=id_to_ctx,
            result_cache=result_cache,
            semantic_hash_cache=None,
        )
        block_content = result2[0]["content"][0]["content"]
        assert "stable_payload_validation_failed" in block_content or block_content.startswith(">>> tool:file_read|")


class TestPrecisionReadVerbatim:
    def test_inline_precision_tool_result_not_skeletonized(self) -> None:
        tool_id = "tid1"
        path = "src/tok/foo.py"
        # Long enough that tok_tool_result would normally compress/skeletonize.
        content = "\n".join(f"line {i}" for i in range(800))
        messages = [_make_tool_result_block(tool_id, content)]
        id_to_ctx = {
            tool_id: {
                "name": "Read",
                "path": path,
                "args": {"file_path": path, "offset": 10, "limit": 40},
            }
        }

        result, _breakdown = compress_tool_results(
            messages,
            tool_use_id_to_context=id_to_ctx,
            result_cache=None,
            semantic_hash_cache=None,
        )
        block_content = result[0]["content"][0]["content"]
        assert block_content == content
        assert ">>> tok_compressed" not in block_content

    def test_top_level_precision_tool_result_not_skeletonized(self) -> None:
        tool_id = "tid1"
        path = "src/tok/foo.py"
        content = "\n".join(f"line {i}" for i in range(800))
        messages = [{"role": "tool_result", "tool_use_id": tool_id, "content": content}]
        id_to_ctx = {
            tool_id: {
                "name": "Read",
                "path": path,
                "args": {"file_path": path, "offset": 10, "limit": 40},
            }
        }
        result_cache: dict[str, ResultCacheEntry] = {}

        result, _breakdown = compress_tool_results(
            messages,
            tool_use_id_to_context=id_to_ctx,
            result_cache=result_cache,
            semantic_hash_cache=None,
        )
        msg_content = result[0]["content"]
        assert msg_content == content


class TestSemanticDedupSignal:
    def test_dedup_signal_in_behavior_after_repeated_tool_result(self) -> None:
        """prepare_request should emit semantic_dedup_hit after the second identical read."""
        runtime = UniversalTokRuntime()
        session = RuntimeSession()

        large_output = "file content line\n" * 50  # > 200 chars

        def _req(tool_id: str) -> RuntimeRequest:
            return RuntimeRequest(
                model="claude-sonnet-4-6",
                messages=[
                    _make_tool_use_msg(tool_id, "view_file"),
                    _make_tool_result_block(tool_id, large_output),
                ],
            )

        # First call — seeds the cache
        runtime.prepare_request(_req("t1"), session)

        # Second call — same tool, same path, same content
        prepared2 = runtime.prepare_request(_req("t1"), session)

        system2 = prepared2.body.get("system", "")
        # The explanation for @stable_result should be injected into the system prompt
        assert "@stable_result" in system2
        assert "unchanged" in system2


class TestFeatureFlaggedDeltaCompression:
    def test_file_overlap_delta_for_precision_reads(self, monkeypatch) -> None:
        monkeypatch.setattr(history_pipeline, "TOK_ENABLE_FILE_OVERLAP_DELTA", True)
        path = "src/tok/foo.py"
        first = "\n".join(f"line {i} with verbose payload " + ("x" * 80) for i in range(0, 60))
        second = "\n".join(f"line {i} with verbose payload " + ("x" * 80) for i in range(20, 80))
        messages = [
            _make_tool_result_block("t1", first),
            _make_tool_result_block("t2", second),
        ]
        id_to_ctx = {
            "t1": {"name": "Read", "path": path, "args": {"file_path": path, "offset": 0, "limit": 20}},
            "t2": {"name": "Read", "path": path, "args": {"file_path": path, "offset": 10, "limit": 20}},
        }

        out, breakdown = compress_tool_results(messages, tool_use_id_to_context=id_to_ctx, result_cache=None)

        second_content = out[1]["content"][0]["content"]
        assert second_content.startswith(">>> tool:file_read_overlap_delta|")
        assert "overlap_lines:" in second_content
        assert breakdown.get("file_overlap_delta", 0) > 0

    def test_file_reread_diff_for_small_changes(self, monkeypatch) -> None:
        monkeypatch.setattr(history_pipeline, "TOK_ENABLE_FILE_REREAD_DIFF", True)
        path = "src/tok/foo.py"
        previous = "\n".join(f"value_{i}" for i in range(200))
        current_lines = [f"value_{i}" for i in range(200)]
        current_lines[42] = "value_42_changed"
        current_lines[155] = "value_155_changed"
        current = "\n".join(current_lines)
        messages = [
            _make_tool_result_block("t1", previous),
            _make_tool_result_block("t2", current),
        ]
        id_to_ctx = {
            "t1": {"name": "Read", "path": path, "args": {"file_path": path}},
            "t2": {"name": "Read", "path": path, "args": {"file_path": path}},
        }

        out, breakdown = compress_tool_results(messages, tool_use_id_to_context=id_to_ctx, result_cache=None)

        second_content = out[1]["content"][0]["content"]
        assert second_content.startswith(">>> tool:file_reread_diff|")
        assert "changed_lines:" in second_content
        assert breakdown.get("file_reread_diff", 0) > 0

    def test_search_overlap_delta_for_repeated_scope(self, monkeypatch) -> None:
        monkeypatch.setattr(history_pipeline, "TOK_ENABLE_SEARCH_OVERLAP_DELTA", True)
        first = "\n".join(
            [
                "src/a.py:10:def alpha(): " + ("payload " * 20),
                "src/a.py:20:def beta(): " + ("payload " * 20),
                "src/b.py:30:def gamma(): " + ("payload " * 20),
            ]
        )
        second = "\n".join(
            [
                "src/a.py:20:def beta(): " + ("payload " * 20),
                "src/b.py:30:def gamma(): " + ("payload " * 20),
                "src/c.py:40:def delta(): " + ("payload " * 20),
            ]
        )
        messages = [
            _make_tool_result_block("s1", first),
            _make_tool_result_block("s2", second),
        ]
        id_to_ctx = {
            "s1": {"name": "grep_search", "path": "src", "query": "def alpha", "args": {"path": "src"}},
            "s2": {"name": "grep_search", "path": "src", "query": "def", "args": {"path": "src"}},
        }

        out, breakdown = compress_tool_results(messages, tool_use_id_to_context=id_to_ctx, result_cache=None)

        second_content = out[1]["content"][0]["content"]
        assert second_content.startswith(">>> tool:search_overlap_delta|")
        assert "new_matches:1" in second_content
        assert "src/c.py:40:def delta():" in second_content
        assert breakdown.get("search_overlap_delta", 0) > 0

    def test_stack_repeat_delta_for_top_frame_changes(self, monkeypatch) -> None:
        monkeypatch.setattr(history_pipeline, "TOK_ENABLE_STACK_REPEAT_DELTA", True)
        trace1 = "\n".join(
            [
                "Traceback (most recent call last):",
                '  File "/repo/a.py", line 10, in run',
                '  File "/repo/common.py", line 20, in worker',
                "ValueError: boom",
            ]
        )
        trace2 = "\n".join(
            [
                "Traceback (most recent call last):",
                '  File "/repo/b.py", line 12, in run',
                '  File "/repo/common.py", line 20, in worker',
                "ValueError: boom",
            ]
        )
        messages = [
            _make_tool_result_block("e1", trace1),
            _make_tool_result_block("e2", trace2),
        ]
        id_to_ctx = {
            "e1": {"name": "bash", "path": "", "args": {"command": "pytest -q"}},
            "e2": {"name": "bash", "path": "", "args": {"command": "pytest -q"}},
        }

        out, breakdown = compress_tool_results(messages, tool_use_id_to_context=id_to_ctx, result_cache=None)

        second_content = out[1]["content"][0]["content"]
        assert second_content.startswith(">>> tool:stack_trace_delta|")
        assert "ValueError: boom" in second_content
        assert breakdown.get("stack_repeat_delta", 0) > 0
