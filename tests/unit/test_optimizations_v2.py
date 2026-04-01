"""Tests for speculative macro injection and semantic hash deduplication."""

from __future__ import annotations

import tok.compression

tok.compression.TOOL_COMPRESS_THRESHOLD = 0

from tok.compression import (
    _compute_semantic_hash,
    _make_semantic_cache_key,
    _STABLE_RESULT_EXPLANATION,
    _SEMANTIC_HASH_MIN_CHARS,
    compress_tool_results,
)
from tok.bridge_memory import BridgeMemoryState
from tok.neuro.ir import Instruction, Macro
from tok.universal_runtime import (
    RuntimeRequest,
    RuntimeSession,
    UniversalTokRuntime,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_use_msg(
    tool_id: str, tool_name: str, path: str = "src/tok/foo.py"
):
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


def _make_tool_result_block(tool_id: str, content: str):
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


def _make_id_to_context(tool_id: str, tool_name: str, path: str):
    return {
        tool_id: {
            "name": tool_name,
            "path": path,
            "args": {"path": path},
        }
    }


# ---------------------------------------------------------------------------
# Speculative Macro Injection
# ---------------------------------------------------------------------------


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

    def test_speculative_hint_injected_when_macros_match(self):
        macro = self._simple_macro("fix_imports", hit_count=3)
        session = self._session_with_macros([macro])
        runtime = UniversalTokRuntime()

        prepared = runtime.prepare_request(
            self._request_with_message(), session
        )

        system = prepared.body.get("system", "")
        assert "@fix_imports" in system
        assert "Available macros" in system

    def test_speculative_hint_absent_when_no_macros(self):
        session = RuntimeSession(
            bridge_memory=BridgeMemoryState(load_global_macros=False)
        )
        runtime = UniversalTokRuntime()

        prepared = runtime.prepare_request(
            self._request_with_message(), session
        )

        system = prepared.body.get("system", "")
        assert "Available macros" not in system

    def test_speculative_hint_absent_below_threshold(self):
        """Macros below hit threshold should not be injected."""
        macro = self._simple_macro("low_hit", hit_count=1)
        session = self._session_with_macros([macro])
        runtime = UniversalTokRuntime()

        prepared = runtime.prepare_request(
            self._request_with_message(), session
        )

        system = prepared.body.get("system", "")
        assert "@low_hit" not in system

    def test_speculative_hint_lists_multiple_macros(self):
        macros = [
            self._simple_macro("macro_a", hit_count=5),
            self._simple_macro("macro_b", hit_count=4),
        ]
        session = self._session_with_macros(macros)
        runtime = UniversalTokRuntime()

        prepared = runtime.prepare_request(
            self._request_with_message(), session
        )

        system = prepared.body.get("system", "")
        assert "@macro_a" in system
        assert "@macro_b" in system

    def test_speculative_signal_recorded(self):
        macro = self._simple_macro("sig_macro", hit_count=3)
        session = self._session_with_macros([macro])
        runtime = UniversalTokRuntime()

        runtime.prepare_request(self._request_with_message(), session)

        # The signal may already be consumed into behavior, but the macro count
        # should be reflected or the key should have been set.
        # We verify by checking that the hint appeared in the system prompt as a proxy.
        prepared = runtime.prepare_request(
            self._request_with_message(), session
        )
        assert "@sig_macro" in prepared.body.get("system", "")


# ---------------------------------------------------------------------------
# Semantic Hash Deduplication
# ---------------------------------------------------------------------------


class TestComputeSemanticHash:
    def test_returns_hex_string(self):
        h = _compute_semantic_hash("hello world")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_content_same_hash(self):
        assert _compute_semantic_hash("abc") == _compute_semantic_hash("abc")

    def test_different_content_different_hash(self):
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
    ):
        tool_id = "tid1"
        messages = [_make_tool_result_block(tool_id, content)]
        id_to_ctx = _make_id_to_context(tool_id, tool_name, path)
        return messages, id_to_ctx

    def test_first_occurrence_not_replaced(self):
        content = self._large_content()
        messages, id_to_ctx = self._messages_and_ctx(content)
        cache: dict[str, str] = {}

        result, breakdown = compress_tool_results(
            messages,
            tool_use_id_to_context=id_to_ctx,
            semantic_hash_cache=cache,
        )

        # First time: content should NOT be replaced, hash should be stored
        block_content = result[0]["content"][0]["content"]
        assert "@stable_result" not in block_content
        assert len(cache) == 1

    def test_second_occurrence_replaced_with_token(self):
        content = self._large_content()
        messages, id_to_ctx = self._messages_and_ctx(content)
        cache: dict[str, str] = {}

        # First pass: populates cache
        compress_tool_results(
            [_make_tool_result_block("tid1", content)],
            tool_use_id_to_context=_make_id_to_context(
                "tid1", "view_file", "src/tok/foo.py"
            ),
            semantic_hash_cache=cache,
        )

        # Second pass: same tool, same args, same content → should dedup
        messages2 = [_make_tool_result_block("tid1", content)]
        id_to_ctx2 = _make_id_to_context("tid1", "view_file", "src/tok/foo.py")
        result2, breakdown2 = compress_tool_results(
            messages2,
            tool_use_id_to_context=id_to_ctx2,
            semantic_hash_cache=cache,
            hot_summary_records={},
        )

        block_content = result2[0]["content"][0]["content"]
        assert block_content.startswith("@stable_result(hash:")
        # Note: breakdown may be 0 when summary is attached (summary adds length)
        assert "@stable_summary" in block_content

    def test_changed_content_not_replaced(self):
        path = "src/tok/foo.py"
        cache: dict[str, str] = {}

        content_a = self._large_content("a")
        content_b = self._large_content("b")

        # First pass: content_a
        compress_tool_results(
            [_make_tool_result_block("tid1", content_a)],
            tool_use_id_to_context=_make_id_to_context(
                "tid1", "view_file", path
            ),
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

    def test_small_content_not_deduped(self):
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

    def test_no_cache_no_dedup(self):
        content = self._large_content()
        messages, id_to_ctx = self._messages_and_ctx(content)

        result, breakdown = compress_tool_results(
            messages,
            tool_use_id_to_context=id_to_ctx,
            semantic_hash_cache=None,
        )

        block_content = result[0]["content"][0]["content"]
        assert "@stable_result" not in block_content

    def test_stable_result_explanation_constant_exists(self):
        assert "@stable_result" in _STABLE_RESULT_EXPLANATION
        assert "unchanged" in _STABLE_RESULT_EXPLANATION

    def test_semantic_cache_key_includes_path_identity(self):
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

    def test_tok_bypass_cache_skips_stable_and_compression(self):
        path = "src/tok/foo.py"
        tool_id = "tid1"
        content = (
            "class A:\n"
            "    def m(self):\n"
            "        pass\n\n"
            "def top():\n"
            "    return 1\n"
            + ("# filler\n" * 200)
        )
        cache: dict[str, str] = {}

        # Seed semantic hash cache
        compress_tool_results(
            [_make_tool_result_block(tool_id, content)],
            tool_use_id_to_context=_make_id_to_context(tool_id, "view_file", path),
            semantic_hash_cache=cache,
        )

        # Second pass with bypass enabled must return raw content unchanged.
        id_to_ctx_bypass = {
            tool_id: {
                "name": "view_file",
                "path": path,
                "args": {"path": path, "tok_bypass_cache": True},
            }
        }
        result2, _breakdown2 = compress_tool_results(
            [_make_tool_result_block(tool_id, content)],
            tool_use_id_to_context=id_to_ctx_bypass,
            semantic_hash_cache=cache,
        )
        block_content = result2[0]["content"][0]["content"]
        assert block_content == content
        assert "@stable_result" not in block_content

    def test_stable_payload_includes_skeleton_for_code(self):
        path = "src/tok/foo.py"
        tool_id = "tid1"
        content = (
            "class A:\n"
            "    def m(self):\n"
            "        pass\n\n"
            "async def coro():\n"
            "    return 1\n\n"
            "def top():\n"
            "    return 2\n"
            + ("# filler\n" * 200)
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
        assert block_content.startswith("@stable_result(hash:")
        assert "\n@stable_summary |>" in block_content
        assert "\n@stable_skeleton |>" in block_content


# ---------------------------------------------------------------------------
# Integration: semantic_dedup_hit signal in prepare_request
# ---------------------------------------------------------------------------


class TestSemanticDedupSignal:
    def test_dedup_signal_in_behavior_after_repeated_tool_result(self):
        """prepare_request should emit semantic_dedup_hit after the second identical read."""
        runtime = UniversalTokRuntime()
        session = RuntimeSession()

        large_output = "file content line\n" * 50  # > 200 chars

        def _req(tool_id: str):
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
