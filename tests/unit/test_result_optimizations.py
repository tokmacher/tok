from tok.compression import tok_tool_result, truncate_large_result
from tok.compression._tool_result_codecs import _compress_file_read, _compress_grep, _detect_tool_content_type
from tok.gateway._anthropic_optimizations import (
    _sift_stdout,
    bpe_translate_request,
    scrub_leaked_tok_context,
    sift_tool_results,
)
from tok.runtime.repeat_targets import build_file_summary, build_search_summary
from tok.universal_runtime import RuntimeSession


def test_semantic_truncation() -> None:
    # 50 lines of garbage
    large_text = "\n".join([f"line {i}: content {i * 7}" for i in range(100)])
    # This is roughly 2000 chars
    assert len(large_text) > 1800

    truncated = truncate_large_result(large_text, limit=1200)
    assert len(truncated) < 1500
    assert "... [TRUNCATED" in truncated
    assert "line 0:" in truncated
    assert "line 99:" in truncated


def test_sift_tool_results_records_cache_marked_block_metrics() -> None:
    raw = "\n".join(f"noise line {index}: {'x' * 80}" for index in range(140))
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": raw,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ]
    }
    signals: dict[str, int] = {}

    sift_tool_results(body, behavior_signals=signals)

    compressed = body["messages"][0]["content"][0]["content"]
    assert len(compressed) < len(raw)
    assert signals["tok_sift_cache_marked_blocks"] == 1
    assert signals["tok_sift_cache_marked_block_tokens"] == len(raw) // 4
    assert signals["tok_sift_cache_marked_saved_tokens"] == (len(raw) - len(compressed)) // 4


def test_semantic_truncation_prefers_structure_boundary() -> None:
    lines = [
        "class Example:",
        "    def alpha(self):",
        "        return 1",
    ]
    lines.extend([f"        alpha payload line {i}" for i in range(6)])
    lines.append("")
    lines.append("class NextExample:")
    lines.extend([f"    beta payload line {i}" for i in range(36)])

    large_text = "\n".join(lines)
    assert len(large_text) > 750

    truncated = truncate_large_result(large_text, limit=500)
    assert "... [TRUNCATED" in truncated
    assert "continue at line" in truncated
    assert "\n\n... [TRUNCATED" in truncated


def test_stable_file_summary_prefers_structural_lines() -> None:
    text = "class Example:\n    def resolve(self):\n        logger.debug('noise')\n        value = compute_value()\n        return value\n\ndef helper():\n    pass"

    summary = build_file_summary(text, max_chars=280, max_lines=6)

    assert "class Example:" in summary
    assert "def resolve(self):" in summary
    assert "value = compute_value()" in summary or "return value" in summary
    assert "logger.debug" not in summary


def test_stable_search_summary_prefers_matched_code_lines() -> None:
    text = "src/example.py:3:        logger.debug('noise')\nsrc/example.py:4:    def resolve(self):\nsrc/example.py:5:        result = compute_value()\nsrc/example.py:6:        return result"

    summary = build_search_summary(text, max_chars=280, max_lines=4)

    assert "def resolve(self):" in summary
    assert "result = compute_value()" in summary or "return result" in summary
    assert "logger.debug" not in summary


def test_result_cache_persistence(tmp_path) -> None:
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()

    session = RuntimeSession(memory_dir=memory_dir)
    session.result_cache = {"some_hash": ("h1", "old content")}
    session._save_result_cache()

    # Reload in new session — legacy 2-tuples are upgraded to dict format
    session2 = RuntimeSession(memory_dir=memory_dir)
    entry = session2.result_cache["some_hash"]
    assert isinstance(entry, dict)
    assert entry["hash"] == "h1"
    assert entry["raw"] == "old content"
    assert entry["first_read_complete"] is False
    assert isinstance(entry["timestamp"], float)


def test_restored_file_cache_entry_serves_one_exact_read_before_stub() -> None:
    from tok.compression import _apply_result_cache

    raw = "\n".join(f"line {i}: stable restored content" for i in range(80))
    context = {"name": "read", "args": {"path": "src/example.py"}}
    cache = {}

    first, _ = _apply_result_cache(raw, context, cache)
    assert first == raw
    cache_key = next(iter(cache))
    cache[cache_key]["first_read_complete"] = False

    restored_first, saved = _apply_result_cache(raw, context, cache)
    assert restored_first == raw
    assert saved == 0
    assert cache[cache_key]["first_read_complete"] is True

    restored_second, saved = _apply_result_cache(raw, context, cache)
    assert restored_second.startswith(">>> tool:read|unchanged|cached")
    assert saved > 0


def test_replayed_tok_cache_stub_serves_cached_raw_content() -> None:
    from tok.compression import _apply_result_cache

    raw = "\n".join(f"line {i}: stable restored content" for i in range(80))
    context = {"name": "read", "args": {"path": "src/example.py"}}
    cache = {}

    first, _ = _apply_result_cache(raw, context, cache)
    assert first == raw
    cache_key = next(iter(cache))
    cache[cache_key]["first_read_complete"] = True

    stub = (
        ">>> tool:read|unchanged|cached|confidence:exact|reason:hash_match|path:src/example.py\n"
        "@stable_hint |> File unchanged - summary shown to save tokens. Read offset=1 for full content."
    )

    replayed, saved = _apply_result_cache(stub, context, cache)

    assert replayed == raw
    assert saved == 0


def test_repeated_safe_command_result_uses_stable_cache_stub() -> None:
    from tok.compression import _apply_result_cache

    raw = "\n".join(f"tests/unit/test_result_optimizations.py::test_{i} PASSED" for i in range(80))
    context = {"name": "bash", "args": {"command": "uv run pytest -q"}}
    cache = {}

    first, first_saved = _apply_result_cache(raw, context, cache)
    assert "unchanged|cached" not in first
    assert first_saved > 0

    second, second_saved = _apply_result_cache(raw, context, cache)
    assert second.startswith(">>> tool:bash|unchanged|cached|")
    assert "command:pytest -q" in second
    assert second_saved > 0


def test_changed_safe_command_result_updates_cache_without_stale_stub() -> None:
    from tok.compression import _apply_result_cache

    context = {"name": "bash", "args": {"command": "git diff -- src/tok/foo.py"}}
    cache = {}
    raw_1 = "diff --git a/src/tok/foo.py b/src/tok/foo.py\n" + "\n".join(f"+first {i}" for i in range(80))
    raw_2 = "diff --git a/src/tok/foo.py b/src/tok/foo.py\n" + "\n".join(f"+second {i}" for i in range(80))

    _apply_result_cache(raw_1, context, cache)
    changed, _ = _apply_result_cache(raw_2, context, cache)
    assert "unchanged|cached" not in changed
    assert "second" in next(iter(cache.values()))["raw"]

    repeated, saved = _apply_result_cache(raw_2, context, cache)
    assert repeated.startswith(">>> tool:bash|unchanged|cached|")
    assert saved > 0


def test_apply_result_cache_command_stored_signal() -> None:
    from tok.compression import _apply_result_cache

    raw = "\n".join(f"src/tok/foo.py:{i}: match" for i in range(80))
    context = {"name": "bash", "args": {"command": "rg 'pattern' src/tok"}}
    cache: dict[str, object] = {}
    signals: dict[str, int] = {}

    _apply_result_cache(raw, context, cache, cache_breakdown=signals)
    assert signals.get("command_cache_stored", 0) == 1
    assert "command_cache_hit" not in signals


def test_apply_result_cache_command_hit_signal() -> None:
    from tok.compression import _apply_result_cache

    raw = "\n".join(f"src/tok/foo.py:{i}: match" for i in range(80))
    context = {"name": "bash", "args": {"command": "rg 'pattern' src/tok"}}
    cache: dict[str, object] = {}

    _apply_result_cache(raw, context, cache)

    signals: dict[str, int] = {}
    _apply_result_cache(raw, context, cache, cache_breakdown=signals)
    assert signals.get("command_cache_hit", 0) >= 1
    assert "command_cache_stored" not in signals


def test_apply_result_cache_ineligible_command_signal() -> None:
    from tok.compression import _apply_result_cache

    raw = "some output"
    context = {"name": "bash", "args": {"command": "rm -rf /tmp/test"}}
    cache: dict[str, object] = {}
    signals: dict[str, int] = {}

    _apply_result_cache(raw, context, cache, cache_breakdown=signals)
    assert signals.get("command_cache_skip_ineligible_cmd", 0) == 1


def test_apply_result_cache_changed_output_signal() -> None:
    from tok.compression import _apply_result_cache

    context = {"name": "bash", "args": {"command": "git diff -- src/tok/foo.py"}}
    cache: dict[str, object] = {}
    raw_1 = "\n".join(f"+first {i}" for i in range(80))
    raw_2 = "\n".join(f"+second {i}" for i in range(80))

    signals: dict[str, int] = {}
    _apply_result_cache(raw_1, context, cache, cache_breakdown=signals)
    assert signals.get("command_cache_stored", 0) == 1

    signals.clear()
    _apply_result_cache(raw_2, context, cache, cache_breakdown=signals)
    assert signals.get("command_cache_replaced_changed", 0) == 1


def test_compress_tool_results_command_cache_hit_signals() -> None:
    from tok.compression import compress_tool_results

    raw = "\n".join(f"src/tok/foo.py:{i}: match" for i in range(80))
    tool_use_id_1 = "toolu_001"
    tool_use_id_2 = "toolu_002"
    context = {"name": "bash", "args": {"command": "rg 'pattern' src/tok"}}

    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": tool_use_id_1, "name": "bash", "input": context["args"]}],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use_id_1, "content": raw}]},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": tool_use_id_2, "name": "bash", "input": context["args"]}],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use_id_2, "content": raw}]},
    ]

    id_to_context = {tool_use_id_1: context, tool_use_id_2: context}
    result_cache: dict[str, object] = {}

    _, breakdown = compress_tool_results(
        messages,
        result_cache=result_cache,
        tool_use_id_to_context=id_to_context,
    )

    assert breakdown.get("command_cacheable_seen", 0) >= 2
    assert breakdown.get("command_cache_stored", 0) >= 1
    assert breakdown.get("command_cache_hit", 0) >= 1
    assert breakdown.get("command_cache_reached_apply", 0) >= 1


def test_compress_tool_results_command_cache_signals_combined() -> None:
    from tok.compression import compress_tool_results

    raw = "\n".join(f"src/tok/foo.py:{i}: match" for i in range(80))
    tool_use_id_1 = "toolu_001"
    tool_use_id_2 = "toolu_002"
    context = {"name": "bash", "args": {"command": "rg 'pattern' src/tok"}}

    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": tool_use_id_1, "name": "bash", "input": context["args"]}],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use_id_1, "content": raw}]},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": tool_use_id_2, "name": "bash", "input": context["args"]}],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use_id_2, "content": raw}]},
    ]

    id_to_context = {tool_use_id_1: context, tool_use_id_2: context}
    result_cache: dict[str, object] = {}

    _, breakdown = compress_tool_results(
        messages,
        result_cache=result_cache,
        tool_use_id_to_context=id_to_context,
    )

    assert breakdown.get("command_cacheable_seen", 0) >= 2, f"Expected >=2 command_cacheable_seen, got {breakdown}"
    assert breakdown.get("command_cache_stored", 0) >= 1, f"Expected >=1 stored, got {breakdown}"
    assert breakdown.get("command_cache_hit", 0) >= 1, f"Expected >=1 hit, got {breakdown}"
    assert breakdown.get("command_cache_reached_apply", 0) >= 1, f"Expected >=1 reached_apply, got {breakdown}"


def test_compress_tool_results_repeated_pass_no_replaced_changed() -> None:
    """compress_tool_results called N times on the same mutated messages must not fire
    command_cache_replaced_changed.  The second pass sees already-compressed content
    (starts with '>>> tool:'); it should pass through unchanged instead of overwriting
    the cache entry with the compressed form and breaking subsequent real hits.
    """
    from tok.compression import compress_tool_results

    raw = "\n".join(f"src/tok/foo.py:{i}: match_token" for i in range(80))
    tool_use_id = "toolu_001"
    context = {"name": "bash", "args": {"command": "rg 'match_token' src/tok"}}

    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": tool_use_id, "name": "bash", "input": context["args"]}],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": raw}]},
    ]
    id_to_context = {tool_use_id: context}
    result_cache: dict[str, object] = {}
    first_exact_evidence_seen: set[str] = set()

    # Pass 1 — stores the entry and compresses the block in-place
    _, bd1 = compress_tool_results(
        messages,
        result_cache=result_cache,
        tool_use_id_to_context=id_to_context,
        first_exact_evidence_seen=first_exact_evidence_seen,
    )
    assert bd1.get("command_cache_stored", 0) >= 1, f"Pass 1 should store: {bd1}"
    assert bd1.get("command_cache_replaced_changed", 0) == 0, f"Pass 1 must not replace: {bd1}"

    # Pass 2 — block content is now tok-compressed; must not fire replaced_changed
    _, bd2 = compress_tool_results(
        messages,
        result_cache=result_cache,
        tool_use_id_to_context=id_to_context,
        first_exact_evidence_seen=first_exact_evidence_seen,
    )
    assert bd2.get("command_cache_replaced_changed", 0) == 0, (
        f"Pass 2 must not fire replaced_changed on already-compressed block: {bd2}"
    )

    # Pass 3 — same invariant
    _, bd3 = compress_tool_results(
        messages,
        result_cache=result_cache,
        tool_use_id_to_context=id_to_context,
        first_exact_evidence_seen=first_exact_evidence_seen,
    )
    assert bd3.get("command_cache_replaced_changed", 0) == 0, f"Pass 3 must not fire replaced_changed: {bd3}"


def test_safe_command_classifier_allows_common_read_only_commands() -> None:
    from tok.compression import _apply_result_cache

    commands = [
        "rg pattern src/tok",
        'rg -n "command_result_cache_hit|command_result_cache_saved_tokens" src/tok tests',
        "grep -R pattern src/tok",
        "sed -n '1,80p' src/tok/foo.py",
        "git diff -- src/tok/foo.py",
    ]
    raw = "\n".join(f"src/tok/foo.py:{i}: pattern" for i in range(50))

    for command in commands:
        cache = {}
        context = {"name": "bash", "args": {"command": command}}
        _apply_result_cache(raw, context, cache)
        repeated, saved = _apply_result_cache(raw, context, cache)
        assert "unchanged|cached" in repeated
        assert saved > 0


def test_unsafe_or_failed_command_results_are_not_cached() -> None:
    from tok.compression import _apply_result_cache

    cases = [
        ("rg pattern src/tok | head", "src/tok/foo.py:1:pattern"),
        ("rg pattern* src/tok", "src/tok/foo.py:1:pattern"),
        ("git add src/tok/foo.py", "added"),
        ("rm -f src/tok/foo.py", "removed"),
        ("uv run pytest -q", "FAILED tests/unit/test_x.py::test_x"),
        ("unknown-tool --version", "unknown output"),
    ]

    for command, raw in cases:
        cache = {}
        context = {"name": "bash", "args": {"command": command}}
        _apply_result_cache(raw, context, cache)
        repeated, _ = _apply_result_cache(raw, context, cache)
        assert "unchanged|cached" not in repeated
        assert cache == {}


def test_tok_tool_result_applies_truncation() -> None:
    # Long result that doesn't match any specific compressor
    long_raw = "A" * 5000
    compressed = tok_tool_result(long_raw)

    # It should be truncated because it's not a known type and it's long
    assert len(compressed) < 2000
    assert "... [TRUNCATED" in compressed


def test_truncate_large_result_preserves_small_files() -> None:
    """Files with < 100 lines should return full content regardless of char count."""
    # 77 lines, ~2000 chars - should NOT be truncated
    lines = [f"def method_{i}(self):" for i in range(77)]
    small_file = "\n".join(lines)
    assert len(small_file) > 1200  # Exceeds char threshold
    assert small_file.count("\n") + 1 < 100  # Below line threshold

    truncated = truncate_large_result(small_file, limit=1200)

    # Should return full content, not truncated
    assert "... [TRUNCATED" not in truncated
    assert truncated == small_file


def test_truncate_large_result_truncates_large_files() -> None:
    """Files with >= 100 lines should still be truncated."""
    # 150 lines - should be truncated
    lines = [f"line {i}: content {i}" for i in range(150)]
    large_file = "\n".join(lines)
    assert len(large_file) > 1200
    assert large_file.count("\n") + 1 >= 100

    truncated = truncate_large_result(large_file, limit=1200)

    # Should be truncated
    assert "... [TRUNCATED" in truncated


def test_truncate_large_result_respects_already_compressed_flag() -> None:
    """Pre-compressed (e.g., skeletonized) output should not be truncated again."""
    large_text = "A" * 5000

    truncated = truncate_large_result(large_text, limit=1200)
    assert "... [TRUNCATED" in truncated

    preserved = truncate_large_result(large_text, limit=1200, already_compressed=True)
    assert preserved == large_text


def test_grep_shows_multiple_snippets_per_file() -> None:
    """Small result sets (≤20 matches) return verbatim — no compression overhead."""
    grep_output = "\n".join(
        [
            "src/main.py:10:def foo():",
            "src/main.py:20:def bar():",
            "src/main.py:30:def baz():",
            "src/main.py:40:def qux():",
            "src/other.py:5:def hello():",
            "src/other.py:15:def world():",
        ]
    )

    compressed = _compress_grep(grep_output)

    # With 6 total matches (≤20), compression adds no value — return verbatim
    assert compressed == grep_output


def test_grep_shows_other_snippets_explicitly() -> None:
    """Small result sets (≤20 matches) return verbatim — no compression overhead."""
    grep_output = "\n".join(
        [
            "src/main.py:10:def foo():",
            "unusual line without colon format",
            "another unusual line",
            "third unusual line",
            "fourth unusual line",
            "fifth unusual line",
        ]
    )

    compressed = _compress_grep(grep_output)

    # With 6 total matches (≤20), compression adds no value — return verbatim
    assert compressed == grep_output


def test_grep_collapses_large_result_sets() -> None:
    """Large grep results (>50 matches) should still collapse per-file snippets."""
    lines = []
    for i in range(60):
        lines.append(f"src/main.py:{i * 10}:def method_{i}():")
    for i in range(10):
        lines.append(f"src/other.py:{i * 5}:def helper_{i}():")
    grep_output = "\n".join(lines)

    compressed = _compress_grep(grep_output)

    # 70 total matches (>50) → 3 per file limit
    assert "more matches" in compressed
    # Should show first 3 snippets for main.py
    assert "def method_0():" in compressed
    assert "def method_1():" in compressed
    assert "def method_2():" in compressed
    # No advisory footer
    assert "tok advisory" not in compressed


def test_files_only_grep_not_misclassified_as_ls() -> None:
    """Files-only grep results preserve grep semantics without lossy compression."""
    # This simulates the second grep in the audit which returned just file paths
    files_only_grep = "\n".join(
        [
            "src/tok/memory/pointers.py",
            "src/tok/memory/__init__.py",
            "src/tok/runtime/memory/bridge_memory.py",
            "tests/unit/test_memory_pointers.py",
            "src/tok/runtime/_session_persistence.py",
            "src/tok/runtime/memory/__init__.py",
            "src/tok/memory/__init__.py",
            "src/tok/runtime/memory/__init__.py",
        ]
    )

    # Heuristics may classify path-only output as ls-like; verify behavior remains safe.
    kind = _detect_tool_content_type(files_only_grep)
    assert kind in {"grep", "ls"}

    # For small result sets, grep compression should preserve verbatim content.
    compressed = _compress_grep(files_only_grep)
    assert compressed == files_only_grep


def test_compress_file_read_preserves_small_files() -> None:
    """Small files (≤100 lines, ≤5000 chars) must not be skeletonized."""
    # Simulate a file like pointers.py: 77 lines, ~2.5KB
    lines = ["class PointerRegistry:"]
    lines.extend(f"    def method_{i}(self):" for i in range(10))
    lines.extend(f"        return self._data[{i}]" for i in range(10))
    lines.extend(f"    # comment {i}" for i in range(56))
    small_file = "\n".join(lines)
    assert small_file.count("\n") + 1 <= 100
    assert len(small_file) <= 5000

    compressed = _compress_file_read(small_file)

    # Must return the original verbatim — no skeleton stub
    assert compressed == small_file
    assert ">>>" not in compressed


def test_compress_file_read_skeletonizes_large_files() -> None:
    """Large files with heavy bodies should still be skeletonized normally."""
    lines = [f"def method_{i}(self):" for i in range(20)]
    # Add lots of body lines that the skeletonizer will collapse
    lines.extend(f"    x = {i} + {i * 2}" for i in range(200))
    lines.extend(f"    y = self.data[{i}]" for i in range(200))
    large_file = "\n".join(lines)
    assert large_file.count("\n") + 1 > 100
    assert len(large_file) > 5000

    compressed = _compress_file_read(large_file)

    # Should be skeletonized (shorter than original)
    assert len(compressed) < len(large_file)


def test_sift_stdout_preserves_small_files() -> None:
    """Small files (≤100 lines, ≤10000 chars) must not be truncated by gateway sifting."""
    lines = ["class PointerRegistry:"]
    lines.extend(f"    def method_{i}(self):" for i in range(10))
    lines.extend(f"        return self._data[{i}]" for i in range(10))
    lines.extend(f"    # comment {i}" for i in range(56))
    small_file = "\n".join(lines)
    assert small_file.count("\n") + 1 <= 100
    assert len(small_file) <= 5000

    result = _sift_stdout(small_file)

    # Must return the original verbatim — no truncation or skeletonization
    assert result == small_file


def test_sift_stdout_preserves_large_code() -> None:
    """Gateway sifting must not skeletonize possible first-read code."""
    lines = [f"def method_{i}(self):" for i in range(20)]
    lines.extend(f"    x = {i} + {i * 2}" for i in range(200))
    lines.extend(f"    y = self.data[{i}]" for i in range(200))
    large_code = "\n".join(lines)
    assert len(large_code) > 3000

    result = _sift_stdout(large_code)

    assert result == large_code


def test_sift_stdout_skips_already_compressed() -> None:
    """Already-compressed Tok content (>>> prefix) must pass through unchanged."""
    compressed = ">>> tool:file_read|original_chars:5000|skeleton_lines:10\nimport os\ndef foo():"
    result = _sift_stdout(compressed)
    assert result == compressed


def test_bpe_translate_request_ignores_traceback_with_stray_tok_pipe() -> None:
    leaked_context = (
        "gitStatus:\n"
        "Traceback (most recent call last):\n"
        '  File "src/tok/runtime/__init__.py", line 45, in __getattr__\n'
        "    |> AttributeError: missing field from compressed annotation\n"
    )
    body = {"messages": [{"role": "user", "content": leaked_context}]}

    result = bpe_translate_request(body)

    assert result["messages"][0]["content"] == leaked_context


def test_bpe_translate_request_still_translates_tok_wire() -> None:
    tok_text = ">>> t:1|state:active\n@msg role:assistant\n  |> ok"
    body = {"messages": [{"role": "assistant", "content": [{"type": "text", "text": tok_text}]}]}

    result = bpe_translate_request(body)
    translated = result["messages"][0]["content"][0]["text"]

    assert "state:active" in translated
    assert "  > ok" in translated


def test_scrub_leaked_tok_context_removes_gitstatus_pointer_dump() -> None:
    leaked = (
        "gitStatus:\n"
        "commit 65300c0 Fix bridge handling\n"
        "@pointers\n"
        "  *A -> src/tok/runtime/core.py\n"
        "  *B -> tests/unit/test_gateway.py\n"
        ">>> t:3027|g:_CLAUDE_SONNET = ModelProfile(...)|s:drift_healed|k:answer_verification:health\n"
        "file[src/tok/runtime/core.py]:50|digest|~120\n"
        " M src/tok/gateway/__init__.py\n"
    )
    body = {"system": [{"type": "text", "text": leaked}]}

    result = scrub_leaked_tok_context(body)
    cleaned = result["system"][0]["text"]

    assert "commit 65300c0" in cleaned
    assert "M src/tok/gateway/__init__.py" in cleaned
    assert "@pointers" not in cleaned
    assert ">>>" not in cleaned
    assert "answer_verification" not in cleaned
    assert "file[src/tok/runtime/core.py]" not in cleaned


def test_scrub_leaked_tok_context_preserves_legitimate_state_without_leak_signature() -> None:
    state = ">>> goal:fix_gateway|turns:3\n"
    body = {"system": state}

    result = scrub_leaked_tok_context(body)

    assert result["system"] == state


def test_scrub_leaked_tok_context_removes_isolated_state_leak() -> None:
    leaked = ">>> t:3027|g:_CLAUDE_SONNET|s:drift_healed|k:answer_file:src/tok/gateway.py\n"
    body = {"system": leaked}

    result = scrub_leaked_tok_context(body)

    assert ">>>" not in result["system"]
    assert "answer_file" not in result["system"]


def test_scrub_leaked_tok_context_drops_empty_text_blocks() -> None:
    body = {
        "system": [
            {"type": "text", "text": "@pointers\n  *A -> src/tok/runtime/core.py\n"},
            {"type": "text", "text": "normal instructions"},
        ]
    }

    result = scrub_leaked_tok_context(body)

    assert result["system"] == [{"type": "text", "text": "normal instructions"}]


def test_cache_hit_preserves_small_files() -> None:
    """First read stays verbatim; repeat read emits guided stable stub."""
    from tok.compression import _apply_result_cache

    # Simulate a small file like test_memory_pointers.py: 65 lines, ~2400 chars
    lines = ["from tok.memory.pointers import PointerRegistry"]
    lines.extend(f"    def test_method_{i}(self):" for i in range(10))
    lines.extend(f"        assert reg.get_pointer('path_{i}') == '*A'" for i in range(10))
    lines.extend(f"    # comment {i}" for i in range(44))
    small_file = "\n".join(lines)
    assert small_file.count("\n") + 1 <= 100
    assert len(small_file) <= 5000

    context = {"name": "view_file", "path": "test_memory_pointers.py", "args": {"path": "test_memory_pointers.py"}}
    cache: dict = {}

    # First call — populates cache
    first_result, _first_saved = _apply_result_cache(small_file, context, cache)
    # Second call — first cache hit delivers verbatim and marks as fully delivered.
    second_result, _second_saved = _apply_result_cache(small_file, context, cache)
    # Third call — file is now fully delivered; emits the stable guided stub.
    third_result, _third_saved = _apply_result_cache(small_file, context, cache)

    assert first_result == small_file
    assert ">>>" not in first_result
    assert second_result == small_file
    assert ">>>" not in second_result
    assert third_result.startswith(">>> tool:view_file|unchanged|cached|")
    assert "Read offset=1 for full content" in third_result
    assert "@stable_skeleton |>" in third_result


def test_compress_file_read_preserves_medium_small_slices() -> None:
    """Targeted code slices (≤100 lines, ≤5000 chars) must not be skeletonized.

    This covers the case where a Bash sed/cat output of a 90-line slice
    from a large file (e.g., bridge_memory.py lines 590-680) should be
    preserved verbatim, not skeletonized.
    """
    lines = [f"def method_{i}(self):" for i in range(15)]
    lines.extend(f"    x = self.data[{i}] + {i * 2} + self.offset[{i}]" for i in range(40))
    lines.extend(f"    y = self.other[{i}] + self.buffer[{i}]" for i in range(35))
    medium_slice = "\n".join(lines)
    assert medium_slice.count("\n") + 1 <= 100
    assert 3000 < len(medium_slice) <= 5000  # Between old and new threshold

    compressed = _compress_file_read(medium_slice)

    # Must return the original verbatim — no skeleton stub
    assert compressed == medium_slice
    assert ">>>" not in compressed


def test_medium_small_file_preservation() -> None:
    """Files between 5000-10000 chars must not be skeletonized (threshold raised to 10000)."""
    # Generate ~90 lines, ~8000 chars - should now be preserved verbatim
    lines = ["class MediumClass:"]
    lines.extend(f"    def method_{i}(self, x: int) -> int:" for i in range(20))
    lines.extend(f"        return x + {i}" for i in range(20))
    lines.extend(
        f"    # This is a long comment to add characters and make it meaningfully longer than the legacy threshold {i}"
        for i in range(55)
    )
    medium_slice = "\n".join(lines)
    assert medium_slice.count("\n") + 1 <= 100
    assert 5000 < len(medium_slice) <= 10000  # Between old and new threshold

    compressed = _compress_file_read(medium_slice)

    # Must return the original verbatim — no skeleton stub
    assert compressed == medium_slice
    assert ">>>" not in compressed


def test_precision_read_not_compressed() -> None:
    """Offset/limit based reads (precision reads) must never be compressed."""
    # Create content that would normally be compressed (>10000 chars or >100 lines)
    lines = ["class LargeClass:"]
    lines.extend(f"    def method_{i}(self, x: int) -> int:" for i in range(220))
    lines.extend(f"        return x + {i}  # keep this line intentionally verbose for size" for i in range(220))
    large_content = "\n".join(lines)
    assert len(large_content) > 10000

    # With offset in tool_context, should return verbatim (precision read)
    tool_context = {"args": {"offset": 100, "limit": 50}}
    compressed_with_offset = _compress_file_read(large_content, tool_context=tool_context)
    assert compressed_with_offset == large_content

    # With limit only, should also return verbatim
    tool_context_limit = {"args": {"limit": 100}}
    compressed_with_limit = _compress_file_read(large_content, tool_context=tool_context_limit)
    assert compressed_with_limit == large_content

    # With start/end, should also return verbatim
    tool_context_start_end = {"args": {"start": 50, "end": 200}}
    compressed_with_start_end = _compress_file_read(large_content, tool_context=tool_context_start_end)
    assert compressed_with_start_end == large_content


def test_grep_with_valueerror_not_error_stub() -> None:
    """Grep results containing 'ValueError' must not be misclassified as error stubs."""
    from tok.compression import _apply_result_cache

    # Grep output that contains "ValueError" (common in source code)
    grep_output = """src/errors.py:10:raise ValueError("Invalid input")
src/errors.py:20:except ValueError as e:
src/main.py:5:from errors import ValueError
"""

    context = {"name": "grep_search", "args": {"pattern": "ValueError"}}
    result_cache = {}

    # Should NOT be converted to error stub
    compressed, _ = _apply_result_cache(grep_output, context, result_cache)
    assert "|err:value_error|" not in compressed
    assert "ValueError" in compressed  # Original content preserved


def test_legacy_1tuple_cache_entry_requires_verbatim_first_read() -> None:
    """Legacy 1-tuple cache entries (hashonly) must have first_read_complete=False.

    This ensures that old cache files with 1-tuple entries still grant
    the first-read guarantee and deliver content verbatim before serving stubs.
    """
    from tok.compression import _unpack_cache_entry

    entry_1tuple = ("abc123def456",)
    hash_val, raw_val, ts, first_read_complete, version = _unpack_cache_entry(entry_1tuple)

    assert hash_val == "abc123def456"
    assert raw_val == ""
    assert first_read_complete is False, (
        "Legacy 1-tuple entries must require verbatim first read (first_read_complete=False)"
    )
    assert version == 1


def test_legacy_2tuple_cache_entry_requires_verbatim_first_read() -> None:
    """Legacy 2-tuple cache entries must have first_read_complete=False."""
    from tok.compression import _unpack_cache_entry

    entry_2tuple = ("abc123def456", "raw file content here")
    hash_val, raw_val, ts, first_read_complete, version = _unpack_cache_entry(entry_2tuple)

    assert hash_val == "abc123def456"
    assert raw_val == "raw file content here"
    assert first_read_complete is False, (
        "Legacy 2-tuple entries must require verbatim first read (first_read_complete=False)"
    )
    assert version == 2


def test_legacy_3tuple_cache_entry_requires_verbatim_first_read() -> None:
    """Legacy 3-tuple cache entries must have first_read_complete=False."""
    from tok.compression import _unpack_cache_entry

    entry_3tuple = ("abc123def456", "raw file content here", 1234567890.0)
    hash_val, raw_val, ts, first_read_complete, version = _unpack_cache_entry(entry_3tuple)

    assert hash_val == "abc123def456"
    assert raw_val == "raw file content here"
    assert ts == 1234567890.0
    assert first_read_complete is False, (
        "Legacy 3-tuple entries must require verbatim first read (first_read_complete=False)"
    )
    assert version == 3
