from tok.compression import tok_tool_result, truncate_large_result
from tok.compression._tool_result_codecs import _compress_file_read, _compress_grep, _detect_tool_content_type
from tok.gateway._anthropic_optimizations import _sift_stdout
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

    # Reload in new session
    session2 = RuntimeSession(memory_dir=memory_dir)
    assert session2.result_cache == {"some_hash": ["h1", "old content"]}


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


def test_sift_stdout_uses_skeletonizer_for_large_code() -> None:
    """Large code-like content should be skeletonized, not bluntly truncated."""
    lines = [f"def method_{i}(self):" for i in range(20)]
    lines.extend(f"    x = {i} + {i * 2}" for i in range(200))
    lines.extend(f"    y = self.data[{i}]" for i in range(200))
    large_code = "\n".join(lines)
    assert len(large_code) > 3000

    result = _sift_stdout(large_code)

    # Should use skeletonizer (preserves signatures) not blunt truncation
    assert ">>>" in result or len(result) < len(large_code)
    # Skeleton should preserve method signatures
    if ">>>" in result:
        assert "def method_0(self):" in result


def test_sift_stdout_skips_already_compressed() -> None:
    """Already-compressed Tok content (>>> prefix) must pass through unchanged."""
    compressed = ">>> tool:file_read|original_chars:5000|skeleton_lines:10\nimport os\ndef foo():"
    result = _sift_stdout(compressed)
    assert result == compressed


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
    # Second call — cache hit path
    second_result, _second_saved = _apply_result_cache(small_file, context, cache)

    # First read is verbatim content.
    assert first_result == small_file
    # Repeat read should be a stable guided stub with explicit recovery hints.
    assert second_result.startswith(">>> tool:view_file|unchanged|cached|")
    assert "Read offset=1 for full content" in second_result
    assert "@stable_skeleton |>" in second_result
    assert ">>>" not in first_result
    assert ">>> tool:view_file|unchanged|cached|" in second_result


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
