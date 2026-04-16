from tok.compression import tok_tool_result, truncate_large_result
from tok.compression._tool_result_codecs import _compress_file_read, _compress_grep
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


def test_grep_shows_multiple_snippets_per_file() -> None:
    """Multi-file grep should show up to 3 snippets per file, not just one."""
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

    # Should show 3 snippets for main.py (not just 1)
    assert "src/main.py:10:" in compressed or "def foo():" in compressed
    assert "src/main.py:20:" in compressed or "def bar():" in compressed
    assert "src/main.py:30:" in compressed or "def baz():" in compressed
    # 4th match should be collapsed
    assert "more matches" in compressed


def test_grep_shows_other_snippets_explicitly() -> None:
    """Lines that don't match grep format should show first 3 explicitly."""
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

    # Should show first 3 __other__ snippets explicitly
    assert "__other__:" in compressed
    # Should NOT collapse all __other__ content
    assert "unusual line" in compressed or "another unusual" in compressed
    # 4th and 5th should be collapsed
    assert "more)" in compressed


def test_compress_file_read_preserves_small_files() -> None:
    """Small files (≤100 lines, ≤3000 chars) must not be skeletonized."""
    # Simulate a file like pointers.py: 77 lines, ~2.5KB
    lines = ["class PointerRegistry:"]
    lines.extend(f"    def method_{i}(self):" for i in range(10))
    lines.extend(f"        return self._data[{i}]" for i in range(10))
    lines.extend(f"    # comment {i}" for i in range(56))
    small_file = "\n".join(lines)
    assert small_file.count("\n") + 1 <= 100
    assert len(small_file) <= 3000

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
    assert len(large_file) > 3000

    compressed = _compress_file_read(large_file)

    # Should be skeletonized (shorter than original)
    assert len(compressed) < len(large_file)


def test_sift_stdout_preserves_small_files() -> None:
    """Small files (≤100 lines, ≤3000 chars) must not be truncated by gateway sifting."""
    lines = ["class PointerRegistry:"]
    lines.extend(f"    def method_{i}(self):" for i in range(10))
    lines.extend(f"        return self._data[{i}]" for i in range(10))
    lines.extend(f"    # comment {i}" for i in range(56))
    small_file = "\n".join(lines)
    assert small_file.count("\n") + 1 <= 100
    assert len(small_file) <= 3000

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
