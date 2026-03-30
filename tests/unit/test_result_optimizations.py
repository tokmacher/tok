from tok.compression import truncate_large_result, tok_tool_result
from tok.universal_runtime import RuntimeSession


def test_semantic_truncation():
    # 50 lines of garbage
    large_text = "\n".join([f"line {i}: content {i * 7}" for i in range(100)])
    # This is roughly 2000 chars
    assert len(large_text) > 1800

    truncated = truncate_large_result(large_text, limit=1200)
    assert len(truncated) < 1500
    assert "... [TRUNCATED" in truncated
    assert "line 0:" in truncated
    assert "line 99:" in truncated


def test_result_cache_persistence(tmp_path):
    memory_dir = tmp_path / ".tok"
    memory_dir.mkdir()

    session = RuntimeSession(memory_dir=memory_dir)
    session.result_cache = {"some_hash": ("h1", "old content")}
    session._save_result_cache()

    # Reload in new session
    session2 = RuntimeSession(memory_dir=memory_dir)
    assert session2.result_cache == {"some_hash": ["h1", "old content"]}


def test_tok_tool_result_applies_truncation():
    # Long result that doesn't match any specific compressor
    long_raw = "A" * 5000
    compressed = tok_tool_result(long_raw)

    # It should be truncated because it's not a known type and it's long
    assert len(compressed) < 2000
    assert "... [TRUNCATED" in compressed
