import tok.compression._pipeline

# Monkeypatch threshold FIRST before any imports from tok.compression
tok.compression._pipeline.TOOL_COMPRESS_THRESHOLD = 0

from tok.compression import (
    ResultCacheEntry,
    _detect_tool_content_type,
    tok_tool_result,
)


def test_stack_trace_filtering() -> None:
    trace = (
        "Traceback (most recent call last):\n"
        '  File "main.py", line 10, in main\n'
        "    run_app()\n"
        '  File "app.py", line 20, in run_app\n'
        "    process_data()\n"
        '  File "venv/lib/python3.11/site-packages/library/core.py", line 50, in process_data\n'
        '    raise ValueError("Invalid data")\n'
        "ValueError: Invalid data"
    )
    # Mocking standard threshold for test
    compressed = tok_tool_result(trace + "\n" + "extra data\n" * 100)
    assert "filtered 1 library frames" in compressed
    assert "core.py" not in compressed
    assert "app.py" in compressed


def test_grep_context_compression() -> None:
    grep_out = (
        "src/main.py-10-def main():\n"
        "src/main.py-11-    print('hello')\n"
        "src/main.py-12-    # comment\n"
        "src/main.py-13-    # more padding\n" * 50
    )
    compressed = tok_tool_result(grep_out)
    assert ">>> tool:grep_context" in compressed
    assert "file://src/main.py:" in compressed
    assert "[11]" in compressed


def test_ps_output_compression() -> None:
    ps_out = (
        "USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND\n"
        "root         1  0.0  0.0 168012  9812 ?        Ss   Mar15   0:02 /sbin/init\n"
        "jfj       1234  5.0  1.2 987654 12345 ?        Sl   00:01   0:10 /usr/bin/python3 main.py\n"
        "root      5678  0.0  0.0      0     0 ?        S    00:05   0:00 [kernel_task]\n"
        "padding" * 100
    )
    compressed = tok_tool_result(ps_out)
    assert ">>> tool:ps" in compressed
    assert "kernel_task" not in compressed
    assert "python3 main.py" in compressed


def test_env_output_compression() -> None:
    env_out = "PATH=/usr/bin:/bin\nHOME=/tmp\nUSER=testuser\nSECRET_API_KEY=sk-12345\nLESS_IMPORTANT=ignored\n"
    env_out += "PADDING=more data\n" * 100
    compressed = tok_tool_result(env_out)
    assert ">>> tool:env" in compressed
    assert "PATH=" in compressed
    assert "SECRET_API_KEY=" in compressed


def test_json_skeletonization() -> None:
    import json

    large_data = {
        "id": 1,
        "nested": {"large_list": list(range(100)), "long_string": "o" * 500},
        "items": [{"id": i} for i in range(20)],
    }
    raw_json = json.dumps(large_data)
    compressed = tok_tool_result(raw_json)
    assert ">>> tool:json_skeleton" in compressed
    assert '"_omitted": 99' in compressed
    assert "..." in compressed


def test_general_result_caching() -> None:
    from tok.compression import _apply_result_cache

    cache: dict[str, ResultCacheEntry] = {}

    # Tool 1: ls
    context_ls = {"name": "ls", "args": {"path": "."}}
    raw_ls = "file1.txt\nfile2.txt\nfile3.txt\n" * 50

    # First call: compressed
    out1, _saved1 = _apply_result_cache(raw_ls, context_ls, cache)
    assert ">>> tool:ls" in out1

    # Second call (unchanged): stub
    out2, _saved2 = _apply_result_cache(raw_ls, context_ls, cache)
    assert "|unchanged|cached" in out2

    # Tool 2: ps aux
    context_ps = {"name": "ps", "args": {"aux": True}}
    raw_ps1 = "USER PID COMMAND\nroot 1 init\njfj 123 python\n" * 50
    raw_ps2 = "USER PID COMMAND\nroot 1 init\nextra 999 process\n" * 50

    _apply_result_cache(raw_ps1, context_ps, cache)

    # Second call (changed): diff
    out3, _saved3 = _apply_result_cache(raw_ps2, context_ps, cache)
    assert "|delta|changed" in out3


def test_detect_stack_trace() -> None:
    trace = 'Traceback (most recent call last):\n  File "main.py", line 1\n    raise E\nException: E'
    assert _detect_tool_content_type(trace) == "stack_trace"


class TestStackTraceMixedFormats:
    """Regression: Java stack frame regex must not match Node.js frames."""

    def test_mixed_node_and_java_frames_no_mangled_prefix(self) -> None:
        trace = "\n".join(
            [
                "Traceback (most recent call last):",
                '  File "/Users/dev/project/src/app.py", line 42, in handle_request',
                "    process(data)",
                "at Module._compile (module.js:653:30)",
                "at Object.Module._extensions..js (internal/module.js:711:10)",
                "at com.example.service.UserService.findById (UserService.java:42)",
                "at com.example.controller.UserController.getUser (UserController.java:15)",
                "ValueError: bad data",
            ]
        )
        compressed = tok_tool_result(trace + "\n" + "extra data\n" * 100)
        assert "module" not in compressed.replace("module.js", "").replace("Module", "") or "module/" not in compressed

    def test_node_frame_path_not_corrupted_by_java_regex(self) -> None:
        from tok.compression._tool_result_codecs import _compress_stack_traces

        trace = "\n".join(
            [
                "at Object.<anonymous> (/Users/dev/project/node_modules/foo/index.js:10:5)",
                "at com.example.Service.run (Service.java:20)",
            ]
        )
        result = _compress_stack_traces(trace)
        assert "module/" not in result.replace("node_modules/", "")

    def test_java_path_collection_excludes_js_file_paths(self) -> None:
        import re

        text = "at Module._compile (module.js:653)\nat com.example.Service.run(Service.java:20)"
        node_paths: set[str] = set()
        for m in re.finditer(r"at [\w.<>$\[\]]+ \((.+):\d+:\d+\)", text):
            node_paths.add(m.group(1))
        _JS_FILE_RE = re.compile(r"\.(js|ts|mjs|cjs|jsx|tsx)$", re.IGNORECASE)
        paths: list[str] = []
        all_candidates: list[str] = []
        for m in re.finditer(r"at [\w.$]+\(([\w.$]+):(\d+)\)", text):
            candidate = m.group(1)
            all_candidates.append(candidate)
            if candidate in node_paths:
                continue
            if _JS_FILE_RE.search(candidate):
                continue
            paths.append(candidate.replace(".", "/"))
        assert all_candidates, f"Java regex matched nothing; text={text!r}"
        assert "module/js" not in paths
        assert "Service/java" in paths
