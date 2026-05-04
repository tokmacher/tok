from __future__ import annotations

import os
import subprocess
import sys


def _import_with_env(module: str, env_updates: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(env_updates)
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    return subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_runtime_config_invalid_integer_env_falls_back_without_import_crash() -> None:
    result = _import_with_env(
        "tok.runtime.core",
        {
            "TOK_FALLBACK_THRESHOLD": "bad",
            "TOK_RESULT_CACHE_TTL": "bad",
            "TOK_REACQUIRE_TRIGGER_COUNT": "bad",
            "TOK_LOOP_DETECTION_THRESHOLD": "bad",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "Invalid integer config TOK_FALLBACK_THRESHOLD" in result.stderr


def test_stream_recovery_invalid_integer_env_falls_back_without_import_crash() -> None:
    result = _import_with_env(
        "tok.gateway._bridge_streaming",
        {"TOK_STREAM_RECOVERY_TOOL_ONLY_REPEAT_LIMIT": "bad"},
    )

    assert result.returncode == 0, result.stderr
    assert "Invalid integer config TOK_STREAM_RECOVERY_TOOL_ONLY_REPEAT_LIMIT" in result.stderr


def test_compression_invalid_integer_env_falls_back_without_import_crash() -> None:
    result = _import_with_env(
        "tok.compression",
        {"TOK_SEMANTIC_HASH_MIN_CHARS": "bad"},
    )

    assert result.returncode == 0, result.stderr
    assert "Invalid integer config TOK_SEMANTIC_HASH_MIN_CHARS" in result.stderr
