"""Shared test fixtures for Tok tests."""

import pytest


@pytest.fixture(autouse=True)
def _isolate_tok_session_state(tmp_path, monkeypatch) -> None:
    """
    Force pytest runs to use an isolated runtime state directory.

    This prevents ambient ~/.tok data from leaking into tests and keeps
    request preparation deterministic across repeated local runs.
    """
    isolated_root = tmp_path / "tok-session-state"
    isolated_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TOK_PROJECT_DIR", str(isolated_root))
    monkeypatch.setenv("TOK_TEST_ISOLATED_SESSION", "1")
    yield


@pytest.fixture(autouse=True)
def _disable_short_session_threshold(request):
    """
    Disable short session threshold for unit tests.

    The short session optimization skips compression for sessions < 8 turns.
    Unit tests need to test compression logic in isolation, so we disable
    this optimization by setting the threshold to 0.

    We need to patch the config module directly since the value is read
    at module import time.

    Note: We only apply this to specific test files that test compression
    and gateway behavior, not to analysis tests that rely on default behavior.
    """
    # Only apply to specific test files that need short session threshold disabled
    test_file = request.fspath.basename
    relevant_tests = {
        "test_compression.py",
        "test_universal_runtime.py",
        "test_gateway.py",
        "test_freshness_signaling.py",
        "test_bridge_fidelity.py",
        "test_request_validation.py",
        "test_reacquisition_control.py",
        "test_speculative_macros.py",
    }

    if test_file not in relevant_tests:
        yield
        return

    from tok.runtime import _request_preparation, config

    original_config = config._SHORT_SESSION_THRESHOLD
    original_rp = _request_preparation._SHORT_SESSION_THRESHOLD
    config._SHORT_SESSION_THRESHOLD = 0
    _request_preparation._SHORT_SESSION_THRESHOLD = 0

    try:
        yield
    finally:
        config._SHORT_SESSION_THRESHOLD = original_config
        _request_preparation._SHORT_SESSION_THRESHOLD = original_rp
