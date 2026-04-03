"""Shared test fixtures for Tok tests."""

import pytest


@pytest.fixture(autouse=True)
def _disable_short_session_threshold(request):
    """Disable short session threshold for unit tests.

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
    }

    if test_file not in relevant_tests:
        yield
        return

    from tok.runtime import config
    from tok.runtime import _request_preparation

    original_config = config._SHORT_SESSION_THRESHOLD
    original_rp = _request_preparation._SHORT_SESSION_THRESHOLD
    config._SHORT_SESSION_THRESHOLD = 0
    _request_preparation._SHORT_SESSION_THRESHOLD = 0

    try:
        yield
    finally:
        config._SHORT_SESSION_THRESHOLD = original_config
        _request_preparation._SHORT_SESSION_THRESHOLD = original_rp
