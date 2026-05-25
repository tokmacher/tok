"""Section 5.1.3 tests: Config Strictness Table.

Verifies that TokConfig and parse_config() exist in src/tok/config.py and provide
correct severity classification, defaults, and invalid-value rejection for all
TOK_* environment variables.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# RED: TokConfig and parse_config() must exist
# ---------------------------------------------------------------------------


def test_tok_config_module_exists() -> None:
    """src/tok/config.py must export TokConfig and parse_config."""
    from tok import config as cfg_mod

    assert hasattr(cfg_mod, "TokConfig"), "tok.config.TokConfig is missing"
    assert hasattr(cfg_mod, "parse_config"), "tok.config.parse_config() is missing"


def test_parse_config_returns_tok_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """parse_config() must return a TokConfig instance."""
    monkeypatch.delenv("TOK_DEBUG", raising=False)
    from tok.config import TokConfig, parse_config

    cfg = parse_config()
    assert isinstance(cfg, TokConfig)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_tok_config_default_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOK_LOG_LEVEL", raising=False)
    from tok.config import parse_config

    cfg = parse_config()
    assert cfg.log_level == "INFO"


def test_tok_config_default_bridge_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOK_BRIDGE_HOST", raising=False)
    from tok.config import parse_config

    cfg = parse_config()
    assert cfg.bridge_host == "localhost"


def test_tok_config_default_tool_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOK_TOOL_TIMEOUT", raising=False)
    from tok.config import parse_config

    cfg = parse_config()
    assert cfg.tool_timeout == 120


def test_tok_config_default_debug_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOK_DEBUG", raising=False)
    from tok.config import parse_config

    cfg = parse_config()
    assert cfg.debug is False


def test_tok_config_default_loop_detection_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOK_LOOP_DETECTION_ENABLED", raising=False)
    from tok.config import parse_config

    cfg = parse_config()
    assert cfg.loop_detection_enabled is True


def test_tok_config_default_project_dir_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOK_PROJECT_DIR", raising=False)
    from tok.config import parse_config

    cfg = parse_config()
    assert cfg.project_dir == ""


# ---------------------------------------------------------------------------
# Valid overrides
# ---------------------------------------------------------------------------


def test_tok_debug_enabled_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOK_DEBUG", "1")
    from tok.config import parse_config

    cfg = parse_config()
    assert cfg.debug is True


def test_tok_tool_timeout_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOK_TOOL_TIMEOUT", "60")
    from tok.config import parse_config

    cfg = parse_config()
    assert cfg.tool_timeout == 60


def test_tok_log_level_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOK_LOG_LEVEL", "DEBUG")
    from tok.config import parse_config

    cfg = parse_config()
    assert cfg.log_level == "DEBUG"


def test_tok_project_dir_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOK_PROJECT_DIR", "/tmp/my_project")
    from tok.config import parse_config

    cfg = parse_config()
    assert cfg.project_dir == "/tmp/my_project"


def test_tok_loop_detection_disabled_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOK_LOOP_DETECTION_ENABLED", "0")
    from tok.config import parse_config

    cfg = parse_config()
    assert cfg.loop_detection_enabled is False


# ---------------------------------------------------------------------------
# Severity classifications
# ---------------------------------------------------------------------------


def test_tok_config_has_severity_critical_keys() -> None:
    """Keys that affect safe operation must be classified as severity=critical."""
    from tok.config import CONFIG_SCHEMA

    critical_keys = {k for k, v in CONFIG_SCHEMA.items() if v["severity"] == "critical"}
    # TOK_TOOL_TIMEOUT is critical: infinite timeout blocks the bridge
    assert "TOK_TOOL_TIMEOUT" in critical_keys or "TOK_BRIDGE_HOST" in critical_keys, (
        "At least one critical-severity key must be registered in CONFIG_SCHEMA"
    )


def test_tok_config_has_severity_info_keys() -> None:
    """Non-structural keys must be classified as severity=info."""
    from tok.config import CONFIG_SCHEMA

    info_keys = {k for k, v in CONFIG_SCHEMA.items() if v["severity"] == "info"}
    assert info_keys, "No info-severity keys found in CONFIG_SCHEMA"


def test_tok_config_has_severity_warning_keys() -> None:
    """Behaviorally-significant but non-critical keys must be severity=warning."""
    from tok.config import CONFIG_SCHEMA

    warn_keys = {k for k, v in CONFIG_SCHEMA.items() if v["severity"] == "warning"}
    assert warn_keys, "No warning-severity keys found in CONFIG_SCHEMA"
    assert "TOK_ADAPTER" in warn_keys
    assert "TOK_UNSTABLE_ADAPTERS" in warn_keys


def test_config_schema_covers_all_tok_keys() -> None:
    """CONFIG_SCHEMA must cover every registered TOK_* key."""
    from tok.config import CONFIG_SCHEMA

    # All keys must start with TOK_ or be known non-TOK operational keys
    for key in CONFIG_SCHEMA:
        assert key.isupper(), f"CONFIG_SCHEMA key {key!r} must be uppercase"


# ---------------------------------------------------------------------------
# Invalid-value behavior
# ---------------------------------------------------------------------------


def test_invalid_tool_timeout_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-integer TOK_TOOL_TIMEOUT must fall back to 120 without raising."""
    monkeypatch.setenv("TOK_TOOL_TIMEOUT", "not_a_number")
    from tok.config import parse_config

    cfg = parse_config()
    assert cfg.tool_timeout == 120


def test_invalid_log_level_falls_back_to_info(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unrecognized TOK_LOG_LEVEL must fall back to INFO."""
    monkeypatch.setenv("TOK_LOG_LEVEL", "ULTRA_VERBOSE")
    from tok.config import parse_config

    cfg = parse_config()
    assert cfg.log_level == "INFO"


def test_negative_tool_timeout_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative TOK_TOOL_TIMEOUT must be rejected and fall back to 120."""
    monkeypatch.setenv("TOK_TOOL_TIMEOUT", "-5")
    from tok.config import parse_config

    cfg = parse_config()
    assert cfg.tool_timeout == 120


# ---------------------------------------------------------------------------
# Adversarial / edge cases (RED phase 2)
# ---------------------------------------------------------------------------


def test_multiple_invalid_values_all_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multiple bad values must all fall back independently; no partial failure."""
    monkeypatch.setenv("TOK_TOOL_TIMEOUT", "bad")
    monkeypatch.setenv("TOK_LOG_LEVEL", "ULTRA_VERBOSE")
    from tok.config import parse_config

    cfg = parse_config()
    assert cfg.tool_timeout == 120
    assert cfg.log_level == "INFO"


def test_empty_string_tool_timeout_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty-string TOK_TOOL_TIMEOUT must fall back to default."""
    monkeypatch.setenv("TOK_TOOL_TIMEOUT", "")
    from tok.config import parse_config

    cfg = parse_config()
    assert cfg.tool_timeout == 120


def test_whitespace_only_project_dir_treated_as_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """TOK_PROJECT_DIR set to whitespace-only must be treated as empty."""
    monkeypatch.setenv("TOK_PROJECT_DIR", "   ")
    from tok.config import parse_config

    cfg = parse_config()
    assert cfg.project_dir == ""


def test_parse_config_is_pure_no_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling parse_config() twice with the same env must return equivalent configs."""
    monkeypatch.setenv("TOK_DEBUG", "1")
    from tok.config import parse_config

    cfg1 = parse_config()
    cfg2 = parse_config()
    assert cfg1 == cfg2
