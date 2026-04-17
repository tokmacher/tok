"""Unit tests for load_gate_config() in tok.cli._gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tok.cli._gate import load_gate_config


class TestLoadGateConfig:
    def test_valid_json_returns_dict(self, tmp_path: Path) -> None:
        """Happy path: a well-formed JSON config file returns a dict."""
        config = {"enabled": True, "threshold": 0.9, "fixtures": ["a", "b"]}
        config_file = tmp_path / "gate-config.json"
        config_file.write_text(json.dumps(config))

        result = load_gate_config(config_file)

        assert isinstance(result, dict)
        assert result["enabled"] is True
        assert result["threshold"] == 0.9
        assert result["fixtures"] == ["a", "b"]

    def test_invalid_json_returns_none(self, tmp_path: Path) -> None:
        """Invalid JSON content must return None (exception caught internally)."""
        bad_file = tmp_path / "gate-config.json"
        bad_file.write_text("{ this is: not valid json !!!")

        result = load_gate_config(bad_file)

        assert result is None

    def test_nonexistent_path_returns_none(self, tmp_path: Path) -> None:
        """A path that does not exist returns None."""
        missing = tmp_path / "no-such-file.json"

        result = load_gate_config(missing)

        assert result is None

    def test_none_path_no_default_returns_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When path=None and cwd has no gate-config.json, returns None."""
        monkeypatch.chdir(tmp_path)  # tmp_path has no gate-config.json

        result = load_gate_config(None)

        assert result is None

    def test_empty_json_object_returns_empty_dict(self, tmp_path: Path) -> None:
        """An empty JSON object {} is valid and returns an empty dict."""
        config_file = tmp_path / "gate-config.json"
        config_file.write_text("{}")

        result = load_gate_config(config_file)

        assert result == {}
