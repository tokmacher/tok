"""Tests asserting tool taxonomy is imported from one canonical source."""

from __future__ import annotations

import tok.compression._tool_taxonomy as _tool_taxonomy
import tok.runtime.repeat_targets as repeat_targets


def test_search_like_tools_is_canonical() -> None:
    assert repeat_targets.SEARCH_LIKE_TOOLS is _tool_taxonomy.SEARCH_LIKE_TOOLS


def test_command_like_tools_is_canonical() -> None:
    assert repeat_targets.COMMAND_LIKE_TOOLS is _tool_taxonomy.COMMAND_LIKE_TOOLS


def test_command_like_tools_includes_bridge_aliases() -> None:
    assert "run_bash" in _tool_taxonomy.COMMAND_LIKE_TOOLS


def test_listing_like_tools_is_canonical() -> None:
    assert repeat_targets.LISTING_LIKE_TOOLS is _tool_taxonomy.LISTING_LIKE_TOOLS
