"""Structural regression tests for adapter consolidation.

These tests enforce:
- Shared utilities live in tok.adapters._utils (single definition).
- Probe adapters are consolidated in tok.adapters.probes.
- No adapter defines _render_text or _system_to_messages independently.

Run order: these tests are RED before the refactor, GREEN after.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# New import paths must exist
# ---------------------------------------------------------------------------


def test_render_text_importable_from_utils() -> None:
    from tok.adapters._utils import _render_text  # noqa: PLC2701

    assert callable(_render_text)


def test_system_to_messages_importable_from_utils() -> None:
    from tok.adapters._utils import _system_to_messages  # noqa: PLC2701

    assert callable(_system_to_messages)


def test_opencode_adapter_importable_from_probes() -> None:
    from tok.adapters.probes import OpenCodeAdapter

    assert OpenCodeAdapter().identify_runtime() == "opencode"


def test_codex_adapter_importable_from_probes() -> None:
    from tok.adapters.probes import CodexAdapter

    assert CodexAdapter().identify_runtime() == "codex-cli"


# ---------------------------------------------------------------------------
# Behaviour is preserved under the new location
# ---------------------------------------------------------------------------


def test_render_text_empty_blocks() -> None:
    from tok.adapters._utils import _render_text  # noqa: PLC2701

    assert _render_text([]) == ""


def test_render_text_skips_non_text_blocks() -> None:
    from tok.adapters._utils import _render_text  # noqa: PLC2701

    blocks = [{"type": "tool_use", "text": "ignored"}, {"type": "text", "text": "kept"}]
    assert _render_text(blocks) == "kept"


def test_render_text_joins_multiple_text_blocks() -> None:
    from tok.adapters._utils import _render_text  # noqa: PLC2701

    blocks = [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]
    assert _render_text(blocks) == "hello\nworld"


def test_render_text_strips_whitespace() -> None:
    from tok.adapters._utils import _render_text  # noqa: PLC2701

    blocks = [{"type": "text", "text": "  spaces  "}]
    assert _render_text(blocks) == "spaces"


def test_render_text_skips_blank_text() -> None:
    from tok.adapters._utils import _render_text  # noqa: PLC2701

    blocks = [{"type": "text", "text": "   "}, {"type": "text", "text": "real"}]
    assert _render_text(blocks) == "real"


def test_system_to_messages_string() -> None:
    from tok.adapters._utils import _system_to_messages  # noqa: PLC2701

    assert _system_to_messages("hello") == [{"role": "system", "content": "hello"}]


def test_system_to_messages_none() -> None:
    from tok.adapters._utils import _system_to_messages  # noqa: PLC2701

    assert _system_to_messages(None) == []


def test_system_to_messages_empty_string() -> None:
    from tok.adapters._utils import _system_to_messages  # noqa: PLC2701

    assert _system_to_messages("") == []


def test_system_to_messages_list() -> None:
    from tok.adapters._utils import _system_to_messages  # noqa: PLC2701

    result = _system_to_messages([{"text": "block one"}, {"text": "block two"}])
    assert result == [
        {"role": "system", "content": "block one"},
        {"role": "system", "content": "block two"},
    ]


# ---------------------------------------------------------------------------
# Single source of truth: no independent definitions allowed
# ---------------------------------------------------------------------------


def test_adapters_py_uses_utils_render_text() -> None:
    """After refactor, adapters._render_text must be the same object as _utils._render_text."""
    import tok.adapters._utils as utils_module  # noqa: PLC2701
    import tok.adapters.adapters as adapters_module

    assert adapters_module._render_text is utils_module._render_text


def test_adapters_py_uses_utils_system_to_messages() -> None:
    """After refactor, adapters._system_to_messages must be the same object as _utils._system_to_messages."""
    import tok.adapters._utils as utils_module  # noqa: PLC2701
    import tok.adapters.adapters as adapters_module

    assert adapters_module._system_to_messages is utils_module._system_to_messages


def test_probes_uses_utils_render_text() -> None:
    """After refactor, probes._render_text must be the same object as _utils._render_text."""
    import tok.adapters._utils as utils_module  # noqa: PLC2701
    import tok.adapters.probes as probes_module

    assert probes_module._render_text is utils_module._render_text
