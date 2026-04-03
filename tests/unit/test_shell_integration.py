import pytest
from pathlib import Path

from tok.utils import shell_integration


def test_install_adds_marked_block_to_zshrc(tmp_path):
    rc_path = shell_integration.install(
        shell_env="/bin/zsh",
        home=tmp_path,
        tok_dir=Path("/repo/tok"),
    )

    assert rc_path == tmp_path / ".zshrc"
    content = rc_path.read_text()
    assert shell_integration.START_MARKER in content
    assert (
        "tok_claude.sh" in content
    )  # packaged asset path (resolved from tok.data)
    assert shell_integration.END_MARKER in content


def test_install_is_idempotent(tmp_path):
    shell_integration.install(
        shell_env="/bin/zsh",
        home=tmp_path,
        tok_dir=Path("/repo/tok"),
    )
    shell_integration.install(
        shell_env="/bin/zsh",
        home=tmp_path,
        tok_dir=Path("/repo/tok"),
    )

    content = (tmp_path / ".zshrc").read_text()
    assert content.count(shell_integration.START_MARKER) == 1


def test_uninstall_removes_marked_block(tmp_path):
    # Create a fake script file for testing
    fake_script = tmp_path / "tok_claude.sh"
    fake_script.write_text("# Fake tok script for testing")

    rc_path = tmp_path / ".zshrc"
    rc_path.write_text(
        "export PATH=/usr/bin\n"
        + shell_integration.integration_block(fake_script)
        + "alias ll='ls -la'\n"
    )

    removed = shell_integration.uninstall(home=tmp_path)

    assert rc_path in removed
    content = rc_path.read_text()
    assert shell_integration.START_MARKER not in content
    assert shell_integration.END_MARKER not in content
    assert "export PATH=/usr/bin" in content
    assert "alias ll='ls -la'" in content


def test_detect_shell_rejects_unsupported_shell():
    """Test that unsupported shells raise RuntimeError with appropriate message."""
    with pytest.raises(RuntimeError, match=r"supports zsh and bash"):
        shell_integration.detect_shell("/opt/homebrew/bin/fish")


def test_tok_claude_script_does_not_override_tok_cli():
    script = Path("scripts/tok_claude.sh").read_text()

    assert "tok()" not in script
    assert "command tok bridge start" in script
    assert "command claude" in script
