"""Packaging smoke tests for Batch 3: install-path hardening.

Verifies that the shell integration script is resolvable from package data
in both source-checkout and installed-wheel scenarios.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


class TestBundledScript:
    def test_bundled_script_exists_via_importlib_resources(self):
        """tok_claude.sh must be resolvable from tok.data package data."""
        ref = files("tok.data").joinpath("tok_claude.sh")
        path = Path(str(ref))

        assert path.exists(), (
            f"tok_claude.sh not found at {path}. "
            "Run Batch 3: the script must live in src/tok/data/ and be included in the wheel."
        )

    def test_bundled_script_is_nonempty(self):
        """tok_claude.sh must not be empty."""
        ref = files("tok.data").joinpath("tok_claude.sh")
        path = Path(str(ref))

        assert path.stat().st_size > 0, "tok_claude.sh is empty"

    def test_bundled_script_contains_claude_function(self):
        """Sanity-check that the sourced script defines a claude() function."""
        ref = files("tok.data").joinpath("tok_claude.sh")
        content = Path(str(ref)).read_text()

        assert (
            "claude()" in content
        ), "tok_claude.sh must define claude() function"


class TestShellIntegrationBlock:
    def test_integration_block_references_bundled_script(self):
        """integration_block() must point at the bundled script, not scripts/."""
        from tok.utils.shell_integration import integration_block

        block = integration_block()
        assert "tok_claude.sh" in block
        # Must NOT reference the old repo-only scripts/ directory.
        assert "/scripts/tok_claude.sh" not in block, (
            "integration_block() must not reference the scripts/ directory. "
            "It should use the bundled package data path."
        )

    def test_integration_block_has_markers(self):
        from tok.utils.shell_integration import (
            END_MARKER,
            START_MARKER,
            integration_block,
        )

        block = integration_block()
        assert START_MARKER in block
        assert END_MARKER in block

    def test_install_writes_block_pointing_at_bundled_path(
        self, tmp_path, monkeypatch
    ):
        """install() must write a source line that points at the bundled script."""
        monkeypatch.setenv("SHELL", "/bin/zsh")
        from tok.utils.shell_integration import install

        rc = install(home=tmp_path)
        content = rc.read_text()
        assert "tok_claude.sh" in content
        assert "/scripts/tok_claude.sh" not in content
