from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

START_MARKER = "# >>> tok shell integration >>>"
END_MARKER = "# <<< tok shell integration <<<"

_DATA_SCRIPT = "tok_claude.sh"


def _bundled_script_path() -> Path:
    """Return path to the packaged tok_claude.sh, whether installed or in-tree."""
    from importlib.resources import files

    ref = files("tok.data").joinpath(_DATA_SCRIPT)
    # In Python 3.9+, files() returns a Traversable; resolve to real Path.
    return Path(str(ref))


def validate_shell_path(shell_path: str) -> bool:
    """Validate shell path is safe and reasonable."""
    if not shell_path or len(shell_path) > 255:
        return False

    # Check for suspicious characters
    suspicious_chars = [";", "&", "|", "`", "$", "(", ")", "<", ">", '"', "'"]
    if any(char in shell_path for char in suspicious_chars):
        return False

    # Check for reasonable shell names
    allowed_shells = ["bash", "zsh", "fish", "sh", "ksh", "csh", "tcsh"]
    shell_name = Path(shell_path).name
    return shell_name in allowed_shells


def detect_shell(shell_env: str | None = None) -> str:
    """Detect shell with proper validation."""
    raw_shell = shell_env or os.getenv("SHELL") or ""
    shell = raw_shell.strip()

    # Validate shell path
    if not validate_shell_path(shell):
        raise RuntimeError(
            f"Invalid shell path detected: {shell}. "
            "tok install currently supports bash, zsh, fish, sh, ksh, csh, and tcsh. "
            "For other shells, source the installed tok_claude.sh path manually."
        )

    # Extract shell name
    name = Path(shell).name
    if name in {"zsh", "bash"}:
        return name

    raise RuntimeError(
        f"Unsupported shell: {name}. "
        "tok install currently supports zsh and bash with automatic integration. "
        "For other shells, source the installed tok_claude.sh path manually."
    )


def validate_home_directory(home: Path | None) -> Path:
    """Validate and return home directory."""
    if home is None:
        home = Path.home()

    # Validate home directory exists and is writable
    if not home.exists():
        raise RuntimeError(f"Home directory does not exist: {home}")

    if not home.is_dir():
        raise RuntimeError(f"Home path is not a directory: {home}")

    # Test write permissions
    test_file = home / ".tok_write_test"
    try:
        test_file.touch()
        test_file.unlink()
    except (OSError, PermissionError) as e:
        raise RuntimeError(f"Home directory is not writable: {home} ({e})")

    return home


def rc_path_for_shell(shell: str, home: Path | None = None) -> Path:
    """Get RC file path with validation."""
    home = validate_home_directory(home)

    if shell == "zsh":
        return home / ".zshrc"
    if shell == "bash":
        return home / ".bashrc"

    raise RuntimeError(f"Unsupported shell for RC file detection: {shell}")


def validate_script_path(script_path: Path) -> bool:
    """Validate script path is safe."""
    try:
        # Check if file exists
        if not script_path.exists():
            logger.error(f"Script file does not exist: {script_path}")
            return False

        # Check if it's a regular file
        if not script_path.is_file():
            logger.error(f"Script path is not a file: {script_path}")
            return False

        # Check file permissions
        if not os.access(script_path, os.R_OK):
            logger.error(f"Script file is not readable: {script_path}")
            return False

        # Basic content validation
        try:
            content = script_path.read_text(encoding="utf-8")
            if len(content) > 100000:  # 100KB limit
                logger.error(f"Script file too large: {script_path}")
                return False

            # Check for suspicious content
            suspicious_patterns = [
                "rm -rf /",
                "sudo rm",
                "chmod 777",
                "curl | sh",
                "wget | sh",
            ]
            content_lower = content.lower()
            if any(
                pattern in content_lower for pattern in suspicious_patterns
            ):
                logger.warning(
                    f"Suspicious content detected in script: {script_path}"
                )
                # Allow but warn

        except (UnicodeDecodeError, OSError) as e:
            logger.error(f"Failed to read script file {script_path}: {e}")
            return False

        return True

    except Exception as e:
        logger.error(f"Error validating script path {script_path}: {e}")
        return False


def integration_block(script_path: Path | None = None) -> str:
    """Generate integration block with validation."""
    resolved = script_path or _bundled_script_path()

    # Validate script path
    if not validate_script_path(resolved):
        raise RuntimeError(f"Invalid script path: {resolved}")

    return f'{START_MARKER}\nsource "{resolved}"\n{END_MARKER}\n'


def install(
    *,
    shell_env: str | None = None,
    home: Path | None = None,
    tok_dir: Path | None = None,
) -> Path:
    """Install shell integration with comprehensive validation."""
    try:
        shell = detect_shell(shell_env)
        rc_path = rc_path_for_shell(shell, home)

        # tok_dir kept for backwards compat but ignored; script resolved from package data.
        _ = tok_dir

        # Validate and generate integration block
        block = integration_block()

        # Ensure parent directory exists
        try:
            rc_path.parent.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise RuntimeError(
                f"Failed to create parent directory for {rc_path}: {e}"
            )

        # Read existing content safely
        existing = ""
        if rc_path.exists():
            try:
                existing = rc_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError) as e:
                logger.warning(
                    f"Failed to read existing RC file {rc_path}: {e}"
                )
                existing = ""

        # Check if integration already exists
        if START_MARKER in existing and END_MARKER in existing:
            logger.info(f"Tok integration already exists in {rc_path}")
            return rc_path

        # Validate existing content size
        if len(existing) > 1000000:  # 1MB limit
            raise RuntimeError(
                f"RC file too large: {rc_path} ({len(existing)} bytes)"
            )

        # Write new content safely
        temp_file = None
        try:
            prefix = "" if not existing or existing.endswith("\n") else "\n"
            new_content = existing + prefix + block

            # Write to temporary file first, then atomic rename
            temp_file = rc_path.with_suffix(".tmp")
            temp_file.write_text(new_content, encoding="utf-8")

            # Verify the temp file was written correctly
            if not temp_file.exists():
                raise RuntimeError("Failed to write temporary file")

            temp_size = temp_file.stat().st_size
            if temp_size == 0:
                raise RuntimeError("Temporary file is empty")

            # Atomic rename
            temp_file.replace(rc_path)

            # Verify the operation succeeded
            if not rc_path.exists():
                raise RuntimeError("RC file not found after atomic rename")

            logger.info(f"Tok shell integration installed to {rc_path}")

        except (OSError, PermissionError) as e:
            raise RuntimeError(f"Failed to write RC file {rc_path}: {e}")
        finally:
            # Cleanup temporary file if it still exists
            if temp_file and temp_file.exists():
                try:
                    temp_file.unlink()
                    logger.debug(f"Cleaned up temporary file: {temp_file}")
                except OSError as e:
                    logger.warning(
                        f"Failed to cleanup temporary file {temp_file}: {e}"
                    )

        return rc_path

    except Exception as e:
        logger.error(f"Shell integration installation failed: {e}")
        raise


def uninstall(*, home: Path | None = None) -> list[Path]:
    removed: list[Path] = []
    roots = [
        ("zsh", rc_path_for_shell("zsh", home)),
        ("bash", rc_path_for_shell("bash", home)),
    ]
    for _, rc_path in roots:
        if not rc_path.exists():
            continue
        content = rc_path.read_text()
        start = content.find(START_MARKER)
        end = content.find(END_MARKER)
        if start == -1 or end == -1:
            continue
        end += len(END_MARKER)
        remainder = content[:start] + content[end:]
        rc_path.write_text(
            remainder.strip("\n") + ("\n" if remainder.strip("\n") else "")
        )
        removed.append(rc_path)
    return removed
