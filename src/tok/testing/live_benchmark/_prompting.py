from __future__ import annotations

from typing import Any

from ._utils import _estimate_tokens


def _minimalize_system_prompt(system: str | dict[str, Any] | list[Any], original_system_prompt: str) -> str:
    minimal_directive = (
        "Use plain text only. History may be compressed. Prefer exact filenames and verification results."
    )

    # Extract @pointers and @macros from the 'system' returned by runtime
    system_text = ""
    if isinstance(system, str):
        system_text = system
    elif isinstance(system, list):
        system_text = "\n".join(
            block.get("text", "") for block in system if isinstance(block, dict) and block.get("type") == "text"
        )

    extra_blocks = []
    for block_name in ["@pointers", "@macros"]:
        if block_name in system_text:
            lines = system_text.splitlines()
            in_block = False
            block_lines = []
            for line in lines:
                if line.strip() == block_name:
                    in_block = True
                    block_lines.append(line)
                    continue
                if in_block:
                    if (
                        line.strip().startswith("@")
                        and not line.strip().startswith("@pointers")
                        and not line.strip().startswith("@macros")
                    ):
                        in_block = False
                        continue
                    if line.strip().startswith("|> ") or not line.strip():
                        block_lines.append(line)
                    else:
                        in_block = False
            if block_lines:
                extra_blocks.append("\n".join(block_lines))

    additions = minimal_directive
    if extra_blocks:
        additions += "\n\n" + "\n\n".join(extra_blocks)

    if isinstance(system, str):
        return original_system_prompt + "\n\n" + additions if original_system_prompt else additions

    # Return as list of blocks if original was likely a list or specifically requested
    return original_system_prompt + "\n\n" + additions if original_system_prompt else additions


def _system_text(system: str | dict[str, Any] | list[Any]) -> str:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "\n".join(
            str(block.get("text", ""))
            for block in system
            if isinstance(block, dict) and str(block.get("text", "")).strip()
        )
    return ""


def _system_breakdown(original_system_prompt: str, system: str | dict[str, Any] | list[Any]) -> tuple[int, int, int]:
    system_text = _system_text(system)
    if not system_text:
        return 0, 0, 0
    total_system_tokens = _estimate_tokens(system_text)
    state_tokens = 0
    marker = "[Tok compressed history]"
    if marker in system_text:
        state_fragment = system_text.split(marker, 1)[1].strip()
        state_tokens = _estimate_tokens(state_fragment)
    directive_tokens = max(
        0,
        total_system_tokens - _estimate_tokens(original_system_prompt) - state_tokens,
    )
    return total_system_tokens, directive_tokens, state_tokens
