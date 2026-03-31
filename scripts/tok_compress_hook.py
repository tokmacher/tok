"""
Tok Compress Hook (Stop event)
==============================
Registered as a Claude Code Stop hook. Fires after each session ends.
Reads the session JSONL transcript, runs TokOrchestrator on it, and
writes a compressed Tok rolling state to the project memory files.

Registration (add to ~/.claude/settings.json under "hooks"):
    "Stop": [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": "cd $TOK_ROOT && uv run python scripts/tok_compress_hook.py"
                }
            ]
        }
    ]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------


def _text_of(content: str | list | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
        return " ".join(p for p in parts if p)
    return str(content)


def parse_transcript(path: Path) -> list[dict]:
    """
    Parse a Claude Code session JSONL file.
    Handles two known formats:
      - {"role": "user"|"assistant", "content": ...}
      - {"type": "message", "message": {"role": ..., "content": ...}}
    Returns list of {"role": str, "content": str}.
    """
    messages = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = entry.get("type", "")

            # Claude Code format: type=user|assistant with nested message
            if entry_type in ("user", "assistant") and "message" in entry:
                msg = entry["message"]
                role = msg.get("role", entry_type)
                text = _text_of(msg.get("content", ""))
                if role in ("user", "assistant") and text:
                    messages.append({"role": role, "content": text})

            # Fallback: top-level role field
            elif "role" in entry:
                role = entry["role"]
                text = _text_of(entry.get("content", ""))
                if role in ("user", "assistant") and text:
                    messages.append({"role": role, "content": text})

    return messages


# ---------------------------------------------------------------------------
# Memory output
# ---------------------------------------------------------------------------

MEMORY_CANDIDATES = [
    Path.home() / ".claude" / "projects" / "-Users-jfj-Desktop-tok" / "memory",
    Path(__file__).parent.parent / ".claude" / "memory",
]


def find_memory_dir() -> Path | None:
    for candidate in MEMORY_CANDIDATES:
        if candidate.exists() and candidate.is_dir():
            return candidate
    # Try to create the project-local one
    local = Path(__file__).parent.parent / ".claude" / "memory"
    try:
        local.mkdir(parents=True, exist_ok=True)
        return local
    except OSError:
        return None


def write_memory(
    memory_dir: Path, tok_state: str, message_count: int, session_id: str
) -> None:
    # Write the raw Tok state as memory.tok (native protocol format)
    tok_file = memory_dir / "memory.tok"
    tok_file.write_text(tok_state, encoding="utf-8")

    # Write a markdown wrapper for the memory index system
    state_file = memory_dir / "tok_session_state.md"
    state_file.write_text(
        f"""---
name: tok_session_state
description: Tok-compressed rolling state from last Claude Code session ({session_id[:8]})
type: project
---

Compressed from {message_count} messages in session `{session_id}`.

```tok
{tok_state}
```
""",
        encoding="utf-8",
    )

    # Ensure MEMORY.md index has a pointer
    memory_index = memory_dir / "MEMORY.md"
    pointer = "- [tok_session_state.md](tok_session_state.md) — Tok-compressed rolling state from last session (raw: memory.tok)\n"
    if memory_index.exists():
        existing = memory_index.read_text(encoding="utf-8")
        if "tok_session_state.md" not in existing:
            memory_index.write_text(
                existing.rstrip() + "\n" + pointer, encoding="utf-8"
            )
    else:
        memory_index.write_text(pointer, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    raw = sys.stdin.read()
    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        sys.exit(0)

    # Prevent infinite loops
    if event.get("stop_hook_active"):
        sys.exit(0)

    session_id = event.get("session_id", "unknown")
    transcript_path_str = event.get("transcript_path", "")
    if not transcript_path_str:
        sys.exit(0)

    transcript_path = Path(transcript_path_str)
    if not transcript_path.exists():
        print(
            f"[tok-hook] transcript not found: {transcript_path}",
            file=sys.stderr,
        )
        sys.exit(0)

    messages = parse_transcript(transcript_path)
    if not messages:
        sys.exit(0)

    # Run TokOrchestrator
    try:
        src_path = str(Path(__file__).parent.parent / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        from tok.tok_orchestrator import TokOrchestrator  # type: ignore

        orch = TokOrchestrator(model="dummy")
        for i, msg in enumerate(messages):
            orch.turn_count = i + 1
            content = msg["content"]
            orch._update_entropy(content)
            # Feed a truncated snapshot into hot_state
            orch.memory.hot_state = content[:300]
            orch._sift_memory()

        tok_state = orch.memory.to_tok()

    except Exception as exc:
        print(f"[tok-hook] orchestrator error: {exc}", file=sys.stderr)
        sys.exit(0)

    memory_dir = find_memory_dir()
    if not memory_dir:
        print(
            "[tok-hook] could not find or create memory directory",
            file=sys.stderr,
        )
        sys.exit(0)

    write_memory(memory_dir, tok_state, len(messages), session_id)
    output_path = memory_dir / "tok_session_state.md"
    print(
        f"[tok-hook] compressed {len(messages)} msgs → {output_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
