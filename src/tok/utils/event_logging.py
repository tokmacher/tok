"""
Colored event logging for Tok internal feature visibility.

Log messages use Rich markup for color coding, visible via `tok bridge logs`.
"""

import logging

logger = logging.getLogger("tok.events")

# Color scheme for different event types
COLORS = {
    "macro_created": "bold green",
    "macro_used": "green",
    "macro_registered": "green",
    "semantic_dedup": "cyan",
    "delta_compress": "blue",
    "rolling_state": "magenta",
    "memory_promotion": "yellow",
    "pointer_created": "dim",
    "drift_detected": "bold red",
    "drift_healed": "yellow",
}

# Icons for visual distinction
ICONS = {
    "macro_created": "🧬",
    "macro_used": "⚡",
    "macro_registered": "✓",
    "semantic_dedup": "🔄",
    "delta_compress": "📉",
    "rolling_state": "📜",
    "memory_promotion": "📦",
    "pointer_created": "🔗",
    "drift_detected": "⚠️",
    "drift_healed": "🔧",
}


def log_event(event_type: str, message: str) -> None:
    """
    Log a colored event message.

    Args:
        event_type: One of the keys in COLORS dict
        message: The detail message to log

    """
    color = COLORS.get(event_type, "white")
    icon = ICONS.get(event_type, "")
    category = event_type.upper().replace("_", " ")

    # Format: [color]ICON CATEGORY[/color] message
    prefix = f"{icon} {category}".strip()
    logger.info(f"[{color}]{prefix}[/{color}] {message}")


def log_macro_created(name: str, ops: tuple[str, ...]) -> None:
    """Log when a new macro is created."""
    op_seq = " → ".join(ops)
    log_event("macro_created", f"@{name} = [{op_seq}]")


def log_macro_registered(name: str, source: str = "mined") -> None:
    """Log when a macro is registered to the registry."""
    log_event("macro_registered", f"@{name} ({source})")


def log_macro_used(name: str, tokens_saved: int = 0) -> None:
    """Log when a macro is invoked."""
    if tokens_saved > 0:
        log_event("macro_used", f"@{name} (saved {tokens_saved} tokens)")
    else:
        log_event("macro_used", f"@{name}")


def log_semantic_dedup(cache_key: str, chars_saved: int) -> None:
    """Log when semantic deduplication hits."""
    # Truncate cache key for readability
    key_display = cache_key[:50] + "..." if len(cache_key) > 50 else cache_key
    log_event("semantic_dedup", f"{key_display} (saved {chars_saved} chars)")


def log_delta_compress(tool_name: str, original_len: int, compressed_len: int) -> None:
    """Log when delta compression is applied."""
    pct = (compressed_len / original_len * 100) if original_len > 0 else 100
    log_event("delta_compress", f"{tool_name} ({pct:.0f}% of original)")


def log_rolling_state(turn: int, entries_trimmed: int = 0) -> None:
    """Log rolling state update."""
    if entries_trimmed > 0:
        log_event("rolling_state", f"turn {turn}, trimmed {entries_trimmed} entries")
    else:
        log_event("rolling_state", f"turn {turn}")


def log_memory_promotion(_field: str, value: str, bucket: str = "durable") -> None:
    """Log when memory is promoted from hot to durable."""
    # Truncate value for readability
    val_display = value[:40] + "..." if len(value) > 40 else value
    log_event("memory_promotion", f'"{val_display}" → {bucket}')


def log_pointer_created(pointer_id: str, target: str) -> None:
    """Log when a pointer is created."""
    log_event("pointer_created", f"{pointer_id} → {target}")


def log_drift_detected(drift_type: str, detail: str = "") -> None:
    """Log when semantic drift is detected."""
    if detail:
        log_event("drift_detected", f"{drift_type}: {detail}")
    else:
        log_event("drift_detected", drift_type)


def log_drift_healed(heal_type: str) -> None:
    """Log when drift is automatically healed."""
    log_event("drift_healed", heal_type)


__all__ = [
    "log_delta_compress",
    "log_drift_detected",
    "log_drift_healed",
    "log_event",
    "log_macro_created",
    "log_macro_registered",
    "log_macro_used",
    "log_memory_promotion",
    "log_pointer_created",
    "log_rolling_state",
    "log_semantic_dedup",
]
