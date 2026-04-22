"""Constants for bridge memory behavior and wire parsing."""

from __future__ import annotations

# Field-specific decay rates. 0 = immortal (never decay).
# Hot bucket uses these rates; durable uses half-rates (floored at 1 for non-zero).
DECAY_RATES: dict[str, int] = {
    "constraints": 0,  # user invariants — never decay
    "goal": 0,  # core orientation — never decay
    "facts": 0,  # file/search snapshots — never decay
    "blockers": 1,
    "errs": 1,
    "files": 1,
    "questions": 1,
    "cmds": 2,  # ephemeral — decay faster
    "tests": 2,
    "next": 2,  # highly transient
    "turns": 1,
    "edited": 1,
}

# Per-field promotion thresholds for hot→durable.
PROMOTION_THRESHOLDS: dict[str, int] = {
    "goal": 2,
    "files": 3,
    "edited": 2,  # edited files promote faster than regular files
    "constraints": 2,
    "facts": 3,
    "blockers": 3,
    "errs": 4,
}

MULTI_VALUE_FIELDS = {
    "files",
    "edited",
    "cmds",
    "tests",
    "errs",
    "blockers",
    "constraints",
    "questions",
    "facts",
    "goal",
}

HOT_LIMITS = {
    "turns": 10,
    "goal": 40,
    "files": 40,
    "edited": 40,
    "cmds": 160,
    "tests": 80,
    "errs": 80,
    "blockers": 40,
    "constraints": 40,
    "questions": 40,
    "next": 20,
    "facts": 160,
}

DURABLE_LIMITS = {
    "turns": 10,
    "goal": 40,
    "files": 80,
    "edited": 80,
    "cmds": 80,
    "tests": 80,
    "errs": 80,
    "blockers": 80,
    "constraints": 80,
    "questions": 80,
    "next": 40,
    "facts": 320,
}

# Hard caps on total entries across all fields in each bucket
HOT_TOTAL_CAP = 600
DURABLE_TOTAL_CAP = 2000

SECTION_HEADERS: dict[str, str] = {
    "@h": "h",
    "@d": "d",
    "@rolling_cmds": "rolling_cmds",
    "@macros": "macros",
}


def dispatch_section_header(
    line: str, section: str | None, current_field: str | None
) -> tuple[str | None, str | None, bool]:
    """Return (new_section, new_current_field, was_handled)."""
    new_section = SECTION_HEADERS.get(line)
    if new_section is not None:
        return new_section, None, True
    return section, current_field, False
