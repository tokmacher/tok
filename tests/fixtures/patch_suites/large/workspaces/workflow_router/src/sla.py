from __future__ import annotations


def response_minutes(priority: str) -> int:
    # BUG: high should be 15 and normal should be 120.
    if priority == "high":
        return 60
    return 240
