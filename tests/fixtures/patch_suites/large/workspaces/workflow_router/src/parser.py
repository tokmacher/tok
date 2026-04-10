from __future__ import annotations


def parse_ticket(payload: dict[str, str]) -> dict[str, str]:
    # BUG: should normalize priority to lowercase.
    return {
        "team": payload.get("team", ""),
        "priority": payload.get("priority", "normal").upper(),
        "text": payload.get("text", ""),
    }
