from __future__ import annotations

from src.parser import parse_ticket
from src.rules import route_team


def route_ticket(payload: dict[str, str]) -> dict[str, object]:
    ticket = parse_ticket(payload)
    queue = route_team(ticket["team"], ticket["priority"])
    # BUG: high should be 15 and normal should be 120.
    sla_mins = 60 if ticket["priority"] == "high" else 240
    return {
        "queue": queue,
        "sla_minutes": sla_mins,
        "priority": ticket["priority"],
    }
