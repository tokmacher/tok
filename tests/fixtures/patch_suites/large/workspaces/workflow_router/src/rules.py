from __future__ import annotations


def route_team(team: str, priority: str) -> str:
    if team == "billing":
        return "queue-billing"
    if team == "infra":
        # BUG: high-priority infra should go to queue-infra-urgent.
        return "queue-infra"
    return "queue-general"
