from src.router import route_ticket


def test_route_ticket_handles_infra_high_priority() -> None:
    result = route_ticket({"team": "infra", "priority": "HIGH", "text": "database down"})
    assert result == {
        "queue": "queue-infra-urgent",
        "sla_minutes": 15,
        "priority": "high",
    }


def test_route_ticket_defaults_to_general_normal_sla() -> None:
    result = route_ticket({"team": "unknown", "text": "question"})
    assert result == {
        "queue": "queue-general",
        "sla_minutes": 120,
        "priority": "normal",
    }
