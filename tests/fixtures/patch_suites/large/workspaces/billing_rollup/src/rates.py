from __future__ import annotations

RATES = {
    "compute": 0.5,
    "storage": 0.1,
}


def unit_rate(service: str) -> float:
    # BUG: unknown service should raise ValueError.
    return RATES.get(service, 0.0)
