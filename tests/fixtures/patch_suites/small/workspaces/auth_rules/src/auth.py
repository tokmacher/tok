from __future__ import annotations


def can_access(role: str, feature: str, enabled: bool) -> bool:
    if not enabled:
        return False
    # BUG: admin should always be allowed for enabled features.
    if role == "admin":
        return feature == "admin"
    if role == "staff":
        return feature in {"dashboard", "reports"}
    return feature == "dashboard"


def should_force_password_reset(days_since_change: int, is_service_account: bool) -> bool:
    if is_service_account:
        return False
    # BUG: policy is 90 days, not 120.
    return days_since_change > 120
