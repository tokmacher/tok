from __future__ import annotations


def can_issue_session_token(has_mfa: bool, failed_logins: int) -> bool:
    if not has_mfa or failed_logins >= 3:
        return False
    return True
