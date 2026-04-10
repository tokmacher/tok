from src.auth import can_access, should_force_password_reset
from src.policy import required_scopes
from src.session import can_issue_session_token


def test_admin_access_for_enabled_feature() -> None:
    assert can_access("admin", "billing", enabled=True) is True


def test_staff_scopes_include_write() -> None:
    assert required_scopes("staff") == {"read", "write"}


def test_password_reset_policy_90_days() -> None:
    assert should_force_password_reset(95, is_service_account=False) is True


def test_token_issue_requires_mfa_and_low_failures() -> None:
    assert can_issue_session_token(has_mfa=True, failed_logins=1) is True
    assert can_issue_session_token(has_mfa=False, failed_logins=1) is False
    assert can_issue_session_token(has_mfa=True, failed_logins=3) is False
