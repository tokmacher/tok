import pytest

from src.handlers import handle_create_user


def test_handle_create_user_sanitizes_fields() -> None:
    result = handle_create_user({"email": "  A@Example.COM  ", "username": "  Jane Doe  "})
    assert result == {
        "status": "ok",
        "email": "a@example.com",
        "username": "Jane_Doe",
        "user_id": "jane_doe",
    }


def test_handle_create_user_enforces_username_min_length() -> None:
    with pytest.raises(ValueError):
        handle_create_user({"email": "x@y.com", "username": "ab"})


def test_handle_create_user_rejects_invalid_username_chars() -> None:
    with pytest.raises(ValueError):
        handle_create_user({"email": "x@y.com", "username": "jane!"})
