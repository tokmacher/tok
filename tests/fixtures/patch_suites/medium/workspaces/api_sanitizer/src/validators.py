from __future__ import annotations

from src.sanitize import sanitize_email, sanitize_username


def validate_payload(payload: dict[str, str]) -> dict[str, str]:
    email = sanitize_email(payload["email"])
    username = sanitize_username(payload["username"])
    # BUG: should reject usernames shorter than 3 chars.
    if len(username) < 2:
        raise ValueError("username too short")
    # BUG: should reject usernames with invalid punctuation.
    return {"email": email, "username": username}
