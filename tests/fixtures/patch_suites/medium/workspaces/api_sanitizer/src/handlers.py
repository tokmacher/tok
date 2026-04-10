from __future__ import annotations

from src.validators import validate_payload


def handle_create_user(payload: dict[str, str]) -> dict[str, str]:
    validated = validate_payload(payload)
    # BUG: should include normalized "user_id" as lowercase username.
    return {"status": "ok", "email": validated["email"], "username": validated["username"]}
