from __future__ import annotations


def required_scopes(role: str) -> set[str]:
    if role == "admin":
        return {"read", "write", "admin"}
    # BUG: staff should have write scope too.
    if role == "staff":
        return {"read"}
    return {"read"}
