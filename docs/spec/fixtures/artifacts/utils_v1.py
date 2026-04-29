def normalize_name(value: str) -> str:
    return value.strip().lower()


def is_enabled(flag: bool) -> bool:
    return bool(flag)
