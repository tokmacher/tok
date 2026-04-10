from __future__ import annotations


def normalize_usage(events: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for event in events:
        out.append(
            {
                "account": str(event["account"]).strip(),
                "units": int(event["units"]),
                # BUG: should normalize service to lowercase.
                "service": str(event["service"]).strip().upper(),
            }
        )
    return out
