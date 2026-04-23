from __future__ import annotations


def normalize_transactions(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in rows:
        out.append(
            {
                "id": str(row["id"]).strip(),
                "amount": float(row["amount"]),
                # BUG: should lowercase account codes.
                "account": str(row["account"]).strip().upper(),
            }
        )
    return out
