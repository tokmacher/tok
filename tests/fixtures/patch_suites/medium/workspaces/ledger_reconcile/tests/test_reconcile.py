from src.reconcile import reconcile
from src.report import render_report


def test_reconcile_deduplicates_transaction_ids_and_normalizes_account() -> None:
    rows = [
        {"id": "a1", "amount": 10.0, "account": "Sales"},
        {"id": "a1", "amount": 10.0, "account": "sales"},
        {"id": "b2", "amount": 2.345, "account": "sales"},
        {"id": "c3", "amount": 5.0, "account": "Ops"},
    ]
    totals = reconcile(rows)
    assert totals == {"sales": 12.345, "ops": 5.0}


def test_render_report_rounds_totals_and_sorts_accounts() -> None:
    assert render_report({"sales": 12.345, "ops": 5.0}) == "account,total\nops,5.00\nsales,12.35"
