from src.invoice import render_invoice


def test_render_invoice_rolls_up_by_account_with_rates() -> None:
    events = [
        {"account": "acme", "service": "Compute", "units": 4},
        {"account": "acme", "service": "Storage", "units": 10},
        {"account": "beta", "service": "compute", "units": 2},
    ]
    # acme: 4*0.5 + 10*0.1 = 3.0 ; beta: 2*0.5 = 1.0
    assert render_invoice(events) == "account,total\nacme,3.00\nbeta,1.00"
