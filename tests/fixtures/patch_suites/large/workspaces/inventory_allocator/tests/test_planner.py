from src.planner import build_plan


def test_build_plan_combines_forecast_allocation_and_reorder_point() -> None:
    plan = build_plan(avg_daily_demand=10, lead_days=3, safety_days=2, buffer_units=5, available=25)
    # required = 10 * (3 + 2) = 50
    # allocated = 25, backorder = 25
    # reorder_point = 10*3 + 5 = 35
    assert plan == {
        "required": 50,
        "allocated": 25,
        "backorder": 25,
        "reorder_point": 35,
    }
