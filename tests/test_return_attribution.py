import pandas as pd
import pytest

from cfquant.engine import BacktestResult
from cfquant.return_attribution import attribute_result


def hand_ledger():
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    # Buy A at 10, sell 4 at 12; buy B at 20 and leave it marked at 18.
    trades = pd.DataFrame([
        {"date": dates[0], "asset": "A", "side": "buy", "units": 10., "notional": 100., "cost": 2.},
        {"date": dates[0], "asset": "B", "side": "buy", "units": 5., "notional": 100., "cost": 3.},
        {"date": dates[1], "asset": "A", "side": "sell", "units": 4., "notional": 48., "cost": 1.},
    ])
    positions = pd.DataFrame([
        {"date": dates[0], "asset": "A", "units": 10., "mark_price": 11., "value": 110.},
        {"date": dates[0], "asset": "B", "units": 5., "mark_price": 19., "value": 95.},
        {"date": dates[1], "asset": "A", "units": 6., "mark_price": 12., "value": 72.},
        {"date": dates[1], "asset": "B", "units": 5., "mark_price": 18., "value": 90.},
    ])
    daily = pd.DataFrame({"date": dates, "cash": [795., 842.], "nav": [1_000., 1_004.], "cost": [5., 1.]})
    return BacktestResult(daily, trades, positions, pd.DataFrame())


def test_hand_cashflow_attribution_and_industry_totals():
    result = attribute_result(hand_ledger(), 1_000., {"A": "TECH", "B": "TECH"})
    stocks = result.stocks.set_index("asset")
    assert stocks.loc["A", "gross_pnl"] == 20.
    assert stocks.loc["A", "cost"] == 3.
    assert stocks.loc["A", "net_pnl"] == 17.
    assert stocks.loc["B", "net_pnl"] == -13.
    assert stocks.return_contribution.sum() == pytest.approx(.004)
    assert stocks.commission.isna().all()  # Components absent from a baseline ledger stay unknown.
    assert result.industries.iloc[0].net_pnl == 4.
    assert result.industries.iloc[0].asset_count == 2
    assert result.reconciliation["passed"]
    assert result.reconciliation["ending_cash"] == 842.


def test_fully_closed_stock_is_retained_even_with_no_final_position():
    ledger = hand_ledger()
    ledger.trades.loc[2, ["units", "notional"]] = [10., 120.]
    ledger.positions = ledger.positions[~((ledger.positions.date == ledger.daily.date.iloc[-1]) & (ledger.positions.asset == "A"))]
    ledger.daily.loc[1, "cash"] = 914.
    attributed = attribute_result(ledger, 1_000.)
    a = attributed.stocks.set_index("asset").loc["A"]
    assert a.ending_units == 0.
    assert a.net_pnl == 17.
    assert attributed.reconciliation["passed"]


def test_reconciliation_detects_cash_and_mark_rewrite_and_unknown_industry():
    ledger = hand_ledger()
    ledger.daily.loc[1, "cash"] += 1
    ledger.positions.loc[3, "value"] += 2
    attributed = attribute_result(ledger, 1_000., {"A": "TECH"})
    assert not attributed.reconciliation["passed"]
    assert {"cash", "position value", "portfolio pnl"}.issubset(attributed.reconciliation["errors"])
    assert set(attributed.industries.industry) == {"TECH", "UNKNOWN"}


def test_attribution_accepts_empty_fills_and_zero_holdings():
    ledger = hand_ledger()
    ledger.trades = ledger.trades.iloc[:0]
    ledger.positions = ledger.positions.iloc[:0]
    ledger.daily["cash"] = ledger.daily["nav"] = 1_000.
    ledger.daily["cost"] = 0.
    result = attribute_result(ledger, 1_000.)
    assert result.stocks.empty
    assert result.reconciliation["passed"]
    assert result.reconciliation["attributed_pnl"] == 0.
