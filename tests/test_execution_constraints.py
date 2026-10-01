from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from cfquant.config import Config
from cfquant.engine import run_backtest
from cfquant.execution_constraints import ExecutionPolicy, audit_constrained_result, run_constrained_backtest


def fixture_execution(cash=10_000.0, n=5):
    dates = pd.bdate_range("2023-08-24", periods=n)
    market = pd.DataFrame({"date": dates, "asset": "A", "open": 10., "close": 10.,
                           "raw_open": 10., "raw_close": 10., "volume": 100_000.,
                           "up_limit": 11., "down_limit": 9.})
    scores = pd.DataFrame(1., index=dates, columns=["A"])
    weights = scores.copy()
    config = Config(start=str(dates[1].date()), end=str(dates[-1].date()),
                    initial_cash=cash, holdings=1, rebalance_every=1)
    policy = ExecutionPolicy(liquidity_window=1, participation=1.)
    return market, dates, scores, weights, config, policy


def test_lots_and_minimum_commission_change_actual_fills():
    market, dates, scores, weights, cfg, policy = fixture_execution(cash=1_000.)
    cfg = replace(cfg, end=str(dates[1].date()))
    baseline = run_backtest(market, dates, scores, cfg, target_weights=weights)
    actual = run_constrained_backtest(market, dates, scores, cfg, target_weights=weights, policy=policy)
    assert not baseline.trades.empty
    assert actual.trades.empty  # A 100-share lot plus its commission is unaffordable.
    assert actual.daily.iloc[0].cash == 1_000.
    assert (actual.orders.reason == "lot_or_cash").all()
    assert audit_constrained_result(actual, cfg.initial_cash, policy)["passed"]


def test_minimum_commission_stamp_schedule_and_slippage_cashflow():
    market, dates, scores, weights, cfg, policy = fixture_execution(n=3)
    weights.loc[dates[1]:] = 0.
    result = run_constrained_backtest(market, dates, scores, cfg, target_weights=weights, policy=policy)
    buy, sell = result.trades.iloc[0], result.trades.iloc[1]
    assert buy.raw_equivalent_shares == pytest.approx(900.)
    assert buy.commission == 5.
    assert buy.stamp_tax == 0.
    assert buy.slippage_cost == pytest.approx(4.5)
    assert sell.date == pd.Timestamp("2023-08-28")
    assert sell.stamp_tax == pytest.approx(4.5)
    assert sell.cost == pytest.approx(14.)
    assert result.daily.iloc[-1].nav == pytest.approx(9_976.5)
    assert audit_constrained_result(result, cfg.initial_cash, policy)["passed"]
    expensive = replace(policy, stamp_tax_schedule=(("1900-01-01", .002),))
    changed = run_constrained_backtest(market, dates, scores, cfg, target_weights=weights, policy=expensive)
    assert changed.daily.iloc[-1].nav == pytest.approx(9_963.)


def test_adjusted_conversion_and_full_odd_lot_sell_do_not_create_actions():
    market, dates, scores, weights, cfg, policy = fixture_execution(cash=2_000., n=3)
    policy = replace(policy, slippage=0., stamp_tax_schedule=(("1900-01-01", 0.),))
    weights.loc[dates[1]:] = 0.
    market.loc[market.date == dates[2], ["open", "close"]] = 15.
    result = run_constrained_backtest(market, dates, scores, cfg, target_weights=weights, policy=policy)
    buy, sell = result.trades.iloc[0], result.trades.iloc[1]
    assert buy.units == pytest.approx(100.)
    assert sell.units == pytest.approx(100.)
    assert sell.raw_equivalent_shares == pytest.approx(150.)
    assert sell.liquidation
    assert result.daily.iloc[-1].cash == pytest.approx(2_490.)
    assert result.positions.date.max() == dates[1]
    assert audit_constrained_result(result, cfg.initial_cash, policy)["passed"]


def test_liquidity_cap_is_lagged_and_lot_rounded():
    market, dates, scores, weights, cfg, policy = fixture_execution(cash=1e8)
    policy = replace(policy, participation=.01)
    cfg = replace(cfg, end=str(dates[2].date()))
    result = run_constrained_backtest(market, dates, scores, cfg, target_weights=weights, policy=policy)
    altered = market.copy()
    altered.loc[altered.date >= dates[2], "volume"] *= 1000
    changed = run_constrained_backtest(altered, dates, scores, cfg, target_weights=weights, policy=policy)
    pd.testing.assert_frame_equal(result.trades, changed.trades)
    assert (result.trades.notional <= result.trades.liquidity_budget + 1e-7).all()
    assert np.allclose(result.trades.raw_equivalent_shares / 100, np.round(result.trades.raw_equivalent_shares / 100))
    assert audit_constrained_result(result, cfg.initial_cash, policy)["passed"]


@pytest.mark.parametrize(("field", "reason"), [("raw_open", "missing_raw_open"), ("open", "missing_open"), ("up_limit", "missing_limit")])
def test_unavailable_execution_inputs_block_fills(field, reason):
    market, dates, scores, weights, cfg, policy = fixture_execution()
    cfg = replace(cfg, end=str(dates[1].date()))
    market.loc[market.date == dates[1], field] = np.nan
    result = run_constrained_backtest(market, dates, scores, cfg, target_weights=weights, policy=policy)
    assert result.trades.empty
    assert result.orders.iloc[0].reason == reason


def test_execution_prefix_ignores_future_prices_and_audit_detects_cost_rewrite():
    market, dates, scores, weights, cfg, policy = fixture_execution()
    weights.loc[dates[2]:] = .4
    result = run_constrained_backtest(market, dates, scores, cfg, target_weights=weights, policy=policy)
    changed_market = market.copy()
    changed_market.loc[changed_market.date >= dates[4], ["open", "close", "raw_open", "raw_close"]] *= 2
    changed = run_constrained_backtest(changed_market, dates, scores, cfg, target_weights=weights, policy=policy)
    pd.testing.assert_frame_equal(result.trades[result.trades.date < dates[4]], changed.trades[changed.trades.date < dates[4]])
    pd.testing.assert_frame_equal(result.daily[result.daily.date < dates[4]], changed.daily[changed.daily.date < dates[4]])
    result.trades.loc[0, "cost"] += 1
    audit = audit_constrained_result(result, cfg.initial_cash, policy)
    assert not audit["passed"]
    assert "cost components" in audit["errors"]
    assert "cash" in audit["errors"]


def test_audit_rejects_same_day_resale_of_new_purchase():
    market, dates, scores, weights, cfg, policy = fixture_execution()
    cfg = replace(cfg, end=str(dates[1].date()))
    result = run_constrained_backtest(market, dates, scores, cfg, target_weights=weights, policy=policy)
    fake = result.trades.iloc[0].copy()
    fake["side"] = "sell"
    fake["stamp_tax"] = fake["notional"] * policy.stamp_tax_rate(fake.date)
    fake["cost"] = fake.commission + fake.stamp_tax + fake.slippage_cost
    result.trades = pd.concat([result.trades, fake.to_frame().T], ignore_index=True)
    assert "T+1" in audit_constrained_result(result, cfg.initial_cash, policy)["errors"]


@pytest.mark.parametrize("kwargs", [{"lot_size": 0}, {"min_commission": -1}, {"participation": 0},
                                    {"stamp_tax_schedule": (("2024-01-01", .001), ("2023-01-01", .001))}])
def test_invalid_execution_policy_rejected(kwargs):
    with pytest.raises(ValueError):
        ExecutionPolicy(**kwargs)
