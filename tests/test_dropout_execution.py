"""Top-K dropout accounting, real fills, fee floors and legacy defaults."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from cfquant.analytics import reconcile
from cfquant.config import Config
from cfquant.engine import run_backtest


def fixture(assets=6, sessions=7, cash=100_000., holdings=3, rate=0.):
    calendar = pd.bdate_range("2024-01-02", periods=sessions)
    codes = [f"S{k:03}" for k in range(assets)]
    market = pd.DataFrame([
        {"date": date, "asset": code, "open": 100., "close": 100.,
         "raw_open": 100., "raw_close": 100., "up_limit": 110., "down_limit": 90.,
         "volume": 1_000_000.}
        for date in calendar for code in codes])
    scores = pd.DataFrame(np.tile(np.arange(assets, 0, -1, dtype=float), (sessions, 1)),
                          index=calendar, columns=codes)
    cfg = Config(start=str(calendar[1].date()), end=str(calendar[-1].date()),
                 holdings=holdings, rebalance_every=1, initial_cash=cash,
                 buy_cost=rate, sell_cost=rate)
    return market, calendar, scores, cfg


def check(run, cfg, minimum=0.):
    assert reconcile(run, cfg.initial_cash, cfg.buy_cost, cfg.sell_cost, minimum)["passed"]
    assert run.daily.cash.ge(0).all()
    assert run.positions.units.ge(0).all()
    assert run.trades.signal_date.lt(run.trades.date).all()


def day_positions(run, date):
    return run.positions.loc[run.positions.date == date].set_index("asset").units


def test_top50_drop5_initialization_replacement_limit_and_retained_units():
    market, calendar, scores, cfg = fixture(assets=80, sessions=6, holdings=50)
    scores.iloc[1:] = np.arange(80, dtype=float)
    run = run_backtest(market, calendar, scores, cfg, dropout=5, minimum_commission=5)
    first = run.trades.loc[run.trades.date == calendar[1]]
    assert first.side.eq("buy").all() and len(first) == 50
    assert set(first.asset) == set(scores.columns[:50])
    previous = day_positions(run, calendar[1])
    for date in calendar[2:]:
        fills = run.trades.loc[run.trades.date == date]
        assert int(fills.side.eq("sell").sum()) <= 5
        assert int(fills.side.eq("buy").sum()) <= 5
        current = day_positions(run, date)
        retained = previous.index.intersection(current.index)
        np.testing.assert_array_equal(previous.reindex(retained), current.reindex(retained))
        assert len(current) == 50
        previous = current
    assert run.daily.holdings_count.le(50).all()
    check(run, cfg, 5)


def test_price_changes_do_not_resize_held_units_or_create_daily_fee_churn():
    market, calendar, scores, cfg = fixture(assets=5, holdings=3)
    for i, date in enumerate(calendar[2:], 1):
        row = (market.date == date) & (market.asset == "S000")
        market.loc[row, ["open", "close", "raw_open", "raw_close"]] = 100. + i * 7
        market.loc[row, ["up_limit", "down_limit"]] = [200., 10.]
    run = run_backtest(market, calendar, scores, cfg, dropout=1, minimum_commission=5)
    assert len(run.trades) == 3
    assert run.daily.cost.iloc[1:].eq(0).all()
    for date in calendar[2:]:
        pd.testing.assert_series_equal(day_positions(run, calendar[1]), day_positions(run, date))
    check(run, cfg, 5)


def test_manual_minimum_fee_and_cash_split_for_new_entries():
    market, calendar, scores, cfg = fixture(assets=3, sessions=4, holdings=2, cash=100., rate=.001)
    scores.iloc[1:, :] = [1., 2., 3.]
    run = run_backtest(market, calendar, scores, cfg, dropout=1, minimum_commission=5)
    first = run.trades.loc[run.trades.date == calendar[1]]
    assert first.notional.tolist() == pytest.approx([45., 45.])
    assert first.cost.tolist() == pytest.approx([5., 5.])
    second = run.trades.loc[run.trades.date == calendar[2]]
    assert second.side.tolist() == ["sell", "buy"]
    assert second.notional.tolist() == pytest.approx([45., 35.])
    assert second.cost.tolist() == pytest.approx([5., 5.])
    assert run.daily.nav.tolist() == pytest.approx([90., 80., 80.])
    assert run.daily.cash.tolist() == pytest.approx([0., 0., 0.])
    assert day_positions(run, calendar[2])["S001"] == pytest.approx(.45)
    assert day_positions(run, calendar[2])["S002"] == pytest.approx(.35)
    check(run, cfg, 5)
    assert not reconcile(run, cfg.initial_cash, cfg.buy_cost, cfg.sell_cost)["passed"]


def test_rate_fee_wins_over_floor_and_zero_cost_mode_has_no_hidden_commission():
    market, calendar, scores, cfg = fixture(assets=3, holdings=2, rate=.001)
    charged = run_backtest(market, calendar, scores, cfg, dropout=1, minimum_commission=5)
    np.testing.assert_allclose(charged.trades.cost, charged.trades.notional * .001)
    free_cfg = replace(cfg, buy_cost=0., sell_cost=0.)
    free = run_backtest(market, calendar, scores, free_cfg, dropout=1, minimum_commission=0)
    assert free.trades.cost.eq(0).all()
    assert free.daily.nav.eq(cfg.initial_cash).all()
    check(charged, cfg, 5)
    check(free, free_cfg)


def test_no_buy_when_minimum_fees_exhaust_cash_and_no_charge_on_unfilled_orders():
    market, calendar, scores, cfg = fixture(assets=3, holdings=2, cash=10.)
    for dropout in (None, 1):
        run = run_backtest(market, calendar, scores, cfg, dropout=dropout, minimum_commission=5)
        assert run.trades.empty
        assert run.daily.cash.eq(10).all() and run.daily.cost.eq(0).all()
        assert run.orders.filled_units.eq(0).all()
        check(run, cfg, 5)


def test_blocked_sell_keeps_units_and_does_not_release_a_buy_slot():
    market, calendar, scores, cfg = fixture(assets=5, sessions=4, holdings=3)
    scores.iloc[1:] = np.arange(5, dtype=float)
    for date in calendar[2:]:
        market.loc[(market.date == date) & (market.asset == "S000"), "down_limit"] = 100.
    run = run_backtest(market, calendar, scores, cfg, dropout=2, minimum_commission=5)
    second = run.trades.loc[run.trades.date == calendar[2]]
    assert second.side.tolist() == ["sell", "buy"]
    assert set(second.asset) == {"S001", "S004"}
    assert set(day_positions(run, calendar[2]).index) == {"S000", "S002", "S004"}
    assert day_positions(run, calendar[2])["S000"] == day_positions(run, calendar[1])["S000"]
    assert run.daily.holdings_count.le(3).all()
    assert "lower_limit" in run.orders.reason.tolist()
    check(run, cfg, 5)


def test_blocked_buy_keeps_its_cash_allocation_and_vacancy_without_replacement_lookup():
    market, calendar, scores, cfg = fixture(assets=5, sessions=3, holdings=3, cash=300.)
    scores.iloc[1:] = np.arange(5, dtype=float)
    market.loc[(market.date == calendar[2]) & (market.asset == "S004"), "up_limit"] = 100.
    run = run_backtest(market, calendar, scores, cfg, dropout=2, minimum_commission=5)
    final = day_positions(run, calendar[2])
    assert set(final.index) == {"S002", "S003"}
    assert run.daily.cash.iloc[-1] == pytest.approx(90.)
    assert run.daily.cost.iloc[-1] == pytest.approx(15.)
    blocked = run.orders.loc[run.orders.reason == "upper_limit"].iloc[0]
    assert blocked.asset == "S004" and blocked.filled_units == 0
    check(run, cfg, 5)


@pytest.mark.parametrize("field", ["open", "raw_open"])
def test_missing_execution_price_blocks_without_fabricating_fee(field):
    market, calendar, scores, cfg = fixture(assets=3, sessions=3, holdings=2)
    market.loc[(market.date == calendar[1]) & (market.asset == "S000"), field] = np.nan
    run = run_backtest(market, calendar, scores, cfg, dropout=1, minimum_commission=5)
    first = run.trades.loc[run.trades.date == calendar[1]]
    assert first.asset.tolist() == ["S001"]
    assert run.daily.cash.iloc[0] == pytest.approx(50_000.)
    assert run.daily.holdings_count.iloc[0] == 1
    assert run.daily.cost.iloc[0] == 5.
    check(run, cfg, 5)


def test_missing_scores_rank_held_assets_last_and_cannot_be_bought():
    market, calendar, scores, cfg = fixture(assets=5, sessions=4, holdings=3)
    scores.iloc[1:, :] = [np.nan, 8., 7., 6., np.nan]
    run = run_backtest(market, calendar, scores, cfg, dropout=1)
    second = run.trades.loc[run.trades.date == calendar[2]]
    assert second.asset.tolist() == ["S000", "S003"]
    assert "S004" not in run.trades.asset.values
    check(run, cfg)


def test_ties_are_deterministic_by_code_even_when_columns_are_reordered():
    market, calendar, scores, cfg = fixture(assets=5, holdings=3)
    scores[:] = 1.
    a = run_backtest(market, calendar, scores, cfg, dropout=1)
    b = run_backtest(market, calendar, scores.loc[:, list(reversed(scores.columns))], cfg, dropout=1)
    assert a.trades.asset.tolist() == b.trades.asset.tolist() == ["S000", "S001", "S002"]
    assert len(a.trades) == 3
    check(a, cfg)
    check(b, cfg)


def test_scarce_candidates_do_not_force_sell_without_replacement():
    market, calendar, scores, cfg = fixture(assets=3, holdings=3)
    run = run_backtest(market, calendar, scores, cfg, dropout=2, minimum_commission=5)
    assert len(run.trades) == 3 and run.trades.side.eq("buy").all()
    assert run.daily.holdings_count.eq(3).all()
    check(run, cfg, 5)


def test_dropout_zero_initializes_and_fills_vacancies_but_never_replaces():
    market, calendar, scores, cfg = fixture(assets=5, holdings=3)
    scores.iloc[1:] = np.arange(5, dtype=float)
    run = run_backtest(market, calendar, scores, cfg, dropout=0)
    assert run.trades.asset.tolist() == ["S000", "S001", "S002"]
    check(run, cfg)


def test_rebalance_timing_and_future_score_mutations_do_not_change_past():
    market, calendar, scores, cfg = fixture(assets=7, holdings=3)
    cfg = replace(cfg, rebalance_every=2)
    scores.iloc[2:] = np.arange(7, dtype=float)
    original = run_backtest(market, calendar, scores, cfg, dropout=1)
    assert set(original.trades.date.unique()) <= {calendar[1], calendar[3], calendar[5]}
    changed = scores.copy()
    changed.iloc[4:] = np.arange(7, dtype=float)[::-1] * 100
    future = run_backtest(market, calendar, changed, cfg, dropout=1)
    pd.testing.assert_frame_equal(original.daily.iloc[:4], future.daily.iloc[:4])
    pd.testing.assert_frame_equal(original.trades.loc[original.trades.date <= calendar[4]],
                                  future.trades.loc[future.trades.date <= calendar[4]])
    check(original, cfg)


def test_partial_liquidity_sell_keeps_slot_and_held_residue():
    market, calendar, scores, cfg = fixture(assets=3, sessions=24, holdings=1, cash=1000.)
    cfg = replace(cfg, start=str(calendar[21].date()))
    scores.iloc[21:] = [0., 5., 1.]
    # Prior 20-session notional mean * 1% = 100 currency units per session.
    market["volume"] = 100.
    market.loc[(market.date == calendar[21]) & (market.asset == "S000"), "volume"] = 0.
    run = run_backtest(market, calendar, scores, cfg, dropout=1, participation_limit=.01)
    first_units = day_positions(run, calendar[21])["S000"]
    assert first_units == pytest.approx(1.)
    second_units = day_positions(run, calendar[22])["S000"]
    assert second_units == pytest.approx(.05)
    assert run.trades.loc[run.trades.date == calendar[22]].side.tolist() == ["sell"]
    assert "S001" not in day_positions(run, calendar[22]).index
    check(run, cfg)


def test_sell_fee_cannot_overdraw_cash_when_position_collapses_below_fee():
    market, calendar, scores, cfg = fixture(assets=2, sessions=3, holdings=1, cash=10.)
    scores.iloc[1:] = [0., 2.]
    row = (market.date == calendar[2]) & (market.asset == "S000")
    market.loc[row, ["open", "close", "raw_open", "raw_close"]] = 1.
    market.loc[row, ["up_limit", "down_limit"]] = [2., .1]
    run = run_backtest(market, calendar, scores, cfg, dropout=1, minimum_commission=5)
    assert len(run.trades) == 1
    assert "insufficient_cash_for_fee" in run.orders.reason.values
    assert run.daily.cash.iloc[-1] == 0
    assert run.daily.nav.iloc[-1] == pytest.approx(.05)
    check(run, cfg, 5)


def test_minimum_fee_also_supported_by_existing_target_weight_path():
    market, calendar, scores, cfg = fixture(assets=3, sessions=3, holdings=2, cash=100., rate=.001)
    weights = scores * 0
    weights.iloc[:, :2] = .5
    run = run_backtest(market, calendar, scores, cfg, target_weights=weights,
                       reserve_cost_buffer=False, minimum_commission=5)
    assert run.trades.iloc[:2].notional.tolist() == pytest.approx([45., 45.])
    assert run.trades.iloc[:2].cost.tolist() == pytest.approx([5., 5.])
    check(run, cfg, 5)


def test_default_parameters_preserve_existing_outputs():
    market, calendar, scores, cfg = fixture(assets=5, holdings=3, rate=.001)
    default = run_backtest(market, calendar, scores, cfg)
    explicit = run_backtest(market, calendar, scores, cfg, dropout=None, minimum_commission=0)
    for field in ("daily", "trades", "positions", "orders"):
        pd.testing.assert_frame_equal(getattr(default, field), getattr(explicit, field))
    check(default, cfg)


@pytest.mark.parametrize("kwargs", [{"dropout": -1}, {"dropout": 4}, {"dropout": True},
                                   {"dropout": 1.5}, {"minimum_commission": -1},
                                   {"minimum_commission": np.nan}, {"minimum_commission": True}])
def test_invalid_dropout_and_fee_arguments_are_rejected(kwargs):
    market, calendar, scores, cfg = fixture()
    with pytest.raises(ValueError):
        run_backtest(market, calendar, scores, cfg, **kwargs)


def test_dropout_cannot_silently_ignore_explicit_target_weights():
    market, calendar, scores, cfg = fixture()
    with pytest.raises(ValueError, match="target_weights"):
        run_backtest(market, calendar, scores, cfg, dropout=1, target_weights=scores * 0)
