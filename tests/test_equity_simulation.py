from dataclasses import replace
from datetime import date, datetime, timedelta
import io

import numpy as np
import pandas as pd
import pytest

from cfquant.equity_simulation import (
    COG_ID, V3_ID, CogSelection, EquityQuote, EquityTargets, build_cogalpha_selection,
    build_v3_targets, fetch_equity_quotes, plan_cogalpha_orders, plan_v3_orders,
)
from cfquant.simulation_signals import SHANGHAI


SIGNAL = date(2026, 10, 20)
EXECUTION = date(2026, 10, 21)
NOW = datetime(2026, 10, 21, 10, tzinfo=SHANGHAI)
# A fixed synthetic official-open-session list, used only by these offline tests.
DAYS = tuple(day.date() for day in pd.bdate_range("2025-12-01", "2027-01-29"))
ASSETS = [f"{600000 + i:06}.SH" for i in range(100)]


@pytest.fixture
def scores():
    return pd.DataFrame({
        "date": pd.Timestamp(SIGNAL), "asset": ASSETS, "eligible": True,
        "v3_lightgbm_frozen_reference": np.arange(100, dtype=float),
        "cogalpha_731_rank_frozen_reference": np.arange(100, dtype=float),
        "volatility": 0.02, "industry": [f"industry{i % 10}" for i in range(100)],
    })


def quotes(assets=ASSETS):
    return {asset: EquityQuote(asset, 10.0, NOW, 11.0, 9.0) for asset in assets}


def cog(scores, **kwargs):
    params = dict(signal_asof=SIGNAL, execution_date=EXECUTION, trading_days=DAYS,
                  account_binding="cogalpha-account", account_nav=200_000.0,
                  available_cash=200_000.0, holdings={}, available_shares={},
                  quotes=quotes(), now=NOW)
    params.update(kwargs)
    return plan_cogalpha_orders(scores, **params)


def test_initial_cog_buys_50_lots_with_fees_inside_actual_cash(scores):
    result = cog(scores)
    assert len(result.intents) == 50
    assert all(order.side == "buy" and order.quantity % 100 == 0 for order in result.intents)
    spend = sum(order.quantity * order.limit_price
                + max(order.quantity * order.limit_price * 0.0025, 5) for order in result.intents)
    assert spend <= 200_000
    assert result.reserved_cash == spend
    assert result.selection is not None
    assert len(result.selection.buy_candidates) == 55
    assert result.metadata["retained_shares_resized"] is False


def test_cog_retains_shares_and_missing_held_scores_rank_last(scores):
    holdings = {asset: 200 for asset in ASSETS[50:]}
    held_missing = ASSETS[-1]
    scores.loc[scores.asset.eq(held_missing), "cogalpha_731_rank_frozen_reference"] = np.nan
    result = cog(scores, holdings=holdings, available_shares=holdings, available_cash=100_000.0)
    sells = [order for order in result.intents if order.side == "sell"]
    assert [order.asset for order in sells] == [held_missing]
    assert sells[0].quantity == holdings[held_missing]
    assert not [order for order in result.intents if order.side == "buy"]
    retained = set(holdings) - {held_missing}
    assert not retained.intersection(order.asset for order in result.intents)


def test_cog_failed_exit_never_releases_slot_or_assumes_sale_cash(scores):
    holdings = {asset: 100 for asset in ASSETS[:50]}
    result = cog(scores, holdings=holdings, available_shares={}, available_cash=150_000.0)
    assert not result.intents
    assert len(result.selection.sell_candidates) == 5
    assert sum(row["reason"] == "no_sellable_lots" for row in result.blocked) == 5
    assert any(row["reason"] == "no_actual_holding_slot" for row in result.blocked)
    assert result.reserved_cash == 0


def test_frozen_cog_selection_prevents_repeated_five_name_exits(scores):
    holdings = {asset: 100 for asset in ASSETS[:50]}
    first = cog(scores, holdings=holdings, available_shares=holdings, available_cash=150_000.0)
    assert len(first.intents) == 5
    assert all(order.side == "sell" for order in first.intents)
    state = CogSelection.from_dict(first.selection.to_dict())
    remaining = {asset: shares for asset, shares in holdings.items() if asset not in state.sell_candidates}
    second = cog(scores, holdings=remaining, available_shares=remaining,
                 available_cash=155_000.0, selection=state)
    assert len(second.intents) == 5
    assert all(order.side == "buy" for order in second.intents)
    assert all(order.asset in state.buy_candidates for order in second.intents)
    assert second.selection.selection_id == first.selection.selection_id


def test_cog_partial_exit_still_occupies_slot(scores):
    holdings = {asset: 100 for asset in ASSETS[:50]}
    holdings[ASSETS[0]] = 5000
    result = cog(scores, holdings=holdings, available_shares=holdings,
                 available_cash=100_000.0, max_order_notional=20_000.0)
    order = next(order for order in result.intents if order.asset == ASSETS[0])
    assert order.quantity == 2000
    assert all(order.side == "sell" for order in result.intents)


def test_quote_limit_and_timestamp_fail_closed(scores):
    current = quotes()
    current[ASSETS[-1]] = replace(current[ASSETS[-1]], price=11.0)
    current[ASSETS[-2]] = replace(current[ASSETS[-2]], asof=NOW - timedelta(seconds=31))
    current[ASSETS[-3]] = replace(current[ASSETS[-3]], asof=NOW + timedelta(seconds=1))
    current[ASSETS[-4]] = replace(current[ASSETS[-4]], price_basis="adjusted")
    result = cog(scores, quotes=current)
    reasons = {row["reason"] for row in result.blocked}
    assert {"upper_limit", "stale_or_future_quote", "missing_raw_quote"} <= reasons
    assert not set(ASSETS[-4:]).intersection(order.asset for order in result.intents)


def test_missing_score_allows_no_candidate_or_forced_exit(scores):
    scores["cogalpha_731_rank_frozen_reference"] = np.nan
    holdings = {ASSETS[0]: 100}
    result = cog(scores, holdings=holdings, available_shares=holdings, available_cash=199_000.0)
    assert not result.intents
    assert not result.selection.sell_candidates


def test_frozen_selection_tamper_and_extra_holdings_rejected(scores):
    initial = cog(scores)
    data = initial.selection.to_dict()
    data["buy_candidates"] = data["buy_candidates"][1:]
    with pytest.raises(ValueError, match="identity changed"):
        CogSelection.from_dict(data)
    with pytest.raises(ValueError, match="permitted assets"):
        cog(scores, holdings={ASSETS[0]: 100}, available_shares={ASSETS[0]: 100},
            available_cash=199_000.0, selection=initial.selection)


def test_intent_identity_independent_of_quote_and_quantity(scores):
    first = cog(scores)
    changed = {asset: replace(quote, price=10.50) for asset, quote in quotes().items()}
    second = cog(scores, quotes=changed)
    one = {order.asset: order.intent_id for order in first.intents}
    two = {order.asset: order.intent_id for order in second.intents}
    assert one == two
    assert all(order.strategy_id == COG_ID for order in first.intents)


def v3_inputs(scores):
    dates = pd.DatetimeIndex([day for day in DAYS if day <= SIGNAL])[-160:]
    t = np.arange(len(dates))
    frames = []
    for i, asset in enumerate(ASSETS):
        close = 10 * np.exp(0.0005 * t + 0.004 * np.sin(t / (3 + i % 5)))
        frames.append(pd.DataFrame({"date": dates, "asset": asset, "close": close}))
    index = pd.Series(4000 + t, index=dates)
    return pd.concat(frames, ignore_index=True), index


def build_v3(scores, **kwargs):
    market, index = v3_inputs(scores)
    params = dict(signal_asof=SIGNAL, execution_date=EXECUTION, trading_days=DAYS)
    params.update(kwargs)
    return build_v3_targets(scores, market, index, **params)


def test_v3_reuses_managed_policy_and_retains_buffer(scores):
    targets = build_v3(scores, previous_target_constituents=[ASSETS[35]])
    assert targets.strategy_id == V3_ID
    assert targets.rebalance_due
    assert len(targets.weights) == 50
    assert ASSETS[35] in targets.weights
    assert sum(targets.weights.values()) <= 0.95 + 1e-8
    assert max(targets.weights.values()) <= 0.04 + 1e-8
    assert targets.metadata["policy"]["target_volatility"] == 0.20
    assert targets.metadata["policy"]["buffer"] == 20
    assert targets.metadata["policy"]["defensive_scale"] == 0.75
    sectors = scores.set_index("asset").industry
    for group in sectors.unique():
        assert sum(weight for asset, weight in targets.weights.items() if sectors[asset] == group) <= 0.25 + 1e-8


def test_v3_twenty_session_schedule_not_weekday_guess(scores):
    later = date(2026, 10, 22)
    updated = scores.assign(date=pd.Timestamp(EXECUTION))
    market, index = v3_inputs(scores)
    # A non-rebalance day does not require a newly computed risk estimate.
    result = build_v3_targets(updated, market, index, signal_asof=EXECUTION,
                              execution_date=later, trading_days=DAYS)
    assert result.rebalance_due is False
    assert result.weights == {}
    twentieth = DAYS[DAYS.index(EXECUTION) + 20]
    previous = DAYS[DAYS.index(twentieth) - 1]
    with pytest.raises(ValueError, match="current"):
        build_v3_targets(scores.assign(date=pd.Timestamp(previous)), market, index,
                          signal_asof=previous, execution_date=twentieth, trading_days=DAYS)


def test_v3_risk_history_blocks_future_or_missing_dates(scores):
    market, index = v3_inputs(scores)
    args = dict(signal_asof=SIGNAL, execution_date=EXECUTION, trading_days=DAYS)
    with pytest.raises(ValueError, match="history misses official"):
        build_v3_targets(scores, market, index.drop(index.index[-10]), **args)
    market.loc[market.index[-1], "date"] = pd.Timestamp(EXECUTION)
    with pytest.raises(ValueError, match="future"):
        build_v3_targets(scores, market, index, **args)


def test_v3_cash_and_slots_do_not_assume_filled_sells(scores):
    targets = build_v3(scores)
    holdings = {asset: 100 for asset in ASSETS[:50]}
    result = plan_v3_orders(targets, account_binding="v3-account", account_nav=200_000.0,
                            available_cash=150_000.0, holdings=holdings, available_shares=holdings,
                            quotes=quotes(), now=NOW)
    assert result.intents
    assert all(order.side == "sell" for order in result.intents)
    assert result.reserved_cash == 0
    assert any(row["reason"] == "no_actual_holding_slot" for row in result.blocked)
    fresh = plan_v3_orders(targets, account_binding="v3-account", account_nav=200_000.0,
                           available_cash=13_000.0, holdings={}, available_shares={},
                           quotes=quotes(), now=NOW)
    assert 0 < fresh.reserved_cash <= 3000 + 1e-8
    assert all(order.quantity % 100 == 0 for order in fresh.intents)


def test_v3_t_plus_one_and_current_down_limit_block_exit(scores):
    targets = EquityTargets(V3_ID, SIGNAL, EXECUTION, {}, True, {})
    holdings = {ASSETS[0]: 1000, ASSETS[1]: 1000}
    prices = quotes()
    prices[ASSETS[1]] = replace(prices[ASSETS[1]], price=9.0)
    result = plan_v3_orders(targets, account_binding="v3-account", account_nav=200_000.0,
                            available_cash=180_000.0, holdings=holdings,
                            available_shares={ASSETS[0]: 300, ASSETS[1]: 1000},
                            quotes=prices, now=NOW)
    assert len(result.intents) == 1
    assert result.intents[0].quantity == 300
    assert any(row["reason"] == "lower_limit" for row in result.blocked)


def test_v3_failed_exits_cannot_spend_risk_budget_cash_reserve():
    targets = EquityTargets(V3_ID, SIGNAL, EXECUTION, {ASSETS[2]: .04}, True, {})
    holdings = {ASSETS[0]: 1000}
    result = plan_v3_orders(targets, account_binding="v3-account", account_nav=20_000.0,
                            available_cash=10_000.0, holdings=holdings, available_shares={},
                            quotes=quotes(), now=NOW)
    assert not result.intents
    assert result.metadata["minimum_cash_reserve"] == 19_200.0


@pytest.mark.parametrize("change", ["fractional", "future_scores", "bad_calendar", "future_order"])
def test_invalid_execution_inputs_rejected(scores, change):
    args = {}
    if change == "fractional":
        args["holdings"] = {ASSETS[0]: 100.5}
    elif change == "future_scores":
        scores = scores.assign(date=pd.Timestamp(EXECUTION))
    elif change == "bad_calendar":
        args["trading_days"] = tuple(day for day in DAYS if day != EXECUTION)
    else:
        args["now"] = NOW - timedelta(days=1)
    with pytest.raises(ValueError):
        cog(scores, **args)


def quote_line(asset, stamp="20261021100000"):
    fields = ["0"] * 52
    fields[2], fields[3], fields[30], fields[47], fields[48] = asset[:6], "10.0", stamp, "11.0", "9.0"
    code = asset[-2:].lower() + asset[:6]
    return 'v_' + code + '="' + "~".join(fields) + '";'


def test_batch_quote_single_call_code_timestamp_and_limits():
    assets = ASSETS[:50]
    calls = []

    def opener(request, timeout):
        calls.append((request.full_url, timeout))
        return io.BytesIO("\n".join(quote_line(asset) for asset in reversed(assets)).encode())

    data = fetch_equity_quotes(assets, now=NOW, opener=opener)
    assert len(calls) == 1 and calls[0][1] == 12
    assert len(data) == 50
    assert data[assets[0]].asof == NOW
    assert data[assets[0]].up_limit == 11
    assert data[assets[0]].down_limit == 9
    assert data[assets[0]].source == "tencent_public"


@pytest.mark.parametrize("problem", ["missing", "wrong_code", "duplicate", "old", "undated", "adjusted_universe"])
def test_bad_batch_quotes_fail_closed(problem):
    assets = ASSETS[:2]
    body = [quote_line(asset) for asset in assets]
    if problem == "missing":
        body.pop()
    elif problem == "wrong_code":
        body[0] = quote_line(ASSETS[9])
    elif problem == "duplicate":
        body.append(body[0])
    elif problem == "old":
        body[0] = quote_line(assets[0], "20261020150000")
    elif problem == "undated":
        body[0] = quote_line(assets[0], "100000")
    else:
        assets = ["000300.SH"]

    def opener(request, timeout):
        return io.BytesIO("\n".join(body).encode())

    with pytest.raises((ValueError, RuntimeError)):
        fetch_equity_quotes(assets, now=NOW, opener=opener)
