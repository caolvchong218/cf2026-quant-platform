"""Independent accounting and causality checks for the P2/P1 integration."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from cfquant.analytics import performance, reconcile
from cfquant.config import Config
from cfquant.engine import run_backtest
from cfquant.rsrs import (SPECS, Specification, decisions, factors, index_to_market,
                          run_rsrs, run_signal, strategy, volatility_weights)


@pytest.fixture(scope="module")
def legacy():
    """Load the untouched preparation code under a distinct module identity."""
    source = Path(__file__).resolve().parents[1] / "projects/project2_rsrs/rsrs.py"
    spec = importlib.util.spec_from_file_location("rsrs_legacy_reference", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prices(n=900):
    rng = np.random.default_rng(189)
    level = 100 * np.exp(np.cumsum(rng.normal(0, 0.025, n)))
    return pd.DataFrame({"low": level * .98,
                         "high": level * (1.02 + rng.uniform(0, .01, n)),
                         "open": level, "close": level * 1.001},
                        index=pd.bdate_range("2010-01-01", periods=n))


def small_engine_inputs():
    dates = pd.bdate_range("2020-01-01", periods=5)
    data = pd.DataFrame({"open": [100., 100., 110., 120., 130.],
                         "close": [100., 105., 108., 120., 130.]}, index=dates)
    data["high"] = data[["open", "close"]].max(axis=1) * 1.01
    data["low"] = data[["open", "close"]].min(axis=1) * .99
    market = index_to_market(data, "INDEX")
    scores = pd.DataFrame(1., index=dates, columns=["INDEX"])
    cfg = Config(start=str(dates[1].date()), end=str(dates[-1].date()),
                 initial_cash=1000., buy_cost=.01, sell_cost=.01,
                 holdings=1, rebalance_every=1, respect_price_limits=False)
    return data, market, scores, cfg


def test_ols_prior_standardization_and_legacy_formula_equivalence(legacy):
    data = prices()
    actual = factors(data)
    pd.testing.assert_frame_equal(actual, legacy.factors(data))
    for t in (17, 200, 650, 899):
        window = data.iloc[t - 17:t + 1]
        design = np.column_stack([np.ones(18), window.low])
        coefficient = np.linalg.lstsq(design, window.high, rcond=None)[0]
        residual = window.high - design @ coefficient
        assert actual.beta.iloc[t] == pytest.approx(coefficient[1], abs=1e-11)
        expected_r2 = 1 - (residual**2).sum() / ((window.high - window.high.mean())**2).sum()
        assert actual.r2.iloc[t] == pytest.approx(expected_r2)
    assert actual.z.iloc[:617].isna().all()
    past = actual.beta.iloc[17:617]
    assert actual.z.iloc[617] == pytest.approx((actual.beta.iloc[617] - past.mean()) / past.std(ddof=0))


@pytest.mark.parametrize("name", [s.name for s in SPECS] + ["buy_hold", "ma20"])
@pytest.mark.parametrize("cost_bps", [0, 10])
def test_shared_engine_matches_independent_legacy_ledger(legacy, name, cost_bps):
    data = prices(780)
    start, end = str(data.index[650].date()), str(data.index[-1].date())
    if name == "buy_hold":
        decision = pd.Series(1, index=data.index, dtype=int)
    elif name == "ma20":
        decision = (data.close > data.close.rolling(20).mean()).astype(int)
    else:
        old_spec = next(s for s in legacy.SPECS if s.name == name)
        _, decision = legacy.strategy(data, old_spec)
    reference = legacy.backtest(data, decision, start, end, cost_bps)
    actual = run_rsrs(data, name, start, end, cost_bps=cost_bps, initial_cash=1.)
    day = actual.result.daily.set_index("date")
    units = (actual.result.positions.groupby("date").units.sum()
             .reindex(day.index, fill_value=0.))
    np.testing.assert_allclose(day.nav, reference.equity, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(day.cash, reference.cash, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(units, reference.units, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(day.cost, reference.cost, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(day["return"], reference["return"], rtol=1e-12, atol=1e-12)
    assert actual.metrics["sharpe"] == pytest.approx(legacy.metrics(reference)["sharpe"], nan_ok=True)
    assert actual.metrics["max_drawdown"] == pytest.approx(legacy.metrics(reference)["max_drawdown"])
    assert actual.checks["passed"]
    assert actual.metadata["is_executable_instrument"] is False
    assert actual.metadata["respect_price_limits"] is False
    legacy.audit_ledger(reference, cost_bps)


def test_improvement_preserves_signal_and_is_prefix_invariant():
    data = prices(180)
    spec = Specification("small", n=5, m=20)
    start, end = str(data.index[30].date()), str(data.index[-1].date())
    base = run_rsrs(data, spec, start, end)
    improved = run_rsrs(data, spec, start, end, vol_target=.15)
    pd.testing.assert_series_equal(base.decision, improved.decision)
    pd.testing.assert_frame_equal(base.factors, improved.factors)
    changed = data.copy()
    changed.iloc[120:] *= np.linspace(2, 4, len(data) - 120)[:, None]
    future = run_rsrs(changed, spec, start, end, vol_target=.15)
    pd.testing.assert_frame_equal(improved.factors.iloc[:120], future.factors.iloc[:120])
    pd.testing.assert_frame_equal(improved.targets.iloc[:120], future.targets.iloc[:120])
    pd.testing.assert_frame_equal(improved.result.daily.iloc[:90], future.result.daily.iloc[:90])
    assert improved.targets.iloc[:, 0].between(0, 1).all()
    assert improved.checks["passed"]


def test_volatility_scaling_uses_observed_returns_and_cash_warmup():
    data = prices(45)
    decision = pd.Series(1, index=data.index, dtype=int)
    weight = volatility_weights(decision, data.close, .15, 20)
    assert weight.iloc[:20].eq(0).all()
    expected = min(1, .15 / (data.close.pct_change(fill_method=None).iloc[1:21].std(ddof=1) * np.sqrt(252)))
    assert weight.iloc[20] == pytest.approx(expected)
    flat = pd.Series(100., index=data.index)
    flat_weight = volatility_weights(decision, flat)
    assert flat_weight.iloc[:20].eq(0).all()
    assert flat_weight.iloc[20:].eq(1).all()
    decision.iloc[30:] = 0
    assert volatility_weights(decision, data.close).iloc[30:].eq(0).all()


def test_hysteresis_strict_thresholds_and_entry_only_filter():
    score = pd.Series([.7, .8, .9, -.7, -.8, .9, .9, np.nan])
    gate = pd.Series([True, True, False, False, False, False, True, True])
    assert decisions(score, .7, -.7, gate).tolist() == [0, 1, 1, 1, 0, 0, 1, 0]


def test_partial_exposure_manual_cash_share_and_fee_ledger():
    data, market, scores, cfg = small_engine_inputs()
    targets = scores * .5
    result = run_backtest(market, data.index, scores, cfg, target_weights=targets,
                          rebalance_mask=pd.Series(True, index=data.index),
                          reserve_cost_buffer=False)
    first, second = result.daily.iloc[0], result.daily.iloc[1]
    assert first.cash == pytest.approx(495.)
    assert first.cost == pytest.approx(5.)
    assert first.nav == pytest.approx(1020.)
    assert second.opening_nav == pytest.approx(1045.)
    assert second.sell_notional == pytest.approx(27.5)
    assert second.cost == pytest.approx(.275)
    assert second.cash == pytest.approx(522.225)
    assert second.nav == pytest.approx(1035.225)
    assert result.trades.iloc[0].units == pytest.approx(5.)
    assert result.trades.iloc[1].units == pytest.approx(.25)
    assert reconcile(result, 1000., .01, .01)["passed"]


def test_mask_is_formation_dated_and_initial_open_establishes_target():
    data, market, scores, cfg = small_engine_inputs()
    targets = scores.copy()
    targets.iloc[2:] = 0
    mask = pd.Series([False, False, True, False, False], index=data.index)
    result = run_backtest(market, data.index, scores, cfg, target_weights=targets,
                          rebalance_mask=mask, reserve_cost_buffer=False)
    assert result.trades.side.tolist() == ["buy", "sell"]
    assert result.trades.date.tolist() == [data.index[1], data.index[3]]
    assert result.trades.signal_date.tolist() == [data.index[0], data.index[2]]
    # A gap before entry is not a return: 1000 cash buys at the first evaluation
    # open, with real buy cost; liquidation uses the later execution open.
    quantity = 1000 / (100 * 1.01)
    assert result.daily.nav.iloc[0] == pytest.approx(quantity * 105)
    assert result.daily.nav.iloc[-1] == pytest.approx(quantity * 120 * .99)


def test_no_mask_default_preserves_periodic_schedule_and_cost_reserve():
    data, market, scores, cfg = small_engine_inputs()
    default = run_backtest(market, data.index, scores, cfg)
    explicit = run_backtest(market, data.index, scores, cfg,
                            rebalance_mask=None, reserve_cost_buffer=True)
    for field in ("daily", "trades", "positions", "orders"):
        pd.testing.assert_frame_equal(getattr(default, field), getattr(explicit, field))
    assert default.daily.rebalance.all()
    assert default.trades.iloc[0].units == pytest.approx(1000 / 1.02 / 100)
    assert default.daily.cash.iloc[0] == pytest.approx(1000 - 1000 / 1.02 * 1.01)


@pytest.mark.parametrize("kind", ["list", "integer", "object", "missing", "wrong_index"])
def test_rebalance_mask_rejects_ambiguous_or_misaligned_inputs(kind):
    data, market, scores, cfg = small_engine_inputs()
    mask = pd.Series(True, index=data.index)
    if kind == "list":
        mask = mask.tolist()
    elif kind == "integer":
        mask = mask.astype(int)
    elif kind == "object":
        mask = mask.astype(object)
    elif kind == "missing":
        mask = mask.astype("boolean")
        mask.iloc[0] = pd.NA
    else:
        mask.index = mask.index + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="Boolean Series"):
        run_backtest(market, data.index, scores, cfg, rebalance_mask=mask)


def test_cost_buffer_requires_an_explicit_boolean():
    data, market, scores, cfg = small_engine_inputs()
    with pytest.raises(ValueError, match="reserve_cost_buffer"):
        run_backtest(market, data.index, scores, cfg, reserve_cost_buffer="false")


def test_fixed_exposure_control_changes_weights_only_and_reports_daily_resize():
    data = prices(50)
    start, end = str(data.index[21].date()), str(data.index[-1].date())
    base = run_rsrs(data, "buy_hold", start, end)
    fixed = run_rsrs(data, "buy_hold", start, end, exposure_scale=.75)
    combined = run_rsrs(data, "buy_hold", start, end, exposure_scale=.75, vol_target=.15)
    pd.testing.assert_series_equal(base.decision, fixed.decision)
    assert fixed.targets.iloc[:, 0].eq(.75).all()
    assert combined.targets.iloc[:, 0].le(.75).all()
    assert fixed.metadata["rebalance"] == "daily"
    assert fixed.result.daily.rebalance.all()
    assert fixed.metrics["exposure"] == pytest.approx(.75)
    assert len(base.result.trades) == 1
    assert fixed.checks["passed"] and combined.checks["passed"]
    shared_metrics = performance(fixed.result.daily, 1_000_000., trades=fixed.result.trades)
    assert fixed.metrics["total_return"] == shared_metrics["total_return"]


def test_public_signal_control_reuses_accounting_without_standardization():
    data = prices(60)
    signal = pd.Series(([1] * 10 + [0] * 10) * 3, index=data.index)
    actual = run_signal(data, signal, str(data.index[5].date()), str(data.index[-1].date()))
    assert actual.checks["passed"]
    assert actual.metadata["specification"]["name"] == "supplied_signal"
    assert actual.factors.empty
    assert actual.result.trades.signal_date.lt(actual.result.trades.date).all()
    with pytest.raises(ValueError, match="align exactly"):
        run_signal(data, signal.iloc[1:], "2010", "2011")
    with pytest.raises(ValueError, match="Scaled targets require"):
        run_signal(data, signal, "2010", "2011", target_weights=signal * .75)
    with pytest.raises(ValueError, match="between zero and the binary signal"):
        run_signal(data, signal, "2010", "2011", target_weights=pd.Series(1., index=data.index))


def test_data_adapter_requires_real_chronology_and_never_fabricates_exchange_limits():
    data = prices(30)
    market = index_to_market(data)
    assert market.up_limit.isna().all() and market.down_limit.isna().all()
    with pytest.raises(ValueError, match="unique, increasing"):
        index_to_market(data.iloc[::-1])
    duplicate = pd.concat([data.iloc[:1], data])
    with pytest.raises(ValueError, match="unique, increasing"):
        index_to_market(duplicate)
    incomplete = data.copy()
    incomplete.iloc[3, incomplete.columns.get_loc("open")] = np.nan
    with pytest.raises(ValueError, match="finite and positive"):
        index_to_market(incomplete)
    with pytest.raises(ValueError, match="pre-start"):
        run_rsrs(data, "buy_hold", str(data.index[0].date()), str(data.index[-1].date()))


@pytest.mark.parametrize("kwargs", [{"cost_bps": -1}, {"cost_bps": np.nan},
                                   {"vol_target": 0}, {"vol_window": 1},
                                   {"exposure_scale": 1.1}, {"exposure_scale": -.1}])
def test_invalid_research_controls_fail_before_simulation(kwargs):
    data = prices(30)
    with pytest.raises(ValueError):
        run_rsrs(data, "buy_hold", str(data.index[21].date()), str(data.index[-1].date()), **kwargs)


def test_degenerate_regression_stays_unavailable():
    data = prices(50)
    data["high"], data["low"] = 3., 2.
    factor = factors(data, 5, 10)
    assert factor.beta.isna().all()
    assert factor.z.isna().all()
