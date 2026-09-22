"""Hand-calculated boundary cases for public-ledger research analytics."""
import numpy as np
import pandas as pd
import pytest

from cfquant.analytics import compare_strategies, drawdown_episodes, monthly_returns, rolling_risk


def ledger(dates, nav, **fields):
    return pd.DataFrame({"date": pd.to_datetime(dates), "nav": nav, **fields})


def test_comparison_includes_initial_loss_and_benchmark_first_day():
    dates = pd.bdate_range("2025-01-02", periods=3)
    daily = ledger(dates, [90., 99., 108.], cost=[10., 0., 0.], cash=[0., 9., 18.],
                   turnover=[1., .2, 0.])
    price = pd.Series([.98, 1.02, 1.04], index=dates)
    result = compare_strategies({"strategy": daily}, {"CSI300": price}, initial_cash=100., window=2)
    summary = result["summary"]
    assert result["returns"].iloc[0].to_dict() == pytest.approx({"strategy": -.1, "CSI300": -.02})
    assert summary.loc["strategy", "total_return"] == pytest.approx(.08)
    assert summary.loc["strategy", "max_drawdown"] == pytest.approx(.1)
    assert summary.loc["strategy", "total_cost"] == 10.
    assert summary.loc["strategy", "cost_over_starting_nav"] == .1
    assert summary.loc["strategy", "total_turnover"] == pytest.approx(1.2)
    assert summary.loc["strategy", "mean_cash_weight"] == pytest.approx((0 + 9/99 + 18/108)/3)
    assert result["relative"].iloc[0].excess_return_pp == pytest.approx(.04)
    assert result["relative"].iloc[0].relative_wealth_return == pytest.approx(1.08/1.04-1)
    assert np.isnan(summary.loc["CSI300", "total_cost"])


def test_subperiod_uses_previous_close_and_keeps_first_selected_day():
    dates = pd.bdate_range("2025-01-02", periods=4)
    a = ledger(dates, [90., 99., 108., 118.8], cost=[10., 1., 2., 3.])
    # A different nominal capital must not affect fair normalized comparison.
    b = ledger(dates, [900., 990., 1080., 1188.])
    result = compare_strategies({"a": a, "b": b}, initial_cash={"a": 100., "b": 1000.},
                               start=dates[1], end=dates[2], window=2)
    assert result["returns"].iloc[0].a == pytest.approx(.1)
    assert result["summary"].loc["a", "total_return"] == pytest.approx(.2)
    assert result["summary"].loc["a", "total_cost"] == 3.
    assert np.isnan(result["summary"].loc["b", "total_cost"])
    assert result["wealth"].a.tolist() == pytest.approx(result["wealth"].b.tolist())
    assert result["assumptions"]["baseline_date"] == "2025-01-02"
    assert result["assumptions"]["period_is_existing_path_slice"] is True


def test_raw_price_benchmark_and_different_coverage_use_exact_overlap():
    dates = pd.bdate_range("2025-01-02", periods=4)
    # Benchmark has a genuine preceding observation; strategy starts in cash.
    a = ledger(dates[1:], [100., 105., 110.])
    price = pd.Series([4000., 4040., 4080., 4200.], index=dates)
    result = compare_strategies({"a": a}, {"price": price}, initial_cash=100.,
                               benchmark_initial_values={"price": 3900.}, window=2)
    assert result["wealth"].index.equals(dates[1:].rename("date"))
    assert result["returns"].iloc[0].price == pytest.approx(.01)
    assert result["summary"].loc["price", "total_return"] == pytest.approx(.05)
    assert result["relative"].iloc[0].excess_return_pp == pytest.approx(.05)


def test_gaps_duplicates_and_conflicting_baselines_are_not_silently_filled():
    dates = pd.bdate_range("2025-01-02", periods=4)
    full = ledger(dates, [100., 101., 102., 103.])
    with pytest.raises(ValueError, match="Internal observation gaps"):
        compare_strategies({"a": full, "b": full.drop(index=1)})
    with pytest.raises(ValueError, match="duplicate"):
        compare_strategies({"a": pd.concat([full, full.iloc[:1]])})
    # Inside the selected span both paths have data, but their preceding close
    # comes from different days, so the first return spans unequal intervals.
    with pytest.raises(ValueError, match="inconsistent previous-close"):
        compare_strategies({"a": full, "b": full.drop(index=1)}, start=dates[2])
    with pytest.raises(ValueError, match="No overlapping"):
        compare_strategies({"a": full}, start="2030-01-01")


def test_monthly_returns_compound_and_leave_unobserved_months_absent():
    r = pd.Series([-.1, .1, .2, -.05], index=pd.to_datetime([
        "2025-01-30", "2025-01-31", "2025-03-03", "2025-03-04"]), name="strategy")
    monthly = monthly_returns(r)
    assert list(monthly.index) == [(2025, 1), (2025, 3)]
    assert monthly.loc[(2025, 1), "strategy"] == pytest.approx(-.01)
    assert monthly.loc[(2025, 3), "strategy"] == pytest.approx(.14)
    assert (monthly.strategy + 1).prod() == pytest.approx((r + 1).prod())


def test_drawdown_initial_peak_recovery_and_open_episode_have_honest_dates():
    dates = pd.bdate_range("2025-01-02", periods=5)
    wealth = pd.Series([.9, .8, 1., 1.1, .88], index=dates)
    episodes = drawdown_episodes(wealth)
    first = episodes[episodes.recovered].iloc[0]
    assert first.depth == pytest.approx(.2)
    assert pd.isna(first.peak_date)  # Initial capital has no observed date.
    assert first.trough_date == dates[1]
    assert first.recovery_date == dates[2]
    assert first.decline_sessions == 2
    assert first.recovery_sessions == 1
    assert first.duration_sessions == 3
    assert first.underwater_sessions == 2
    assert np.isnan(first.calendar_days)
    active = episodes[~episodes.recovered].iloc[0]
    assert active.peak_date == dates[3]
    assert active.depth == pytest.approx(.2)
    assert pd.isna(active.recovery_date)
    assert np.isnan(active.recovery_sessions)
    assert active.duration_sessions == 1
    assert drawdown_episodes(wealth, min_depth=.21).empty


def test_rolling_risk_is_trailing_and_downside_denominator_is_all_days():
    dates = pd.bdate_range("2025-01-02", periods=4)
    r = pd.Series([-.1, .1, .2, -.9], index=dates, name="strategy")
    result = rolling_risk(r, window=2, annualization=2)
    assert result.iloc[0].isna().all()
    assert result.iloc[1].rolling_return == pytest.approx(-.01)
    assert result.iloc[1].annualized_volatility == pytest.approx(.2)
    assert result.iloc[1].downside_deviation == pytest.approx(.1)
    assert result.iloc[1].sharpe == pytest.approx(0.)
    altered = r.copy()
    altered.iloc[-1] = 10.
    pd.testing.assert_frame_equal(result.iloc[:-1], rolling_risk(altered, window=2, annualization=2).iloc[:-1])


def test_constant_paths_have_undefined_risk_ratios_and_absent_metadata():
    dates = pd.bdate_range("2025-01-02", periods=3)
    result = compare_strategies({"cash": ledger(dates, [100., 100., 100.])}, initial_cash=100., window=2)
    summary = result["summary"].loc["cash"]
    assert summary.total_return == 0.
    assert summary.annualized_volatility == 0.
    assert summary.max_drawdown == 0.
    assert np.isnan(summary.sharpe) and np.isnan(summary.sortino) and np.isnan(summary.calmar)
    assert np.isnan(summary.mean_cash_weight)  # Unknown allocation is not inferred.
    assert result["episodes"].empty
    assert result["relative"].empty


def test_sharpe_nonzero_risk_free_matches_hand_calculation_and_keeps_nav():
    dates = pd.bdate_range('2025-01-02', periods=3)
    returns = pd.Series([.01, -.02, .03], index=dates)
    daily = ledger(dates, 100 * (1 + returns).cumprod().to_numpy())
    # Two periods per year: 4.04% annual risk-free return becomes 2% per period.
    # Mean excess = -1/75; sample variance = 19/30000.
    expected = -0.7492686492653554
    result = compare_strategies({'strategy': daily}, initial_cash=100.,
                               annualization=2, risk_free_annual=.0404, window=3)
    base = compare_strategies({'strategy': daily}, initial_cash=100., annualization=2, window=3)
    assert result['summary'].loc['strategy', 'sharpe'] == pytest.approx(expected)
    assert result['rolling'].iloc[-1].sharpe == pytest.approx(expected)
    assert rolling_risk(returns, 3, 2, .0404).iloc[-1].sharpe == pytest.approx(expected)
    assert result['assumptions']['risk_free_annual'] == .0404
    pd.testing.assert_frame_equal(result['wealth'], base['wealth'])
    pd.testing.assert_frame_equal(result['returns'], base['returns'])


def test_var_expected_shortfall_and_relative_risk_match_hand_calculation():
    dates = pd.bdate_range("2025-01-02", periods=20)
    daily_returns = np.r_[-.2, np.repeat(.01, 19)]
    benchmark_returns = np.r_[-.1, np.repeat(.005, 19)]
    nav = 100 * np.cumprod(1 + daily_returns)
    benchmark = pd.Series(np.cumprod(1 + benchmark_returns), index=dates)
    result = compare_strategies({"strategy": ledger(dates, nav)}, {"benchmark": benchmark},
                               initial_cash=100., window=2)
    summary = result["summary"].loc["strategy"]
    # Numpy/pandas linear 5% quantile interpolates 95% toward the second sample.
    assert summary.daily_var_95 == pytest.approx(.0005)
    assert summary.daily_expected_shortfall_95 == pytest.approx(.2)
    assert result["relative"].iloc[0].beta == pytest.approx(2.)
    assert result["correlation"].loc["strategy", "benchmark"] == pytest.approx(1.)


def test_partial_unknown_ledger_fields_do_not_understate_fees():
    dates = pd.bdate_range("2025-01-02", periods=3)
    result = compare_strategies({"a": ledger(dates, [100., 101., 102.], cost=[1., np.nan, 1.])},
                               initial_cash=100., window=2)
    assert np.isnan(result["summary"].loc["a", "total_cost"])
    assert np.isnan(result["summary"].loc["a", "cost_over_starting_nav"])
