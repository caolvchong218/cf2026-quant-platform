import json

import numpy as np
import pandas as pd
import pytest

from cfquant.rsrs import Specification, run_signal
from cfquant import rsrs_diagnostics as diag


def prices(n=380):
    rng = np.random.default_rng(231)
    level = 100 * np.exp(np.cumsum(rng.normal(0, .02, n)))
    return pd.DataFrame({"low": level * .98, "high": level * (1.02 + rng.uniform(0, .01, n)),
                         "open": level, "close": level * 1.001},
                        index=pd.bdate_range("2015-01-01", periods=n))


def test_partial_history_uses_only_available_prior_slopes_and_never_fills_zero():
    data = prices(40)
    spec = Specification("short", n=5, m=10)
    strict, _ = diag._variant_signal(data, spec)
    partial, _ = diag._variant_signal(data, spec, partial=True)
    assert strict.z.iloc[:14].isna().all()
    assert partial.z.iloc[:6].isna().all()
    prior = partial.beta.iloc[4:6]
    assert partial.z.iloc[6] == pytest.approx((partial.beta.iloc[6] - prior.mean()) / prior.std(ddof=0))
    pd.testing.assert_series_equal(strict.z.iloc[14:], partial.z.iloc[14:])
    altered = data.copy()
    altered.iloc[20:] *= 2
    changed, changed_signal = diag._variant_signal(altered, spec, partial=True)
    _, signal = diag._variant_signal(data, spec, partial=True)
    pd.testing.assert_frame_equal(partial.iloc[:20], changed.iloc[:20])
    pd.testing.assert_series_equal(signal.iloc[:20], changed_signal.iloc[:20])


def test_standard_deviation_and_current_inclusion_are_independent_controls():
    data = prices(40)
    spec = Specification("short", n=5, m=10)
    prior, _ = diag._variant_signal(data, spec)
    current, _ = diag._variant_signal(data, spec, include_current=True)
    sample, _ = diag._variant_signal(data, spec, ddof=1)
    window = prior.beta.iloc[11:21]
    assert current.z.iloc[20] == pytest.approx((prior.beta.iloc[20] - window.mean()) / window.std(ddof=0))
    assert sample.z.iloc[20] == pytest.approx(prior.z.iloc[20] * np.sqrt(9 / 10))
    pd.testing.assert_series_equal(prior.beta, current.beta)
    pd.testing.assert_series_equal(prior.r2, sample.r2)


def test_formula_manual_check_uses_previous_session_and_three_hundred_independent_regressions(tmp_path, monkeypatch):
    data = prices()
    execution_date = data.index[340]
    monkeypatch.setattr(diag, "FORMULA_EXECUTION_DATE", str(execution_date.date()))
    checked = diag._formula_check(data, tmp_path)
    assert checked["passed"]
    assert checked["formation_date"] == str(data.index[339].date())
    assert checked["prior_slope_last_date"] == str(data.index[338].date())
    assert checked["prior_slope_count"] == 300
    assert checked["prior_contains_current"] is False
    assert max(checked["absolute_errors"].values()) < 1e-8
    points = pd.read_csv(tmp_path / "formula_ols_points.csv", parse_dates=["date"])
    assert len(points) == 16
    assert points.date.max() < execution_date
    np.testing.assert_allclose(points.high - points.fitted_high, points.residual, atol=1e-10)


def test_next_close_control_preserves_t_signal_but_changes_execution_price():
    data = prices(30)
    signal = pd.Series(0, index=data.index)
    signal.iloc[22:25] = 1
    close_fill = data.copy()
    close_fill["open"] = close_fill.close
    run = run_signal(close_fill, signal, str(data.index[22].date()), str(data.index[-1].date()))
    buy = run.result.trades.iloc[0]
    assert buy.signal_date == data.index[22]
    assert buy.date == data.index[23]
    assert buy.adjusted_price == data.close.iloc[23]
    assert run.checks["passed"]


def test_small_complete_diagnostic_run_writes_control_rows_and_audited_daily_files(tmp_path, monkeypatch):
    data = prices()
    monkeypatch.setattr(diag, "PERIODS", (
        ("historical_overlap", "历史重叠期", str(data.index[325].date()), str(data.index[335].date())),
        ("post_publication", "发表后历史期", str(data.index[345].date()), str(data.index[355].date())),
    ))
    monkeypatch.setattr(diag, "SOURCE_START", str(data.index[300].date()))
    monkeypatch.setattr(diag, "SOURCE_ENDS", (
        ("march_end", str(data.index[350].date())),
        ("last_session_before_publication", str(data.index[370].date())),
    ))
    monkeypatch.setattr(diag, "FORMULA_EXECUTION_DATE", str(data.index[340].date()))
    receipt = diag.run_diagnostics(data, "sh000300", tmp_path)
    assert receipt["passed"]
    assert receipt["diagnostic_runs"] == 16
    assert receipt["source_compatible_runs"] == 24
    assert receipt["benchmark_runs"] == 14
    assert all(check["passed"] for check in receipt["reconciliation"])
    controlled = pd.read_csv(tmp_path / "diagnostics.csv")
    source = pd.read_csv(tmp_path / "source_compatible.csv")
    assert len(controlled) == 16 and controlled.compare_id.nunique() == 16
    baseline = controlled.loc[controlled.case == diag.BASELINE_CASE]
    assert baseline.delta_cagr.eq(0).all() and baseline.delta_max_drawdown.eq(0).all()
    for _, rows in controlled.groupby("period_id"):
        base = rows.loc[rows.case == diag.BASELINE_CASE].iloc[0]
        np.testing.assert_allclose(rows.delta_cagr, rows.cagr - base.cagr, atol=1e-12)
        np.testing.assert_allclose(rows.delta_max_drawdown, rows.max_drawdown - base.max_drawdown, atol=1e-12)
    assert len(source) == 24
    assert set(source.warmup) == {"strict_prior_M", "partial_prior_M_min2"}
    assert set(source.strategy_id) == {"standard_ma", "corrected_ma", "right_ma"}
    assert set(source.cost_bps) == {0., 10.}
    for filename in receipt["written_files"]:
        assert (tmp_path / filename).is_file()
    for filename in source.daily_file:
        daily = pd.read_csv(tmp_path / filename)
        assert {"date", "nav", "benchmark_nav", "exposure", "drawdown", "capital_nav"}.issubset(daily)
        np.testing.assert_allclose(daily.capital_nav / 1_000_000, daily.nav, atol=1e-12)
        np.testing.assert_allclose(np.cumprod(1 + daily["return"]), daily.nav, atol=1e-12)
        expected_dd = 1 - daily.nav / np.maximum.accumulate(np.r_[1., daily.nav])[1:]
        np.testing.assert_allclose(expected_dd, daily.drawdown, atol=1e-12)
    assumptions = json.loads((tmp_path / "diagnostic_assumptions.json").read_text(encoding="utf-8"))
    assert "p.22" in assumptions["source_partial_warmup"]
    assert "not uniquely" in assumptions["source_ambiguities"]


def test_diagnostics_reject_wrong_index_and_incomplete_snapshot(tmp_path):
    data = prices(30)
    with pytest.raises(ValueError, match="only for sh000300"):
        diag.run_diagnostics(data, "sh000905", tmp_path)
    with pytest.raises(ValueError, match="cover all fixed"):
        diag.run_diagnostics(data, "sh000300", tmp_path)
