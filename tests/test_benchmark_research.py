import numpy as np
import pandas as pd
import pytest
from cfquant.benchmark_research import annual_windows, training_mask, benchmark_metrics, choose_candidate


def test_training_label_exit_is_strictly_before_first_signal():
    cal = pd.bdate_range('2021-12-01', '2023-01-10')
    _, cutoff, _ = next(annual_windows(cal, [2023]))
    f = pd.DataFrame({'date': [cutoff-pd.Timedelta(days=30)]*3,
                      'label_exit': [cutoff-pd.Timedelta(days=1), cutoff, cutoff+pd.Timedelta(days=1)],
                      'label': [.1, 100., -100.]})
    assert training_mask(f, pd.Series(True, index=f.index), cutoff).tolist() == [True, False, False]
    f.loc[1:, 'label'] = [1e20, -1e20]
    assert training_mask(f, pd.Series(True, index=f.index), cutoff).tolist() == [True, False, False]


def test_annual_prediction_windows_do_not_overlap_or_gap():
    cal = pd.bdate_range('2022-12-01', '2026-09-18')
    windows = list(annual_windows(cal, range(2023, 2027)))
    for left, right in zip(windows, windows[1:]):
        assert left[2] == right[1]
    covered = sum(((cal >= a) & (cal < b)).astype(int) for _, a, b in windows)
    assert (covered[cal >= windows[0][1]] == 1).all()


def test_benchmark_relative_metrics_match_hand_calculation_and_fail_on_gap():
    dates = pd.bdate_range('2025-01-01', periods=4)
    b = pd.Series([100., 105., 100., 110.], index=dates)
    daily = pd.DataFrame({'date': dates[1:], 'nav': [1.04e6, 1.02e6, 1.2e6]})
    m = benchmark_metrics(daily, b)
    assert m['excess_return_pp'] == pytest.approx(.10)
    assert m['relative_wealth_return'] == pytest.approx(1.2/1.1-1)
    assert m['benchmark_return'] == pytest.approx(.1)
    with pytest.raises(ValueError, match='Missing'):
        benchmark_metrics(daily, b.drop(dates[2]))


def test_selection_requires_positive_net_and_excess_with_drawdown_constraint():
    t = pd.DataFrame([
        {'candidate':'bad_dd','total_return':1.,'excess_return_pp':.9,'max_drawdown':.8,'worst_year_excess':.5},
        {'candidate':'robust','total_return':.3,'excess_return_pp':.2,'max_drawdown':.15,'worst_year_excess':.08},
        {'candidate':'high_total','total_return':.6,'excess_return_pp':.5,'max_drawdown':.2,'worst_year_excess':-.1}])
    assert choose_candidate(t) == ('robust', True)
