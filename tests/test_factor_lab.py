"""Hand calculations and future perturbations for the causal factor DSL."""
import numpy as np
import pandas as pd
import pytest

from cfquant.factor_lab import (builtin_candidates, evaluate_expression,
                                summarize_factor, validate_expression)


def frame(values):
    return pd.DataFrame(values, index=pd.bdate_range("2024-01-02", periods=len(values)),
                        columns=list("ABC"), dtype=float)


def market_fields(periods=90):
    rng = np.random.default_rng(47)
    index = pd.bdate_range("2024-01-02", periods=periods)
    close = pd.DataFrame(50 * np.exp(np.cumsum(rng.normal(.001, .02, (periods, 6)), axis=0)),
                         index=index, columns=list("ABCDEF"))
    volume = pd.DataFrame(rng.uniform(100, 1000, close.shape), index=index, columns=close.columns)
    return {"close": close, "open": close * .99, "high": close * 1.03,
            "low": close * .97, "volume": volume}


def test_arithmetic_safe_division_and_no_mutation_match_hand_calculation():
    close = frame([[10, 0, 20], [12, 5, 18], [15, 10, 9]])
    volume = frame([[2, 0, 4], [3, 5, 0], [5, 2, 3]])
    original = close.copy(deep=True)
    result = evaluate_expression("safe_divide(close, volume) + 2 * close - close / 2", {"close": close, "volume": volume})
    assert result.iloc[0].A == pytest.approx(20)
    assert np.isnan(result.iloc[0].B) and np.isnan(result.iloc[1].C)
    assert result.iloc[2].A == pytest.approx(25.5)
    inverse = evaluate_expression("safe_divide(1, close)", {"close": close})
    assert inverse.iloc[0].A == .1 and np.isnan(inverse.iloc[0].B)
    pd.testing.assert_frame_equal(close, original)
    assert not np.isinf(result.to_numpy()).any()


def test_returns_require_complete_sessions_and_delay_zero_is_allowed():
    close = frame([[10, 10, 10], [np.nan, 11, 12], [20, 12, 15], [22, 13, 18]])
    result = evaluate_expression("returns(close, 2)", {"close": close})
    assert result.iloc[:2].isna().all().all()
    assert np.isnan(result.iloc[2].A) and np.isnan(result.iloc[3].A)
    assert result.iloc[2].B == pytest.approx(.2)
    pd.testing.assert_frame_equal(evaluate_expression("delay(close, 0)", {"close": close}), close)
    pd.testing.assert_frame_equal(evaluate_expression("delay(close, 2)", {"close": close}), close.shift(2))


def test_rolling_statistics_cross_section_and_clip_have_explicit_conventions():
    close = frame([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    fields = {"close": close, "volume": close * 2}
    mean = evaluate_expression("rolling_mean(close, 2)", fields)
    assert mean.iloc[0].isna().all()
    assert mean.iloc[1].tolist() == pytest.approx([1.5, 3, 4.5])
    std = evaluate_expression("rolling_std(close, 2)", fields)
    assert std.iloc[1].A == pytest.approx(np.sqrt(.5))
    assert evaluate_expression("rolling_min(close, 2)", fields).iloc[-1].C == 6
    assert evaluate_expression("rolling_max(close, 2)", fields).iloc[-1].C == 9
    corr = evaluate_expression("rolling_corr(close, volume, 3)", fields)
    assert corr.iloc[:2].isna().all().all()
    assert corr.iloc[-1].tolist() == pytest.approx([1, 1, 1])
    ranked = evaluate_expression("rank(close)", fields)
    assert ranked.iloc[0].tolist() == pytest.approx([1/3, 2/3, 1])
    scored = evaluate_expression("clip(zscore(close), -1, 1)", fields)
    assert scored.iloc[0].tolist() == pytest.approx([-1, 0, 1])
    constant = frame([[1, 1, 1]])
    assert evaluate_expression("zscore(close)", {"close": constant}).isna().all().all()


@pytest.mark.parametrize("expression", [
    "__import__('os').system('whoami')", "close.shift(-1)", "close.iloc[0]",
    "close['A']", "[x for x in close]", "lambda: close", "close if True else volume",
    "(close, volume)", "open('/tmp/file')", "eval(close)", "exec(close)",
    "delay(close, -1)", "returns(close, 0)", "returns(close, -20)",
    "rolling_mean(close, 0)", "rolling_std(close, 2521)", "rolling_corr(close, volume, 2.5)",
    "delay(close, True)", "rolling_mean(close, 1 + 2)", "delay(close, lag=1)",
    "close ** 1000", "clip(close, 1, -1)", "close + 1e309", "close + 'str'",
    "rolling_mean(1, 2)", "2", "safe_divide(1, 2)", "future_returns + close",
])
def test_arbitrary_python_future_lags_and_invalid_parameters_are_rejected(expression):
    with pytest.raises(ValueError):
        validate_expression(expression)


def test_ast_complexity_is_bounded_and_field_axes_are_not_silently_aligned():
    with pytest.raises(ValueError, match="complex|long|deep"):
        validate_expression(" + ".join(["close"] * 100))
    close = frame([[1, 2, 3], [2, 3, 4]])
    with pytest.raises(ValueError, match="identical"):
        evaluate_expression("close + volume", {"close": close, "volume": close[["B", "A", "C"]]})
    with pytest.raises(ValueError, match="unique"):
        evaluate_expression("close", {"close": pd.concat([close, close.iloc[-1:]])})
    with pytest.raises(ValueError, match="increasing"):
        evaluate_expression("close", {"close": close.iloc[::-1]})
    with pytest.raises(ValueError, match="numeric"):
        evaluate_expression("close", {"close": close.astype(str)})
    with pytest.raises(ValueError, match="mapping"):
        evaluate_expression("close", {})


def test_every_builtin_is_prefix_invariant_and_ignores_future_perturbations():
    fields = market_fields()
    candidates = builtin_candidates()
    assert 12 <= len(candidates) <= 20
    assert len({item["id"] for item in candidates}) == len(candidates)
    for item in candidates:
        assert item["label"] and item["hypothesis"]
        expression = item["expression"]
        full = evaluate_expression(expression, fields)
        assert full.notna().any().any(), item["id"]
        for cutoff in (35, 70):
            prefix = {name: panel.iloc[:cutoff] for name, panel in fields.items()}
            pd.testing.assert_frame_equal(evaluate_expression(expression, prefix), full.iloc[:cutoff],
                                          atol=1e-10, rtol=1e-10, obj=item["id"])
            changed = {name: panel.copy() for name, panel in fields.items()}
            for panel in changed.values():
                panel.iloc[cutoff:] *= np.arange(2, 8)[None, :]
            pd.testing.assert_frame_equal(evaluate_expression(expression, changed).iloc[:cutoff], full.iloc[:cutoff],
                                          atol=1e-10, rtol=1e-10, obj=item["id"])


def test_factor_summary_uses_only_selected_pairs_and_honest_undefined_metrics():
    values = frame([[1, 2, 3], [3, 2, 1], [1, np.nan, 3], [1, 1, 1]])
    labels = frame([[.1, .2, .3], [.1, .2, .3], [.1, .2, .3], [.1, .2, .3]])
    result = summarize_factor(values, labels, start=values.index[0], end=values.index[2])
    assert result["daily"].n_pairs.tolist() == [3, 3, 2]
    assert result["daily"].ic.iloc[:2].tolist() == pytest.approx([1, -1])
    assert result["daily"].rank_ic.iloc[:2].tolist() == pytest.approx([1, -1])
    assert np.isnan(result["daily"].ic.iloc[2])
    assert result["ic_mean"] == pytest.approx(0)
    assert result["ic_ir"] == pytest.approx(0)
    assert result["coverage"] == pytest.approx(8/9)
    assert result["observations"] == 8 and result["ic_dates"] == 2
    constant = summarize_factor(values, labels, start=values.index[-1])
    assert constant["ic_mean"] is None and constant["rank_ic_ir"] is None
    empty = summarize_factor(values, labels, start="2030-01-01")
    assert empty["n_dates"] == 0 and empty["coverage"] is None
    assert empty["daily"].empty
    with pytest.raises(ValueError, match="start"):
        summarize_factor(values, labels, start="2025", end="2024")


def test_tied_ranks_and_missing_labels_use_same_pair_mask():
    dates = pd.bdate_range("2024-01-02", periods=1)
    values = pd.DataFrame([[1, 1, 3, 10]], index=dates, columns=list("ABCD"))
    labels = pd.DataFrame([[1, 2, 3, np.nan]], index=dates, columns=list("ABCD"))
    result = summarize_factor(values, labels)
    assert result["daily"].iloc[0].rank_ic == pytest.approx(np.sqrt(.75))
    assert result["coverage"] == .75
    assert result["ic_ir"] is None
