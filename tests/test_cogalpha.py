"""Independent arithmetic, train-boundary, safety, and factor-selection checks."""
import math

import numpy as np
import pandas as pd
import pytest

from cfquant import cogalpha
from cfquant.cogalpha import (ELITE_FLOORS, METRIC_NAMES, QUALIFIED_FLOORS,
                             causal_quality_check, factor_metrics,
                             paper_literal_candidates, qualify_factors,
                             quantile_mutual_information, select_diverse)
from cfquant.factor_lab import evaluate_expression, validate_expression


def _panels(n=100, assets=40):
    dates = pd.bdate_range("2020-01-02", periods=n)
    rng = np.random.default_rng(17)
    close = pd.DataFrame(50. * np.exp(np.cumsum(rng.normal(0., .01, (n, assets)), axis=0)),
                         index=dates, columns=[f"stock_{i}" for i in range(assets)])
    fields = {"close": close, "open": close * .99, "high": close * 1.02,
              "low": close * .98, "volume": close * 100_000}
    return fields, pd.DataFrame(True, index=dates, columns=close.columns)


def test_mutual_information_matches_known_independent_and_deterministic_tables():
    # Five equally likely classes have entropy log(5); diagonal joint table.
    x = np.repeat(np.arange(5), 10)
    assert quantile_mutual_information(x, x) == pytest.approx(math.log(5))
    assert quantile_mutual_information(x, -x) == pytest.approx(math.log(5))
    # Every one of 25 joint categories occurs twice: MI is exactly zero.
    independent_x = np.repeat(np.arange(5), 10)
    independent_y = np.tile(np.repeat(np.arange(5), 2), 5)
    assert quantile_mutual_information(independent_x, independent_y) == pytest.approx(0., abs=1e-14)


def test_mi_ties_are_not_split_by_order_missing_pairs_and_constants_are_honest():
    x = np.repeat([0., 1., 2.], [15, 20, 15])
    y = np.arange(50, dtype=float)
    permutation = np.random.default_rng(43).permutation(50)
    assert quantile_mutual_information(x, y) == pytest.approx(quantile_mutual_information(x[permutation], y[permutation]))
    assert math.isnan(quantile_mutual_information(np.ones(40), np.arange(40)))
    assert math.isnan(quantile_mutual_information(np.arange(29), np.arange(29)))
    assert quantile_mutual_information(np.r_[x, np.nan, np.inf], np.r_[y, 1., 2.]) == pytest.approx(quantile_mutual_information(x, y))
    with pytest.raises(ValueError):
        quantile_mutual_information(x, y, min_pairs=29)
    with pytest.raises(ValueError):
        quantile_mutual_information(x, y, bins=True)
    with pytest.raises(ValueError):
        quantile_mutual_information(np.ones((5, 10)), np.ones((5, 10)))


def test_metrics_match_manual_daily_ic_and_nonannualized_sample_ir():
    fields, eligible = _panels(4)
    index, columns = eligible.index, eligible.columns
    sequence = np.arange(len(columns), dtype=float)
    scores = pd.DataFrame(np.tile(sequence, (4, 1)), index=index, columns=columns)
    labels = scores.mul(pd.Series([1., 1., 1., -1.], index=index), axis=0)
    exits = pd.Series(index, index=index)
    result = factor_metrics(scores, labels, eligible, date_start=index[0], date_end=index[-1], label_exit=exits)
    assert result["daily"].ic.tolist() == pytest.approx([1., 1., 1., -1.])
    assert result["summary"]["ic"] == pytest.approx(.5)
    assert result["summary"]["rank_ic"] == pytest.approx(.5)
    # Sample std of [1, 1, 1, -1] is 1, giving IR=.5, not .5*sqrt(252).
    assert result["summary"]["ic_ir"] == pytest.approx(.5)
    assert result["summary"]["rank_ic_ir"] == pytest.approx(.5)
    assert result["summary"]["mi"] == pytest.approx(math.log(5))
    assert result["summary"]["coverage"] == 1.
    assert result["summary"]["valid_days"] == 4
    assert result["assumptions"]["ic_ir_annualized"] is False
    assert result["assumptions"]["mi_matches_unpublished_original_estimator"] is False


def test_training_boundary_excludes_equal_cutoff_and_future_label_perturbations():
    fields, eligible = _panels(8)
    scores = fields["close"]
    labels = scores.copy()
    index = scores.index
    exits = pd.Series(index + pd.Timedelta(days=1), index=index)
    cutoff = index[4]
    exits.iloc[2] = cutoff  # An exactly equal exit is excluded, not just later exits.
    exits.iloc[3] = pd.NaT
    args = dict(date_start=index[0], date_end=index[4], label_exit=exits, split_end_nextday=cutoff)
    baseline = factor_metrics(scores, labels, eligible, **args)
    assert baseline["daily"].date.tolist() == [index[0], index[1]]
    assert baseline["summary"]["dropped_unrealized_dates"] == 3
    changed = labels.copy()
    changed.iloc[2:] = -1e9
    perturbed = factor_metrics(scores, changed, eligible, **args)
    pd.testing.assert_frame_equal(baseline["daily"], perturbed["daily"])
    assert math.isnan(baseline["summary"]["ic_ir"])  # Constant daily IC has undefined IR.


def test_coverage_uses_eligible_cells_and_insufficient_cross_sections_fail():
    fields, eligible = _panels(3)
    scores = fields["close"].copy()
    labels = scores.copy()
    eligible.iloc[0, :5] = False
    scores.iloc[0, 5:7] = np.nan
    scores.iloc[1, :11] = np.inf  # 29 finite pairs, below the required 30.
    scores.iloc[2] = 1.  # Constant factor has no defined IC or MI.
    index = scores.index
    result = factor_metrics(scores, labels, eligible, date_start=index[0], date_end=index[-1],
                            label_exit=pd.Series(index, index=index))
    assert result["daily"].n_pairs.tolist() == [33, 29, 40]
    assert result["daily"].coverage.iloc[0] == pytest.approx(33/35)
    assert result["summary"]["coverage"] == pytest.approx(102/115)
    assert result["summary"]["valid_days"] == 1
    assert result["daily"].mi.iloc[1:].isna().all()
    assert math.isnan(result["summary"]["rank_ic_ir"])


def test_metric_axes_and_split_parameters_are_not_silently_repaired():
    fields, eligible = _panels(3)
    scores = fields["close"]
    args = dict(date_start=scores.index[0], date_end=scores.index[-1],
                label_exit=pd.Series(scores.index, index=scores.index))
    with pytest.raises(ValueError, match="exactly"):
        factor_metrics(scores, scores.iloc[:, ::-1], eligible, **args)
    with pytest.raises(ValueError, match="Boolean"):
        factor_metrics(scores, scores, eligible.astype(int), **args)
    with pytest.raises(ValueError, match="ordered"):
        factor_metrics(scores, scores, eligible, **{**args, "date_end": "2010-01-01"})
    with pytest.raises(ValueError, match="label_exit"):
        factor_metrics(scores, scores, eligible, **{**args, "label_exit": args["label_exit"].iloc[::-1]})
    with pytest.raises(ValueError, match="min_pairs"):
        factor_metrics(scores, scores, eligible, **args, min_pairs=10)


def test_no_realized_labels_produce_unknown_metrics_and_zero_valid_days():
    fields, eligible = _panels(3)
    scores = fields["close"]
    result = factor_metrics(scores, scores, eligible, date_start=scores.index[0], date_end=scores.index[-1],
                            label_exit=pd.Series(pd.NaT, index=scores.index))
    assert result["daily"].empty
    assert result["summary"]["valid_days"] == 0
    assert result["summary"]["dropped_unrealized_dates"] == 3
    assert all(math.isnan(result["summary"][key]) for key in METRIC_NAMES)
    qualified, _ = qualify_factors(pd.DataFrame(columns=["id", *METRIC_NAMES]))
    assert qualified.qualified.dtype == bool
    assert select_diverse(qualified, pd.DataFrame()) == []


def test_quality_counts_nested_rolling_warmup_without_filling_or_using_labels():
    fields, eligible = _panels()
    check = causal_quality_check("rolling_mean(rolling_mean(close, 10), 10)", fields, eligible, fields["close"].index)
    assert check["passed"]
    assert check["missing_rate"] == pytest.approx(.18)
    assert check["coverage"] == pytest.approx(.82)
    assert check["uses_labels"] is False
    assert all(row["prefix_passed"] and row["future_perturbation_passed"] for row in check["checks"])
    assert check["checks"][-1]["synthetic_future_added"]
    failed = causal_quality_check("rolling_mean(close, 40)", fields, eligible, fields["close"].index)
    assert not failed["passed"] and failed["missing_rate"] == pytest.approx(.39)
    assert "nonfinite_rate_exceeds_limit" in failed["reasons"]


def test_quality_uses_only_selected_training_eligibility_and_reports_invalid_formula():
    fields, eligible = _panels()
    fields["close"].iloc[80:] = np.nan
    check = causal_quality_check("close", fields, eligible, eligible.index[:60])
    assert check["passed"] and check["coverage"] == 1.
    invalid = causal_quality_check("delay(close, -1)", fields, eligible, eligible.index[:60])
    assert not invalid["passed"] and any("evaluation_error" in x for x in invalid["reasons"])
    labeled = causal_quality_check("label", {**fields, "label": fields["close"]}, eligible, eligible.index[:60])
    assert not labeled["passed"] and any("never labels" in x for x in labeled["reasons"])
    none = causal_quality_check("close", fields, eligible & False, eligible.index[:60])
    assert not none["passed"] and "no_eligible_training_cells" in none["reasons"]
    with pytest.raises(ValueError):
        causal_quality_check("close", fields, eligible, eligible.index, max_missing_rate=-1)


def test_quality_catches_future_dependence_independently_of_dsl_whitelist(monkeypatch):
    fields, eligible = _panels()

    def leaking_evaluator(expression, panels):
        close = panels["close"]
        # Plausible unsafe normalization by the full-sample time mean.
        return close.div(close.mean(axis=0), axis=1)

    monkeypatch.setattr(cogalpha, "evaluate_expression", leaking_evaluator)
    check = causal_quality_check("close", fields, eligible, eligible.index[:70])
    assert not check["passed"]
    assert any("prefix_mismatch" in reason for reason in check["reasons"])
    assert any("future_dependence" in reason for reason in check["reasons"])


def test_five_metric_qualification_preserves_thresholds_and_requires_positive_direction():
    rows = [{"id": "weak", **{key: floor * .1 for key, floor in QUALIFIED_FLOORS.items()}},
            {"id": "qualified", **{key: floor * 2 for key, floor in QUALIFIED_FLOORS.items()}},
            {"id": "elite", **{key: floor * 4 for key, floor in ELITE_FLOORS.items()}},
            {"id": "negative", **{key: -.9 for key in METRIC_NAMES}},
            {"id": "unknown", **{key: np.nan for key in METRIC_NAMES}}]
    table, thresholds = qualify_factors(pd.DataFrame(rows))
    indexed = table.set_index("id")
    assert indexed.loc["elite", "elite"] and indexed.loc["elite", "qualified"]
    assert not indexed.loc["negative", "qualified"]
    assert not indexed.loc["unknown", "qualified"]
    assert len(indexed.loc["unknown", "qualified_reasons"]) == 5
    expected = np.quantile([.0005, .01, .04, -.9], .65)
    assert thresholds["qualified"]["ic"]["quantile_value"] == pytest.approx(expected)
    assert thresholds["qualified"]["ic"]["threshold"] == pytest.approx(max(.005, expected))
    assert thresholds["qualified"]["ic"]["finite_pool_count"] == 4
    assert thresholds["elite"]["mi"]["absolute_floor"] == .02


def test_absolute_floors_equality_nan_pool_and_duplicate_ids():
    minimum = pd.DataFrame([{"id": "on_floor", **QUALIFIED_FLOORS}])
    qualified, thresholds = qualify_factors(minimum)
    assert bool(qualified.qualified.iloc[0])
    assert not bool(qualified.elite.iloc[0])
    unknown, unknown_thresholds = qualify_factors(pd.DataFrame([{"id": "missing", **dict.fromkeys(METRIC_NAMES, np.nan)}]))
    assert not bool(unknown.qualified.iloc[0])
    assert math.isnan(unknown_thresholds["qualified"]["ic"]["threshold"])
    with pytest.raises(ValueError, match="unique"):
        qualify_factors(pd.concat([minimum, minimum]))
    with pytest.raises(ValueError):
        qualify_factors(minimum, qualified_quantile=.9, elite_quantile=.8)


def test_top20_positive_direction_and_diversity_reject_duplicate_and_mirrored_paths():
    metric = pd.DataFrame({"id": ["primary", "duplicate", "mirror", "independent", "negative", "unknown"],
                           "rank_ic": [.1, .09, .08, .07, -.8, np.nan]})
    dates = pd.bdate_range("2020-01-01", periods=40)
    first = np.tile([-1., -1., 1., 1.], 10)
    second = np.tile([-1., 1., -1., 1.], 10)
    daily = pd.DataFrame({"primary": first, "duplicate": first * 2, "mirror": -first,
                          "independent": second, "negative": first, "unknown": first}, index=dates)
    assert select_diverse(metric) == ["primary", "duplicate", "mirror", "independent"]
    assert select_diverse(metric, daily) == ["primary", "independent"]
    assert select_diverse(metric, daily, max_factors=1) == ["primary"]
    # Candidate sort uses positive training mean only, and stable ID tie-breaks.
    tied = pd.DataFrame({"id": ["b", "a"], "rank_ic": [.1, .1]})
    assert select_diverse(tied) == ["a", "b"]
    with pytest.raises(ValueError):
        select_diverse(metric, daily[["primary"]])
    with pytest.raises(ValueError):
        select_diverse(metric, max_abs_correlation=1.01)


def test_diversity_does_not_impute_undefined_correlations_to_zero():
    dates = pd.bdate_range("2020-01-01", periods=6)
    metric = pd.DataFrame({"id": ["a", "b", "constant"], "rank_ic": [.1, .09, .08]})
    daily = pd.DataFrame({"a": [1., 2., 3., np.nan, np.nan, np.nan],
                          "b": [np.nan, np.nan, np.nan, 1., 2., 3.], "constant": [1.] * 6}, index=dates)
    assert select_diverse(metric, daily) == ["a"]


def test_paper_literal_formulas_preserve_epsilon_and_use_share_volume():
    fields, _ = _panels(2)
    # This is already shares; no implicit lot conversion is permitted here.
    fields["volume"].iloc[0, 0] = 0.
    cards = paper_literal_candidates()
    assert len(cards) == 3 and len({card["id"] for card in cards}) == 3
    expected = [(fields["high"] - fields["close"]) / (fields["volume"] + 1e-9),
                (fields["high"] - fields["low"]) / (fields["volume"] + 1e-9),
                np.tanh(abs(fields["close"] - fields["open"]) / (fields["volume"] * fields["close"] + 1e-9))]
    for card, calculation in zip(cards, expected):
        validate_expression(card["expression"])
        pd.testing.assert_frame_equal(evaluate_expression(card["expression"], fields), calculation)
        assert card["denominator_epsilon"] == 1e-9 and card["volume_unit"] == "shares"
