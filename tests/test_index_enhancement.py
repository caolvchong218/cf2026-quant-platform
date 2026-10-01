"""Complete snapshot, as-of, and hand-verifiable allocation constraints."""
import numpy as np
import pandas as pd
import pytest

from cfquant.index_enhancement import index_enhanced_targets, validate_index_weights


ASSETS = list("ABCDEF")
BASE = np.array([.30, .25, .15, .12, .10, .08])


def panels(periods=12):
    dates = pd.bdate_range("2024-01-02", periods=periods)
    returns = np.resize(np.array([.01, -.02, .03, -.01]), periods)
    prices = np.cumprod(1 + returns)[:, None] * np.arange(10., 16.)[None, :]
    close = pd.DataFrame(prices, index=dates, columns=ASSETS)
    scores = pd.DataFrame(np.tile(np.arange(1., 7.), (periods, 1)), index=dates, columns=ASSETS)
    industry = pd.DataFrame(np.tile(["X", "X", "Y", "Y", "Z", "Z"], (periods, 1)), index=dates, columns=ASSETS)
    return scores, close, industry


def snapshot(date="2024-01-02", known_at=None, asset_weights=None):
    item = asset_weights or dict(zip(ASSETS, BASE))
    return pd.DataFrame({"date": date, "asset": list(item), "weight": list(item.values()),
                         "known_at": known_at or date})


def test_snapshot_validation_preserves_full_weights_and_percent_is_explicit():
    raw = snapshot()
    original = raw.copy(deep=True)
    actual = validate_index_weights(raw)
    assert actual.weight.sum() == pytest.approx(1)
    assert actual.asset.tolist() == ASSETS
    assert actual.date.dtype == "datetime64[ns]"
    pd.testing.assert_frame_equal(raw, original)
    percent = raw.assign(weight=raw.weight * 100)
    with pytest.raises(ValueError, match="percent"):
        validate_index_weights(percent)
    converted = validate_index_weights(percent, weights_in_percent=True)
    np.testing.assert_allclose(converted.weight, raw.weight)
    assert converted.attrs["weight_sums_repaired"] is False


@pytest.mark.parametrize("mutation, message", [
    (lambda data: data.iloc[:-1], "complete"),
    (lambda data: pd.concat([data, data.iloc[:1]]), "duplicate"),
    (lambda data: data.assign(weight=np.nan), "finite"),
    (lambda data: data.assign(weight=-data.weight), "finite"),
    (lambda data: data.assign(weight=True), "boolean"),
    (lambda data: data.assign(asset=600000), "strings"),
    (lambda data: data.assign(known_at=None), "nonmissing"),
    (lambda data: data.assign(date="invalid"), "timestamps"),
    (lambda data: data.drop(columns="known_at"), "known_at"),
])
def test_invalid_or_incomplete_snapshots_are_not_repaired(mutation, message):
    with pytest.raises(ValueError, match=message):
        validate_index_weights(mutation(snapshot()))


def test_allocation_meets_sum_long_only_stock_and_industry_caps():
    scores, close, industry = panels()
    targets, diagnostics = index_enhanced_targets(scores, close, industry, snapshot(),
        max_active_stock=.02, max_active_industry=.015, max_exposure=.95, lookback=3)
    assert diagnostics.constraints_passed.all()
    assert not targets.isna().any().any()
    assert (targets >= -1e-10).all().all()
    np.testing.assert_allclose(targets.sum(axis=1), .95, atol=1e-9)
    active = targets - BASE * .95
    assert active.abs().max().max() <= .02 + 1e-9
    for group in ("X", "Y", "Z"):
        members = industry.columns[industry.iloc[0] == group]
        assert active[members].sum(axis=1).abs().max() <= .015 + 1e-9
    assert diagnostics.iloc[0].status == "ready_baseline_only"
    assert diagnostics.iloc[-1].status == "ready"
    assert diagnostics.iloc[-1].history_coverage == 1
    assert diagnostics.iloc[-1].cash_target == pytest.approx(.05)
    assert diagnostics.iloc[-1].allocation_is_trade == False
    assert diagnostics.iloc[-1].active_reference == "exposure_scaled_benchmark"
    assert targets.iloc[-1].F > BASE[-1] * .95
    assert targets.iloc[-1].A < BASE[0] * .95


def test_zero_industry_active_cap_holds_sector_exposure_exactly():
    scores, close, industry = panels()
    # Give each group an internal score ordering so its two stocks may tilt.
    scores.loc[:] = [2, -2, 2, -2, 2, -2]
    targets, diagnostics = index_enhanced_targets(scores, close, industry, snapshot(),
        max_active_stock=.01, max_active_industry=0., max_exposure=.95, lookback=3)
    active = targets.iloc[-1] - BASE * .95
    assert diagnostics.constraints_passed.all()
    for a, b in (("A", "B"), ("C", "D"), ("E", "F")):
        assert active[a] + active[b] == pytest.approx(0, abs=1e-9)
        assert active[a] == pytest.approx(.005, abs=1e-9)
        assert active[b] == pytest.approx(-.005, abs=1e-9)


def test_missing_scores_and_history_keep_members_at_baseline_with_zero_alpha():
    scores, close, industry = panels()
    scores.loc[:, "A"] = np.nan
    # A missing historical quote does not get fabricated into a risk sample.
    close.iloc[-2, close.columns.get_loc("B")] = np.nan
    targets, diagnostics = index_enhanced_targets(scores, close, industry, snapshot(), lookback=3)
    assert targets.iloc[-1].A == pytest.approx(BASE[0] * .95)
    assert targets.iloc[-1].B == pytest.approx(BASE[1] * .95)
    last = diagnostics.iloc[-1]
    assert last.score_coverage == pytest.approx(5/6)
    assert last.history_coverage == pytest.approx(5/6)
    assert last.alpha_members == 4
    assert last.constraints_passed
    assert diagnostics.iloc[-2].status == "blocked"
    assert targets.iloc[-2].isna().all()
    all_missing = scores * np.nan
    baseline, log = index_enhanced_targets(all_missing, close.ffill(), industry, snapshot(), lookback=3)
    np.testing.assert_allclose(baseline, np.tile(BASE * .95, (len(baseline), 1)), atol=1e-10)
    assert (log.status == "ready_baseline_only").all()
    assert (log.score_coverage == 0).all()


def test_missing_benchmark_member_or_price_blocks_without_partial_normalization():
    scores, close, industry = panels()
    outside = snapshot(asset_weights={"A": .5, "G": .5})
    targets, log = index_enhanced_targets(scores, close, industry, outside, lookback=3)
    assert targets.isna().all().all()
    assert (log.status == "blocked").all()
    assert (log.block_reason == "benchmark_members_outside_price_panel").all()
    assert (log.member_coverage == .5).all()
    close.loc[close.index[-1], "C"] = 0
    targets, log = index_enhanced_targets(scores, close, industry, snapshot(), lookback=3)
    assert targets.iloc[-1].isna().all()
    assert log.iloc[-1].block_reason == "benchmark_member_formation_price_unavailable"
    assert log.iloc[-1].price_coverage == pytest.approx(5/6)
    assert log.iloc[-1].price_weight_coverage == pytest.approx(.85)


def test_missing_or_unknown_industry_blocks_sector_claims():
    scores, close, industry = panels()
    industry.loc[industry.index[-1], "A"] = "UNKNOWN"
    targets, log = index_enhanced_targets(scores, close, industry, snapshot(), lookback=3)
    assert targets.iloc[-1].isna().all()
    assert log.iloc[-1].block_reason == "benchmark_member_industry_unavailable"
    static = industry.iloc[0]
    targets, log = index_enhanced_targets(scores, close, static, snapshot(), lookback=3)
    assert log.constraints_passed.all()
    assert "static_labels_supplied" in log.iloc[-1].industry_source


def test_as_of_selection_waits_for_full_publication_and_effective_date():
    scores, close, industry = panels()
    dates = scores.index
    old = snapshot(date=dates[0], known_at=dates[2])
    new_weights = dict(zip(ASSETS, BASE[::-1]))
    new = snapshot(date=dates[4], known_at=dates[6], asset_weights=new_weights)
    # One late-known row makes the whole snapshot unavailable until date 8.
    new.loc[new.asset == "F", "known_at"] = dates[8]
    early_known_future_effective = snapshot(date=dates[10], known_at=dates[5])
    data = pd.concat([old, new, early_known_future_effective], ignore_index=True)
    targets, log = index_enhanced_targets(scores, close, industry, data, max_active_stock=0, lookback=3)
    assert targets.iloc[:2].isna().all().all()
    assert (log.iloc[:2].block_reason == "no_complete_snapshot_known_by_formation").all()
    assert (log.iloc[2:8].snapshot_date == dates[0]).all()
    assert log.iloc[8].snapshot_date == dates[4]
    assert log.iloc[10].snapshot_date == dates[10]
    np.testing.assert_allclose(targets.iloc[7], BASE * .95)
    np.testing.assert_allclose(targets.iloc[8], BASE[::-1] * .95)


def test_every_target_and_diagnostic_is_prefix_invariant_and_future_robust():
    scores, close, industry = panels(periods=18)
    dates = scores.index
    data = pd.concat([snapshot(date=dates[0]), snapshot(date=dates[13], known_at=dates[15],
                     asset_weights=dict(zip(ASSETS, BASE[::-1])))], ignore_index=True)
    full, log = index_enhanced_targets(scores, close, industry, data, lookback=3)
    for cutoff in (7, 14):
        prefix, prefix_log = index_enhanced_targets(scores.iloc[:cutoff], close.iloc[:cutoff], industry.iloc[:cutoff], data, lookback=3)
        pd.testing.assert_frame_equal(prefix, full.iloc[:cutoff], atol=1e-10, rtol=1e-10)
        pd.testing.assert_frame_equal(prefix_log, log.iloc[:cutoff], check_dtype=False)
        altered_close, altered_scores, altered_industry = close.copy(), scores.copy(), industry.copy()
        altered_close.iloc[cutoff:] *= np.arange(2, 8)[None, :]
        altered_scores.iloc[cutoff:] = 1000
        altered_industry.iloc[cutoff:] = "Future sector"
        altered, altered_log = index_enhanced_targets(altered_scores, altered_close, altered_industry, data, lookback=3)
        pd.testing.assert_frame_equal(altered.iloc[:cutoff], full.iloc[:cutoff], atol=1e-10, rtol=1e-10)
        pd.testing.assert_frame_equal(altered_log.iloc[:cutoff], log.iloc[:cutoff], check_dtype=False)


def test_nonmember_scores_never_add_holdings_and_cap_zero_retains_benchmark():
    scores, close, industry = panels()
    scores["G"] = 100000
    close["G"] = 20
    industry["G"] = "X"
    targets, log = index_enhanced_targets(scores, close, industry, snapshot(), max_active_stock=0, lookback=3)
    assert (targets.G == 0).all()
    np.testing.assert_allclose(targets[ASSETS], np.tile(BASE * .95, (len(scores), 1)), atol=1e-10)
    assert (log.status == "ready_baseline_only").all()


@pytest.mark.parametrize("kwargs", [{"max_active_stock": -.1}, {"max_active_industry": np.inf},
                                    {"max_exposure": 1.1}, {"lookback": 1}, {"lookback": True}])
def test_invalid_configuration_is_rejected(kwargs):
    scores, close, industry = panels()
    with pytest.raises(ValueError):
        index_enhanced_targets(scores, close, industry, snapshot(), **kwargs)


def test_panel_axes_are_not_implicitly_aligned():
    scores, close, industry = panels()
    with pytest.raises(ValueError, match="identical"):
        index_enhanced_targets(scores, close[ASSETS[::-1]], industry, snapshot())
    with pytest.raises(ValueError, match="identical"):
        index_enhanced_targets(scores, close, industry.iloc[:-1], snapshot())
