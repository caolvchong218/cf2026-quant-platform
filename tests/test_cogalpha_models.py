"""Frozen combination models: cutoff enforcement and future independence."""
import numpy as np
import pandas as pd
import pytest

from cfquant.cogalpha_models import fit_predict


def _inputs():
    dates = pd.bdate_range("2022-11-01", periods=70)
    assets = pd.Index(["stock_" + str(i) for i in range(50)])
    rng = np.random.default_rng(10)
    factors = {name: pd.DataFrame(rng.normal(size=(len(dates), len(assets))), index=dates, columns=assets)
               for name in ["value", "momentum"]}
    labels = .3 * factors["value"] + .2 * factors["momentum"] + pd.DataFrame(rng.normal(0, .3, (len(dates), len(assets))), index=dates, columns=assets)
    eligible = pd.DataFrame(True, index=dates, columns=assets)
    exits = pd.Series(dates, index=dates).shift(-11)
    training = dates[:30]
    predicting = dates[dates >= "2023-01-01"][:15]
    return factors, list(factors), labels, eligible, training, predicting, exits


def test_fixed_models_emit_only_requested_dates_and_preserve_missing_rows():
    factors, selected, labels, eligible, training, predicting, exits = _inputs()
    eligible.loc[predicting[0], "stock_0"] = False
    factors["value"].loc[predicting[1], "stock_1"] = np.nan
    scores, metadata = fit_predict(factors, selected, labels, eligible, training, predicting, label_exit=exits)
    assert set(scores) == {"rank", "ridge", "lightgbm"}
    for score in scores.values():
        assert score.index.equals(labels.index) and score.columns.equals(labels.columns)
        assert score.loc[~score.index.isin(predicting)].isna().all().all()
        assert pd.isna(score.loc[predicting[0], "stock_0"])
        assert pd.isna(score.loc[predicting[1], "stock_1"])
        assert score.loc[predicting].notna().sum().sum() == len(predicting) * 50 - 2
    expected = sum(factors[name].loc[predicting].where(eligible.loc[predicting]).rank(axis=1, pct=True) - .5 for name in selected) / len(selected)
    pd.testing.assert_frame_equal(scores["rank"].loc[predicting], expected)
    assert metadata["training_rows"] == len(training) * 50
    assert metadata["prediction_rows"] == len(predicting) * 50 - 2
    assert metadata["status"] == "complete"
    assert metadata["training_fit_count_per_model"] == 1 and not metadata["annual_retraining"]
    assert not metadata["validation_tuning"]
    assert metadata["parameters"]["ridge"] == {"alpha": 100.}
    assert metadata["parameters"]["lightgbm"]["n_estimators"] == 200
    assert metadata["parameters"]["lightgbm"]["deterministic"]
    assert metadata["parameters"]["lightgbm"]["random_state"] == 20261019
    assert set(metadata["coefficients"]) == set(selected) == set(metadata["importances"])


def test_future_label_and_feature_perturbation_cannot_change_fit_or_earlier_predictions():
    factors, selected, labels, eligible, training, predicting, exits = _inputs()
    original, first = fit_predict(factors, selected, labels, eligible, training, predicting, label_exit=exits)
    changed_factors = {name: frame.copy() for name, frame in factors.items()}
    for frame in changed_factors.values():
        frame.loc[frame.index >= predicting[8]] *= np.arange(1, 51)[None, :] * 100.
    changed_labels = labels.copy()
    changed_labels.loc[changed_labels.index >= "2023-01-01"] = -9e20
    changed, second = fit_predict(changed_factors, selected, changed_labels, eligible, training, predicting, label_exit=exits)
    assert first["training_sha256"] == second["training_sha256"]
    assert first["coefficients"] == second["coefficients"]
    assert first["intercept"] == second["intercept"]
    assert first["importances"] == second["importances"]
    for name in original:
        pd.testing.assert_frame_equal(original[name].loc[predicting[:8]], changed[name].loc[predicting[:8]])


def test_ineligible_stock_values_do_not_affect_eligible_feature_or_label_ranks():
    factors, selected, labels, eligible, training, predicting, exits = _inputs()
    eligible.iloc[:, 0] = False
    original, first = fit_predict(factors, selected, labels, eligible, training, predicting, label_exit=exits)
    changed_factors = {name: frame.copy() for name, frame in factors.items()}
    for frame in changed_factors.values():
        frame.iloc[:, 0] = 1e30
    labels.iloc[:, 0] = -1e30
    changed, second = fit_predict(changed_factors, selected, labels, eligible, training, predicting, label_exit=exits)
    assert first["training_sha256"] == second["training_sha256"]
    for name in original:
        pd.testing.assert_frame_equal(original[name], changed[name])


def test_empty_pool_has_nan_scores_and_does_not_import_or_fit_optional_models(monkeypatch):
    factors, _, labels, eligible, training, predicting, exits = _inputs()
    import builtins
    original_import = builtins.__import__

    def no_model_import(name, *args, **kwargs):
        if name.startswith(("lightgbm", "sklearn")):
            raise AssertionError("Empty candidate set must not instantiate a model")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_model_import)
    scores, metadata = fit_predict({}, [], labels, eligible, training, predicting, label_exit=exits)
    assert metadata["status"] == "empty_pool"
    assert metadata["training_rows"] == 0 and metadata["training_fit_count_per_model"] == 0
    assert metadata["training_sha256"] is None
    assert all(score.isna().all().all() for score in scores.values())


@pytest.mark.parametrize("bad_exit", ["2023-01-01", "2023-01-02", None])
def test_training_label_exit_must_be_strictly_before_cutoff(bad_exit):
    factors, selected, labels, eligible, training, predicting, exits = _inputs()
    exits.loc[training[-1]] = pd.Timestamp(bad_exit) if bad_exit is not None else pd.NaT
    with pytest.raises(ValueError, match="label exit"):
        fit_predict(factors, selected, labels, eligible, training, predicting, label_exit=exits)


def test_invalid_training_prediction_axes_and_duplicate_candidates_are_rejected():
    factors, selected, labels, eligible, training, predicting, exits = _inputs()
    with pytest.raises(ValueError, match="unique"):
        fit_predict(factors, ["value", "value"], labels, eligible, training, predicting, label_exit=exits)
    with pytest.raises(ValueError, match="exactly"):
        fit_predict(factors, selected, labels.iloc[:, ::-1], eligible, training, predicting, label_exit=exits)
    with pytest.raises(ValueError, match="training date"):
        fit_predict(factors, selected, labels, eligible, predicting, predicting, label_exit=exits)
    with pytest.raises(ValueError, match="prediction date"):
        fit_predict(factors, selected, labels, eligible, training, training, label_exit=exits)
    with pytest.raises(ValueError, match="Boolean"):
        fit_predict(factors, selected, labels, eligible.astype(int), training, predicting, label_exit=exits)
    with pytest.raises(ValueError, match="seed"):
        fit_predict(factors, selected, labels, eligible, training, predicting, label_exit=exits, seed=True)


def test_unusable_training_data_fail_instead_of_manufacturing_scores():
    factors, selected, labels, eligible, training, predicting, exits = _inputs()
    labels.loc[training] = np.nan
    with pytest.raises(ValueError, match="training rows"):
        fit_predict(factors, selected, labels, eligible, training, predicting, label_exit=exits)
