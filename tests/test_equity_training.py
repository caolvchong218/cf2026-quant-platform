from datetime import date
import json

import numpy as np
import pandas as pd
import pytest

from cfquant import equity_training as training


def _rows(days, assets=None, **fields):
    days = pd.to_datetime(days)
    if assets is None:
        assets = ["600000.SH"] * len(days)
    return pd.DataFrame({
        "date": days, "asset": assets, "eligible": True,
        **{name: np.arange(len(days), dtype=float) + i / 10
           for i, name in enumerate(training.INPUT_FIELDS)},
        **fields,
    })


def _official_fixture():
    # Explicit fixture sessions; production callers must supply the official calendar.
    return pd.bdate_range("2022-12-01", "2028-02-01").difference(
        pd.to_datetime(["2024-01-01", "2025-01-01", "2026-01-01", "2026-01-02",
                        "2027-01-01"])
    )


@pytest.mark.parametrize(
    "signal,execution,cutoff,start",
    [
        ("2026-10-20", "2026-10-21", "2025-12-31", "2022-12-31"),
        ("2026-12-31", "2027-01-04", "2026-12-31", "2023-12-31"),
        ("2027-12-31", "2028-01-03", "2027-12-31", "2024-12-31"),
    ],
)
def test_execution_year_owns_three_year_annual_window(signal, execution, cutoff, start):
    window = training.resolve_annual_window(
        _official_fixture(), signal_asof=signal, execution_date=execution,
    )
    assert window.model_year == pd.Timestamp(execution).year
    assert window.cutoff_exclusive == pd.Timestamp(cutoff).date()
    assert window.training_start == pd.Timestamp(start).date()
    assert window.first_execution_session.year == window.model_year


@pytest.mark.parametrize("signal,execution", [
    ("2026-10-20", "2026-10-22"),
    ("2026-12-31", "2027-01-01"),
    ("2026-10-20 12:00", "2026-10-21"),
    ("2026-10-20T00:00:00+08:00", "2026-10-21"),
])
def test_window_refuses_non_session_or_nonadjacent_execution(signal, execution):
    with pytest.raises(ValueError):
        training.resolve_annual_window(
            _official_fixture(), signal_asof=signal, execution_date=execution,
        )


def test_window_cannot_guess_prior_year_close_from_truncated_calendar():
    with pytest.raises(ValueError, match="prior December"):
        training.resolve_annual_window(
            pd.bdate_range("2026-10-01", "2026-10-30"),
            signal_asof="2026-10-20", execution_date="2026-10-21",
        )


def _label_fixture():
    calendar = pd.bdate_range("2026-09-01", periods=50)
    seed = _rows([calendar[0]])
    added = _rows([calendar[1]], value_kept=["immutable"])
    market = pd.DataFrame({
        "date": calendar, "asset": "600000.SH", "open": 100. + np.arange(len(calendar)),
    })
    return calendar, seed, added, market


def _extend(seed, added, market, calendar, asof, **kwargs):
    return training.extend_training_history(
        seed, added, market.loc[market.date <= asof], calendar,
        observed_asof=asof, seed_identity={"sha256": "original-seed"},
        reference_identity=kwargs.pop("reference_identity", {"available_by": "2026-09-01"}),
        **kwargs,
    )


def test_twenty_session_label_matures_only_at_exit_and_never_changes_features():
    calendar, seed, added, market = _label_fixture()
    before = seed.copy(deep=True)
    before_added = added.copy(deep=True)
    immature = _extend(seed, added, market, calendar, calendar[20])
    assert immature.labels.label.isna().all()
    mature = _extend(seed, added, market, calendar, calendar[21])
    assert mature.labels.loc[0, "label_entry"] == calendar[1]
    assert mature.labels.loc[0, "label_exit"] == calendar[21]
    assert mature.labels.loc[0, "label"] == pytest.approx(121 / 101 - 1)
    assert pd.isna(mature.labels.loc[1, "label"])
    assert "label" not in mature.features
    assert mature.features.loc[1, "value_kept"] == "immutable"
    pd.testing.assert_frame_equal(seed, before)
    pd.testing.assert_frame_equal(added, before_added)
    assert mature.receipt["newly_mature_seed_labels"] == 1


@pytest.mark.parametrize("missing", [1, 21])
def test_missing_label_endpoint_is_not_filled_or_shifted(missing):
    calendar, seed, added, market = _label_fixture()
    market = market.loc[market.date != calendar[missing]]
    result = _extend(seed, added, market, calendar, calendar[30])
    assert pd.isna(result.labels.loc[0, "label"])
    assert result.labels.loc[0, "label_exit"] == calendar[21]


def test_midperiod_missing_price_does_not_move_endpoint_contract():
    calendar, seed, added, market = _label_fixture()
    market = market.loc[market.date != calendar[10]]
    result = _extend(seed, added, market, calendar, calendar[30])
    assert result.labels.loc[0, "label"] == pytest.approx(121 / 101 - 1)
    with pytest.raises(ValueError, match="missing-calendar"):
        _extend(seed, added, market, calendar.delete(15), calendar[30])


def test_known_seed_label_is_retained_with_tail_only_and_wrong_old_prices_fail():
    calendar, seed, added, market = _label_fixture()
    seed["label"] = 121 / 101 - 1
    seed["label_exit"] = calendar[21]
    result = _extend(seed, added, market.loc[market.date >= calendar[25]], calendar, calendar[30])
    assert result.labels.loc[0, "label"] == pytest.approx(seed.label.iloc[0])
    assert result.receipt["retained_seed_labels"] == 1
    changed = market.copy()
    changed.loc[changed.date == calendar[21], "open"] *= 2
    with pytest.raises(ValueError, match="endpoints changed"):
        _extend(seed, added, changed, calendar, calendar[30])
    seed["label_exit"] = calendar[22]
    with pytest.raises(ValueError, match="21-session"):
        _extend(seed, added, market, calendar, calendar[30])


def test_future_known_seed_label_cannot_be_used_before_maturity():
    calendar, seed, added, market = _label_fixture()
    seed["label"] = 999.
    seed["label_exit"] = calendar[21]
    result = _extend(seed, added, market, calendar, calendar[20])
    assert pd.isna(result.labels.loc[0, "label"])
    assert result.receipt["retained_seed_labels"] == 0


def test_no_new_partition_may_be_a_columnless_empty_frame():
    calendar, seed, _, market = _label_fixture()
    result = _extend(seed, pd.DataFrame(), market, calendar, calendar[30])
    assert result.receipt["new_feature_rows"] == 0
    assert result.labels.label.notna().all()


def test_later_reference_vintage_cannot_backfill_past_features():
    calendar, seed, added, market = _label_fixture()
    with pytest.raises(ValueError, match="postdates"):
        _extend(seed, added, market, calendar, calendar[30],
                reference_identity={"available_by": str(calendar[2].date())})
    result = _extend(seed, added, market, calendar, calendar[30],
        reference_identity={"by_date": {
            str(calendar[1].date()): {"available_by": str(calendar[1].date()), "fina_sha256": "observed"},
        }})
    assert result.receipt["new_feature_reference_available_by"] == {
        str(calendar[1].date()): str(calendar[1].date()),
    }


def test_no_future_features_prices_or_historical_feature_replacement():
    calendar, seed, added, market = _label_fixture()
    with pytest.raises(ValueError, match="future"):
        _extend(seed, added, market, calendar, calendar[0])
    with pytest.raises(ValueError, match="replacement"):
        _extend(seed, seed.copy(), market, calendar, calendar[30])
    with pytest.raises(ValueError, match="future"):
        training.extend_training_history(seed, added, market, calendar,
            observed_asof=calendar[30], seed_identity={"sha": "seed"},
            reference_identity={"available_by": "2026-09-01"})


def _window(year=2026):
    return training.AnnualWindow(
        year, date(year - 1, 12, 31), date(year - 4, 12, 31),
        date(year, 1, 5) if year == 2026 else date(year, 1, 4),
    )


def _history():
    return _rows(
        ["2022-12-30", "2022-12-31", "2024-06-03", "2024-06-03",
         "2025-11-20", "2025-11-21", "2025-12-31", "2026-01-05"],
        ["600000.SH", "600000.SH", "600000.SH", "600001.SH",
         "600000.SH", "600000.SH", "600000.SH", "600000.SH"],
        label=[-10., .2, .4, -.3, .1, 999., 999., 999.],
        label_exit=pd.to_datetime([
            "2023-01-30", "2023-01-31", "2024-07-01", "2024-07-01",
            "2025-12-30", "2025-12-31", "2026-01-30", "2026-02-03",
        ]),
    )


@pytest.fixture
def fake_backend(monkeypatch):
    calls = []

    def fit(x, y, parameters, path):
        calls.append((x.copy(), y.copy(), dict(parameters)))
        path.write_text("verified fake model", encoding="utf-8")

    class Predictor:
        def predict(self, x):
            return x[:, 0] + .125

    monkeypatch.setattr(training, "_fit_text_model", fit)
    monkeypatch.setattr(training, "_load_predictor", lambda path: Predictor())
    monkeypatch.setattr(training, "_backend_version", lambda: "unit-test-backend")
    return calls


def _fit(tmp_path, history=None, window=None, identity=None):
    return training.load_or_fit_annual_v3(
        _history() if history is None else history, _window() if window is None else window,
        cache_root=tmp_path, feature_identity={"seed_sha": "same"} if identity is None else identity,
    )


def test_fit_enforces_strict_label_cutoff_and_three_year_lower_boundary(tmp_path, fake_backend):
    model = _fit(tmp_path)
    snapshot = pd.read_parquet(model.training_path)
    assert list(snapshot.date.dt.strftime("%Y-%m-%d")) == [
        "2022-12-31", "2024-06-03", "2024-06-03", "2025-11-20",
    ]
    assert model.metadata["latest_training_label_exit"] == "2025-12-30"
    x, y, params = fake_backend[0]
    assert list(x[:, 0]) == [1., 2., 3., 4.]
    assert list(y) == [.5, .5, 0., .5]
    assert params == training.LGB_PARAMETERS


def test_cache_hit_does_not_call_history_loader_or_fit_again(tmp_path, fake_backend):
    first = _fit(tmp_path, history=lambda: _history())
    modified = first.model_path.stat().st_mtime_ns

    def forbidden_loader():
        raise AssertionError("Full feature history must not be read on a cache hit")

    second = _fit(tmp_path, history=forbidden_loader)
    assert second.cache_hit
    assert first.model_path == second.model_path
    assert second.model_path.stat().st_mtime_ns == modified
    assert len(fake_backend) == 1


@pytest.mark.parametrize("filename", ["model.txt", "training.parquet", "metadata.json"])
def test_corrupt_cache_fails_without_overwriting_it(tmp_path, fake_backend, filename):
    first = _fit(tmp_path)
    target = first.model_path.parent / filename
    target.write_bytes(b"corrupt")
    with pytest.raises(training.CacheIntegrityError):
        _fit(tmp_path, history=lambda: pytest.fail("Must not reload training on corruption"))
    assert target.read_bytes() == b"corrupt"
    assert len(fake_backend) == 1


def test_changed_training_input_requires_new_manifest_identity(tmp_path, fake_backend):
    first = _fit(tmp_path)
    changed = _history()
    changed.loc[2, training.INPUT_FIELDS[0]] += 5.
    with pytest.raises(training.CacheIntegrityError, match="inputs changed"):
        _fit(tmp_path, history=changed)
    second = _fit(tmp_path, history=changed, identity={"seed_sha": "changed"})
    assert not second.cache_hit
    assert first.model_path != second.model_path
    assert len(fake_backend) == 2


def test_changed_raw_labels_with_unchanged_rank_targets_are_detected(tmp_path, fake_backend):
    _fit(tmp_path)
    changed = _history()
    changed["label"] *= 2
    with pytest.raises(training.CacheIntegrityError, match="inputs changed"):
        _fit(tmp_path, history=changed)
    assert len(fake_backend) == 1


def test_later_append_rows_do_not_retrain_existing_annual_model(tmp_path, fake_backend):
    first = _fit(tmp_path)
    later = _rows(["2026-10-20"], label=[.9], label_exit=pd.to_datetime(["2026-11-18"]))
    second = _fit(tmp_path, history=pd.concat([_history(), later], ignore_index=True))
    assert second.cache_hit
    assert first.model_path == second.model_path
    assert len(fake_backend) == 1


def test_new_year_model_includes_newly_matured_forward_observations(tmp_path, fake_backend):
    earlier = _fit(tmp_path)
    appended = _rows(["2026-10-20"], label=[.2], label_exit=pd.to_datetime(["2026-11-18"]))
    history = pd.concat([_history(), appended], ignore_index=True)
    next_year = _fit(tmp_path, history=history, window=_window(2027),
                     identity={"seed_sha": "same", "annual_append_sha": "2026-observed"})
    snapshot = pd.read_parquet(next_year.training_path)
    assert pd.Timestamp("2022-12-31") not in set(snapshot.date)
    assert pd.Timestamp("2026-10-20") in set(snapshot.date)
    assert next_year.model_path != earlier.model_path
    assert len(fake_backend) == 2


def test_interrupted_fit_is_preserved_and_later_attempt_recovers(tmp_path, monkeypatch, fake_backend):
    successful_fit = training._fit_text_model

    def failed_fit(x, y, params, path):
        path.write_text("incomplete", encoding="utf-8")
        raise RuntimeError("Interrupted fitting")

    monkeypatch.setattr(training, "_fit_text_model", failed_fit)
    with pytest.raises(RuntimeError, match="Interrupted"):
        _fit(tmp_path)
    assert len(list(tmp_path.rglob("training.parquet"))) == 1
    leftovers = list(tmp_path.rglob("model.txt"))
    assert ".pending-" in leftovers[0].parent.name
    monkeypatch.setattr(training, "_fit_text_model", successful_fit)
    recovered = _fit(tmp_path)
    assert not recovered.cache_hit
    assert ".pending-" not in recovered.model_path.parent.name
    assert leftovers[0].read_text(encoding="utf-8") == "incomplete"
    assert len(list(tmp_path.rglob("model.txt"))) == 2
    assert _fit(tmp_path).cache_hit
    assert len(fake_backend) == 1


def test_prediction_is_cross_sectional_eligible_and_does_not_consume_labels(tmp_path, fake_backend):
    model = _fit(tmp_path)
    current = _rows(["2026-10-20"] * 3, ["600000.SH", "600001.SH", "600002.SH"])
    current.index = [15, 42, 88]
    current.loc[42, "eligible"] = False
    current["label"] = [np.inf, -np.inf, 999.]
    current["label_exit"] = pd.Timestamp("2100-01-01")
    scores = training.predict_current(model, current, execution_date="2026-10-21")
    assert list(scores.index) == [15, 42, 88]
    assert scores.loc[15] == .125
    assert pd.isna(scores.loc[42])
    assert scores.loc[88] == 2.125
    assert len(fake_backend) == 1
    current.loc[88, ["n_" + name for name in training.FEATURES[:3]]] = np.nan
    assert pd.isna(training.predict_current(model, current).loc[88])


def test_last_year_end_signal_requires_the_new_execution_year_model(tmp_path, fake_backend):
    old = _fit(tmp_path)
    new = _fit(tmp_path, window=_window(2027))
    current = _rows(["2026-12-31"])
    with pytest.raises(ValueError, match="execution year"):
        training.predict_current(old, current, execution_date="2027-01-04")
    assert training.predict_current(new, current, execution_date="2027-01-04").notna().all()
    with pytest.raises(ValueError, match="formation range"):
        training.predict_current(new, current.assign(date=pd.Timestamp("2026-12-30")))
    old.model_path.write_text("changed model", encoding="utf-8")
    with pytest.raises(training.CacheIntegrityError, match="changed after"):
        training.predict_current(old, current)


def test_metadata_identity_is_verified_before_accepting_cached_model(tmp_path, fake_backend):
    first = _fit(tmp_path)
    metadata = json.loads(first.metadata_path.read_text(encoding="utf-8"))
    metadata["identity"]["window"]["model_year"] = 1900
    first.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(training.CacheIntegrityError, match="identity changed"):
        _fit(tmp_path)


def test_lightgbm_text_model_round_trip_produces_identical_cached_predictions(tmp_path):
    pytest.importorskip("lightgbm")
    parameters = {
        **training.LGB_PARAMETERS, "n_estimators": 3, "min_child_samples": 1,
        "num_leaves": 3, "n_jobs": 1,
    }
    arguments = dict(cache_root=tmp_path, feature_identity={"seed_sha": "round-trip"},
                     parameters=parameters)
    first = training.load_or_fit_annual_v3(_history(), _window(), **arguments)
    current = _rows(["2026-10-20"] * 2, ["600000.SH", "600001.SH"])
    scores = training.predict_current(first, current, execution_date="2026-10-21")
    cached = training.load_or_fit_annual_v3(
        lambda: pytest.fail("Cache hit must not read historical parquet"),
        _window(), **arguments,
    )
    assert cached.cache_hit
    assert "Tree=" in first.model_path.read_text(encoding="utf-8")
    pd.testing.assert_series_equal(
        scores, training.predict_current(cached, current, execution_date="2026-10-21"),
    )
