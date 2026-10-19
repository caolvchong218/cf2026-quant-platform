"""Synthetic orchestration checks: no vendor data, credentials, or paid API."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

import cfquant.cogalpha as factor_checks
import cfquant.cogalpha_study as study_module
from cfquant.cogalpha_study import Study


@pytest.fixture
def study_factory(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "configs/project2_cogalpha.json").read_text(encoding="utf-8"))
    protocol["split"] = {"train": ["2022-01-03", "2022-03-11"],
                         "validation": ["2022-03-14", "2022-04-15"],
                         "test": ["2022-04-18", "2022-05-13"]}
    protocol["portfolio"]["initial_cash_cny"] = 1000.
    dates = pd.bdate_range("2022-01-03", periods=105)
    codes = pd.Index([f"S{k:03}" for k in range(60)], name="asset")
    day = np.arange(len(dates), dtype=float)[:, None]
    asset = np.arange(len(codes), dtype=float)[None, :]
    opening = 20. + asset * .1 + day * .02 + .03 * np.sin(day / 3. + asset)
    closing = opening * (1. + .002 * np.cos(day / 5. + asset))
    adj = np.broadcast_to(np.where(day < 50, 1., 1.1), opening.shape)
    idx = pd.MultiIndex.from_product([dates, codes], names=["date", "asset"])
    market = pd.DataFrame({"open": opening.ravel(), "close": closing.ravel(),
                           "high": (np.maximum(opening, closing) + 1.).ravel(),
                           "low": (np.minimum(opening, closing) - 1.).ravel(),
                           "raw_open": (opening / adj).ravel(),
                           "raw_close": (closing / adj).ravel(),
                           "volume": np.broadcast_to(20_000. * (asset + 1.), opening.shape).ravel(),
                           "adj_factor": adj.ravel(), "adjustment_reference": 1.,
                           "up_limit": 100., "down_limit": 1.}, index=idx).reset_index()
    features = market[["date", "asset"]].copy()
    features["eligible"] = True
    features.loc[(features.date == dates[12]) & (features.asset == codes[0]), "eligible"] = False
    benchmark = pd.DataFrame({"open": 100. + day.ravel() * .1,
                              "close": 100.1 + day.ravel() * .1}, index=dates)
    for key in ("market", "features", "benchmark"):
        protocol["data"][key] = f"synthetic/{key}.fixture"
        path = tmp_path / protocol["data"][key]
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(("Synthetic identity: " + key).encode())
    config = tmp_path / "protocol.json"
    config.write_text(json.dumps(protocol), encoding="utf-8")

    def create(name="first", mutate_market=None):
        local_market = market.copy(deep=True)
        if mutate_market is not None:
            mutate_market(local_market)
        monkeypatch.setattr(study_module, "load_inputs", lambda _root: (
            features.copy(deep=True), local_market, dates.copy(), benchmark.copy(deep=True), None))
        return Study(root, tmp_path, tmp_path / name / "public", tmp_path / name / "private", config)

    return create


def card(expression="close", factor_id="fixture_factor"):
    return {"id": factor_id, "label": "Synthetic factor", "expression": expression,
            "hypothesis": "A deterministic test fixture, not an investment claim."}


def test_label_is_next_open_ten_sessions_and_training_exits_are_purged(study_factory):
    study = study_factory()
    dates, code = study.calendar, study.assets[1]
    assert study.labels.loc[dates[15], code] == pytest.approx(
        study.panels["open"].loc[dates[26], code] / study.panels["open"].loc[dates[16], code] - 1.)
    assert study.exits.loc[dates[15]] == dates[26]
    assert study.labels.iloc[-11:].isna().all().all()
    assert np.isnan(study.labels.loc[dates[12], study.assets[0]])
    assert study.exits.reindex(study.train_dates).lt(study.cutoff).all()
    assert study.train_dates[-1] == dates[38]
    assert study.exits.loc[dates[39]] >= study.cutoff
    assert study.predict_dates[0] == dates[50]
    raw = study.market.set_index(["date", "asset"])
    key = (dates[60], code)
    assert study.panels["volume"].loc[key[0], key[1]] == pytest.approx(
        raw.loc[key, "volume"] / raw.loc[key, "adj_factor"])
    assert study.panels["volume"].loc[key[0], key[1]] * study.panels["close"].loc[key[0], key[1]] == pytest.approx(
        raw.loc[key, "volume"] * raw.loc[key, "raw_close"])


def test_future_prices_and_labels_do_not_change_training_evaluation(study_factory):
    first = study_factory("original")
    original = first.evaluate(card("returns(close, 5)"))

    def change_future(market):
        mask = market.date >= first.cutoff
        for field in ("open", "high", "low", "close", "volume"):
            market.loc[mask, field] = market.loc[mask, field] * 100. + np.arange(mask.sum())

    second = study_factory("changed_future", change_future)
    changed = second.evaluate(card("returns(close, 5)"))
    assert original["metrics"] == changed["metrics"]
    assert original["quality_passed"] == changed["quality_passed"]
    assert original["evaluation_scope"] == changed["evaluation_scope"] == "training_only"
    assert not first.labels.loc[first.predict_dates].equals(second.labels.loc[second.predict_dates])


def test_repaired_expression_has_an_immutable_checkpoint_and_can_resume_old_attempt(study_factory, monkeypatch):
    study = study_factory()
    first = study.evaluate(card("close"))
    first_path, _ = study._factor_paths("fixture_factor")
    first_hash = hashlib.sha256(first_path.read_bytes()).hexdigest()
    study.evaluate(card("close * 2"))
    second_path, _ = study._factor_paths("fixture_factor")
    assert first_path != second_path and first_path.exists() and second_path.exists()
    assert hashlib.sha256(first_path.read_bytes()).hexdigest() == first_hash

    def forbidden(*args, **kwargs):
        pytest.fail("A complete immutable factor checkpoint must not be recomputed")

    monkeypatch.setattr(study_module, "evaluate_expression", forbidden)
    assert study.evaluate(card("close")) == first
    assert study._factor_paths("fixture_factor")[0] == first_path


def test_modified_factor_cache_is_rejected_before_loading(study_factory):
    study = study_factory()
    study.evaluate(card())
    path, _ = study._factor_paths("fixture_factor")
    with path.open("ab") as stream:
        stream.write(b"fixture corruption")
    with pytest.raises(ValueError, match="changed"):
        study.factor("fixture_factor")
    with pytest.raises(ValueError, match="changed"):
        study.evaluate(card())


def test_modified_training_ic_cache_is_rejected_by_resume_and_diversity_selection(study_factory):
    study = study_factory()
    study.evaluate(card())
    path, _ = study._factor_paths("fixture_factor")
    with path.with_suffix(".ic.csv").open("ab") as stream:
        stream.write(b"fixture corruption")
    with pytest.raises(ValueError, match="Training IC checkpoint"):
        study.evaluate(card())
    with pytest.raises(ValueError, match="Training IC checkpoint"):
        study.evaluate_pool(["fixture_factor"], 731, "llm_evolved_diverse", 49)


def test_evaluator_source_change_gets_a_new_checkpoint_instead_of_reusing_old_metrics(study_factory):
    study = study_factory()
    previous = study.evaluate(card())
    old_path, _ = study._factor_paths("fixture_factor")
    original_hash = hashlib.sha256(old_path.read_bytes()).hexdigest()
    study.evaluator_sha256 = {**study.evaluator_sha256, "factor_lab.py": "0" * 64}
    with pytest.raises(ValueError, match="source changed"):
        study.factor("fixture_factor")
    updated = study.evaluate(card())
    new_path, _ = study._factor_paths("fixture_factor")
    assert new_path != old_path
    assert old_path.exists() and hashlib.sha256(old_path.read_bytes()).hexdigest() == original_hash
    assert updated["metrics"] == previous["metrics"]
    assert not study.factor("fixture_factor").empty


def test_missing_accepted_factor_does_not_silently_shrink_the_experiment(study_factory):
    study = study_factory()
    with pytest.raises(ValueError, match="checkpoint missing"):
        study.evaluate_pool(["missing_accepted"], 731, "llm_evolved49", 49)
    assert not study.selection and not study.performance_rows


def test_nonfinite_expression_is_rejected_with_a_replayable_receipt(study_factory):
    study = study_factory()
    result = study.evaluate(card("close / (high - high)"))
    assert not result["quality_passed"]
    assert result["metrics"]["coverage"] == 0.
    assert result["metrics"]["valid_days"] == 0
    assert result["metrics"]["rank_ic"] is None
    assert study.factor("fixture_factor").isna().all().all()
    assert study.evaluate(card("close / (high - high)")) == result


def test_expression_runtime_failure_is_a_quality_rejection_not_a_study_abort(study_factory, monkeypatch):
    study = study_factory()

    def failed_expression(*args, **kwargs):
        raise ValueError("synthetic numerical operator failure")

    monkeypatch.setattr(factor_checks, "evaluate_expression", failed_expression)
    monkeypatch.setattr(study_module, "evaluate_expression", failed_expression)
    result = study.evaluate(card())
    assert not result["quality_passed"]
    assert result["metrics"]["valid_days"] == 0
    assert result["metrics"]["coverage"] == 0.
    assert study.factor("fixture_factor").isna().all().all()
    assert study.evaluate(card()) == result


def test_saved_prediction_is_bitwise_identical_to_actual_backtest_input(study_factory, monkeypatch):
    study = study_factory()
    study.evaluate(card())
    # float32 collapses these distinct scores and can change the top-K tie break.
    precise = pd.DataFrame(np.tile(1. + np.arange(len(study.assets)) * 1e-9,
                                  (len(study.calendar), 1)),
                           index=study.calendar, columns=study.assets)
    precise.loc[~precise.index.isin(study.predict_dates)] = np.nan
    assert precise.iloc[-1].nunique() > precise.astype("float32").iloc[-1].nunique()
    monkeypatch.setattr(study_module, "fit_predict", lambda *args, **kwargs: ({"rank": precise}, {"status": "fixture"}))
    used = []
    monkeypatch.setattr(study, "backtest", lambda score, *args: used.append(score.copy(deep=True)))
    study.evaluate_pool(["fixture_factor"], 0, "fixed16", 16)
    saved = pd.read_parquet(study.private / "models/0_fixed16/rank_scores.parquet")
    assert len(used) == 2
    for actual in used:
        pd.testing.assert_frame_equal(actual, saved, check_exact=True, check_freq=False)


@pytest.mark.parametrize("cost_case,first_cost", [("paper", 250.), ("zero", 0.)])
def test_backtest_uses_prior_close_scores_real_fee_floors_and_same_open_benchmark(study_factory, cost_case, first_cost):
    study = study_factory()
    scores = pd.DataFrame(np.tile(np.arange(len(study.assets), dtype=float), (len(study.calendar), 1)),
                          index=study.calendar, columns=study.assets)
    study.backtest(scores, 731, "fixture", "rank", "validation", cost_case)
    key = f"s731__fixture__rank__validation__{cost_case}"
    folder = study.private / "backtests" / key
    daily = pd.read_parquet(folder / "daily.parquet")
    trades = pd.read_parquet(folder / "trades.parquet")
    public = pd.read_csv(study.output / "daily" / f"{key}.csv")
    assert trades.signal_date.lt(trades.date).all()
    assert len(trades) == 50
    assert trades.date.nunique() == 1
    assert daily.cost.iloc[0] == pytest.approx(first_cost)
    assert daily.cost.iloc[1:].eq(0.).all()
    assert daily.cash.ge(0.).all()
    benchmark = study.benchmark.loc[pd.DatetimeIndex(daily.date)]
    np.testing.assert_allclose(public.benchmark_nav, benchmark.close / benchmark.open.iloc[0], rtol=1e-10)
    np.testing.assert_allclose(public.nav, daily.nav / 1000., rtol=1e-10)
    assert study.validations[-1]["passed"]
    assert study.performance_rows[-1]["total_cost"] == pytest.approx(first_cost)


def test_empty_selected_pool_remains_cash_in_all_models_and_periods(study_factory):
    study = study_factory()
    study.evaluate_pool([], 731, "llm_initial21", 21)
    assert study.selection[-1]["selected"] == []
    assert len(study.performance_rows) == 6
    assert all(row["total_return"] == 0. and row["total_cost"] == 0. for row in study.performance_rows)
    assert all(row["valid_days"] == 0 and np.isnan(row["rank_ic"]) for row in study.performance_rows)
    assert all(check["passed"] for check in study.validations)
    for file in (study.output / "daily").glob("*.csv"):
        daily = pd.read_csv(file)
        assert daily.nav.eq(1.).all() and daily.exposure.eq(0.).all()


@pytest.fixture
def offline_cli(tmp_path, monkeypatch):
    """Replace all market computation; fail if the CLI reaches paid generation."""
    import cfquant.cogalpha_mining as mining_module

    protocol = {"experiments": {"replicate_seeds": [731]}}
    metrics = {"rank_ic": .04, "coverage": 1., "valid_days": 15}
    config_hash = hashlib.sha256(json.dumps(protocol, ensure_ascii=False, sort_keys=True,
                                            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    mining_result = {"config_hash": config_hash, "completed": True,
                     "cards": {"accepted_factor": {"id": "accepted_factor", "completed": True,
                          "status": "accepted", "card": card(factor_id="accepted_factor"),
                          "train_metrics": dict(metrics)}},
                     "accepted_ids": {arm: ["accepted_factor"] for arm in ("initial", "fresh", "evolved")},
                     "arms": {arm: ["accepted_factor"] for arm in ("initial", "fresh", "evolved")}}
    private, output = tmp_path / "private", tmp_path / "public"
    mining_folder = private / "mining/731"
    mining_folder.mkdir(parents=True)
    result_path = mining_folder / "mining_result.json"
    output.mkdir()
    budget = {"charged_cost_cny": .01, "reserved_cost_cny": 0., "calls": 1}
    (output / "budget.json").write_text(json.dumps(budget), encoding="utf-8")
    calls = {"instances": [], "evaluated": [], "registered": [], "pools": [], "finalized": []}

    class OfflineStudy:
        def __init__(self, root, data_root, output, private, config_path):
            self.root, self.output, self.private = Path(root), Path(output), Path(private)
            self.protocol = protocol
            calls["instances"].append(self)

        def evaluate(self, candidate):
            calls["evaluated"].append(candidate["id"])
            return {"quality_passed": True, "metrics": dict(metrics)}

        def register_mining(self, result):
            calls["registered"].append(result)

        def evaluate_pool(self, ids, replicate, arm, slots, **kwargs):
            calls["pools"].append({"ids": ids, "replicate": replicate, "arm": arm, "slots": slots})

        def finalize(self, budget, results):
            calls["finalized"].append({"budget": budget, "results": results})

    def forbidden(*args, **kwargs):
        pytest.fail("Offline evaluation must not construct an API client or invoke mining")

    monkeypatch.setattr(study_module, "Study", OfflineStudy)
    monkeypatch.setattr(study_module, "ResearchAPI", forbidden)
    monkeypatch.setattr(mining_module, "mine", forbidden)
    monkeypatch.setattr(study_module, "builtin_candidates", lambda: [])
    monkeypatch.setattr(study_module, "paper_literal_candidates", lambda: [])
    monkeypatch.setattr(sys, "argv", ["cogalpha_study", "--root", str(tmp_path),
                        "--data-root", str(tmp_path / "never_read_market"),
                        "--output", str(output), "--private-output", str(private),
                        "--stage", "evaluate"])

    def run(edit=None):
        if edit is not None:
            edit(mining_result)
        result_path.write_text(json.dumps(mining_result), encoding="utf-8")
        study_module.main()
        return calls

    return run, calls


def test_evaluate_cli_replays_completed_receipts_without_constructing_api_or_calling_mine(offline_cli):
    run, calls = offline_cli
    assert run() is calls
    assert calls["evaluated"] == ["accepted_factor"]
    assert len(calls["registered"]) == 1
    assert {pool["arm"] for pool in calls["pools"]} == {
        "fixed16", "llm_initial21", "llm_fresh49", "llm_evolved49", "llm_evolved_diverse"}
    assert len(calls["finalized"]) == 1
    assert calls["finalized"][0]["budget"]["charged_cost_cny"] == .01


@pytest.mark.parametrize("case, message", [
    ("config_changed", "Offline mining protocol changed"),
    ("metrics_changed", "Offline training metric changed"),
    ("unfinished_slot", "completed mining slots"),
])
def test_evaluate_cli_fails_closed_on_changed_or_incomplete_mining_without_api(offline_cli, case, message):
    run, calls = offline_cli

    def corrupt(result):
        if case == "config_changed":
            result["config_hash"] = "f" * 64
        elif case == "metrics_changed":
            result["cards"]["accepted_factor"]["train_metrics"]["rank_ic"] = .08
        else:
            result["cards"]["accepted_factor"]["completed"] = False

    with pytest.raises(ValueError, match=message):
        run(corrupt)
    assert not calls["registered"] and not calls["pools"] and not calls["finalized"]
