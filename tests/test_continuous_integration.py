"""No-network regression tests for preparation progress and annual integration."""
from datetime import datetime
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cfquant import equity_continuous as continuous
from cfquant import equity_training as training
from cfquant import simulation_refresh
from cfquant import simulation_service
from cfquant.data import digest


def _bootstrap(tmp_path, monkeypatch):
    source = tmp_path / "data/private/mainboard1000_20260918/data/processed"
    extra = tmp_path / "data/private/research_v2"
    source.mkdir(parents=True)
    extra.mkdir(parents=True)
    base = pd.DataFrame([dict(
        date=pd.Timestamp("2026-10-16"), asset="600000.SH",
        open=10., high=11., low=9., close=10., raw_open=10., raw_high=11.,
        raw_low=9., raw_close=10., volume=1000., adj_factor=2.,
        adjustment_reference=2., up_limit=11., down_limit=9.,
    )])
    base.to_csv(source / "market.csv", index=False)
    pd.DataFrame({"date": ["2026-10-16"]}).to_csv(source / "calendar.csv", index=False)
    basic = pd.DataFrame([dict(ts_code="600000.SH", trade_date="20261016",
        total_mv=10000., pe_ttm=10., pb=1., dv_ttm=2., turnover_rate=1.)])
    basic.to_parquet(extra / "daily_basic.parquet", index=False)
    for name in ("features", "financials", "industries"):
        pd.DataFrame({"fixture": [1]}).to_parquet(extra / f"{name}.parquet", index=False)
    (extra / "features_cache.json").write_text(json.dumps({
        "output": digest(extra / "features.parquet"), "inputs": {},
    }), encoding="utf-8")
    monkeypatch.setattr(continuous, "load_market", lambda path: base.copy())
    days = pd.date_range("2025-12-01", "2026-10-22")
    calendar = pd.DataFrame({"cal_date": days, "is_open": (days.dayofweek < 5).astype(int)})
    return calendar


class FailingProvider:
    def __init__(self, failed_day):
        self.failed_day = failed_day
        self.calls = []

    def query(self, api, params, fields):
        self.calls.append((api, params["trade_date"]))
        if params["trade_date"] == self.failed_day:
            raise RuntimeError("synthetic provider outage; no secret")
        row = dict(ts_code="600000.SH", trade_date=params["trade_date"],
                   open=10., high=11., low=9., close=10., vol=10., adj_factor=2.,
                   up_limit=11., down_limit=9., total_mv=10000., pe_ttm=10.,
                   pb=1., dv_ttm=2., turnover_rate=1.)
        return pd.DataFrame([{field: row[field] for field in fields.split(",")}])


def test_committed_partial_session_survives_api_error_and_reports_scheduler_progress(tmp_path, monkeypatch):
    calendar = _bootstrap(tmp_path, monkeypatch)
    now = datetime.fromisoformat("2026-10-20T18:00:00+08:00")
    provider = FailingProvider("20261020")
    monkeypatch.setattr(simulation_refresh, "_provider", lambda *args: provider)
    monkeypatch.setattr(simulation_service, "refresh_signals",
        lambda *args, **kwargs: {"fixture": SimpleNamespace(signal_asof=now.date())})
    monkeypatch.setattr(simulation_service, "refresh_equities",
        lambda root, now, provider: continuous.prepare_equity_forward(
            root, root / "runs/simulation_auto/equities", now.date(),
            now=now, provider=provider, official_calendar=calendar,
        ))
    result = simulation_refresh.perform_refresh(
        tmp_path, SimpleNamespace(kind="data", target_date=now.date()), now=now,
    )
    assert result["progress"] and not result["success"]
    assert result["completed_target_date"] is None
    assert result["equity_status"] == "catching_up"
    state = tmp_path / "runs/simulation_auto"
    assert (state / "equity_market_store/20261019/commit.json").exists()
    assert not (state / "equity_market_store/20261020/commit.json").exists()
    receipt = json.loads(next((state / "equities").glob("*/receipt.json")).read_text(encoding="utf-8"))
    assert receipt["progress_count"] == 1
    assert receipt["scores_ready"] is False and receipt["ready_for_auto_orders"] is False


def test_api_error_without_any_persistent_progress_remains_a_real_failure(tmp_path, monkeypatch):
    calendar = _bootstrap(tmp_path, monkeypatch)
    now = datetime.fromisoformat("2026-10-20T18:00:00+08:00")
    with pytest.raises(RuntimeError, match="synthetic provider outage"):
        continuous.prepare_equity_forward(
            tmp_path, tmp_path / "runs/simulation_auto/equities", now.date(),
            now=now, provider=FailingProvider("20261019"), official_calendar=calendar,
        )
    receipt = json.loads(next((tmp_path / "runs/simulation_auto/equities").glob(
        "*/receipt.json")).read_text(encoding="utf-8"))
    assert receipt["progress_count"] == 0 and receipt["status"] == "blocked"
    assert receipt["scores_ready"] is False


def test_standalone_preparation_extends_existing_official_calendar(tmp_path, monkeypatch):
    from cfquant import simulation_calendar
    calendar = _bootstrap(tmp_path, monkeypatch)
    now = datetime.fromisoformat("2026-10-20T18:00:00+08:00")
    old = calendar[calendar.cal_date <= "2026-10-16"].copy()
    simulation_calendar.seed_calendar(tmp_path / "runs/simulation_auto/calendar_store", old, now=now)
    calls = []

    def refreshed(root, provider, *, now):
        calls.append(root)
        return calendar

    monkeypatch.setattr(simulation_calendar, "refresh_calendar", refreshed)
    # Reaching the market request proves the extended next-session calendar was
    # used. The intentional provider failure keeps this test entirely offline.
    with pytest.raises(RuntimeError, match="synthetic provider outage"):
        continuous.prepare_equity_forward(
            tmp_path, tmp_path / "runs/simulation_auto/equities", now.date(),
            now=now, provider=FailingProvider("20261019"),
        )
    assert calls == [tmp_path / "runs/simulation_auto/calendar_store"]


def test_next_year_training_matures_old_seed_tail_but_excludes_late_features(tmp_path, monkeypatch):
    calendar = pd.bdate_range("2025-01-01", "2027-02-15").difference(
        pd.to_datetime(["2026-01-01", "2027-01-01"]),
    )

    def features(dates):
        return pd.DataFrame({
            "date": pd.to_datetime(dates), "asset": "600000.SH", "eligible": True,
            **{field: np.arange(len(dates), dtype=float) for field in training.INPUT_FIELDS},
        })

    # The immutable September seed does not know the exit date of its last row.
    seed = features(["2025-01-20", "2026-09-18"])
    seed["label"] = np.nan
    seed["label_exit"] = pd.NaT
    added = features(["2026-09-21", "2026-11-30", "2026-12-01",
                      "2026-12-02", "2026-12-31", "2027-01-04"])
    observed = pd.Timestamp("2027-01-04")
    market = pd.DataFrame({
        "date": calendar[calendar <= observed], "asset": "600000.SH",
        "open": 100. + np.arange((calendar <= observed).sum()),
    })
    history = training.extend_training_history(
        seed, added, market, calendar, observed_asof=observed,
        seed_identity={"seed": "20260918"},
        reference_identity={"available_by": "2026-09-18"},
    )
    window = training.resolve_annual_window(
        calendar, signal_asof="2026-12-31", execution_date="2027-01-04",
    )
    monkeypatch.setattr(training, "_fit_text_model",
                        lambda x, y, params, path: path.write_text("fixture", encoding="utf-8"))
    model = training.load_or_fit_annual_v3(
        history, window, cache_root=tmp_path, feature_identity={"annual": "2027"},
    )
    selected = pd.read_parquet(model.training_path)
    assert pd.Timestamp("2026-09-18") in set(selected.date)
    assert pd.Timestamp("2026-12-01") in set(selected.date)
    assert pd.Timestamp("2026-12-02") not in set(selected.date)  # exits exactly at cutoff
    assert pd.Timestamp("2026-12-31") not in set(selected.date)
    assert pd.Timestamp("2027-01-04") not in set(selected.date)
    assert selected.label_exit.max() < pd.Timestamp("2026-12-31")
    assert seed.label.isna().all() and seed.label_exit.isna().all()
