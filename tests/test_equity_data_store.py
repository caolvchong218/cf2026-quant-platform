import json

import pandas as pd
import pytest

from cfquant.equity_data_store import (
    DAILY_FIELDS, EquityDataStore, load_feature_days, query_pages, save_feature_day,
)
from cfquant.equity_forward import append_daily_bars


def inputs():
    base = pd.DataFrame([dict(date=pd.Timestamp("2026-12-30"), asset="600000.SH",
        open=10., high=11., low=9., close=10., raw_open=10., raw_high=11., raw_low=9.,
        raw_close=10., volume=1000., adj_factor=2., adjustment_reference=2.,
        up_limit=11., down_limit=9.)])
    basic = pd.DataFrame([dict(ts_code="600000.SH", trade_date="20261230",
        total_mv=10000., pe_ttm=10., pb=1., dv_ttm=2., turnover_rate=1.)])
    return base, basic


class Provider:
    def __init__(self, fail=None):
        self.calls, self.fail = [], fail

    def query(self, api, params, fields):
        self.calls.append((api, dict(params)))
        if self.fail == (api, params["trade_date"]):
            raise RuntimeError("fixture interruption")
        row = dict(ts_code="600000.SH", trade_date=params["trade_date"],
                   open=10., high=11., low=9., close=10., vol=10., adj_factor=2.,
                   up_limit=11., down_limit=9., total_mv=10000., pe_ttm=10., pb=1.,
                   dv_ttm=2., turnover_rate=1.)
        return pd.DataFrame([{field: row[field] for field in fields.split(",")}])


def store(tmp_path):
    return EquityDataStore(tmp_path, source_identity={"market": "fixture-v1"},
                           base_end="2026-12-30", assets=["600000.SH"])


def test_more_than_40_sessions_resume_across_year_and_never_refetch_commits(tmp_path):
    base, basic = inputs()
    dates = pd.bdate_range("2026-12-30", periods=90)
    provider = Provider()
    first = store(tmp_path)
    _, _, report = first.refresh(provider, base, basic, dates, dates[-1],
                                append_bars=append_daily_bars, max_sessions=20)
    assert report["remaining_sessions"] == 69 and not report["ready"]
    for _ in range(4):
        _, _, report = store(tmp_path).refresh(provider, base, basic, dates, dates[-1],
                                             append_bars=append_daily_bars, max_sessions=20)
    assert report["ready"] and report["total_extension_sessions"] == 89
    assert len(provider.calls) == 89 * 4
    market, new_basic, report = store(tmp_path).refresh(provider, base, basic, dates, dates[-1],
                                                      append_bars=append_daily_bars)
    assert len(provider.calls) == 89 * 4
    assert len(market) == len(new_basic) == 90
    pd.testing.assert_frame_equal(market.iloc[:1].reset_index(drop=True), base)


def test_interrupted_day_is_not_published_and_restarts_after_last_complete_day(tmp_path):
    base, basic = inputs()
    dates = pd.bdate_range("2026-12-30", periods=4)
    provider = Provider(fail=("daily_basic", dates[2].strftime("%Y%m%d")))
    with pytest.raises(RuntimeError, match="interruption"):
        store(tmp_path).refresh(provider, base, basic, dates, dates[-1], append_bars=append_daily_bars)
    assert (tmp_path / dates[1].strftime("%Y%m%d") / "commit.json").exists()
    assert not (tmp_path / dates[2].strftime("%Y%m%d") / "commit.json").exists()
    restarted = Provider()
    _, _, report = store(tmp_path).refresh(restarted, base, basic, dates, dates[-1], append_bars=append_daily_bars)
    assert report["ready"] and len(restarted.calls) == 8
    assert restarted.calls[0][1]["trade_date"] == dates[2].strftime("%Y%m%d")


def test_checkpoint_corruption_and_changed_bootstrap_fail_closed(tmp_path):
    base, basic = inputs()
    dates = pd.bdate_range("2026-12-30", periods=2)
    store(tmp_path).refresh(Provider(), base, basic, dates, dates[-1], append_bars=append_daily_bars)
    path = tmp_path / dates[-1].strftime("%Y%m%d") / "commit.json"
    record = json.loads(path.read_text())
    record["files"]["market.parquet"] = "bad"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="identity"):
        store(tmp_path).load(base, basic, dates, dates[-1])
    with pytest.raises(ValueError, match="bootstrap"):
        EquityDataStore(tmp_path, source_identity={"market": "fixture-v2"},
                        base_end="2026-12-30", assets=["600000.SH"])


def test_paginated_daily_api_rejects_ignored_offset():
    class Ignored:
        def query(self, *args):
            return pd.DataFrame({"ts_code": ["a", "b"], "trade_date": ["20270104"] * 2})
    with pytest.raises(ValueError, match="offset"):
        query_pages(Ignored(), "daily", {}, "ts_code,trade_date", page_size=2)


def test_daily_pages_can_exceed_old_6000_limit_without_duplicates():
    class Paged:
        def query(self, api, params, fields):
            codes = [str(i) for i in range(params["offset"], min(8001, params["offset"] + params["limit"]))]
            return pd.DataFrame({"ts_code": codes, "trade_date": ["20270104"] * len(codes)})
    result = query_pages(Paged(), "daily", {}, "ts_code,trade_date")
    assert len(result) == 8001 and result.ts_code.is_unique


def test_daily_features_cannot_be_rewritten_by_later_references(tmp_path):
    frame = pd.DataFrame({"date": pd.to_datetime(["2026-12-31"]), "asset": ["600000.SH"], "quality_roe": [.1]})
    save_feature_day(tmp_path, frame, reference_identity={"available_by": "2026-12-30"})
    result = save_feature_day(tmp_path, frame.assign(quality_roe=999.),
                              reference_identity={"available_by": "2027-01-04"})
    assert result.quality_roe.iloc[0] == .1
    loaded, identities = load_feature_days(tmp_path, through="2027-01-01")
    assert len(loaded) == 1 and "2026-12-31" in identities
    with pytest.raises(ValueError, match="labels"):
        save_feature_day(tmp_path, frame.assign(label=999), reference_identity={})
