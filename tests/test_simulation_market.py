"""Synthetic market/calendar responses only; no real data, credentials or orders."""
from datetime import date, datetime
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from cfquant.data import digest
from cfquant.simulation_market import (
    HOMOGENEOUS_SOURCE, LEGACY_PRELOAD_SOURCE,
    latest_verified_market, refresh_incremental_market, signal_clock,
)
from cfquant.simulation_signals import (
    EXECUTION_ASSET, SHANGHAI, SIGNAL_ASSET, session_dates,
)


NOW = datetime(2022, 1, 18, 18, tzinfo=SHANGHAI)


def calendar(end="2022-02-28"):
    days = pd.date_range("2019-01-01", end)
    # This weekday pattern is a clearly labelled test fixture, never an
    # implementation fallback for missing official holiday evidence.
    return pd.DataFrame({"cal_date": days, "is_open": (days.dayofweek < 5).astype(int)})


class SyntheticProvider:
    def __init__(self, cal, *, fail_start=None, change=None, fail_api=None):
        self.calendar = cal
        self.fail_start = fail_start
        self.fail_api = fail_api
        self.change = change
        self.calls = []

    def query(self, api, params, fields):
        assert api in ("index_daily", "fund_daily")
        assert set(params) == {"ts_code", "start_date", "end_date"}
        assert fields == "ts_code,trade_date,open,high,low,close"
        assert params["ts_code"] == (SIGNAL_ASSET if api == "index_daily" else EXECUTION_ASSET)
        self.calls.append((api, dict(params)))
        if params["start_date"] == self.fail_start or api == self.fail_api:
            raise RuntimeError("private-provider-secret-never-display")
        start, end = pd.Timestamp(params["start_date"]), pd.Timestamp(params["end_date"])
        assert (end - start).days < 366
        days = self.calendar.loc[
            self.calendar.is_open.eq(1) & self.calendar.cal_date.between(start, end), "cal_date"]
        t = (pd.DatetimeIndex(days) - pd.Timestamp("2019-01-01")).days.to_numpy(dtype=float)
        low = 100 + .02 * t + 4 * np.sin(t / 17)
        high = low + 1.2 + .4 * np.cos(t / 13)
        if api == "fund_daily":
            low, high = low / 30, high / 30
        response = pd.DataFrame({
            "trade_date": pd.DatetimeIndex(days).strftime("%Y%m%d"), "ts_code": params["ts_code"],
            "low": low, "high": high, "open": (high + low) / 2, "close": (high + low) / 2 + .001,
        })
        return self.change(api, response) if self.change else response


def refresh(root, *, cal=None, now=NOW, provider=None):
    cal = calendar() if cal is None else cal
    provider = SyntheticProvider(cal) if provider is None else provider
    return refresh_incremental_market(root, provider=provider, calendar=cal, now=now)


def test_initial_refresh_preserves_2019_anchor_and_requests_year_chunks(tmp_path):
    cal = calendar()
    provider = SyntheticProvider(cal)
    result = refresh(tmp_path, cal=cal, provider=provider)
    index_calls = [params for api, params in provider.calls if api == "index_daily"]
    assert [item["start_date"] for item in index_calls] == ["20190101", "20200101", "20210101", "20220101"]
    assert [item["end_date"] for item in index_calls] == ["20191231", "20201231", "20211231", "20220118"]
    expected = pd.DatetimeIndex(cal.loc[cal.is_open.eq(1) & cal.cal_date.le("2022-01-18"), "cal_date"])
    assert result.index_prices.index.equals(expected)
    assert result.manifest["acquired_index_rows"] == len(expected)
    assert result.manifest["signal_asof"] == "2022-01-18"
    assert result.manifest["execution_date"] == "2022-01-19"
    assert "homogeneous fixed 2019 anchor" in result.manifest["index_source"]
    loaded = latest_verified_market(tmp_path)
    pd.testing.assert_frame_equal(loaded.index_prices, result.index_prices)
    pd.testing.assert_frame_equal(loaded.etf_daily, result.etf_daily)


@pytest.mark.parametrize("closed_year", [2019, 2020, 2022])
def test_officially_all_closed_year_chunks_are_skipped_without_price_calls(tmp_path, closed_year):
    cal = calendar("2023-02-28")
    cal.loc[cal.cal_date.dt.year.eq(closed_year), "is_open"] = 0
    provider = SyntheticProvider(cal)
    result = refresh(tmp_path, cal=cal, provider=provider,
                     now=datetime(2023, 1, 3, 18, tzinfo=SHANGHAI))
    assert not any(params["start_date"].startswith(str(closed_year)) for _, params in provider.calls)
    assert not (result.index_prices.index.year == closed_year).any()
    assert result.manifest["signal_asof"] == "2023-01-03"


def test_failed_year_acquisition_resumes_verified_chunks_without_refetch(tmp_path):
    cal = calendar()
    with pytest.raises(ValueError, match="provider request failed") as error:
        refresh(tmp_path, cal=cal, provider=SyntheticProvider(cal, fail_start="20210101"))
    assert "private-provider-secret" not in str(error.value)
    assert latest_verified_market(tmp_path) is None
    assert len(list((tmp_path / "_index_checkpoints").glob("*.json"))) == 2
    provider = SyntheticProvider(cal)
    result = refresh(tmp_path, cal=cal, provider=provider)
    assert [params["start_date"] for api, params in provider.calls if api == "index_daily"] == [
        "20210101", "20220101"]
    assert result.manifest["acquired_index_rows"] < len(result.index_prices)


def test_failed_etf_request_resumes_without_reacquiring_complete_index(tmp_path):
    cal = calendar()
    with pytest.raises(ValueError, match="provider request failed"):
        refresh(tmp_path, cal=cal, provider=SyntheticProvider(cal, fail_api="fund_daily"))
    assert latest_verified_market(tmp_path) is None
    provider = SyntheticProvider(cal)
    result = refresh(tmp_path, cal=cal, provider=provider)
    assert [api for api, _ in provider.calls] == ["fund_daily"]
    assert result.manifest["acquired_index_rows"] == 0


def test_next_day_fetches_one_overlap_and_new_session_same_target_has_no_requests(tmp_path):
    cal = calendar()
    first = refresh(tmp_path, cal=cal, now=datetime(2022, 1, 17, 18, tzinfo=SHANGHAI))
    provider = SyntheticProvider(cal)
    result = refresh(tmp_path, cal=cal, provider=provider)
    assert provider.calls == [
        ("index_daily", {"ts_code": SIGNAL_ASSET, "start_date": "20220117", "end_date": "20220118"}),
        ("fund_daily", {"ts_code": EXECUTION_ASSET, "start_date": "20220118", "end_date": "20220118"}),
    ]
    pd.testing.assert_frame_equal(result.index_prices.loc[first.index_prices.index], first.index_prices)
    again = SyntheticProvider(cal)
    repeated = refresh(tmp_path, cal=cal, provider=again)
    assert not again.calls
    assert repeated.manifest["acquired_index_rows"] == 0


def test_year_boundary_uses_official_holiday_closures_and_contiguous_history(tmp_path):
    cal = calendar()
    cal.loc[cal.cal_date.between("2021-12-31", "2022-01-03"), "is_open"] = 0
    first = refresh(tmp_path, cal=cal, now=datetime(2021, 12, 30, 18, tzinfo=SHANGHAI))
    provider = SyntheticProvider(cal)
    result = refresh(tmp_path, cal=cal, provider=provider,
                     now=datetime(2022, 1, 4, 18, tzinfo=SHANGHAI))
    assert [(params["start_date"], params["end_date"]) for api, params in provider.calls
            if api == "index_daily"] == [("20211230", "20211231"), ("20220101", "20220104")]
    assert len(result.index_prices) == len(first.index_prices) + 1
    assert result.index_prices.index[-1] == pd.Timestamp("2022-01-04")


@pytest.mark.parametrize("hour, minute, expected", [
    (9, 35, date(2022, 1, 17)), (15, 29, date(2022, 1, 17)),
    (15, 30, date(2022, 1, 17)), (16, 30, date(2022, 1, 17)),
    (17, 4, date(2022, 1, 17)), (17, 5, date(2022, 1, 18)),
])
def test_publication_gate_is_1705_not_market_close(hour, minute, expected):
    now = datetime(2022, 1, 18, hour, minute, tzinfo=SHANGHAI)
    assert session_dates(calendar(), now=signal_clock(now))[0] == expected


def test_refresh_before_publication_reuses_previous_session_without_fetching_today(tmp_path):
    cal = calendar()
    refresh(tmp_path, cal=cal, now=datetime(2022, 1, 17, 18, tzinfo=SHANGHAI))
    provider = SyntheticProvider(cal)
    result = refresh(tmp_path, cal=cal, provider=provider,
                     now=datetime(2022, 1, 18, 16, 30, tzinfo=SHANGHAI))
    assert not provider.calls
    assert result.manifest["signal_asof"] == "2022-01-17"
    assert result.manifest["execution_date"] == "2022-01-18"


@pytest.mark.parametrize("variant", ["missing", "duplicate", "extra", "wrong_asset", "nan", "empty"])
def test_invalid_index_chunks_never_create_checkpoint_or_ready_snapshot(tmp_path, variant):
    def change(api, data):
        assert api == "index_daily"
        if variant == "missing":
            return data.drop(index=3)
        if variant == "duplicate":
            return pd.concat([data, data.iloc[[3]]])
        if variant == "extra":
            return pd.concat([data, data.iloc[[3]].assign(trade_date="20200101")])
        if variant == "wrong_asset":
            return data.assign(ts_code=EXECUTION_ASSET)
        if variant == "nan":
            data.loc[3, "close"] = float("nan")
            return data
        return data.iloc[:0]
    cal = calendar()
    with pytest.raises(ValueError):
        refresh(tmp_path, cal=cal, provider=SyntheticProvider(cal, change=change))
    assert not list(tmp_path.glob("*/manifest.json"))
    assert not list((tmp_path / "_index_checkpoints").glob("*.json"))


def test_historical_overlap_revision_is_rejected_without_poisoning_checkpoint(tmp_path):
    cal = calendar()
    first = refresh(tmp_path, cal=cal, now=datetime(2022, 1, 17, 18, tzinfo=SHANGHAI))
    manifest_before = (first.directory / "manifest.json").read_bytes()

    def change(api, data):
        assert api == "index_daily"
        data.loc[data.trade_date.eq("20220117"), ["open", "high", "low", "close"]] += .1
        return data

    with pytest.raises(ValueError, match="historical index data changed"):
        refresh(tmp_path, cal=cal, provider=SyntheticProvider(cal, change=change))
    assert (first.directory / "manifest.json").read_bytes() == manifest_before
    assert not (tmp_path / "_index_checkpoints/20220117_20220118.json").exists()
    assert latest_verified_market(tmp_path).directory == first.directory
    recovered = refresh(tmp_path, cal=cal)
    assert recovered.manifest["signal_asof"] == "2022-01-18"


@pytest.mark.parametrize("calendar_variant", ["gap", "duplicate", "late_anchor", "closed_past"])
def test_invalid_calendar_or_revised_past_blocks_before_supplier_call(tmp_path, calendar_variant):
    cal = calendar()
    if calendar_variant == "closed_past":
        refresh(tmp_path, cal=cal, now=datetime(2022, 1, 17, 18, tzinfo=SHANGHAI))
        cal.loc[cal.cal_date.eq("2020-01-02"), "is_open"] = 0
    elif calendar_variant == "gap":
        cal = cal.drop(index=3)
    elif calendar_variant == "duplicate":
        cal = pd.concat([cal, cal.iloc[[3]]])
    else:
        cal = cal.iloc[3:]
    provider = SyntheticProvider(cal)
    with pytest.raises(ValueError):
        refresh(tmp_path, cal=cal, provider=provider)
    assert not provider.calls


def test_cached_index_chunk_is_rechecked_against_current_official_calendar(tmp_path):
    cal = calendar()
    with pytest.raises(ValueError, match="provider request failed"):
        refresh(tmp_path, cal=cal, provider=SyntheticProvider(cal, fail_start="20200101"))
    revised = cal.copy()
    revised.loc[revised.cal_date.eq("2019-02-04"), "is_open"] = 0
    provider = SyntheticProvider(revised)
    with pytest.raises(ValueError, match="checkpoint differs from official"):
        refresh(tmp_path, cal=revised, provider=provider)
    assert not provider.calls


@pytest.mark.parametrize("variant", ["sha", "count"])
def test_checkpoint_corruption_blocks_resume(tmp_path, variant):
    cal = calendar()
    with pytest.raises(ValueError):
        refresh(tmp_path, cal=cal, provider=SyntheticProvider(cal, fail_start="20200101"))
    path = tmp_path / "_index_checkpoints/20190101_20191231.json"
    record = json.loads(path.read_text())
    record["sha256" if variant == "sha" else "rows"] = "0" * 64 if variant == "sha" else 1
    path.write_text(json.dumps(record))
    provider = SyntheticProvider(cal)
    with pytest.raises(ValueError, match="checkpoint"):
        refresh(tmp_path, cal=cal, provider=provider)
    assert not provider.calls


@pytest.mark.parametrize("variant", ["file_sha", "date", "count", "missing_session"])
def test_latest_snapshot_checks_file_identity_manifest_and_official_coverage(tmp_path, variant):
    market = refresh(tmp_path)
    manifest_path = market.directory / "manifest.json"
    receipt = json.loads(manifest_path.read_text())
    if variant == "file_sha":
        path = market.directory / "index_daily.csv"
        path.write_bytes(path.read_bytes() + b"\n")
    elif variant == "date":
        receipt["execution_date"] = "2022-01-20"
    elif variant == "count":
        receipt["index_rows"] -= 1
    else:
        path = market.directory / "index_daily.csv"
        frame = pd.read_csv(path).drop(index=3)
        frame.to_csv(path, index=False)
        receipt["files"]["index_daily.csv"] = digest(path)
        receipt["index_rows"] = len(frame)
    manifest_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        latest_verified_market(tmp_path)


def test_backwards_target_and_mixed_vendor_bootstrap_are_rejected(tmp_path):
    market = refresh(tmp_path)
    with pytest.raises(ValueError, match="backwards"):
        refresh(tmp_path, now=datetime(2022, 1, 17, 18, tzinfo=SHANGHAI))
    path = market.directory / "manifest.json"
    receipt = json.loads(path.read_text())
    receipt["index_source"] = "Historical preload plus Tushare index_daily"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="homogeneous"):
        refresh(tmp_path, now=datetime(2022, 1, 19, 18, tzinfo=SHANGHAI))


def test_missing_store_read_is_nonmutating_and_naive_clock_rejected(tmp_path):
    missing = tmp_path / "missing"
    assert latest_verified_market(missing) is None and not missing.exists()
    with pytest.raises(ValueError, match="timezone"):
        refresh(missing, now=NOW.replace(tzinfo=None))
    assert not missing.exists()


def legacy_snapshot(root):
    """Construct a labelled synthetic legacy snapshot, with no target checkpoints."""
    original = refresh(root.parent / "synthetic_fixture_origin")
    folder = root / "20220118T100000000000Z"
    shutil.copytree(original.directory, folder)
    path = folder / "index_daily.csv"
    prices = pd.read_csv(path)
    prices[["open", "high", "low", "close"]] += 2.0
    prices.to_csv(path, index=False)
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(schema_version=1, index_source=LEGACY_PRELOAD_SOURCE)
    manifest["files"]["index_daily.csv"] = digest(path)
    manifest.pop("acquired_index_rows")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return folder, {path.name: path.read_bytes() for path in folder.iterdir()}


def test_known_legacy_preload_is_rebuilt_from_anchor_without_splicing_or_overwriting(tmp_path):
    root = tmp_path / "market"
    legacy, original_bytes = legacy_snapshot(root)
    original_hash = digest(legacy / "manifest.json")
    old = latest_verified_market(root)
    cal = calendar()
    provider = SyntheticProvider(cal)
    rebuilt = refresh(root, cal=cal, provider=provider)
    assert [params["start_date"] for api, params in provider.calls if api == "index_daily"] == [
        "20190101", "20200101", "20210101", "20220101"]
    assert rebuilt.manifest["index_source"] == HOMOGENEOUS_SOURCE
    assert rebuilt.manifest["acquired_index_rows"] == len(rebuilt.index_prices)
    assert np.allclose(old.index_prices - rebuilt.index_prices, 2.0)
    assert rebuilt.manifest["bootstrap_transition"] == {
        "kind": "legacy_mixed_preload_to_homogeneous_tushare",
        "history_start": "2019-01-01",
        "previous_manifest_sha256": original_hash,
        "previous_index_source": LEGACY_PRELOAD_SOURCE,
        "previous_signal_asof": "2022-01-18",
        "previous_index_rows": len(old.index_prices),
        "rebuild_policy": "full_reacquisition_no_splice",
    }
    assert {path.name: path.read_bytes() for path in legacy.iterdir()} == original_bytes
    assert latest_verified_market(root).directory == rebuilt.directory
    again = SyntheticProvider(cal)
    repeated = refresh(root, cal=cal, provider=again)
    assert not again.calls
    assert repeated.manifest["bootstrap_transition"] == rebuilt.manifest["bootstrap_transition"]


def test_legacy_rebuild_interruption_resumes_only_new_pure_tushare_checkpoints(tmp_path):
    root = tmp_path / "market"
    legacy, original_bytes = legacy_snapshot(root)
    cal = calendar()
    with pytest.raises(ValueError, match="provider request failed"):
        refresh(root, cal=cal, provider=SyntheticProvider(cal, fail_start="20210101"))
    assert latest_verified_market(root).directory == legacy
    assert {path.name: path.read_bytes() for path in legacy.iterdir()} == original_bytes
    assert len(list((root / "_index_checkpoints").glob("*.json"))) == 2
    provider = SyntheticProvider(cal)
    rebuilt = refresh(root, cal=cal, provider=provider)
    assert [params["start_date"] for api, params in provider.calls if api == "index_daily"] == [
        "20210101", "20220101"]
    assert rebuilt.manifest["acquired_index_rows"] < len(rebuilt.index_prices)
    assert rebuilt.manifest["bootstrap_transition"]["previous_manifest_sha256"] == digest(
        legacy / "manifest.json")
    assert {path.name: path.read_bytes() for path in legacy.iterdir()} == original_bytes


@pytest.mark.parametrize("schema, source", [
    (2, LEGACY_PRELOAD_SOURCE),
    (1, "Historical preload plus Tushare index_daily"),
    (1, LEGACY_PRELOAD_SOURCE + " unknown variation"),
    (1, "Tushare Pro index_daily, unknown bootstrap"),
])
def test_only_explicit_known_legacy_version_can_request_migration(tmp_path, schema, source):
    root = tmp_path / "market"
    legacy, _ = legacy_snapshot(root)
    path = legacy / "manifest.json"
    receipt = json.loads(path.read_text())
    receipt.update(schema_version=schema, index_source=source)
    path.write_text(json.dumps(receipt))
    provider = SyntheticProvider(calendar())
    with pytest.raises(ValueError, match="homogeneous"):
        refresh(root, provider=provider)
    assert not provider.calls
    assert not (root / "_index_checkpoints").exists()
