"""Synthetic official-calendar responses; no network or credential access."""
from datetime import date, datetime, timedelta
import json

import pandas as pd
import pytest

from cfquant.simulation_calendar import (
    CalendarDataError, CalendarFetchError, SHANGHAI,
    load_calendar_cache, refresh_calendar, seed_calendar,
)


NOW = datetime(2026, 10, 20, 18, tzinfo=SHANGHAI)


def frame(start, end):
    days = pd.date_range(start, end)
    return pd.DataFrame({
        "cal_date": [day.strftime("%Y%m%d") for day in days],
        "is_open": [int(day.weekday() < 5) for day in days],
    })


class SyntheticProvider:
    def __init__(self, *, published_through="2030-12-31", fail_start=None, change=None):
        self.published_through = pd.Timestamp(published_through)
        self.fail_start = fail_start
        self.change = change
        self.calls = []

    def query(self, api, params, fields):
        assert api == "trade_cal" and params["exchange"] == "SSE"
        assert fields == "cal_date,is_open"
        assert set(params) == {"exchange", "start_date", "end_date"}
        self.calls.append(dict(params))
        if params["start_date"] == self.fail_start:
            raise RuntimeError("private-token-should-not-escape")
        start, end = pd.Timestamp(params["start_date"]), pd.Timestamp(params["end_date"])
        assert (end - start).days + 1 < 400
        end = min(end, self.published_through)
        result = frame(start, end) if start <= end else pd.DataFrame(columns=["cal_date", "is_open"])
        return self.change(result) if self.change else result


def test_initial_calendar_is_year_chunked_including_leap_year_and_normalized(tmp_path):
    provider = SyntheticProvider()
    result = refresh_calendar(tmp_path, provider, now=NOW, history_start="2024-01-01")
    assert [item["start_date"] for item in provider.calls] == ["20240101", "20250101", "20260101"]
    assert provider.calls[0]["end_date"] == "20241231"
    assert provider.calls[-1]["end_date"] == "20261204"
    assert result.cal_date.dtype.kind == "M" and result.is_open.dtype.kind in ("i", "u")
    assert result.cal_date.iloc[0] == pd.Timestamp("2024-01-01")
    assert result.cal_date.iloc[-1] == pd.Timestamp("2026-12-04")
    assert result.attrs["coverage_complete"] is True and not result.attrs["request_incomplete"]
    cached = load_calendar_cache(tmp_path)
    pd.testing.assert_frame_equal(result, cached)
    assert len(list((tmp_path / "checkpoints").glob("*.json"))) == 3


def test_incremental_refresh_queries_recent_week_and_future_without_reloading_old_years(tmp_path):
    seed_calendar(tmp_path, frame("2026-01-01", "2026-10-27"), now=NOW)
    provider = SyntheticProvider()
    result = refresh_calendar(tmp_path, provider, now=NOW, history_start="2026-01-01")
    assert provider.calls == [{
        "exchange": "SSE", "start_date": "20261013", "end_date": "20261204",
    }]
    assert result.cal_date.iloc[0] == pd.Timestamp("2026-01-01")
    assert result.cal_date.iloc[-1] == pd.Timestamp("2026-12-04")


def test_failed_initialization_reuses_verified_year_checkpoints_after_restart(tmp_path):
    first = SyntheticProvider(fail_start="20250101")
    with pytest.raises(CalendarFetchError, match="official_calendar_request_failed") as caught:
        refresh_calendar(tmp_path, first, now=NOW, history_start="2024-01-01")
    assert "private-token" not in str(caught.value)
    assert load_calendar_cache(tmp_path) is None
    assert (tmp_path / "checkpoints/20240101_20241231.json").exists()
    resumed = SyntheticProvider()
    result = refresh_calendar(tmp_path, resumed, now=NOW, history_start="2024-01-01")
    assert [item["start_date"] for item in resumed.calls] == ["20250101", "20260101"]
    assert result.attrs["coverage_complete"]


def test_failed_extension_preserves_last_published_manifest(tmp_path):
    seed_calendar(tmp_path, frame("2026-01-01", "2026-10-27"), now=NOW)
    original = (tmp_path / "manifest.json").read_bytes()
    provider = SyntheticProvider(fail_start="20261013")
    with pytest.raises(CalendarFetchError):
        refresh_calendar(tmp_path, provider, now=NOW, history_start="2026-01-01")
    assert (tmp_path / "manifest.json").read_bytes() == original
    assert load_calendar_cache(tmp_path).cal_date.iloc[-1] == pd.Timestamp("2026-10-27")


def test_unpublished_next_year_remains_explicitly_incomplete_and_is_retried(tmp_path):
    now = datetime(2026, 12, 20, 18, tzinfo=SHANGHAI)
    provider = SyntheticProvider(published_through="2026-12-31")
    result = refresh_calendar(tmp_path, provider, now=now, history_start="2026-01-01")
    assert result.cal_date.iloc[-1] == pd.Timestamp("2026-12-31")
    assert not result.attrs["coverage_complete"] and result.attrs["request_incomplete"]
    assert result.attrs["required_through"] == "2027-02-03"
    loaded = load_calendar_cache(tmp_path)
    assert not loaded.attrs["coverage_complete"] and loaded.attrs["request_incomplete"]
    available = SyntheticProvider()
    updated = refresh_calendar(tmp_path, available, now=now, history_start="2026-01-01")
    assert any(item["start_date"] == "20270101" for item in available.calls)
    assert updated.cal_date.iloc[-1] == pd.Timestamp("2027-02-03")
    assert updated.attrs["coverage_complete"] and not updated.attrs["request_incomplete"]


def test_contiguous_partial_future_year_is_saved_but_not_reused_as_complete(tmp_path):
    now = datetime(2026, 12, 20, 18, tzinfo=SHANGHAI)
    early = SyntheticProvider(published_through="2027-01-10")
    result = refresh_calendar(tmp_path, early, now=now, history_start="2026-01-01")
    assert result.cal_date.iloc[-1] == pd.Timestamp("2027-01-10")
    assert result.attrs["request_incomplete"] and not result.attrs["coverage_complete"]
    later = SyntheticProvider()
    updated = refresh_calendar(tmp_path, later, now=now, history_start="2026-01-01")
    assert later.calls[0]["start_date"] == "20261213"
    assert updated.attrs["coverage_complete"]


@pytest.mark.parametrize("variant", ["gap", "duplicate", "bad_flag", "bad_start", "past_truncation"])
def test_missing_or_malformed_calendar_rows_are_not_filled_in(tmp_path, variant):
    def change(data):
        if variant == "gap":
            return data.drop(index=3)
        if variant == "duplicate":
            return pd.concat([data, data.iloc[[3]]])
        if variant == "bad_flag":
            data.loc[3, "is_open"] = 2
            return data
        if variant == "bad_start":
            return data.iloc[1:]
        return data.iloc[:30]

    provider = SyntheticProvider(change=change)
    with pytest.raises(CalendarDataError):
        refresh_calendar(tmp_path, provider, now=NOW, history_start="2026-01-01")
    assert not (tmp_path / "manifest.json").exists()


def test_empty_historical_response_is_an_error_not_a_guessed_calendar(tmp_path):
    provider = SyntheticProvider(published_through="2025-12-31")
    with pytest.raises(CalendarDataError, match="missing_past_dates"):
        refresh_calendar(tmp_path, provider, now=NOW, history_start="2026-01-01")
    assert load_calendar_cache(tmp_path) is None


def test_existing_past_session_revision_is_rejected_without_republishing(tmp_path):
    seed_calendar(tmp_path, frame("2026-01-01", "2026-10-26"), now=NOW)
    original = (tmp_path / "manifest.json").read_bytes()

    def change(data):
        data.loc[data.cal_date.eq("20261019"), "is_open"] = 0
        return data

    with pytest.raises(CalendarDataError, match="changed_existing_past_session"):
        refresh_calendar(tmp_path, SyntheticProvider(change=change), now=NOW, history_start="2026-01-01")
    assert (tmp_path / "manifest.json").read_bytes() == original
    cached = load_calendar_cache(tmp_path).set_index("cal_date")
    assert cached.loc[pd.Timestamp("2026-10-19"), "is_open"] == 1


def test_published_future_session_revision_can_be_updated(tmp_path):
    seed_calendar(tmp_path, frame("2026-01-01", "2026-10-27"), now=NOW)

    def change(data):
        data.loc[data.cal_date.eq("20261022"), "is_open"] = 0
        return data

    result = refresh_calendar(tmp_path, SyntheticProvider(change=change), now=NOW, history_start="2026-01-01")
    assert result.set_index("cal_date").loc[pd.Timestamp("2026-10-22"), "is_open"] == 0


def test_far_future_cache_still_rechecks_tomorrows_official_session(tmp_path):
    seed_calendar(tmp_path, frame("2026-01-01", "2026-12-04"), now=NOW)

    def change(data):
        data.loc[data.cal_date.eq("20261021"), "is_open"] = 0
        return data

    provider = SyntheticProvider(change=change)
    result = refresh_calendar(tmp_path, provider, now=NOW, history_start="2026-01-01")
    assert provider.calls == [{
        "exchange": "SSE", "start_date": "20261013", "end_date": "20261204",
    }]
    assert result.set_index("cal_date").loc[pd.Timestamp("2026-10-21"), "is_open"] == 0
    assert load_calendar_cache(tmp_path).cal_date.iloc[-1] == pd.Timestamp("2026-12-04")


def test_far_future_cache_rejects_a_recent_past_revision_without_republishing(tmp_path):
    seed_calendar(tmp_path, frame("2026-01-01", "2026-12-04"), now=NOW)
    original = (tmp_path / "manifest.json").read_bytes()

    def change(data):
        data.loc[data.cal_date.eq("20261019"), "is_open"] = 0
        return data

    with pytest.raises(CalendarDataError, match="changed_existing_past_session"):
        refresh_calendar(tmp_path, SyntheticProvider(change=change), now=NOW, history_start="2026-01-01")
    assert (tmp_path / "manifest.json").read_bytes() == original


def test_future_year_checkpoints_are_reused_same_day_but_rechecked_on_next_local_date(tmp_path):
    seed_calendar(tmp_path, frame("2026-01-01", "2027-12-31"), now=NOW)
    first = SyntheticProvider()
    refresh_calendar(tmp_path, first, now=NOW, history_start="2026-01-01")
    assert len(first.calls) == 2
    same_day = SyntheticProvider()
    refresh_calendar(tmp_path, same_day, now=NOW + timedelta(hours=1), history_start="2026-01-01")
    assert not same_day.calls

    def change(data):
        data.loc[data.cal_date.isin(["20261022", "20270104"]), "is_open"] = 0
        return data

    later = SyntheticProvider(change=change)
    updated = refresh_calendar(tmp_path, later, now=NOW + timedelta(days=1), history_start="2026-01-01")
    assert [item["start_date"] for item in later.calls] == ["20261014", "20270101"]
    assert updated.set_index("cal_date").loc[pd.Timestamp("2026-10-22"), "is_open"] == 0
    assert updated.set_index("cal_date").loc[pd.Timestamp("2027-01-04"), "is_open"] == 0


def test_current_session_is_also_frozen_to_avoid_rewriting_a_trading_day(tmp_path):
    seed_calendar(tmp_path, frame("2026-01-01", "2026-10-27"), now=NOW)

    def change(data):
        data.loc[data.cal_date.eq("20261020"), "is_open"] = 0
        return data

    with pytest.raises(CalendarDataError, match="changed_existing_past_session"):
        refresh_calendar(tmp_path, SyntheticProvider(change=change), now=NOW, history_start="2026-01-01")


def test_complete_history_seed_is_idempotent_and_rejects_replacement(tmp_path):
    supplied = frame("2026-01-01", "2026-10-27")
    first = seed_calendar(tmp_path, supplied, now=NOW)
    original = (tmp_path / "manifest.json").read_bytes()
    second = seed_calendar(tmp_path, supplied.iloc[::-1], now=NOW)
    pd.testing.assert_frame_equal(first, second)
    assert (tmp_path / "manifest.json").read_bytes() == original
    with pytest.raises(CalendarDataError, match="seed_would_replace"):
        seed_calendar(tmp_path, frame("2026-01-01", "2026-10-20"), now=NOW)


def test_corrupt_snapshot_hash_and_path_are_never_trusted(tmp_path):
    seed_calendar(tmp_path, frame("2026-01-01", "2026-10-27"), now=NOW)
    manifest_path = tmp_path / "manifest.json"
    metadata = json.loads(manifest_path.read_text())
    snapshot = tmp_path / metadata["file"]
    snapshot.write_bytes(snapshot.read_bytes() + b"\n")
    with pytest.raises(CalendarDataError, match="hash_mismatch"):
        load_calendar_cache(tmp_path)
    metadata["file"] = "../external.csv"
    manifest_path.write_text(json.dumps(metadata))
    with pytest.raises(CalendarDataError, match="path_invalid"):
        load_calendar_cache(tmp_path)


def test_corrupt_checkpoint_blocks_resume_without_silently_discarding_history(tmp_path):
    with pytest.raises(CalendarFetchError):
        refresh_calendar(tmp_path, SyntheticProvider(fail_start="20250101"),
                         now=NOW, history_start="2024-01-01")
    metadata = json.loads((tmp_path / "checkpoints/20240101_20241231.json").read_text())
    path = tmp_path / metadata["file"]
    path.write_bytes(path.read_bytes() + b"\n")
    provider = SyntheticProvider()
    with pytest.raises(CalendarDataError, match="hash_mismatch"):
        refresh_calendar(tmp_path, provider, now=NOW, history_start="2024-01-01")
    assert not provider.calls


def test_missing_store_read_is_non_mutating_and_naive_refresh_clock_fails(tmp_path):
    missing = tmp_path / "missing"
    assert load_calendar_cache(missing) is None and not missing.exists()
    with pytest.raises(ValueError, match="timezone_required"):
        refresh_calendar(missing, SyntheticProvider(), now=NOW.replace(tzinfo=None))
    assert not missing.exists()
