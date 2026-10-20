"""Reference updates must be resumable and unable to revise earlier decisions."""
from __future__ import annotations

import json
import hashlib
import sqlite3

import pandas as pd
import pytest

from cfquant.data import TushareProvider
from cfquant.equity_reference import FINANCIAL_FIELDS, ReferenceStore


ASSET = "600000.SH"
OTHER = "000001.SZ"


def base_financials():
    return pd.DataFrame([{
        "ts_code": ASSET, "ann_date": "20260331", "end_date": "20251231", "roe": 10.0,
    }])


def base_industries():
    return pd.DataFrame([{
        "ts_code": ASSET, "l1_code": "OLD", "in_date": "20200101", "out_date": None,
    }])


def make_store(tmp_path, *, assets=None, baseline="2026-10-19", financials=None):
    return ReferenceStore(tmp_path, base_financials() if financials is None else financials,
                          base_industries(), assets or [ASSET], baseline)


class Provider:
    """Small deterministic endpoint contract; never contacts a network."""

    def __init__(self):
        self.calls = []
        self.roe = 20.0
        self.ann = "20260331"
        self.actual_ann = "20260401"
        self.industry = "NEW"
        self.triggers = {}
        self.fail_once = set()
        self.empty_financials = False
        self.empty_industry = False
        self.cap_industry = False
        self.cap_financial = False
        self.conflict = False

    def query(self, api, params, fields):
        self.calls.append((api, dict(params)))
        if api in self.fail_once:
            self.fail_once.remove(api)
            raise RuntimeError("must-not-log-provider-secret-or-message")
        columns = fields.split(",")
        rows = []
        if api == "disclosure_date":
            rows = [{"ts_code": asset, "actual_date": params["actual_date"]}
                    for asset in self.triggers.get(params["actual_date"], [])]
        elif api == "index_classify":
            rows = [{"index_code": self.industry, "industry_name": "sector", "src": "SW2021"}]
        elif api == "index_member_all":
            if params["is_new"] == "Y" and not self.empty_industry:
                rows = [{"ts_code": params.get("ts_code", ASSET), "l1_code": self.industry,
                         "l1_name": "sector", "in_date": "20200101", "out_date": None,
                         "is_new": "Y"}]
                if self.cap_industry and "ts_code" not in params:
                    rows *= 2000
        elif api == "fina_indicator":
            if not self.empty_financials and params["start_date"] <= "20251231" <= params["end_date"]:
                rows = [{"ts_code": params["ts_code"], "ann_date": self.ann,
                         "end_date": "20251231", "roe": self.roe, "update_flag": "1"}]
                span = (pd.Timestamp(params["end_date"]) - pd.Timestamp(params["start_date"])).days
                if self.cap_financial and span > 800:
                    rows *= 100
                if self.conflict:
                    rows.append({**rows[0], "roe": self.roe + 1})
        elif api == "income":
            if params["start_date"] <= self.ann <= params["end_date"]:
                rows = [{"ts_code": params["ts_code"], "ann_date": self.ann,
                         "f_ann_date": self.actual_ann, "end_date": "20251231",
                         "report_type": "1", "update_flag": "1"}]
        else:
            raise AssertionError(api)
        return pd.DataFrame(rows, columns=columns)


def run(store, provider, day="2026-10-20", hour="20:00", budget=100):
    return store.refresh(provider, day, observed_at=f"{day}T{hour}:00+08:00",
                         max_requests=budget)


def snapshot(store, day, hour="21:00"):
    return store.snapshot_asof(day, observed_at=f"{day}T{hour}:00+08:00")


def test_revision_first_observed_this_evening_does_not_rewrite_yesterday(tmp_path):
    store, provider = make_store(tmp_path), Provider()
    result = run(store, provider)
    assert result["ready"]
    old, _, old_meta = snapshot(store, "2026-10-19")
    fresh, _, fresh_meta = snapshot(store, "2026-10-20")
    assert old.roe.tolist() == [10.0]
    assert fresh.roe.tolist() == [20.0]
    assert fresh.ann_date.tolist() == ["20260401"]
    assert fresh.source_ann_date.tolist() == ["20260331"]
    assert fresh.first_seen_at.tolist() == ["2026-10-20T12:00:00+00:00"]
    assert old_meta["observed_financial_rows"] == 0
    assert fresh_meta["full_scan_complete"]


def test_precise_observation_time_is_respected_within_same_formation_day(tmp_path):
    store, provider = make_store(tmp_path), Provider()
    run(store, provider, hour="20:05")
    before, _, _ = snapshot(store, "2026-10-20", hour="20:04")
    after, _, _ = snapshot(store, "2026-10-20", hour="20:06")
    assert before.roe.tolist() == [10.0]
    assert after.roe.tolist() == [20.0]


def test_morning_catchup_cannot_put_new_observation_into_previous_day(tmp_path):
    store, provider = make_store(tmp_path), Provider()
    result = store.refresh(provider, "2026-10-20", observed_at="2026-10-21T08:00:00+08:00",
                           max_requests=100)
    previous, _, _ = store.snapshot_asof("2026-10-20",
                                         observed_at="2026-10-21T08:30:00+08:00")
    assert previous.roe.tolist() == [10.0]
    assert result["ready"] and result["maintenance_ready"]
    assert result["full_scan_complete"] and not result["snapshot_full_scan_complete"]
    assert result["uses_prior_reference_versions"]
    assert result["observed_financial_rows"] == 0


def test_announcement_is_strictly_before_formation_date(tmp_path):
    store, provider = make_store(tmp_path), Provider()
    provider.ann, provider.actual_ann = "20261020", "20261020"
    run(store, provider)
    same_day, _, _ = snapshot(store, "2026-10-20")
    tomorrow, _, _ = snapshot(store, "2026-10-21")
    assert same_day.roe.tolist() == [10.0]
    assert tomorrow.sort_values("ann_date").iloc[-1].roe == 20.0


def test_actual_announcement_delays_value_even_when_vendor_ann_is_earlier(tmp_path):
    store, provider = make_store(tmp_path), Provider()
    provider.ann, provider.actual_ann = "20261019", "20261021"
    result = run(store, provider)
    assert result["future_announcement_rows_ignored"] == 1
    without_refetch, _, _ = snapshot(store, "2026-10-22")
    assert without_refetch.roe.tolist() == [10.0]
    provider.triggers["20261021"] = [ASSET]
    run(store, provider, "2026-10-21")
    on_actual_day, _, _ = snapshot(store, "2026-10-21")
    after_actual_day, _, _ = snapshot(store, "2026-10-22")
    assert on_actual_day.roe.tolist() == [10.0]
    assert after_actual_day.sort_values("ann_date").iloc[-1].roe == 20.0


def test_same_report_revisions_keep_each_historical_version(tmp_path):
    store, provider = make_store(tmp_path), Provider()
    run(store, provider)
    provider.roe = 30.0
    provider.triggers["20261021"] = [ASSET, "999999.SH"]
    run(store, provider, "2026-10-21")
    first, _, _ = snapshot(store, "2026-10-20")
    second, _, _ = snapshot(store, "2026-10-21")
    assert first.roe.tolist() == [20.0]
    assert second.roe.tolist() == [30.0]
    assert set(second.ts_code) == {ASSET}
    assert second.first_seen_at.tolist() == ["2026-10-21T12:00:00+00:00"]


def test_unchanged_refetch_keeps_first_seen_time(tmp_path):
    store, provider = make_store(tmp_path), Provider()
    run(store, provider)
    provider.triggers["20261021"] = [ASSET]
    run(store, provider, "2026-10-21")
    frame, _, _ = snapshot(store, "2026-10-21")
    assert frame.first_seen_at.tolist() == ["2026-10-20T12:00:00+00:00"]


def test_complete_industry_versions_do_not_fill_old_history(tmp_path):
    store, provider = make_store(tmp_path), Provider()
    run(store, provider)
    provider.industry = "CHANGED"
    run(store, provider, "2026-10-26")
    _, old, _ = snapshot(store, "2026-10-19")
    _, first, _ = snapshot(store, "2026-10-20")
    _, second, metadata = snapshot(store, "2026-10-26")
    assert old.l1_code.tolist() == ["OLD"]
    assert first.l1_code.tolist() == ["NEW"]
    assert second.l1_code.tolist() == ["CHANGED"]
    assert metadata["ready"]


def test_budget_checkpoint_survives_new_store_instance_without_repeat_calls(tmp_path):
    provider = Provider()
    first = run(make_store(tmp_path), provider, budget=5)
    assert first["requests_attempted"] == 5 and first["backlog_tasks"] > 0
    assert first["completed_this_run"] == 5
    before = list(provider.calls)
    second = run(make_store(tmp_path), provider, budget=10)
    assert second["ready"]
    assert second["requests_attempted"] == 1
    assert second["completed_this_run"] == 1
    assert provider.calls[:-1] == before
    third = run(make_store(tmp_path), provider, budget=100)
    assert third["requests_attempted"] == 0


def test_failed_request_retries_later_without_repeating_completed_partner(tmp_path):
    store, provider = make_store(tmp_path), Provider()
    provider.fail_once.add("fina_indicator")
    failed = run(store, provider)
    before, _, _ = snapshot(store, "2026-10-20")
    assert not failed["ready"] and before.roe.tolist() == [10.0]
    assert failed["completed_this_run"] == failed["requests_attempted"] - 1
    assert failed["errors_this_refresh"][0]["error_type"] == "RuntimeError"
    assert "must-not-log" not in json.dumps(failed)
    succeeded = run(make_store(tmp_path), provider)
    assert succeeded["ready"]
    assert sum(api == "income" for api, _ in provider.calls) == 1
    assert sum(api == "fina_indicator" for api, _ in provider.calls) == 2
    with sqlite3.connect(store.path) as db:
        assert "must-not-log" not in str(db.execute("SELECT last_error FROM jobs").fetchall())


def test_empty_financial_response_preserves_prior_value_and_is_disclosed(tmp_path):
    store, provider = make_store(tmp_path), Provider()
    provider.empty_financials = True
    result = run(store, provider)
    frame, _, _ = snapshot(store, "2026-10-20")
    assert frame.roe.tolist() == [10.0]
    assert result["empty_annual_refresh_assets"] == [ASSET]
    assert result["observed_financial_rows"] == 0
    # "Full scan" means complete requests; it does not invent a financial value.
    assert result["financial_full_scan_complete"]


def test_empty_complete_industry_scan_is_blocked_and_cannot_erase_baseline(tmp_path):
    store, provider = make_store(tmp_path), Provider()
    provider.empty_industry = True
    result = run(store, provider)
    _, industry, _ = snapshot(store, "2026-10-20")
    assert industry.l1_code.tolist() == ["OLD"]
    assert result["blocked_groups"] and not result["ready"]
    assert not result["industry_full_scan_complete"]


def test_industry_row_cap_splits_by_fixed_symbols_without_assuming_offset(tmp_path):
    store, provider = make_store(tmp_path, assets=[ASSET, OTHER]), Provider()
    provider.cap_industry = True
    result = run(store, provider)
    _, industries, _ = snapshot(store, "2026-10-20")
    assert result["ready"]
    assert set(industries.ts_code) == {ASSET, OTHER}
    fallbacks = [params for api, params in provider.calls
                 if api == "index_member_all" and "ts_code" in params]
    assert {params["ts_code"] for params in fallbacks} == {ASSET, OTHER}
    assert all("offset" not in params for _, params in provider.calls)


def test_financial_row_cap_splits_report_bounds_and_combines_complete_leaves(tmp_path):
    store, provider = make_store(tmp_path), Provider()
    provider.cap_financial = True
    result = run(store, provider)
    assert result["ready"]
    assert sum(api == "fina_indicator" for api, _ in provider.calls) > 1
    frame, _, _ = snapshot(store, "2026-10-20")
    assert frame.roe.tolist() == [20.0]


def test_conflicting_same_public_identity_is_missing_not_arbitrarily_selected(tmp_path):
    store, provider = make_store(tmp_path), Provider()
    provider.conflict = True
    run(store, provider)
    frame, _, _ = snapshot(store, "2026-10-20")
    assert frame.roe.isna().all()
    assert frame.conflicting_values.all()


def test_future_announcements_and_outside_universe_are_excluded(tmp_path):
    future = pd.concat([base_financials(), pd.DataFrame([
        {"ts_code": ASSET, "ann_date": "20261201", "end_date": "20251231", "roe": 900.0},
        {"ts_code": OTHER, "ann_date": "20260331", "end_date": "20251231", "roe": 800.0},
    ])], ignore_index=True)
    store = make_store(tmp_path, financials=future)
    frame, _, _ = snapshot(store, "2026-10-20")
    assert frame.roe.tolist() == [10.0]
    with pytest.raises(ValueError, match="future"):
        store.snapshot_asof("2026-10-22", observed_at="2026-10-21T20:00:00+08:00")


def test_universe_and_inherited_reference_identity_cannot_change(tmp_path):
    make_store(tmp_path)
    with pytest.raises(ValueError, match="identity"):
        make_store(tmp_path, assets=[ASSET, OTHER])
    changed = base_financials()
    changed.loc[0, "roe"] = 500
    with pytest.raises(ValueError, match="identity"):
        make_store(tmp_path, financials=changed)


def test_long_gap_is_checkpointed_without_a_forty_day_limit(tmp_path):
    store, provider = make_store(tmp_path, baseline="2026-08-01"), Provider()
    first = run(store, provider, budget=7)
    assert first["requests_attempted"] == 7
    assert first["backlog_tasks"] > 40 and not first["ready"]
    final = run(make_store(tmp_path, baseline="2026-08-01"), provider, budget=200)
    assert final["ready"]
    assert sum(api == "disclosure_date" for api, _ in provider.calls) == 80


def test_zero_budget_does_not_query_and_invalid_clock_or_budget_is_rejected(tmp_path):
    store, provider = make_store(tmp_path), Provider()
    assert run(store, provider, budget=0)["requests_attempted"] == 0
    assert provider.calls == []
    with pytest.raises(ValueError, match="timezone"):
        store.refresh(provider, "2026-10-20", observed_at="2026-10-20T20:00:00")
    with pytest.raises(ValueError, match="nonnegative"):
        run(store, provider, budget=-1)


def test_empty_or_wrong_schema_is_not_a_successful_refresh(tmp_path):
    class BrokenProvider(Provider):
        def query(self, api, params, fields):
            if api == "fina_indicator":
                self.calls.append((api, dict(params)))
                return pd.DataFrame()
            return super().query(api, params, fields)

    store = make_store(tmp_path)
    result = run(store, BrokenProvider())
    frame, _, _ = snapshot(store, "2026-10-20")
    assert not result["ready"]
    assert result["errors_this_refresh"][0]["error_type"] == "ValueError"
    assert frame.roe.tolist() == [10.0]


def test_provider_ignoring_report_bounds_cannot_complete_scan(tmp_path):
    class UnboundedProvider(Provider):
        def query(self, api, params, fields):
            result = super().query(api, params, fields)
            if api == "fina_indicator":
                result.loc[0, "end_date"] = "20201231"
            return result

    result = run(make_store(tmp_path), UnboundedProvider())
    assert not result["ready"] and result["backlog_tasks"] > 0


def test_year_change_enqueues_current_cycle_without_refetching_all_missed_months(tmp_path):
    store, provider = make_store(tmp_path), Provider()
    run(store, provider)
    result = run(store, provider, "2027-01-04", budget=200)
    assert result["ready"]
    financial = [params for api, params in provider.calls if api == "fina_indicator"]
    assert len(financial) == 2
    assert financial[-1]["start_date"] == "20240101"


def test_per_request_clock_keeps_actual_completion_time_and_midnight_boundary(tmp_path):
    store, provider = make_store(tmp_path), Provider()
    moments = iter([
        "2026-10-20T23:59:00+08:00",  # schedule
        "2026-10-20T23:59:10+08:00",  # disclosure
        "2026-10-20T23:59:20+08:00",  # industry catalogue
        "2026-10-20T23:59:30+08:00",  # members N
        "2026-10-20T23:59:40+08:00",  # members Y, industry complete
        "2026-10-20T23:59:50+08:00",  # income
        "2026-10-21T00:00:10+08:00",  # indicators, financial complete
        "2026-10-21T00:00:20+08:00",  # final maintenance report
    ])
    result = store.refresh(provider, "2026-10-20", max_requests=100,
                           observation_clock=lambda: next(moments))
    old, industry, _ = store.snapshot_asof(
        "2026-10-20", observed_at="2026-10-21T00:00:30+08:00")
    assert result["ready"] and result["uses_prior_reference_versions"]
    assert old.roe.tolist() == [10.0] and industry.l1_code.tolist() == ["NEW"]
    today, _, _ = snapshot(store, "2026-10-21")
    assert today.roe.tolist() == [20.0]
    assert today.first_seen_at.tolist() == ["2026-10-20T16:00:10+00:00"]


def test_late_restatement_of_older_year_does_not_replace_latest_known_annual_report(tmp_path):
    class OlderRestatement(Provider):
        def query(self, api, params, fields):
            result = super().query(api, params, fields)
            if api in {"fina_indicator", "income"}:
                result.loc[0, "end_date"] = "20241231"
                result.loc[0, "ann_date"] = "20261001"
                if api == "income":
                    result.loc[0, "f_ann_date"] = "20261001"
            return result

    store = make_store(tmp_path)
    run(store, OlderRestatement())
    frame, _, metadata = snapshot(store, "2026-10-20")
    assert frame.end_date.tolist() == ["20251231"]
    assert frame.roe.tolist() == [10.0]
    assert metadata["annual_period_selected_assets"] == 1
    assert metadata["inherited_financial_rows"] == 1


def test_inherited_only_historical_dates_keep_original_financial_selection(tmp_path):
    legacy = pd.concat([base_financials(), pd.DataFrame([{
        "ts_code": ASSET, "ann_date": "20261001", "end_date": "20241231", "roe": 40.0,
    }])], ignore_index=True)
    store = make_store(tmp_path, financials=legacy)
    frame, _, metadata = snapshot(store, "2026-10-19")
    assert set(frame.end_date) == {"20241231", "20251231"}
    assert metadata["annual_period_selected_assets"] == 0


@pytest.mark.parametrize("legacy_pending_payload", [False, True])
def test_late_completed_monthly_group_cannot_undo_newer_event_revision(tmp_path, legacy_pending_payload):
    store, provider = make_store(tmp_path), Provider()
    provider.fail_once.add("income")
    first = run(store, provider)
    assert not first["ready"]
    if legacy_pending_payload:
        with sqlite3.connect(store.path) as db:
            for key, result in db.execute("SELECT key,result FROM jobs WHERE result IS NOT NULL").fetchall():
                rows = json.loads(result)
                for row in rows:
                    row.pop("_data_observed_at", None)
                db.execute("UPDATE jobs SET result=? WHERE key=?", (json.dumps(rows), key))
    provider.roe = 30.0
    provider.triggers["20261021"] = [ASSET]
    finished = run(store, provider, "2026-10-21")
    frame, _, _ = snapshot(store, "2026-10-21")
    assert finished["ready"]
    assert frame.roe.tolist() == [30.0]
    assert frame.data_observed_at.tolist() == ["2026-10-21T12:00:00+00:00"]
    old, _, _ = snapshot(store, "2026-10-20")
    assert old.roe.tolist() == [10.0]  # The delayed group was not available yet.


def test_old_bootstrap_observations_recover_data_clock_from_retained_jobs(tmp_path):
    store, provider = make_store(tmp_path), Provider()
    provider.fail_once.add("income")
    run(store, provider)
    provider.roe = 30.0
    provider.triggers["20261021"] = [ASSET]
    run(store, provider, "2026-10-21")
    # Simulate persisted output of the previous worker version in this test DB.
    with sqlite3.connect(store.path) as db:
        for key, result in db.execute("SELECT key,result FROM jobs WHERE result IS NOT NULL").fetchall():
            rows = json.loads(result)
            for row in rows:
                row.pop("_data_observed_at", None)
            db.execute("UPDATE jobs SET result=? WHERE key=?", (json.dumps(rows), key))
        for identity, payload in db.execute("SELECT id,payload FROM observations").fetchall():
            data = json.loads(payload)
            for row in data["rows"]:
                row.pop("data_observed_at", None)
                row.pop("_data_observed_at", None)
            db.execute("UPDATE observations SET payload=? WHERE id=?", (json.dumps(data), identity))
    frame, industry, _ = snapshot(make_store(tmp_path), "2026-10-21")
    assert frame.roe.tolist() == [30.0]
    assert frame.data_observed_at.tolist() == ["2026-10-21T12:00:00+00:00"]
    assert industry.l1_code.tolist() == ["NEW"]


def test_cached_response_keeps_original_fetch_time_after_checkpoint_interruption(tmp_path):
    class CachedProvider(TushareProvider):
        """Local fixture implementing the existing provider receipt contract."""
        def __init__(self, cache):
            self.cache = cache
            self.cache.mkdir()
            self.backend = Provider()
            self.fetched_at = "2026-10-20T12:00:00+00:00"

        def query(self, api, params, fields):
            identity = {"api_name": api, "params": params, "fields": fields}
            key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
            path = self.cache / f"{api}_{key}.json"
            if path.exists():
                data = json.loads(path.read_text())["data"]
                return pd.DataFrame(data["items"], columns=data["fields"])
            frame = self.backend.query(api, params, fields)
            data = json.loads(frame.to_json(orient="split"))
            path.write_text(json.dumps({
                "request": identity, "retrieved_utc": self.fetched_at,
                "data": {"fields": data["columns"], "items": data["data"]},
            }))
            return frame

    store = make_store(tmp_path / "store")
    provider = CachedProvider(tmp_path / "cache")
    run(store, provider, budget=0)
    # Network response was cached, but the process stopped before its DB checkpoint.
    provider.query("fina_indicator", {
        "ts_code": ASSET, "start_date": "20230101", "end_date": "20261020", "limit": 100,
    }, FINANCIAL_FIELDS)
    provider.backend.roe = 30.0
    provider.backend.triggers["20261021"] = [ASSET]
    provider.fetched_at = "2026-10-21T12:00:00+00:00"
    result = run(store, provider, "2026-10-21")
    frame, _, _ = snapshot(store, "2026-10-21")
    assert result["ready"]
    assert frame.roe.tolist() == [30.0]
    with sqlite3.connect(store.path) as db:
        raw = db.execute(
            "SELECT result FROM jobs WHERE key=?",
            (f"financial:monthly:2026-10:{ASSET}:financial_indicator",)).fetchone()[0]
    assert json.loads(raw)[0]["_data_observed_at"] == "2026-10-20T12:00:00+00:00"


def test_late_old_week_industry_completion_cannot_replace_new_week_snapshot(tmp_path):
    class DelayedIndustry(Provider):
        def __init__(self):
            super().__init__()
            self.industry = "WEEK_OLD"
            self.fail_old = True

        def query(self, api, params, fields):
            if (self.fail_old and api == "index_member_all" and
                    params.get("l1_code") == "WEEK_OLD" and params["is_new"] == "N"):
                self.calls.append((api, dict(params)))
                raise RuntimeError("transient")
            return super().query(api, params, fields)

    store, provider = make_store(tmp_path), DelayedIndustry()
    assert not run(store, provider)["ready"]
    provider.industry = "WEEK_NEW"
    assert not run(store, provider, "2026-10-26")["ready"]
    provider.fail_old = False
    result = run(store, provider, "2026-10-27")
    _, industry, _ = snapshot(store, "2026-10-27")
    assert result["ready"]
    assert industry.l1_code.tolist() == ["WEEK_NEW"]


def test_empty_industry_scan_retries_same_cycle_with_budget_and_preserves_failure(tmp_path):
    store, provider = make_store(tmp_path), Provider()
    provider.empty_industry = True
    failed = run(store, provider)
    assert failed["blocked_groups"]
    provider.empty_industry = False
    partial = run(store, provider, budget=1)
    assert partial["requests_attempted"] == 1
    assert not partial["ready"] and partial["backlog_tasks"] == 2
    assert not partial["blocked_groups"] and partial["historical_blocked_groups"]
    complete = run(store, provider)
    assert complete["ready"]
    assert complete["historical_blocked_groups"] == failed["blocked_groups"]
    assert run(store, provider)["requests_attempted"] == 0


def test_empty_industry_retry_does_not_loop_unbounded_within_one_refresh(tmp_path):
    store, provider = make_store(tmp_path), Provider()
    provider.empty_industry = True
    run(store, provider)
    result = run(store, provider, budget=100)
    assert result["requests_attempted"] == 3
    assert result["completed_this_run"] == 3
    assert not result["ready"]
    with sqlite3.connect(store.path) as db:
        count = db.execute("SELECT count(*) FROM groups WHERE kind='industry'").fetchone()[0]
    assert count == 2


def test_successful_new_week_is_not_blocked_by_previous_empty_week(tmp_path):
    store, provider = make_store(tmp_path), Provider()
    provider.empty_industry = True
    failed = run(store, provider)
    provider.empty_industry = False
    result = run(store, provider, "2026-10-26")
    assert result["ready"]
    assert result["blocked_groups"] == []
    assert result["historical_blocked_groups"] == failed["blocked_groups"]
