from concurrent.futures import Future
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Event
from types import SimpleNamespace

import pandas as pd
import pytest

from cfquant import simulation_refresh
from cfquant.simulation_refresh import ContinuousRefresh, perform_refresh
from cfquant.simulation_schedule import ScheduleStore


class ManualPool:
    def __init__(self):
        self.calls = []

    def submit(self, function, *args, **kwargs):
        future = Future()
        self.calls.append((function, args, kwargs, future))
        return future

    def shutdown(self, **kwargs):
        pass


def setup(tmp_path):
    now = datetime.fromisoformat("2026-12-30T18:00:00+08:00")
    dates = pd.date_range("2026-12-01", "2027-02-28")
    calendar = pd.DataFrame({"cal_date": dates, "is_open": (dates.dayofweek < 5).astype(int)})
    state = ScheduleStore(tmp_path / "runs/simulation_auto/refresh_schedule.json")
    attempt = state.begin("calendar", (now + timedelta(days=14)).date(), now=now - timedelta(minutes=1))
    state.finish(attempt, success=True, now=now,
                 completed_target_date=calendar.cal_date.max().date())
    return now, calendar, state


def test_data_thread_does_not_block_heartbeat_and_writes_claim_before_submission(tmp_path):
    now, calendar, store = setup(tmp_path)
    pool = ManualPool()
    controller = ContinuousRefresh(tmp_path, pool=pool, calendar_reader=lambda _: calendar)
    first = controller.tick(now=now)
    assert first["running"] and first["job"] == "data"
    assert store.load()["data"]["2026-12-30"]["status"] == "inflight"
    for second in range(1, 8):
        result = controller.tick(now=now + timedelta(seconds=second))
        assert result["running"]
    assert len(pool.calls) == 1
    pool.calls[0][3].set_result({"success": True, "completed_target_date": now.date()})
    result = controller.tick(now=now + timedelta(seconds=8))
    assert not result["running"]
    assert store.load()["data"]["2026-12-30"]["status"] == "succeeded"
    assert len(pool.calls) == 1


def test_restart_retains_failure_backoff_and_sanitizes_exception(tmp_path):
    now, calendar, store = setup(tmp_path)
    pool = ManualPool()
    controller = ContinuousRefresh(tmp_path, pool=pool, calendar_reader=lambda _: calendar)
    controller.tick(now=now)
    pool.calls[0][3].set_exception(RuntimeError("PRIVATE_PROVIDER_MESSAGE"))
    failed = controller.tick(now=now + timedelta(seconds=1))
    assert failed["last_result"]["error_type"] == "RuntimeError"
    assert "PRIVATE_PROVIDER_MESSAGE" not in str(failed)
    restarted_pool = ManualPool()
    restarted = ContinuousRefresh(tmp_path, pool=restarted_pool, calendar_reader=lambda _: calendar)
    assert not restarted.tick(now=now + timedelta(seconds=30))["running"]
    assert restarted.tick(now=now + timedelta(minutes=2))["running"]
    assert store.load()["data"]["2026-12-30"]["attempts"] == 2


def test_real_progress_resumes_after_one_minute_without_exponential_penalty(tmp_path):
    now, calendar, store = setup(tmp_path)
    pool = ManualPool()
    controller = ContinuousRefresh(tmp_path, pool=pool, calendar_reader=lambda _: calendar)
    controller.tick(now=now)
    pool.calls[0][3].set_result({"success": False, "progress": True,
                                "remaining_sessions": 61})
    controller.tick(now=now + timedelta(seconds=1))
    assert store.load()["data"]["2026-12-30"]["status"] == "partial"
    assert not controller.tick(now=now + timedelta(seconds=30))["running"]
    assert controller.tick(now=now + timedelta(seconds=62))["running"]
    assert store.load()["data"]["2026-12-30"]["attempts"] == 1


def test_data_failed_on_previous_date_does_not_block_next_session(tmp_path):
    now, calendar, _ = setup(tmp_path)
    pool = ManualPool()
    controller = ContinuousRefresh(tmp_path, pool=pool, calendar_reader=lambda _: calendar)
    controller.tick(now=now)
    pool.calls[0][3].set_exception(ValueError("synthetic"))
    controller.tick(now=now + timedelta(seconds=1))
    assert controller.tick(now=now + timedelta(days=1))["target_date"] == "2026-12-31"


def test_calendar_publication_failure_allows_data_during_independent_backoff(tmp_path):
    now = datetime.fromisoformat("2026-12-20T18:00:00+08:00")
    dates = pd.date_range("2026-12-01", "2026-12-31")
    calendar = pd.DataFrame({"cal_date": dates, "is_open": (dates.dayofweek < 5).astype(int)})
    pool = ManualPool()
    controller = ContinuousRefresh(tmp_path, pool=pool, calendar_reader=lambda _: calendar)
    first = controller.tick(now=now)
    assert first["job"] == "calendar"
    assert first["decision"]["data_due"]
    pool.calls[0][3].set_result({
        "success": False, "progress": False, "coverage_end": "2026-12-31",
    })
    second = controller.tick(now=now + timedelta(seconds=1))
    assert second["job"] == "data" and second["target_date"] == "2026-12-18"
    assert len(pool.calls) == 2
    assert controller.store.load()["calendar"]["2027-01-03"]["status"] == "failed"
    assert not second["decision"]["calendar_due"]


def test_unfinished_data_attempt_survives_process_restart_without_immediate_repeat(tmp_path):
    now, calendar, store = setup(tmp_path)
    original_pool = ManualPool()
    original = ContinuousRefresh(tmp_path, pool=original_pool, calendar_reader=lambda _: calendar)
    original.tick(now=now)
    restarted_pool = ManualPool()
    restarted = ContinuousRefresh(tmp_path, pool=restarted_pool, calendar_reader=lambda _: calendar)
    assert not restarted.tick(now=now + timedelta(seconds=59))["running"]
    assert not restarted_pool.calls
    assert restarted.tick(now=now + timedelta(seconds=60))["job"] == "data"
    assert store.load()["data"]["2026-12-30"]["attempts"] == 2


def test_pool_failure_preserves_durable_claim_and_restart_backoff(tmp_path):
    now, calendar, store = setup(tmp_path)

    class BrokenPool(ManualPool):
        def submit(self, *args, **kwargs):
            raise RuntimeError("synthetic closed executor")

    controller = ContinuousRefresh(tmp_path, pool=BrokenPool(), calendar_reader=lambda _: calendar)
    with pytest.raises(RuntimeError, match="synthetic closed executor"):
        controller.tick(now=now)
    assert store.load()["data"]["2026-12-30"]["status"] == "inflight"
    pool = ManualPool()
    restored = ContinuousRefresh(tmp_path, pool=pool, calendar_reader=lambda _: calendar)
    assert not restored.tick(now=now + timedelta(seconds=30))["running"]
    assert not pool.calls


def test_cross_midnight_job_completes_original_target_without_creating_future_session(tmp_path):
    now, calendar, store = setup(tmp_path)
    now = now.replace(hour=23, minute=59)
    pool = ManualPool()
    controller = ContinuousRefresh(tmp_path, pool=pool, calendar_reader=lambda _: calendar)
    controller.tick(now=now)
    pool.calls[0][3].set_result({"success": True, "completed_target_date": date(2026, 12, 30)})
    result = controller.tick(now=now + timedelta(minutes=2))
    assert not result["running"]
    assert result["decision"]["target_date"] == "2026-12-30"
    assert store.load()["data"]["2026-12-30"]["status"] == "succeeded"
    assert "2026-12-31" not in store.load()["data"]


def test_job_crossing_next_publication_finishes_old_target_before_starting_new_one(tmp_path):
    now, calendar, store = setup(tmp_path)
    pool = ManualPool()
    controller = ContinuousRefresh(tmp_path, pool=pool, calendar_reader=lambda _: calendar)
    controller.tick(now=now)
    tomorrow = now + timedelta(days=1)
    still_running = controller.tick(now=tomorrow)
    assert still_running["target_date"] == "2026-12-30"
    assert still_running["decision"]["target_date"] == "2026-12-31"
    assert len(pool.calls) == 1
    pool.calls[0][3].set_result({"success": True, "completed_target_date": date(2026, 12, 30)})
    next_job = controller.tick(now=tomorrow + timedelta(seconds=1))
    assert next_job["target_date"] == "2026-12-31" and len(pool.calls) == 2
    state = store.load()
    assert state["data"]["2026-12-30"]["status"] == "succeeded"
    assert state["data"]["2026-12-31"]["status"] == "inflight"


def test_start_jobs_false_still_records_completion_but_does_not_start_next_target(tmp_path):
    now, calendar, store = setup(tmp_path)
    pool = ManualPool()
    controller = ContinuousRefresh(tmp_path, pool=pool, calendar_reader=lambda _: calendar)
    controller.tick(now=now)
    pool.calls[0][3].set_result({"success": True, "completed_target_date": now.date()})
    result = controller.tick(now=now + timedelta(days=1), start_jobs=False)
    assert not result["running"] and result["decision"]["data_due"]
    assert len(pool.calls) == 1
    assert store.load()["data"]["2026-12-30"]["status"] == "succeeded"


def test_real_background_thread_does_not_wait_for_provider_before_next_tick(tmp_path):
    now, calendar, _ = setup(tmp_path)
    entered, release, completed = Event(), Event(), Event()

    def worker(root, attempt, *, now):
        entered.set()
        try:
            assert release.wait(3), "Test driver did not release the synthetic background job"
            return {"success": True, "completed_target_date": attempt.target_date}
        finally:
            completed.set()

    controller = ContinuousRefresh(tmp_path, worker=worker, calendar_reader=lambda _: calendar)
    try:
        assert controller.tick(now=now)["running"]
        assert entered.wait(1)
        assert controller.tick(now=now + timedelta(seconds=1))["running"]
        assert not completed.is_set()
    finally:
        release.set()
        controller.wait_for_current()
        controller.close()
    assert completed.is_set()


def test_refresh_worker_pins_injected_clock_and_rejects_wrong_signal_target(tmp_path, monkeypatch):
    from cfquant import simulation_service as service
    now, calendar, store = setup(tmp_path)
    attempt = store.begin("data", now.date(), now=now)
    provider = object()
    clocks = []
    monkeypatch.setattr(simulation_refresh, "_provider", lambda *args: provider)
    monkeypatch.setattr(simulation_refresh, "read_calendar", lambda root: calendar)

    def signals(root, *, now, provider):
        clocks.append(now)
        return {"synthetic": SimpleNamespace(signal_asof=attempt.target_date,
                                             execution_date=date(2026, 12, 31))}

    def equities(root, *, now, provider):
        clocks.append(now)
        return {"schema": 2, "ready_for_auto_orders": True, "signal_date": str(attempt.target_date),
                "execution_date": "2026-12-31", "model": {"model_year": 2026}}

    monkeypatch.setattr(service, "refresh_signals", signals)
    monkeypatch.setattr(service, "refresh_equities", equities)
    result = perform_refresh(tmp_path, attempt, now=now)
    assert result["success"] and result["completed_target_date"] == attempt.target_date
    assert clocks == [now, now]
    monkeypatch.setattr(service, "refresh_signals", lambda *a, **kw: {
        "synthetic": SimpleNamespace(signal_asof=attempt.target_date + timedelta(days=1))})
    monkeypatch.setattr(service, "refresh_equities", lambda *a, **kw: pytest.fail("Wrong target reached equities"))
    with pytest.raises(ValueError, match="target no longer agrees"):
        perform_refresh(tmp_path, attempt, now=now)


@pytest.mark.parametrize("damaged", ["{unfinished", "[]", "null"])
def test_non_authoritative_status_cache_corruption_does_not_prevent_restart(tmp_path, damaged):
    now, calendar, _ = setup(tmp_path)
    path = tmp_path / "runs/simulation_auto/refresh_status.json"
    path.write_text(damaged, encoding="utf-8")
    pool = ManualPool()
    controller = ContinuousRefresh(tmp_path, pool=pool, calendar_reader=lambda _: calendar)
    assert controller.tick(now=now)["job"] == "data"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["last_result"] == {}


def test_worker_entrypoint_binds_its_checkout_before_external_editable_install(tmp_path):
    """--help imports modules but never starts a worker, supplier or system task."""
    package = tmp_path / "foreign_checkout/cfquant"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "raise RuntimeError('WRONG_CHECKOUT_WAS_IMPORTED')\n", encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts/run_simulation_service.py"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(package.parent)
    result = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=tmp_path, env=environment,
        capture_output=True, text=True, timeout=20,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 0, result.stderr
    assert "--watch" in result.stdout
    assert not (tmp_path / "runs").exists()


def test_successful_target_is_invalidated_once_when_next_session_moves_to_new_year(tmp_path, monkeypatch):
    from cfquant import simulation_service as service
    now, calendar, store = setup(tmp_path)
    attempt = store.begin("data", now.date(), now=now)
    store.finish(attempt, success=True, completed_target_date=now.date(), now=now)
    receipt = {
        "schema": 2, "signal_date": "2026-12-30", "execution_date": "2026-12-31",
        "model": {"model_year": 2026}, "reference_snapshot_date": "2026-12-30",
        "score_coverage": {}, "variant_labels": {},
    }
    calendar.loc[calendar.cal_date.between("2026-12-31", "2027-01-03"), "is_open"] = 0
    monkeypatch.setattr(service, "_equity_bundle", lambda root: (tmp_path, receipt))
    monkeypatch.setattr(service, "market_calendar", lambda root: calendar)
    with pytest.raises(service.EquityExecutionMismatch):
        service.latest_equity_status(tmp_path)
    pool = ManualPool()
    controller = ContinuousRefresh(tmp_path, pool=pool, calendar_reader=lambda root: calendar)
    result = controller.tick(now=now + timedelta(seconds=1))
    assert not result["running"] and not pool.calls
    invalidated = store.load()["data"]["2026-12-30"]
    assert invalidated["status"] == "invalidated"
    assert invalidated["maintenance"]["required_execution_date"] == "2027-01-04"
    persisted = store.path.read_bytes()
    restarted = ContinuousRefresh(tmp_path, pool=pool, calendar_reader=lambda root: calendar)
    assert not restarted.tick(now=now + timedelta(seconds=30))["running"]
    assert not restarted.tick(now=now + timedelta(seconds=60))["running"]
    assert store.path.read_bytes() == persisted
    assert restarted.tick(now=now + timedelta(seconds=61))["job"] == "data"
    assert store.load()["data"]["2026-12-30"]["attempts"] == 1
    # The re-prepared score is bound to the new session and new annual model.
    receipt.update(execution_date="2027-01-04", model={"model_year": 2027})
    pool.calls[0][3].set_result({"success": True, "completed_target_date": now.date()})
    assert not restarted.tick(now=now + timedelta(seconds=62))["running"]
    assert store.load()["data"]["2026-12-30"]["status"] == "succeeded"
    assert service.latest_equity_status(tmp_path)["model_year"] == 2027
    # Extending known future dates does not change the relevant binding.
    extra = pd.date_range("2027-03-01", "2027-03-31")
    calendar = pd.concat([calendar, pd.DataFrame({
        "cal_date": extra, "is_open": (extra.dayofweek < 5).astype(int),
    })], ignore_index=True)
    assert not restarted.tick(now=now + timedelta(seconds=90))["running"]
    assert len(pool.calls) == 1


@pytest.mark.parametrize("execution_date,model_year", [
    ("2027-01-04", 2026), ("2026-12-31", 2025),
])
def test_ready_wrong_execution_or_model_receipt_cannot_complete_refresh(
        tmp_path, monkeypatch, execution_date, model_year):
    from cfquant import simulation_service as service
    now, calendar, store = setup(tmp_path)
    attempt = store.begin("data", now.date(), now=now)
    monkeypatch.setattr(simulation_refresh, "_provider", lambda *args: object())
    monkeypatch.setattr(simulation_refresh, "read_calendar", lambda root: calendar)
    monkeypatch.setattr(service, "refresh_signals", lambda *a, **kw: {
        "synthetic": SimpleNamespace(signal_asof=now.date(), execution_date=date(2026, 12, 31))})
    monkeypatch.setattr(service, "refresh_equities", lambda *a, **kw: {
        "schema": 2, "ready_for_auto_orders": True, "signal_date": str(now.date()),
        "execution_date": execution_date, "model": {"model_year": model_year},
    })
    with pytest.raises(service.EquityExecutionMismatch):
        perform_refresh(tmp_path, attempt, now=now)
