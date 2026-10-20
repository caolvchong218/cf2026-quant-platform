"""Durable automatic data refresh, independent of the order-execution loop."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from .data import TushareProvider
from .simulation_schedule import ScheduleStore, decide_refresh


def _base(root):
    return Path(root) / "runs/simulation_auto"


def _json_dates(value):
    if isinstance(value, dict):
        return {key: _json_dates(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_dates(item) for item in value]
    return value.isoformat() if hasattr(value, "isoformat") else value


def _provider(root, kind, now):
    from .simulation_service import load_settings
    path = _base(root) / "refresh_raw" / (
        now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + kind)
    return TushareProvider(path, load_settings(root)["token_file"], delay=1.3)


def read_calendar(root):
    from .simulation_calendar import load_calendar_cache
    cached = load_calendar_cache(_base(root) / "calendar_store")
    if cached is not None:
        return cached
    from .simulation_service import market_calendar
    return market_calendar(root, prefer_store=False)


def perform_refresh(root, attempt, *, now):
    """Read-only acquisition job; has no access to an order-submission adapter."""
    from .simulation_service import refresh_equities, refresh_signals, validate_equity_execution
    provider = _provider(root, attempt.kind, now)
    if attempt.kind == "calendar":
        from .simulation_calendar import load_calendar_cache, refresh_calendar, seed_calendar
        target = _base(root) / "calendar_store"
        if load_calendar_cache(target) is None:
            old = read_calendar(root)
            if old is not None:
                seed_calendar(target, old)
        calendar = refresh_calendar(target, provider, now=now)
        coverage = calendar.cal_date.max().date()
        return {"success": coverage >= attempt.target_date, "completed_target_date": coverage,
                "progress": False, "coverage_end": str(coverage)}
    # Each component independently resumes immutable completed partitions.
    # A partial equity batch cannot hold up the worker's execution heartbeat.
    signals = refresh_signals(root, now=now, provider=provider)
    if not signals or any(bundle.signal_asof != attempt.target_date for bundle in signals.values()):
        raise ValueError("Refresh target no longer agrees with completed-session clock")
    receipt = refresh_equities(root, now=now, provider=provider)
    success = (receipt.get("ready_for_auto_orders") is True
               and receipt.get("signal_date") == str(attempt.target_date))
    if success:
        execution = validate_equity_execution(receipt, read_calendar(root))
        if any(bundle.execution_date != execution for bundle in signals.values()):
            raise ValueError("Completed signals disagree with the verified execution session")
    return {"success": success, "completed_target_date": attempt.target_date if success else None,
            "progress": not success and receipt.get("progress_count", 0) > 0,
            "equity_status": receipt.get("status"),
            "remaining_sessions": receipt.get("market_progress", {}).get("remaining_sessions", 0),
            "references": receipt.get("reference_status", {}),
            "model_year": receipt.get("model", {}).get("model_year")}


class ContinuousRefresh:
    """One background data job; the main worker stays responsive to pause/orders.

    The caller must own worker.lock. A crash loses no committed partitions and
    leaves a durable attempt deadline. Late or mismatched completion is rejected
    by ScheduleStore. The data thread never submits any simulated trade.
    """
    def __init__(self, root, *, pool=None, worker=perform_refresh, calendar_reader=read_calendar):
        self.root = Path(root)
        self.store = ScheduleStore(_base(root) / "refresh_schedule.json")
        self.pool = pool or ThreadPoolExecutor(max_workers=1, thread_name_prefix="qingxu-data")
        self.worker, self.calendar_reader = worker, calendar_reader
        self.future = self.attempt = None
        self.last = {}
        path = _base(root) / "refresh_status.json"
        if path.exists():
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
                last = saved.get("last_result", {}) if isinstance(saved, dict) else {}
                self.last = last if isinstance(last, dict) else {}
            except (OSError, ValueError, TypeError):
                # Display-only status is reconstructible. The authoritative
                # refresh_schedule.json is still strictly checked by the store.
                self.last = {}

    def tick(self, *, now=None, start_jobs=True):
        from .simulation_service import equity_maintenance_target, write_json
        now = now or datetime.now(timezone.utc)
        if self.future is not None and self.future.done():
            try:
                result = self.future.result()
            except Exception as exc:
                # No provider messages or secrets enter status files.
                result = {"success": False, "progress": False, "error_type": type(exc).__name__}
            self.store.finish(self.attempt, success=result["success"],
                              progress=result.get("progress", False), now=now,
                              completed_target_date=result.get("completed_target_date"))
            self.last = _json_dates(result)
            self.future = self.attempt = None
        try:
            calendar = self.calendar_reader(self.root)
        except (OSError, ValueError, KeyError, TypeError):
            calendar = None
        decision = decide_refresh(calendar, self.store.load(), now=now)
        if self.future is None and decision.data_reason == "already_complete":
            execution = equity_maintenance_target(self.root, calendar, decision.target_date)
            if execution is not None and self.store.invalidate_data(
                    decision.target_date, expected_execution_date=execution, now=now):
                decision = decide_refresh(calendar, self.store.load(), now=now)
        if self.future is None and start_jobs:
            kind = "calendar" if decision.calendar_due else "data" if decision.data_due else None
            if kind is not None:
                target = decision.calendar_required_through if kind == "calendar" else decision.target_date
                self.attempt = self.store.begin(kind, target, now=now)
                self.future = self.pool.submit(self.worker, self.root, self.attempt, now=now)
        status = {
            "asof": now.isoformat(), "running": self.future is not None,
            "job": self.attempt.kind if self.attempt else None,
            "target_date": str(self.attempt.target_date) if self.attempt else None,
            "decision": _json_dates(asdict(decision)), "last_result": self.last,
        }
        write_json(_base(self.root) / "refresh_status.json", status)
        return status

    def wait_for_current(self):
        """One-shot CLI support; the continuous worker never calls this."""
        if self.future is not None:
            try:
                self.future.result()
            except Exception:
                pass  # tick records the sanitized failure and its durable backoff.

    def close(self):
        self.pool.shutdown(wait=False)
