"""Durable catch-up scheduling, without fetching data or placing orders.

The caller already holds the simulation worker's process lock.  ``begin`` must
be committed before contacting a supplier; a crash therefore retains a retry
deadline.  ``finish(success=True)`` is permitted only after the caller verifies
all required data, reference inputs and scores for the exact target session.

Calendar maintenance targets today plus fourteen calendar days and a verified
refresh within seven days. Data preparation may continue from an older or
shorter calendar if it still proves today's coverage and the next actual open
session. Weekdays are never used to guess exchange sessions.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import json
import os
from pathlib import Path
import re
from typing import Literal
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd


SHANGHAI = ZoneInfo("Asia/Shanghai")
PUBLICATION_TIME = time(17, 5)
CALENDAR_LOOKAHEAD_DAYS = 14
CALENDAR_MAX_AGE = timedelta(days=7)
RETRY_DELAYS = tuple(timedelta(seconds=value) for value in (60, 300, 900, 3600, 10800, 21600))
SCHEMA_VERSION = 1
Kind = Literal["data", "calendar"]
_ATTEMPT_ID = re.compile(r"[0-9a-f]{32}\Z")


class ScheduleStateError(ValueError):
    """A fixed-message local state error; never reset retry history silently."""


@dataclass(frozen=True)
class RefreshDecision:
    target_date: date | None
    data_due: bool
    calendar_due: bool
    reason_code: str
    next_attempt_at: datetime | None
    calendar_required_through: date
    data_reason: str
    calendar_reason: str
    data_next_attempt_at: datetime | None = None
    calendar_next_attempt_at: datetime | None = None


@dataclass(frozen=True)
class RefreshAttempt:
    kind: Kind
    target_date: date
    attempt_id: str
    number: int
    started_at: datetime
    next_attempt_at: datetime


def _aware(value: object) -> bool:
    return isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None


def _now(value: datetime) -> datetime:
    if not _aware(value):
        raise ValueError("schedule_timezone_required")
    return value.astimezone(timezone.utc)


def _stamp(value: datetime) -> str:
    return _now(value).isoformat()


def _instant(value: object) -> datetime:
    try:
        instant = datetime.fromisoformat(value) if isinstance(value, str) else None
    except (ValueError, TypeError):
        instant = None
    if not _aware(instant):
        raise ScheduleStateError("schedule_state_invalid_timestamp")
    return instant.astimezone(timezone.utc)


def _day(value: object) -> date:
    try:
        day = date.fromisoformat(value) if isinstance(value, str) else None
    except (ValueError, TypeError):
        day = None
    if day is None or day.isoformat() != value:
        raise ScheduleStateError("schedule_state_invalid_date")
    return day


def new_schedule_state() -> dict:
    return {
        "schema_version": SCHEMA_VERSION, "updated_at": None,
        "last_calendar_success": None, "last_calendar_coverage": None,
        "data": {}, "calendar": {},
    }


def validate_schedule_state(state: dict) -> None:
    """Validate strict persisted shape; corruption must not erase backoff."""
    if not isinstance(state, dict) or set(state) != set(new_schedule_state()):
        raise ScheduleStateError("schedule_state_invalid_schema")
    if type(state["schema_version"]) is not int or state["schema_version"] != SCHEMA_VERSION:
        raise ScheduleStateError("schedule_state_unsupported_version")
    updated = _instant(state["updated_at"]) if state["updated_at"] is not None else None
    success = _instant(state["last_calendar_success"]) if state["last_calendar_success"] is not None else None
    coverage = _day(state["last_calendar_coverage"]) if state["last_calendar_coverage"] is not None else None
    if (success is None) != (coverage is None) or (success and (updated is None or success > updated)):
        raise ScheduleStateError("schedule_state_invalid_calendar_success")
    for kind in ("data", "calendar"):
        entries = state[kind]
        if not isinstance(entries, dict):
            raise ScheduleStateError("schedule_state_invalid_entries")
        for target, entry in entries.items():
            _day(target)
            expected = {"attempt_id", "attempts", "status", "started_at", "next_attempt_at", "completed_at"}
            if not isinstance(entry, dict) or set(entry) not in (expected, expected | {"maintenance"}):
                raise ScheduleStateError("schedule_state_invalid_attempt")
            if (type(entry["attempts"]) is not int or entry["attempts"] < 1
                    or not isinstance(entry["attempt_id"], str)
                    or not _ATTEMPT_ID.fullmatch(entry["attempt_id"])
                    or entry["status"] not in ("inflight", "failed", "partial", "succeeded", "invalidated")):
                raise ScheduleStateError("schedule_state_invalid_attempt")
            started = _instant(entry["started_at"])
            next_at = _instant(entry["next_attempt_at"]) if entry["next_attempt_at"] is not None else None
            completed = _instant(entry["completed_at"]) if entry["completed_at"] is not None else None
            if updated is None or started > updated or (completed and (completed < started or completed > updated)):
                raise ScheduleStateError("schedule_state_invalid_chronology")
            if entry["status"] == "inflight" and (completed is not None or next_at is None):
                raise ScheduleStateError("schedule_state_invalid_inflight")
            if entry["status"] != "inflight" and completed is None:
                raise ScheduleStateError("schedule_state_missing_completion")
            if entry["status"] in ("failed", "partial", "invalidated") and next_at is None:
                raise ScheduleStateError("schedule_state_missing_retry")
            if next_at is not None and next_at <= started:
                raise ScheduleStateError("schedule_state_invalid_retry")
            if entry["status"] == "failed" and next_at < completed + _retry_delay(entry["attempts"]):
                raise ScheduleStateError("schedule_state_shortened_retry")
            if entry["status"] in ("partial", "invalidated") and next_at < completed + RETRY_DELAYS[0]:
                raise ScheduleStateError("schedule_state_shortened_retry")
            if entry["status"] == "invalidated" and (kind != "data" or "maintenance" not in entry):
                raise ScheduleStateError("schedule_state_invalid_maintenance")
            if "maintenance" in entry:
                maintenance = entry["maintenance"]
                if (kind != "data" or not isinstance(maintenance, dict) or set(maintenance) != {
                        "required_execution_date", "invalidated_at", "previous_completed_at"}
                        or _day(maintenance["required_execution_date"]) <= _day(target)
                        or not (_instant(maintenance["previous_completed_at"])
                                <= _instant(maintenance["invalidated_at"]) <= updated)):
                    raise ScheduleStateError("schedule_state_invalid_maintenance")
            if kind == "data" and entry["status"] == "succeeded" and next_at is not None:
                raise ScheduleStateError("schedule_state_invalid_data_success")
            if kind == "calendar" and entry["status"] == "succeeded" and (
                    success is None or completed > success or coverage < _day(target) or next_at is None):
                raise ScheduleStateError("schedule_state_invalid_calendar_success")


def load_schedule_state(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return new_schedule_state()
    try:
        if path.stat().st_size > 16_000_000:
            raise ScheduleStateError("schedule_state_oversize")
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        if isinstance(exc, ScheduleStateError):
            raise
        raise ScheduleStateError("schedule_state_unreadable") from None
    validate_schedule_state(state)
    return state


def save_schedule_state(path: str | Path, state: dict) -> None:
    """Atomic same-directory replacement; retain a failed temporary write."""
    validate_schedule_state(state)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + "." + uuid4().hex + ".pending")
    payload = (json.dumps(state, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()
    with pending.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def _retry_delay(attempts: int) -> timedelta:
    return RETRY_DELAYS[min(attempts - 1, len(RETRY_DELAYS) - 1)]


def _retry_gate(state: dict, kind: Kind, target: date, now: datetime) -> tuple[bool, str, datetime | None]:
    entry = state[kind].get(target.isoformat())
    if entry is None:
        return True, "not_attempted", now
    if entry["status"] == "succeeded" and kind == "data":
        return False, "already_complete", None
    deadline = _instant(entry["next_attempt_at"]) if entry["next_attempt_at"] else None
    if deadline is not None and now < deadline:
        return False, "retry_backoff" if entry["status"] != "succeeded" else "already_checked", deadline
    return True, "retry_due", now


def _calendar(calendar: pd.DataFrame | None) -> tuple[list[date], date | None, date | None, str]:
    if calendar is None or not isinstance(calendar, pd.DataFrame) or calendar.empty:
        return [], None, None, "calendar_missing"
    if not {"cal_date", "is_open"} <= set(calendar):
        return [], None, None, "calendar_invalid"
    try:
        dates = pd.to_datetime(calendar["cal_date"].astype(str), format="mixed", errors="raise")
        flags = pd.to_numeric(calendar["is_open"], errors="raise")
        if (dates.isna().any() or dates.dt.tz is not None
                or not dates.eq(dates.dt.normalize()).all() or dates.duplicated().any()
                or not flags.isin((0, 1)).all()):
            return [], None, None, "calendar_invalid"
        first, last = dates.min().date(), dates.max().date()
        if len(dates) != (last - first).days + 1:
            return [], first, last, "calendar_has_gaps"
        opens = sorted(dates[flags.eq(1)].dt.date.tolist())
        return opens, first, last, ""
    except (ValueError, TypeError, AttributeError, OverflowError):
        return [], None, None, "calendar_invalid"


def decide_refresh(calendar: pd.DataFrame | None, state: dict, *, now: datetime) -> RefreshDecision:
    """Determine catch-up work from the Shanghai clock and official sessions.

    Prefer refreshing a due calendar first, then recompute the decision. If a
    future calendar is not published yet, data preparation can still proceed
    during calendar backoff when existing rows prove the target and next open
    session. Both due flags can therefore be true independently.
    """
    now = _now(now)
    validate_schedule_state(state)
    local = now.astimezone(SHANGHAI)
    today = local.date()
    required = today + timedelta(days=CALENDAR_LOOKAHEAD_DAYS)
    cutoff = today if local.time().replace(tzinfo=None) >= PUBLICATION_TIME else today - timedelta(days=1)
    opens, first, last, calendar_problem = _calendar(calendar)
    candidates = [day for day in opens if day <= cutoff]
    target = candidates[-1] if candidates else None
    next_opens = [day for day in opens if target is not None and day > target]
    if state["updated_at"] is not None and _instant(state["updated_at"]) > now:
        return RefreshDecision(target, False, False, "clock_moved_backwards",
                               _instant(state["updated_at"]), required,
                               "clock_moved_backwards", "clock_moved_backwards")
    data_calendar_problem = calendar_problem
    if not data_calendar_problem:
        if first > cutoff or target is None:
            data_calendar_problem = "calendar_history_insufficient"
        elif last < today:
            data_calendar_problem = "calendar_current_coverage_short"
        elif not next_opens:
            data_calendar_problem = "calendar_next_session_unknown"
    if not calendar_problem:
        if first > cutoff or target is None:
            calendar_problem = "calendar_history_insufficient"
        elif last < today:
            calendar_problem = "calendar_current_coverage_short"
        elif not next_opens:
            calendar_problem = "calendar_next_session_unknown"
        elif last < required:
            calendar_problem = "calendar_future_coverage_short"
        elif state["last_calendar_success"] is None:
            calendar_problem = "calendar_freshness_unverified"
        elif now - _instant(state["last_calendar_success"]) >= CALENDAR_MAX_AGE:
            calendar_problem = "calendar_refresh_stale"
        elif _day(state["last_calendar_coverage"]) < required:
            calendar_problem = "calendar_verified_coverage_short"
    if calendar_problem:
        calendar_due, _, calendar_refresh_at = _retry_gate(state, "calendar", required, now)
    else:
        calendar_due = False
        calendar_refresh_at = _instant(state["last_calendar_success"]) + CALENDAR_MAX_AGE
    if data_calendar_problem:
        data_due, data_reason, retry_at = False, data_calendar_problem, None
    else:
        data_due, data_reason, retry_at = _retry_gate(state, "data", target, now)
    # Find the next known release instant, even when today is a holiday.
    future_releases = [
        datetime.combine(day, PUBLICATION_TIME, SHANGHAI).astimezone(timezone.utc)
        for day in next_opens
    ]
    next_release = next((instant for instant in future_releases if instant > now), None)
    wakeups = [instant for instant in (retry_at, calendar_refresh_at,
                                      next_release if not data_due else None) if instant is not None]
    return RefreshDecision(
        target, data_due, calendar_due,
        "calendar_update_required" if calendar_due or data_calendar_problem else
        ("data_refresh_due" if data_due else
         ("data_retry_backoff" if data_reason == "retry_backoff" else "up_to_date")),
        min(wakeups) if wakeups else None, required, data_reason, calendar_problem or "calendar_ready",
        data_next_attempt_at=retry_at, calendar_next_attempt_at=calendar_refresh_at,
    )


class ScheduleStore:
    """Single-writer durable JSON store; use the worker's existing process lock.

    Persisted state is never silently repaired or reset.  ``begin`` is a claim
    with a fresh attempt ID; duplicate, early and delayed completion calls fail
    closed.  No method sleeps, performs network I/O or executes a trade.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict:
        return load_schedule_state(self.path)

    def invalidate_data(self, target_date: date, *, expected_execution_date: date,
                        now: datetime) -> bool:
        """Reopen a success whose verified execution session/model year changed.

        Only a completed success is eligible. Repeated checks cannot shorten a
        failure cooldown or reset an in-progress retry. The invalidation and
        prior completion time survive process restarts.
        """
        now = _now(now)
        if (type(target_date) is not date or type(expected_execution_date) is not date
                or expected_execution_date <= target_date):
            raise ValueError("schedule_invalid_maintenance_target")
        state = self.load()
        if state["updated_at"] is not None and _instant(state["updated_at"]) > now:
            raise ScheduleStateError("schedule_clock_moved_backwards")
        entry = state["data"].get(target_date.isoformat())
        if entry is None or entry["status"] != "succeeded":
            return False
        entry["maintenance"] = {
            "required_execution_date": expected_execution_date.isoformat(),
            "invalidated_at": _stamp(now), "previous_completed_at": entry["completed_at"],
        }
        entry.update(status="invalidated", completed_at=_stamp(now),
                     next_attempt_at=_stamp(now + RETRY_DELAYS[0]))
        state["updated_at"] = _stamp(now)
        save_schedule_state(self.path, state)
        return True

    def begin(self, kind: Kind, target_date: date, *, now: datetime) -> RefreshAttempt:
        now = _now(now)
        if kind not in ("data", "calendar") or type(target_date) is not date:
            raise ValueError("schedule_invalid_target")
        state = self.load()
        if state["updated_at"] is not None and _instant(state["updated_at"]) > now:
            raise ScheduleStateError("schedule_clock_moved_backwards")
        permitted, _, _ = _retry_gate(state, kind, target_date, now)
        if not permitted:
            raise ScheduleStateError("schedule_attempt_not_due")
        old = state[kind].get(target_date.isoformat())
        number = 1 if old is None or old["status"] in ("succeeded", "partial", "invalidated") else old["attempts"] + 1
        deadline = now + _retry_delay(number)
        attempt = RefreshAttempt(kind, target_date, uuid4().hex, number, now, deadline)
        state[kind][target_date.isoformat()] = {
            "attempt_id": attempt.attempt_id, "attempts": number, "status": "inflight",
            "started_at": _stamp(now), "next_attempt_at": _stamp(deadline), "completed_at": None,
        }
        if old is not None and "maintenance" in old:
            state[kind][target_date.isoformat()]["maintenance"] = old["maintenance"]
        state["updated_at"] = _stamp(now)
        save_schedule_state(self.path, state)
        return attempt

    def finish(self, attempt: RefreshAttempt, *, success: bool, now: datetime,
               completed_target_date: date | None = None, progress: bool = False) -> dict:
        """Complete only the currently recorded attempt.

        Successful data completion must name exactly the requested session.
        Successful calendar completion must name its verified coverage end,
        which must include the requested fourteen-day horizon.
        ``progress=True`` requires independently verified newly saved inputs or
        completed reference tasks, not just another request. An unfinished batch
        with actual progress waits one minute and resets the next failure count.
        """
        now = _now(now)
        if (not isinstance(attempt, RefreshAttempt) or type(success) is not bool
                or type(progress) is not bool or (progress and success)):
            raise ValueError("schedule_invalid_completion")
        if success:
            if type(completed_target_date) is not date:
                raise ValueError("schedule_completion_date_required")
            if (attempt.kind == "data" and completed_target_date != attempt.target_date
                    or attempt.kind == "calendar" and completed_target_date < attempt.target_date):
                raise ValueError("schedule_completion_target_mismatch")
        state = self.load()
        if state["updated_at"] is not None and _instant(state["updated_at"]) > now:
            raise ScheduleStateError("schedule_clock_moved_backwards")
        entry = state.get(attempt.kind, {}).get(attempt.target_date.isoformat())
        if (entry is None or entry["attempt_id"] != attempt.attempt_id
                or entry["status"] != "inflight"
                or entry["attempts"] != attempt.number
                or _instant(entry["started_at"]) != attempt.started_at
                or _instant(entry["next_attempt_at"]) != attempt.next_attempt_at):
            raise ScheduleStateError("schedule_attempt_no_longer_current")
        state = deepcopy(state)
        entry = state[attempt.kind][attempt.target_date.isoformat()]
        entry["status"] = "succeeded" if success else ("partial" if progress else "failed")
        entry["completed_at"] = _stamp(now)
        if success:
            # Healthy calendars are protected by the seven-day freshness gate.
            # A short local cooldown also allows recovery if a once-valid
            # calendar file is later missing/corrupt, without hammering every run.
            entry["next_attempt_at"] = None if attempt.kind == "data" else _stamp(now + RETRY_DELAYS[0])
            if attempt.kind == "calendar":
                state["last_calendar_success"] = _stamp(now)
                state["last_calendar_coverage"] = completed_target_date.isoformat()
        elif progress:
            entry["next_attempt_at"] = _stamp(now + RETRY_DELAYS[0])
        else:
            entry["next_attempt_at"] = _stamp(max(attempt.next_attempt_at, now + _retry_delay(attempt.number)))
        state["updated_at"] = _stamp(now)
        save_schedule_state(self.path, state)
        return state
