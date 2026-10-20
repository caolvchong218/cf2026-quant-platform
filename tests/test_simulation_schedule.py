"""Clock-driven scheduling with explicit synthetic exchange-calendar fixtures."""
from dataclasses import replace
from datetime import date, datetime, timedelta
import json

import pandas as pd
import pytest

from cfquant import simulation_schedule as schedule
from cfquant.simulation_schedule import (
    CALENDAR_MAX_AGE, RETRY_DELAYS, SHANGHAI, ScheduleStateError, ScheduleStore,
    decide_refresh, load_schedule_state, new_schedule_state, save_schedule_state,
)


NOW = datetime(2026, 10, 20, 17, 5, tzinfo=SHANGHAI)


def calendar(start="2026-09-01", end="2027-02-01", *, closed=(), opened=()):
    """Test fixture only; production scheduling never guesses weekday sessions."""
    days = pd.date_range(start, end)
    closed, opened = set(closed), set(opened)
    return pd.DataFrame({
        "cal_date": [int(day.strftime("%Y%m%d")) for day in days],
        "is_open": [int((day.weekday() < 5 and day.date() not in closed) or day.date() in opened) for day in days],
    })


def seed_calendar(tmp_path, now=NOW, *, coverage=date(2027, 2, 1)):
    store = ScheduleStore(tmp_path / "schedule.json")
    attempt = store.begin("calendar", now.astimezone(SHANGHAI).date() + timedelta(days=14), now=now)
    state = store.finish(attempt, success=True, completed_target_date=coverage, now=now)
    return store, state


def complete_data(store, target, now):
    attempt = store.begin("data", target, now=now)
    return store.finish(attempt, success=True, completed_target_date=target, now=now)


def test_execution_binding_invalidation_survives_restart_without_resetting_backoff(tmp_path):
    store, _ = seed_calendar(tmp_path)
    complete_data(store, NOW.date(), NOW)
    invalidated_at = NOW + timedelta(seconds=1)
    assert store.invalidate_data(NOW.date(), expected_execution_date=date(2026, 10, 22), now=invalidated_at)
    first = store.load()["data"][str(NOW.date())]
    assert first["status"] == "invalidated"
    assert first["maintenance"]["previous_completed_at"] == NOW.astimezone(schedule.timezone.utc).isoformat()
    restarted = ScheduleStore(store.path)
    assert not restarted.invalidate_data(
        NOW.date(), expected_execution_date=date(2026, 10, 23), now=NOW + timedelta(seconds=30))
    assert restarted.load()["data"][str(NOW.date())] == first
    with pytest.raises(ScheduleStateError, match="not_due"):
        restarted.begin("data", NOW.date(), now=NOW + timedelta(seconds=60))
    retry = restarted.begin("data", NOW.date(), now=NOW + timedelta(seconds=61))
    assert retry.number == 1
    restarted.finish(retry, success=False, now=NOW + timedelta(seconds=62))
    assert not restarted.invalidate_data(
        NOW.date(), expected_execution_date=date(2026, 10, 22), now=NOW + timedelta(seconds=70))
    retry = restarted.begin("data", NOW.date(), now=NOW + timedelta(seconds=122))
    assert retry.number == 2
    restarted.finish(retry, success=False, now=NOW + timedelta(seconds=123))
    assert not restarted.invalidate_data(
        NOW.date(), expected_execution_date=date(2026, 10, 22), now=NOW + timedelta(seconds=200))
    assert not decide_refresh(calendar(), restarted.load(), now=NOW + timedelta(seconds=422)).data_due
    assert decide_refresh(calendar(), restarted.load(), now=NOW + timedelta(seconds=423)).data_due


def test_invalidation_requires_future_execution_and_cannot_rewrite_inflight_or_missing_target(tmp_path):
    store, _ = seed_calendar(tmp_path)
    assert not store.invalidate_data(NOW.date(), expected_execution_date=date(2026, 10, 21), now=NOW)
    attempt = store.begin("data", NOW.date(), now=NOW)
    assert not store.invalidate_data(NOW.date(), expected_execution_date=date(2026, 10, 21), now=NOW)
    assert store.load()["data"][str(NOW.date())]["attempt_id"] == attempt.attempt_id
    with pytest.raises(ValueError, match="invalid_maintenance_target"):
        store.invalidate_data(NOW.date(), expected_execution_date=NOW.date(), now=NOW)


@pytest.mark.parametrize("hour,minute,second,target", [
    (17, 4, 59, date(2026, 10, 19)),
    (17, 5, 0, date(2026, 10, 20)),
    (23, 59, 59, date(2026, 10, 20)),
])
def test_publication_cutoff_inclusive_and_late_catchup(tmp_path, hour, minute, second, target):
    now = NOW.replace(hour=hour, minute=minute, second=second)
    _, state = seed_calendar(tmp_path, now=now)
    decision = decide_refresh(calendar(), state, now=now)
    assert decision.target_date == target
    assert decision.data_due and not decision.calendar_due


def test_utc_clock_maps_to_shanghai_release_boundary(tmp_path):
    from datetime import timezone
    _, state = seed_calendar(tmp_path)
    decision = decide_refresh(calendar(), state, now=NOW.astimezone(timezone.utc))
    assert decision.target_date == date(2026, 10, 20) and decision.data_due


def test_missed_window_catches_up_on_next_morning_and_keeps_success_across_restart(tmp_path):
    morning = datetime(2026, 10, 21, 8, tzinfo=SHANGHAI)
    store, state = seed_calendar(tmp_path, now=morning)
    decision = decide_refresh(calendar(), state, now=morning)
    assert decision.data_due and decision.target_date == date(2026, 10, 20)
    complete_data(store, decision.target_date, morning)
    restarted = ScheduleStore(store.path)
    later = decide_refresh(calendar(), restarted.load(), now=morning + timedelta(minutes=1))
    assert not later.data_due and later.reason_code == "up_to_date"
    assert later.next_attempt_at == datetime(2026, 10, 21, 17, 5, tzinfo=SHANGHAI)


def test_weekend_and_explicit_weekday_holiday_use_actual_calendar(tmp_path):
    holiday_monday = date(2026, 10, 26)
    frame = calendar(closed=(holiday_monday,))
    sunday = datetime(2026, 10, 25, 18, tzinfo=SHANGHAI)
    store, state = seed_calendar(tmp_path, now=sunday)
    decision = decide_refresh(frame, state, now=sunday)
    assert decision.target_date == date(2026, 10, 23)
    state = complete_data(store, decision.target_date, sunday)
    monday = sunday + timedelta(days=1)
    decision = decide_refresh(frame, state, now=monday)
    assert decision.target_date == date(2026, 10, 23) and not decision.data_due
    assert decision.next_attempt_at == datetime(2026, 10, 27, 17, 5, tzinfo=SHANGHAI)


def test_calendar_can_declare_a_weekend_session_without_weekday_guess(tmp_path):
    saturday = datetime(2026, 10, 24, 18, tzinfo=SHANGHAI)
    frame = calendar(opened=(saturday.date(),))
    _, state = seed_calendar(tmp_path, now=saturday)
    assert decide_refresh(frame, state, now=saturday).target_date == saturday.date()


def test_new_year_holiday_and_midnight_do_not_invent_january_session(tmp_path):
    now = datetime(2027, 1, 1, 0, 1, tzinfo=SHANGHAI)
    frame = calendar(closed=(date(2027, 1, 1),))
    store, state = seed_calendar(tmp_path, now=now)
    decision = decide_refresh(frame, state, now=now)
    assert decision.target_date == date(2026, 12, 31)
    state = complete_data(store, decision.target_date, now)
    holiday_evening = now.replace(hour=18)
    decision = decide_refresh(frame, state, now=holiday_evening)
    assert not decision.data_due and decision.target_date == date(2026, 12, 31)


def test_calendar_missing_requires_explicit_refresh_before_data(tmp_path):
    decision = decide_refresh(None, new_schedule_state(), now=NOW)
    assert decision.target_date is None
    assert decision.calendar_due and not decision.data_due
    assert decision.calendar_required_through == date(2026, 11, 3)
    assert decision.calendar_reason == "calendar_missing"


@pytest.mark.parametrize("end,data_due,reason", [
    ("2026-10-19", False, "calendar_current_coverage_short"),
    ("2026-10-20", False, "calendar_next_session_unknown"),
    ("2026-11-02", True, "calendar_future_coverage_short"),
])
def test_calendar_requires_maintenance_but_known_next_session_can_proceed(end, data_due, reason):
    decision = decide_refresh(calendar(end=end), new_schedule_state(), now=NOW)
    assert decision.calendar_due and decision.data_due is data_due
    assert decision.calendar_reason == reason


def test_old_calendar_freshness_is_not_inferred_from_a_far_future_end(tmp_path):
    early = NOW - CALENDAR_MAX_AGE
    store, state = seed_calendar(tmp_path, now=early)
    almost = decide_refresh(calendar(), state, now=NOW - timedelta(microseconds=1))
    assert not almost.calendar_due
    expired = decide_refresh(calendar(), state, now=NOW)
    assert expired.calendar_due and expired.calendar_reason == "calendar_refresh_stale"
    attempt = store.begin("calendar", expired.calendar_required_through, now=NOW)
    store.finish(attempt, success=False, now=NOW)
    blocked = decide_refresh(calendar(), ScheduleStore(store.path).load(), now=NOW + timedelta(seconds=30))
    assert not blocked.calendar_due and blocked.calendar_next_attempt_at == NOW + timedelta(minutes=1)
    assert blocked.data_due


def test_unknown_calendar_refresh_time_is_explicitly_unverified():
    decision = decide_refresh(calendar(), new_schedule_state(), now=NOW)
    assert decision.calendar_due and decision.calendar_reason == "calendar_freshness_unverified"


@pytest.mark.parametrize("variant", ["gap", "duplicate", "bad_flag", "bad_date", "timezone"])
def test_invalid_official_calendar_is_not_repaired_by_guessing(variant):
    frame = calendar()
    if variant == "gap":
        frame = frame.drop(index=5)
    elif variant == "duplicate":
        frame = pd.concat([frame, frame.iloc[[5]]], ignore_index=True)
    elif variant == "bad_flag":
        frame.loc[5, "is_open"] = 2
    elif variant == "bad_date":
        frame["cal_date"] = frame.cal_date.astype(str)
        frame.loc[5, "cal_date"] = "bad-date"
    else:
        frame["cal_date"] = pd.to_datetime(frame.cal_date.astype(str)).dt.tz_localize(SHANGHAI)
    result = decide_refresh(frame, new_schedule_state(), now=NOW)
    assert result.calendar_due and not result.data_due
    assert result.calendar_reason in ("calendar_invalid", "calendar_has_gaps")


def test_calendar_success_must_prove_required_horizon(tmp_path):
    store = ScheduleStore(tmp_path / "state.json")
    attempt = store.begin("calendar", date(2026, 11, 3), now=NOW)
    with pytest.raises(ValueError, match="target_mismatch"):
        store.finish(attempt, success=True, completed_target_date=date(2026, 11, 2), now=NOW)
    assert store.load()["calendar"]["2026-11-03"]["status"] == "inflight"
    store.finish(attempt, success=False, now=NOW)
    decision = decide_refresh(calendar(end="2026-11-02"), store.load(), now=NOW)
    assert not decision.calendar_due and decision.data_due


def test_unpublished_next_year_does_not_block_known_december_sessions_during_calendar_backoff(tmp_path):
    now = datetime(2026, 12, 20, 18, tzinfo=SHANGHAI)
    known = calendar(start="2026-12-01", end="2026-12-31")
    store = ScheduleStore(tmp_path / "state.json")
    decision = decide_refresh(known, store.load(), now=now)
    assert decision.calendar_due and decision.data_due
    assert decision.target_date == date(2026, 12, 18)
    assert decision.calendar_required_through == date(2027, 1, 3)
    attempt = store.begin("calendar", decision.calendar_required_through, now=now)
    store.finish(attempt, success=False, now=now)
    during_backoff = decide_refresh(known, store.load(), now=now + timedelta(seconds=30))
    assert not during_backoff.calendar_due and during_backoff.data_due
    complete_data(store, during_backoff.target_date, now + timedelta(seconds=30))
    prepared = decide_refresh(known, store.load(), now=now + timedelta(seconds=31))
    assert not prepared.data_due
    monday = datetime(2026, 12, 21, 18, tzinfo=SHANGHAI)
    assert decide_refresh(known, store.load(), now=monday).target_date == date(2026, 12, 21)


def test_year_end_without_any_known_next_open_session_remains_blocked():
    now = datetime(2026, 12, 31, 18, tzinfo=SHANGHAI)
    known = calendar(start="2026-12-01", end="2026-12-31")
    decision = decide_refresh(known, new_schedule_state(), now=now)
    assert decision.calendar_due and not decision.data_due
    assert decision.data_reason == "calendar_next_session_unknown"


def test_begin_persists_crash_backoff_and_increments_after_restart(tmp_path):
    store, state = seed_calendar(tmp_path)
    attempt = store.begin("data", NOW.date(), now=NOW)
    assert attempt.number == 1
    restarted = ScheduleStore(store.path)
    waiting = decide_refresh(calendar(), restarted.load(), now=NOW + timedelta(seconds=59))
    assert waiting.reason_code == "data_retry_backoff" and not waiting.data_due
    retry_at = NOW + timedelta(minutes=1)
    assert decide_refresh(calendar(), restarted.load(), now=retry_at).data_due
    second = restarted.begin("data", NOW.date(), now=retry_at)
    assert second.number == 2 and second.next_attempt_at == retry_at + timedelta(minutes=5)
    with pytest.raises(ScheduleStateError, match="no_longer_current"):
        restarted.finish(attempt, success=True, completed_target_date=NOW.date(), now=retry_at)


def test_failure_backoff_escalates_to_six_hours_without_infinite_minute_polling(tmp_path):
    store, _ = seed_calendar(tmp_path)
    target, instant = NOW.date(), NOW
    for index, delay in enumerate((*RETRY_DELAYS, RETRY_DELAYS[-1], RETRY_DELAYS[-1]), start=1):
        attempt = store.begin("data", target, now=instant)
        assert attempt.number == index
        finished = instant + timedelta(seconds=10)
        state = store.finish(attempt, success=False, now=finished)
        deadline = datetime.fromisoformat(state["data"][target.isoformat()]["next_attempt_at"])
        assert deadline == finished + delay
        with pytest.raises(ScheduleStateError, match="not_due"):
            ScheduleStore(store.path).begin("data", target, now=deadline - timedelta(microseconds=1))
        instant = deadline


def test_partial_progress_survives_restart_and_resets_only_the_next_failure_count(tmp_path):
    store, _ = seed_calendar(tmp_path)
    first = store.begin("data", NOW.date(), now=NOW)
    store.finish(first, success=False, now=NOW)
    second = store.begin("data", NOW.date(), now=NOW + timedelta(minutes=1))
    assert second.number == 2
    state = store.finish(second, success=False, progress=True, now=NOW + timedelta(minutes=1))
    assert state["data"][NOW.date().isoformat()]["status"] == "partial"
    restarted = ScheduleStore(store.path)
    with pytest.raises(ScheduleStateError, match="not_due"):
        restarted.begin("data", NOW.date(), now=NOW + timedelta(seconds=119))
    third = restarted.begin("data", NOW.date(), now=NOW + timedelta(minutes=2))
    assert third.number == 1
    restarted.finish(third, success=False, now=NOW + timedelta(minutes=2))
    fourth = restarted.begin("data", NOW.date(), now=NOW + timedelta(minutes=3))
    assert fourth.number == 2
    failed = restarted.finish(fourth, success=False, now=NOW + timedelta(minutes=3))
    assert datetime.fromisoformat(failed["data"][NOW.date().isoformat()]["next_attempt_at"]) == NOW + timedelta(minutes=8)


def test_progress_cannot_relabel_success_or_an_already_completed_attempt(tmp_path):
    store, _ = seed_calendar(tmp_path)
    attempt = store.begin("data", NOW.date(), now=NOW)
    with pytest.raises(ValueError, match="invalid_completion"):
        store.finish(attempt, success=True, progress=True, completed_target_date=NOW.date(), now=NOW)
    store.finish(attempt, success=True, completed_target_date=NOW.date(), now=NOW)
    with pytest.raises(ScheduleStateError, match="no_longer_current"):
        store.finish(attempt, success=False, progress=True, now=NOW)


def test_new_data_target_is_not_blocked_by_old_target_failure(tmp_path):
    store, _ = seed_calendar(tmp_path)
    old = store.begin("data", date(2026, 10, 19), now=NOW)
    store.finish(old, success=False, now=NOW)
    result = decide_refresh(calendar(), store.load(), now=NOW)
    assert result.target_date == date(2026, 10, 20) and result.data_due
    new = store.begin("data", result.target_date, now=NOW)
    assert new.number == 1


def test_new_calendar_horizon_is_not_blocked_by_yesterdays_failure(tmp_path):
    previous = NOW - timedelta(days=1)
    store = ScheduleStore(tmp_path / "state.json")
    old_horizon = previous.date() + timedelta(days=14)
    attempt = store.begin("calendar", old_horizon, now=previous)
    store.finish(attempt, success=False, now=previous)
    result = decide_refresh(None, store.load(), now=NOW)
    assert result.calendar_due
    current = store.begin("calendar", result.calendar_required_through, now=NOW)
    assert current.target_date != old_horizon and current.number == 1


def test_exact_target_required_before_marking_data_success(tmp_path):
    store, _ = seed_calendar(tmp_path)
    attempt = store.begin("data", NOW.date(), now=NOW)
    for mismatched in (None, date(2026, 10, 19), date(2026, 10, 21)):
        with pytest.raises(ValueError, match="completion"):
            store.finish(attempt, success=True, completed_target_date=mismatched, now=NOW)
        assert store.load()["data"][NOW.date().isoformat()]["status"] == "inflight"
    state = store.finish(attempt, success=True, completed_target_date=NOW.date(), now=NOW)
    assert not decide_refresh(calendar(), state, now=NOW).data_due
    with pytest.raises(ScheduleStateError, match="not_due"):
        store.begin("data", NOW.date(), now=NOW + timedelta(days=1))


def test_completed_attempt_cannot_be_completed_again_or_forged(tmp_path):
    store, _ = seed_calendar(tmp_path)
    attempt = store.begin("data", NOW.date(), now=NOW)
    with pytest.raises(ScheduleStateError, match="no_longer_current"):
        store.finish(replace(attempt, number=999), success=False, now=NOW)
    store.finish(attempt, success=True, completed_target_date=NOW.date(), now=NOW)
    with pytest.raises(ScheduleStateError, match="no_longer_current"):
        store.finish(attempt, success=False, now=NOW)


def test_clock_rollback_does_not_create_extra_attempts_or_erase_history(tmp_path):
    store, state = seed_calendar(tmp_path)
    before = store.path.read_bytes()
    rolled_back = NOW - timedelta(hours=2)
    result = decide_refresh(calendar(), state, now=rolled_back)
    assert not result.data_due and not result.calendar_due
    assert result.reason_code == "clock_moved_backwards"
    assert result.next_attempt_at == NOW
    with pytest.raises(ScheduleStateError, match="clock_moved_backwards"):
        store.begin("data", date(2026, 10, 19), now=rolled_back)
    assert store.path.read_bytes() == before


def test_forward_clock_jump_catches_up_latest_target_instead_of_replaying_every_day(tmp_path):
    store, state = seed_calendar(tmp_path)
    complete_data(store, NOW.date(), NOW)
    later = NOW + timedelta(days=3)
    decision = decide_refresh(calendar(), store.load(), now=later)
    assert decision.data_due and decision.target_date == later.date()
    assert decision.target_date != NOW.date() + timedelta(days=1)


@pytest.mark.parametrize("raw", [b"{", b"[]", b'{"schema_version":99}', b"\xff\xfe\xff"])
def test_corrupt_state_is_never_reset_silently(tmp_path, raw):
    path = tmp_path / "state.json"
    path.write_bytes(raw)
    with pytest.raises(ScheduleStateError):
        ScheduleStore(path).load()
    with pytest.raises(ScheduleStateError):
        ScheduleStore(path).begin("data", NOW.date(), now=NOW)
    assert path.read_bytes() == raw


def test_valid_json_with_shortened_retry_is_rejected(tmp_path):
    store, _ = seed_calendar(tmp_path)
    attempt = store.begin("data", NOW.date(), now=NOW)
    state = store.finish(attempt, success=False, now=NOW)
    state["data"][NOW.date().isoformat()]["next_attempt_at"] = (NOW + timedelta(seconds=1)).isoformat()
    store.path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ScheduleStateError, match="shortened_retry"):
        store.load()


def test_atomic_replace_failure_keeps_last_good_state_and_pending_evidence(tmp_path, monkeypatch):
    store, _ = seed_calendar(tmp_path)
    original = store.path.read_bytes()

    def failed_replace(*_):
        raise OSError("synthetic local replace failure")

    monkeypatch.setattr(schedule.os, "replace", failed_replace)
    with pytest.raises(OSError):
        store.begin("data", NOW.date(), now=NOW)
    assert store.path.read_bytes() == original
    assert len(list(tmp_path.glob("*.pending"))) == 1
    assert load_schedule_state(store.path)["data"] == {}


def test_fresh_missing_state_load_is_read_only_and_naive_clock_rejected(tmp_path):
    path = tmp_path / "not_created.json"
    assert load_schedule_state(path) == new_schedule_state()
    assert not path.exists()
    with pytest.raises(ValueError, match="timezone_required"):
        decide_refresh(calendar(), new_schedule_state(), now=NOW.replace(tzinfo=None))
    save_schedule_state(path, new_schedule_state())
    assert load_schedule_state(path) == new_schedule_state()
