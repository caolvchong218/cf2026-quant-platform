"""Incremental, verified official exchange calendar storage.

Only ``trade_cal`` is requested. Existing calendar dates through the current
Shanghai date are immutable; future published dates may change. Missing dates
are never synthesized from weekdays. Each request is split at year boundaries.

The caller supplies a provider with ``query(api, params, fields)`` and owns
retry/backoff. Use a fresh raw provider cache per refresh attempt: an old cached
empty response must not permanently hide a newly published future calendar.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
from typing import Protocol
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd


SHANGHAI = ZoneInfo("Asia/Shanghai")
_SNAPSHOT = re.compile(r"calendar_([0-9a-f]{64})\.csv\Z")


class CalendarDataError(ValueError):
    """Invalid evidence or a historical calendar revision; no raw response."""


class CalendarFetchError(RuntimeError):
    """A bounded provider call failed; already verified checkpoints remain."""


class CalendarIncompleteError(RuntimeError):
    """The provider has not supplied any usable contiguous calendar prefix."""

    def __init__(self, required_through: date):
        super().__init__("official_calendar_not_published")
        self.required_through = required_through


class CalendarProvider(Protocol):
    def query(self, api: str, params: dict, fields: str) -> pd.DataFrame: ...


def _clock(now: datetime | None) -> datetime:
    now = datetime.now(timezone.utc) if now is None else now
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("calendar_timezone_required")
    return now.astimezone(SHANGHAI)


def _date(value: str | date) -> date:
    if type(value) is date:
        return value
    try:
        result = date.fromisoformat(value)
    except (TypeError, ValueError):
        raise CalendarDataError("calendar_invalid_date") from None
    if result.isoformat() != value:
        raise CalendarDataError("calendar_invalid_date")
    return result


def _instant(value: object) -> datetime:
    try:
        result = datetime.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        result = None
    if result is None or result.tzinfo is None or result.utcoffset() is None:
        raise CalendarDataError("calendar_invalid_timestamp")
    return result


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty or not {"cal_date", "is_open"} <= set(frame):
        raise CalendarDataError("calendar_missing_required_rows")
    try:
        dates = pd.to_datetime(frame.cal_date.astype(str), format="mixed", errors="raise")
        flags = pd.to_numeric(frame.is_open, errors="raise")
        if (dates.isna().any() or dates.dt.tz is not None or dates.duplicated().any()
                or not dates.eq(dates.dt.normalize()).all() or not flags.isin((0, 1)).all()):
            raise CalendarDataError("calendar_invalid_values")
        result = pd.DataFrame({"cal_date": dates.to_numpy(), "is_open": flags.to_numpy(dtype=int)})
        result = result.sort_values("cal_date").reset_index(drop=True)
        if len(result) != (result.cal_date.iloc[-1] - result.cal_date.iloc[0]).days + 1:
            raise CalendarDataError("calendar_has_missing_dates")
        return result
    except (TypeError, ValueError, AttributeError, OverflowError) as exc:
        if isinstance(exc, CalendarDataError):
            raise
        raise CalendarDataError("calendar_invalid_values") from None


def _json(path: Path) -> dict:
    try:
        if path.stat().st_size > 1_000_000:
            raise CalendarDataError("calendar_metadata_oversize")
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError, UnicodeError) as exc:
        if isinstance(exc, CalendarDataError):
            raise
        raise CalendarDataError("calendar_metadata_unreadable") from None
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise CalendarDataError("calendar_metadata_invalid")
    return data


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".pending")
    with temporary.open("xb") as stream:
        stream.write((json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _snapshot(root: Path, frame: pd.DataFrame) -> dict:
    raw = frame.to_csv(index=False, date_format="%Y-%m-%d", lineterminator="\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    filename = f"calendar_{digest}.csv"
    folder = root / "snapshots"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise CalendarDataError("calendar_snapshot_identity_mismatch")
    else:
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    return {
        "file": "snapshots/" + filename, "sha256": digest, "rows": len(frame),
        "coverage_start": frame.cal_date.iloc[0].date().isoformat(),
        "coverage_end": frame.cal_date.iloc[-1].date().isoformat(),
    }


def _read_snapshot(root: Path, metadata: dict) -> pd.DataFrame:
    relative = metadata.get("file")
    if not isinstance(relative, str) or not relative.startswith("snapshots/"):
        raise CalendarDataError("calendar_snapshot_path_invalid")
    match = _SNAPSHOT.fullmatch(relative.removeprefix("snapshots/"))
    if match is None or match.group(1) != metadata.get("sha256"):
        raise CalendarDataError("calendar_snapshot_identity_invalid")
    path = root / relative
    try:
        if path.stat().st_size > 10_000_000:
            raise CalendarDataError("calendar_snapshot_oversize")
        raw = path.read_bytes()
    except OSError:
        raise CalendarDataError("calendar_snapshot_unreadable") from None
    if hashlib.sha256(raw).hexdigest() != metadata["sha256"]:
        raise CalendarDataError("calendar_snapshot_hash_mismatch")
    try:
        frame = _normalize(pd.read_csv(io.BytesIO(raw)))
    except (ValueError, pd.errors.ParserError):
        raise CalendarDataError("calendar_snapshot_invalid_rows") from None
    if (type(metadata.get("rows")) is not int or len(frame) != metadata["rows"]
            or frame.cal_date.iloc[0].date() != _date(metadata.get("coverage_start"))
            or frame.cal_date.iloc[-1].date() != _date(metadata.get("coverage_end"))):
        raise CalendarDataError("calendar_snapshot_metadata_mismatch")
    return frame


def _publish(root: Path, frame: pd.DataFrame, *, required_start: date,
             required_end: date, now: datetime, origin: str,
             request_incomplete: bool = False) -> pd.DataFrame:
    frame = _normalize(frame)
    metadata = {
        "schema_version": 1, "provider": "tushare", "api": "trade_cal", "exchange": "SSE",
        "updated_at": now.astimezone(timezone.utc).isoformat(), "origin": origin,
        "request_incomplete": request_incomplete,
        "required_start": required_start.isoformat(), "required_through": required_end.isoformat(),
        "coverage_complete": (frame.cal_date.iloc[0].date() <= required_start
                              and frame.cal_date.iloc[-1].date() >= required_end),
        **_snapshot(root, frame),
    }
    _atomic_json(root / "manifest.json", metadata)
    frame.attrs.update(metadata)
    return frame


def load_calendar_cache(root: str | Path) -> pd.DataFrame | None:
    """Verify manifest, SHA, continuous rows and coverage; absence returns None.

    Incomplete but verified prefixes are returned with ``coverage_complete``
    false in ``DataFrame.attrs``. The scheduler must independently check whether
    the known prefix proves today's target and the next open session.
    """
    root = Path(root)
    if not (root / "manifest.json").exists():
        return None
    metadata = _json(root / "manifest.json")
    if (metadata.get("provider") != "tushare" or metadata.get("api") != "trade_cal"
            or metadata.get("exchange") != "SSE" or type(metadata.get("coverage_complete")) is not bool
            or type(metadata.get("request_incomplete")) is not bool):
        raise CalendarDataError("calendar_manifest_invalid_source")
    _instant(metadata.get("updated_at"))
    frame = _read_snapshot(root, metadata)
    required_start, required_end = _date(metadata.get("required_start")), _date(metadata.get("required_through"))
    if required_start > required_end:
        raise CalendarDataError("calendar_manifest_invalid_range")
    actual_complete = frame.cal_date.iloc[0].date() <= required_start and frame.cal_date.iloc[-1].date() >= required_end
    if metadata["coverage_complete"] != actual_complete:
        raise CalendarDataError("calendar_manifest_invalid_completeness")
    frame.attrs.update(metadata)
    return frame


def seed_calendar(root: str | Path, verified_frame: pd.DataFrame, *,
                  now: datetime | None = None) -> pd.DataFrame:
    """Migrate an already verified market snapshot into an empty calendar store.

    Identical seeding is idempotent. A different existing store is not silently
    overwritten with an older source snapshot.
    """
    root, now = Path(root), _clock(now)
    frame = _normalize(verified_frame)
    old = load_calendar_cache(root)
    if old is not None:
        if not old.equals(frame):
            raise CalendarDataError("calendar_seed_would_replace_existing")
        return old
    return _publish(root, frame, required_start=frame.cal_date.iloc[0].date(),
                    required_end=frame.cal_date.iloc[-1].date(), now=now, origin="verified_market_seed")


def _year_chunks(start: date, end: date) -> list[tuple[date, date]]:
    result = []
    while start <= end:
        stop = min(end, date(start.year, 12, 31))
        result.append((start, stop))
        start = stop + timedelta(days=1)
    return result


def _checkpoint_path(root: Path, start: date, end: date) -> Path:
    return root / "checkpoints" / f"{start:%Y%m%d}_{end:%Y%m%d}.json"


def _checkpoint(root: Path, start: date, end: date, now: datetime) -> pd.DataFrame | None:
    path = _checkpoint_path(root, start, end)
    if not path.exists():
        return None
    metadata = _json(path)
    if (metadata.get("requested_start") != start.isoformat()
            or metadata.get("requested_end") != end.isoformat()
            or type(metadata.get("complete")) is not bool):
        raise CalendarDataError("calendar_checkpoint_invalid")
    fetched = _instant(metadata.get("fetched_at")).astimezone(SHANGHAI)
    if metadata.get("file") is None:
        if metadata.get("status") != "unpublished" or metadata["complete"]:
            raise CalendarDataError("calendar_checkpoint_invalid_empty")
        return None
    frame = _read_snapshot(root, metadata)
    if (frame.cal_date.iloc[0].date() != start or frame.cal_date.iloc[-1].date() > end
            or metadata["complete"] != (frame.cal_date.iloc[-1].date() == end)):
        raise CalendarDataError("calendar_checkpoint_range_mismatch")
    # Only complete chunks are reusable. Recheck future-containing chunks on a
    # later Shanghai date; permanently reuse completed historical years.
    if metadata["complete"] and (end < now.date() or fetched.date() == now.date()):
        return frame
    return None


def _save_checkpoint(root: Path, start: date, end: date, frame: pd.DataFrame | None,
                     now: datetime) -> None:
    metadata = {
        "schema_version": 1, "requested_start": start.isoformat(), "requested_end": end.isoformat(),
        "fetched_at": now.astimezone(timezone.utc).isoformat(),
        "complete": frame is not None and frame.cal_date.iloc[-1].date() == end,
        "status": "verified" if frame is not None else "unpublished",
    }
    if frame is not None:
        metadata.update(_snapshot(root, frame))
    else:
        metadata["file"] = None
    _atomic_json(_checkpoint_path(root, start, end), metadata)


def _merge(old: pd.DataFrame | None, part: pd.DataFrame, *, today: date) -> pd.DataFrame:
    if old is None:
        return part.copy()
    left, right = old.set_index("cal_date"), part.set_index("cal_date")
    overlap = left.index.intersection(right.index)
    frozen = overlap[overlap.date <= today]
    if len(frozen) and not left.loc[frozen, "is_open"].equals(right.loc[frozen, "is_open"]):
        raise CalendarDataError("official_calendar_changed_existing_past_session")
    merged = pd.concat([left.loc[~left.index.isin(right.index)], right]).sort_index().reset_index()
    # Intermediate backward/forward chunks can temporarily be disconnected.
    # The complete merged result is checked before publication.
    return merged


def refresh_calendar(root: str | Path, provider: CalendarProvider, *, now: datetime,
                     history_start: str = "2019-01-01", horizon_days: int = 45) -> pd.DataFrame:
    """Recheck recent and future dates, with resumable historical year chunks.

    Partial future publication is explicit in the returned attrs and manifest.
    Network failure leaves the current manifest unchanged and preserves earlier
    complete chunks for the next externally scheduled attempt. A long future
    cache must not hide a revision to tomorrow's session: refresh from the
    earlier of its tail minus seven days and the current date minus seven days.
    """
    root, now = Path(root), _clock(now)
    start = _date(history_start)
    if type(horizon_days) is not int or not 0 <= horizon_days <= 366:
        raise ValueError("calendar_invalid_horizon")
    required_end = now.date() + timedelta(days=horizon_days)
    if start > now.date() or now.year - start.year > 150:
        raise ValueError("calendar_invalid_history_start")
    old = load_calendar_cache(root)
    ranges = []
    if old is None:
        ranges.append((start, required_end))
    else:
        old_start, old_end = old.cal_date.iloc[0].date(), old.cal_date.iloc[-1].date()
        if start < old_start:
            ranges.append((start, old_start - timedelta(days=1)))
        # Preserve any already published farther future; never trim the cache.
        tail_end = max(required_end, old_end)
        tail_start = max(start, min(old_end, now.date()) - timedelta(days=7))
        ranges.append((tail_start, tail_end))
    combined = old.copy() if old is not None else None
    unpublished = False
    for begin, stop in (chunk for span in ranges for chunk in _year_chunks(*span)):
        part = _checkpoint(root, begin, stop, now)
        if part is None:
            try:
                raw = provider.query("trade_cal", {
                    "exchange": "SSE", "start_date": begin.strftime("%Y%m%d"),
                    "end_date": stop.strftime("%Y%m%d"),
                }, "cal_date,is_open")
            except Exception:
                raise CalendarFetchError("official_calendar_request_failed") from None
            if not isinstance(raw, pd.DataFrame):
                raise CalendarDataError("official_calendar_response_invalid")
            if raw.empty:
                if begin <= now.date():
                    raise CalendarDataError("official_calendar_missing_past_dates")
                _save_checkpoint(root, begin, stop, None, now)
                unpublished = True
                break
            part = _normalize(raw)
            if part.cal_date.iloc[0].date() != begin or part.cal_date.iloc[-1].date() > stop:
                raise CalendarDataError("official_calendar_response_wrong_range")
            if part.cal_date.iloc[-1].date() < min(stop, now.date()):
                raise CalendarDataError("official_calendar_missing_past_dates")
            # Check historical revisions BEFORE checkpointing the response.
            if old is not None:
                _merge(old, part, today=now.date())
            _save_checkpoint(root, begin, stop, part, now)
        combined = _merge(combined, part, today=now.date())
        if part.cal_date.iloc[-1].date() < stop:
            unpublished = True
            break
    if combined is None or combined.empty:
        raise CalendarIncompleteError(required_end)
    combined = _normalize(combined)
    # Existing cached future rows remain usable even if this request publishes
    # a shorter prefix; attrs still record that this refresh was incomplete.
    result = _publish(root, combined, required_start=start, required_end=required_end,
                      now=now, origin="official_trade_cal_refresh", request_incomplete=unpublished)
    return result
