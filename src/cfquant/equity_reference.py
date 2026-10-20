"""Persistent, observation-dated equity references for forward simulation.

The inherited research snapshot is not a complete historical-vintage database.
New observations are usable only on/after their Shanghai observation date, and
financial announcements remain strictly earlier than the formation date.
Call ``snapshot_asof`` separately for each date before freezing its features.
The caller supplies a provider with a fresh refresh-attempt cache namespace; successful
tasks and intermediate responses are independently checkpointed here.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Callable

import pandas as pd

from .data import TushareProvider


SHANGHAI = "Asia/Shanghai"
FINANCIAL_FIELDS = (
    "ts_code,ann_date,end_date,roe,roa,debt_to_assets,ocf_to_or,update_flag"
)
INCOME_FIELDS = "ts_code,ann_date,f_ann_date,end_date,report_type,update_flag"
INDUSTRY_FIELDS = "ts_code,l1_code,l1_name,in_date,out_date,is_new"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _day(value: Any) -> pd.Timestamp:
    day = pd.Timestamp(value)
    if pd.isna(day):
        raise ValueError("Reference date is missing")
    if day.tzinfo is not None:
        day = day.tz_convert(SHANGHAI).tz_localize(None)
    return day.normalize()


def _stamp(value: Any | None) -> pd.Timestamp:
    stamp = pd.Timestamp(value if value is not None else datetime.now(timezone.utc))
    if stamp.tzinfo is None:
        raise ValueError("observed_at must include an explicit timezone")
    return stamp.tz_convert("UTC")


def _ymd(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value)
    if text.endswith(".0") and len(text) == 10:
        text = text[:-2]
    try:
        parsed = pd.to_datetime(text, format="%Y%m%d") if len(text) == 8 else pd.Timestamp(text)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(parsed) else parsed.strftime("%Y%m%d")


def _load_frame(source: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    source = Path(source)
    return pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)


def _response_observed_at(provider, identity, received_at):
    """Recover the fetch clock on cache hits; a cache read is not a new vintage."""
    if not isinstance(provider, TushareProvider):
        return received_at
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    path = provider.cache / f"{identity['api_name']}_{key}.json"
    if not path.is_file():
        raise ValueError("Tushare reference response has no acquisition receipt")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("request") != identity:
        raise ValueError("Tushare reference response identity changed")
    stamp = _stamp(receipt.get("retrieved_utc"))
    if receipt.get("retrieved_utc") is None or stamp > received_at:
        raise ValueError("Tushare reference response has an invalid acquisition time")
    return stamp


def _financial_revision(row):
    # Acquisition timestamps describe the observation, not a changed reported value.
    return _hash({key: value for key, value in row.items()
                  if key not in {"data_observed_at", "_data_observed_at"}})


class ReferenceStore:
    """A single-writer checkpoint queue and append-only reference observations.

    ``refresh`` performs at most ``max_requests`` network/provider calls. A
    successful empty financial response is retained as evidence but never
    erases an older financial value. A full scan means all scheduled queries
    finished, not that every issuer has a usable financial history.
    """

    def __init__(self, root: str | Path, base_financials: pd.DataFrame | str | Path,
                 base_industries: pd.DataFrame | str | Path, assets: list[str],
                 baseline_date: Any):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "references.sqlite3"
        self.assets = tuple(sorted(set(map(str, assets))))
        if not self.assets:
            raise ValueError("Reference universe must not be empty")
        self.baseline_date = _day(baseline_date)
        financials, industries = _load_frame(base_financials), _load_frame(base_industries)
        for frame, required in (
            (financials, {"ts_code", "ann_date", "end_date", "roe"}),
            (industries, {"ts_code", "l1_code", "in_date", "out_date"}),
        ):
            if not required.issubset(frame.columns):
                raise ValueError("Inherited reference schema is incomplete")
        financials = financials[financials.ts_code.isin(self.assets)]
        industries = industries[industries.ts_code.isin(self.assets)]
        base = {"financials": _records(financials), "industries": _records(industries)}
        identity = {"schema": 1, "assets": self.assets,
                    "baseline_date": self.baseline_date.strftime("%Y-%m-%d"),
                    "base_hash": _hash(base)}
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs(
                    key TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL,
                    priority INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                    created_date TEXT NOT NULL, completed_at TEXT, result TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0, last_error TEXT);
                CREATE TABLE IF NOT EXISTS groups(
                    key TEXT PRIMARY KEY, kind TEXT NOT NULL, asset TEXT,
                    cycle TEXT NOT NULL, created_date TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending', completed_at TEXT);
                CREATE TABLE IF NOT EXISTS observations(
                    id INTEGER PRIMARY KEY, kind TEXT NOT NULL, group_key TEXT NOT NULL UNIQUE,
                    asset TEXT, observed_at TEXT NOT NULL, observed_date TEXT NOT NULL,
                    payload TEXT NOT NULL, content_hash TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS jobs_by_group
                    ON jobs(json_extract(payload, '$.group'));
            """)
            stored = self._meta(db, "identity")
            if stored is not None and stored != json.loads(_json(identity)):
                raise ValueError("Reference store universe or inherited data identity changed")
            if stored is None:
                self._set_meta(db, "identity", identity)
                self._set_meta(db, "base", base)
                self._set_meta(db, "disclosure_enqueued_through",
                               self.baseline_date.strftime("%Y-%m-%d"))

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _meta(db, key, default=None):
        row = db.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row is not None else default

    @staticmethod
    def _set_meta(db, key, value):
        db.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)",
                   (key, _json(value)))

    @staticmethod
    def _enqueue(db, key, kind, payload, priority, created_date):
        db.execute(
            "INSERT OR IGNORE INTO jobs(key,kind,payload,priority,created_date) VALUES(?,?,?,?,?)",
            (key, kind, _json(payload), priority, created_date))

    def _financial_group(self, db, group_key, asset, cycle, day, priority):
        day_text = day.strftime("%Y-%m-%d")
        existed = db.execute("SELECT 1 FROM groups WHERE key=?", (group_key,)).fetchone()
        if existed:
            return
        db.execute("INSERT INTO groups(key,kind,asset,cycle,created_date) VALUES(?,?,?,?,?)",
                   (group_key, "financial", asset, cycle, day_text))
        # Bounds for fina_indicator are REPORT PERIODS; income bounds are ANNOUNCEMENTS.
        start = f"{day.year - 3}0101"
        end = day.strftime("%Y%m%d")
        for kind in ("financial_indicator", "financial_income"):
            self._enqueue(db, f"{group_key}:{kind}", kind,
                          {"group": group_key, "asset": asset, "start": start, "end": end},
                          priority, day_text)

    def _schedule(self, db, day):
        day_text = day.strftime("%Y-%m-%d")
        through = _day(self._meta(db, "disclosure_enqueued_through"))
        if day > through:
            for current in pd.date_range(through + pd.Timedelta(days=1), day, freq="D"):
                value = current.strftime("%Y%m%d")
                self._enqueue(db, f"disclosure:{value}", "disclosure", {"date": value},
                              0, current.strftime("%Y-%m-%d"))
            self._set_meta(db, "disclosure_enqueued_through", day_text)
        month = day.strftime("%Y-%m")
        if not self._meta(db, f"scheduled_month:{month}", False):
            for asset in self.assets:
                self._financial_group(db, f"financial:monthly:{month}:{asset}", asset,
                                      f"monthly:{month}", day, 100)
            self._set_meta(db, f"scheduled_month:{month}", True)
        year, week, _ = day.isocalendar()
        cycle = f"{year}-W{week:02d}"
        attempts = db.execute(
            "SELECT key,status FROM groups WHERE kind='industry' AND cycle=?", (cycle,)).fetchall()
        # Retain the failed response/attempt for audit. Retry at most one full
        # scan per refresh invocation, under the same request budget.
        if not any(row["status"] in {"pending", "done"} for row in attempts):
            key = f"industry:{cycle}" if not attempts else f"industry:{cycle}:retry:{len(attempts)}"
            db.execute("INSERT INTO groups(key,kind,cycle,created_date) VALUES(?,?,?,?)",
                       (key, "industry", cycle, day_text))
            self._enqueue(db, f"{key}:catalog", "industry_catalog", {"group": key},
                          20, day_text)

    def _query(self, provider, job):
        payload = json.loads(job["payload"])
        kind = job["kind"]
        if kind == "disclosure":
            api, params, fields, cap = (
                "disclosure_date", {"actual_date": payload["date"]},
                "ts_code,ann_date,end_date,actual_date", 6000)
        elif kind in {"financial_indicator", "financial_income"}:
            api = "fina_indicator" if kind == "financial_indicator" else "income"
            params = {"ts_code": payload["asset"], "start_date": payload["start"],
                      "end_date": payload["end"], "limit": 100}
            fields = FINANCIAL_FIELDS if kind == "financial_indicator" else INCOME_FIELDS
            cap = 100
        elif kind == "industry_catalog":
            api, params, fields, cap = (
                "index_classify", {"level": "L1", "src": "SW2021"},
                "index_code,industry_name,src", 2000)
        else:
            api, fields, cap = "index_member_all", INDUSTRY_FIELDS, 2000
            params = {"is_new": payload["is_new"]}
            params["ts_code" if "asset" in payload else "l1_code"] = (
                payload["asset"] if "asset" in payload else payload["l1_code"])
        frame = provider.query(api, params, fields)
        if not isinstance(frame, pd.DataFrame):
            raise ValueError("Provider returned a non-tabular reference response")
        required = {
            "disclosure": {"ts_code", "actual_date"},
            "financial_indicator": {"ts_code", "ann_date", "end_date", "roe"},
            "financial_income": {"ts_code", "ann_date", "f_ann_date", "end_date"},
            "industry_catalog": {"index_code"},
            "industry_members": {"ts_code", "l1_code", "in_date", "out_date"},
        }[kind]
        if not required.issubset(frame.columns):
            raise ValueError("Reference response is missing required columns")
        if kind == "industry_catalog" and "src" in frame.columns:
            if not frame.src.dropna().astype(str).eq("SW2021").all():
                raise ValueError("Industry catalogue changed classification version")
        if kind.startswith("financial") and not frame.empty:
            if not frame.ts_code.astype(str).eq(payload["asset"]).all():
                raise ValueError("Financial response contains an unexpected asset")
            bound_field = "end_date" if kind == "financial_indicator" else "ann_date"
            returned_dates = frame[bound_field].map(_ymd)
            if (returned_dates.isna() | returned_dates.lt(payload["start"]) |
                    returned_dates.gt(payload["end"])).any():
                raise ValueError("Financial response violates its endpoint date bounds")
        return frame, cap, {"api_name": api, "params": params, "fields": fields}

    def _split_capped(self, db, job, payload, stamp):
        """Use documented filters, not an assumed generic offset contract."""
        kind = job["kind"]
        if kind in {"financial_indicator", "financial_income"}:
            start, end = _day(payload["start"]), _day(payload["end"])
            if start >= end:
                raise ValueError("One-day reference response still reaches the row cap")
            midpoint = start + pd.Timedelta(days=(end - start).days // 2)
            ranges = [(start, midpoint), (midpoint + pd.Timedelta(days=1), end)]
            for left, right in ranges:
                child = {**payload, "start": left.strftime("%Y%m%d"),
                         "end": right.strftime("%Y%m%d")}
                self._enqueue(db, f"{job['key']}:{child['start']}:{child['end']}",
                              kind, child, job["priority"], job["created_date"])
        elif kind == "industry_members" and "asset" not in payload:
            # The fixed universe is finite. A single-symbol fallback avoids
            # relying on undocumented l1+offset or comma-list behavior.
            for asset in self.assets:
                self._enqueue(db, f"{job['key']}:{asset}", kind,
                              {**payload, "asset": asset}, job["priority"],
                              job["created_date"])
        else:
            raise ValueError("Reference response reaches an unsplittable row cap")
        db.execute("UPDATE jobs SET status='split',completed_at=?,last_error=NULL WHERE key=?",
                   (stamp.isoformat(), job["key"]))

    def _accept(self, db, job, frame, cap, stamp, source_stamp):
        payload = json.loads(job["payload"])
        kind = job["kind"]
        if len(frame) >= cap:
            self._split_capped(db, job, payload, stamp)
            return
        if kind == "industry_catalog":
            codes = sorted(set(frame.index_code.dropna().astype(str)))
            if not codes:
                raise ValueError("Industry catalogue is empty")
            for code in codes:
                for flag in ("Y", "N"):
                    self._enqueue(db, f"{payload['group']}:{code}:{flag}", "industry_members",
                                  {"group": payload["group"], "l1_code": code, "is_new": flag},
                                  30, job["created_date"])
        elif kind == "disclosure":
            # Only an actual publication on the requested date is a trigger.
            actual = frame.actual_date.map(_ymd)
            triggered = frame.loc[actual.eq(payload["date"]), "ts_code"].astype(str)
            for asset in sorted(set(triggered).intersection(self.assets)):
                self._financial_group(
                    db, f"financial:event:{payload['date']}:{asset}", asset,
                    f"event:{payload['date']}", _day(stamp.tz_convert(SHANGHAI)), 10)
        elif kind == "industry_members":
            frame = frame[frame.ts_code.isin(self.assets) &
                          frame.l1_code.astype(str).eq(payload["l1_code"])]
        frame = frame.copy()
        frame["_data_observed_at"] = source_stamp.isoformat()
        db.execute(
            "UPDATE jobs SET status='done',result=?,completed_at=?,last_error=NULL WHERE key=?",
            (_json(_records(frame)), stamp.isoformat(), job["key"]))

    @staticmethod
    def _normal_financial_rows(indicators, income):
        """Match actual publication dates only to the same report/public identity."""
        actual_dates = {}
        for row in income:
            report_type = str(row.get("report_type", "1")).removesuffix(".0")
            if report_type not in {"1", "4", "5"}:
                continue
            key = (str(row.get("ts_code")), _ymd(row.get("end_date")),
                   _ymd(row.get("ann_date")))
            actual = _ymd(row.get("f_ann_date"))
            if actual is not None:
                actual_dates.setdefault(key, set()).add(actual)
        groups = {}
        for row in indicators:
            period, ann = _ymd(row.get("end_date")), _ymd(row.get("ann_date"))
            if period is None or ann is None or not period.endswith("1231") or period > ann:
                continue
            key = (str(row.get("ts_code")), period, ann)
            groups.setdefault(key, []).append(row)
        output = []
        for (asset, period, ann), rows in groups.items():
            values = {row.get("roe") for row in rows}
            actuals = actual_dates.get((asset, period, ann), set())
            # A conflicting match is not evidence that the earliest date is safe.
            actual = max(actuals) if actuals else None
            item = dict(rows[0])
            item.pop("_data_observed_at", None)
            source_clocks = [_stamp(row["_data_observed_at"]) for row in rows
                             if row.get("_data_observed_at") is not None]
            item.update(ts_code=asset, end_date=period, source_ann_date=ann,
                        ann_date=max(ann, actual) if actual else ann,
                        actual_ann_date=actual,
                        announcement_match=("exact" if len(actuals) == 1 else
                                            "ambiguous_delayed" if actuals else "unmatched"),
                        conflicting_values=len(values) > 1)
            if source_clocks:
                item["data_observed_at"] = max(source_clocks).isoformat()
            if len(values) > 1:
                item["roe"] = None
            output.append(item)
        return output

    def _finalize_groups(self, db, group_key):
        if group_key is None:
            return
        groups = db.execute("SELECT * FROM groups WHERE status='pending' AND key=?",
                            (group_key,)).fetchall()
        for group in groups:
            jobs = db.execute("SELECT * FROM jobs WHERE json_extract(payload,'$.group')=?",
                              (group["key"],)).fetchall()
            if not jobs or any(row["status"] not in {"done", "split"} for row in jobs):
                continue
            completed = [row for row in jobs if row["status"] == "done"]
            stamp = max(_stamp(row["completed_at"]) for row in completed)
            if group["kind"] == "financial":
                indicators, income = [], []
                for job in completed:
                    if job["kind"] == "financial_indicator":
                        indicators.extend(
                            {**row, "_data_observed_at": row.get("_data_observed_at") or job["completed_at"]}
                            for row in json.loads(job["result"]))
                    elif job["kind"] == "financial_income":
                        income.extend(json.loads(job["result"]))
                data = self._normal_financial_rows(indicators, income)
                def already_published(row):
                    data_day = _stamp(row["data_observed_at"]).tz_convert(SHANGHAI).strftime("%Y%m%d")
                    return row["ann_date"] <= data_day

                future_rows = sum(not already_published(row) for row in data)
                # A future-dated vendor record must be fetched again after its
                # publication; it must not silently become usable as time passes.
                data = [row for row in data if already_published(row)]
                payload = {"rows": data, "empty_indicator_response": not indicators,
                           "empty_annual_response": not data,
                           "future_announcement_rows_ignored": future_rows,
                           "cycle": group["cycle"]}
            else:
                data = []
                for job in completed:
                    if job["kind"] == "industry_members":
                        data.extend(json.loads(job["result"]))
                # A completely empty full scan must not erase the inherited sectors.
                if not data:
                    db.execute("UPDATE groups SET status='blocked' WHERE key=?", (group["key"],))
                    continue
                # Duplicate rows from overlap/split are harmless; conflicting
                # memberships are intentionally preserved for industry_history.
                data = list({_json(row): row for row in data}.values())
                payload = {"rows": data, "cycle": group["cycle"]}
            observed_date = stamp.tz_convert(SHANGHAI).strftime("%Y-%m-%d")
            db.execute(
                """INSERT OR IGNORE INTO observations
                   (kind,group_key,asset,observed_at,observed_date,payload,content_hash)
                   VALUES(?,?,?,?,?,?,?)""",
                (group["kind"], group["key"], group["asset"], stamp.isoformat(),
                 observed_date, _json(payload), _hash(payload)))
            db.execute("UPDATE groups SET status='done',completed_at=? WHERE key=?",
                       (stamp.isoformat(), group["key"]))

    def refresh(self, provider, signal_date: Any, *, observed_at: Any | None = None,
                max_requests: int = 24,
                observation_clock: Callable[[], Any] | None = None) -> dict:
        if isinstance(max_requests, bool) or not isinstance(max_requests, int) or max_requests < 0:
            raise ValueError("max_requests must be a nonnegative integer")
        if observed_at is not None and observation_clock is not None:
            raise ValueError("Choose either a fixed observed_at or an observation_clock")

        def current_stamp():
            return _stamp(observation_clock() if observation_clock is not None else observed_at)

        day, initial_stamp = _day(signal_date), current_stamp()
        if day < self.baseline_date or day > _day(initial_stamp.tz_convert(SHANGHAI)):
            raise ValueError("Reference refresh date must be between baseline and observed date")
        with self._db() as db:
            self._schedule(db, day)
        attempted = set()
        errors = []
        completed = 0
        while len(attempted) < max_requests:
            with self._db() as db:
                pending = db.execute(
                    "SELECT * FROM jobs WHERE status='pending' ORDER BY priority,created_date,key"
                ).fetchall()
                job = next((row for row in pending if row["key"] not in attempted), None)
            if job is None:
                break
            attempted.add(job["key"])
            try:
                frame, cap, identity = self._query(provider, job)
                stamp = current_stamp()
                source_stamp = _response_observed_at(provider, identity, stamp)
                with self._db() as db:
                    db.execute("UPDATE jobs SET attempts=attempts+1 WHERE key=?", (job["key"],))
                    self._accept(db, job, frame, cap, stamp, source_stamp)
                    self._finalize_groups(db, json.loads(job["payload"]).get("group"))
                completed += 1
            except Exception as exc:
                # Never persist the raw request, token, provider error or response text.
                code = type(exc).__name__
                with self._db() as db:
                    db.execute(
                        "UPDATE jobs SET attempts=attempts+1,last_error=? WHERE key=?",
                        (code, job["key"]))
                errors.append({"task": job["key"], "error_type": code})
        stamp = current_stamp()
        _, _, metadata = self.snapshot_asof(day, observed_at=stamp)
        return {**metadata, "requests_attempted": len(attempted),
                "completed_this_run": completed,
                "errors_this_refresh": errors}

    def snapshot_asof(self, signal_date: Any, *, observed_at: Any | None = None):
        day, stamp = _day(signal_date), _stamp(observed_at)
        if day > _day(stamp.tz_convert(SHANGHAI)):
            raise ValueError("Cannot construct a future reference snapshot")
        day_text, day_ymd = day.strftime("%Y-%m-%d"), day.strftime("%Y%m%d")
        with self._db() as db:
            base = self._meta(db, "base")
            all_observations = db.execute(
                "SELECT * FROM observations ORDER BY observed_at,id").fetchall()
            all_observations = [row for row in all_observations if _stamp(row["observed_at"]) <= stamp]
            observations = [row for row in all_observations if row["observed_date"] <= day_text]
            # Maintenance is measured at the current wall clock. A morning
            # catch-up can finish its queue while the historical feature still
            # correctly uses an older, already-known financial/industry version.
            maintenance_day = stamp.tz_convert(SHANGHAI).strftime("%Y-%m-%d")
            jobs = db.execute("SELECT * FROM jobs WHERE created_date<=?",
                              (maintenance_day,)).fetchall()
            groups = db.execute("SELECT * FROM groups WHERE created_date<=?",
                                (maintenance_day,)).fetchall()
        financial = {}
        financial_source_clocks = {}
        revision_first_seen = {}
        # Older bootstrap workers store no per-row data clock. Reconstruct it
        # from their retained successful indicator jobs without changing the DB.
        legacy_source_clocks = {}
        missing_clock_groups = {
            observation["group_key"] for observation in observations
            if observation["kind"] == "financial" and
            any("data_observed_at" not in row
                for row in json.loads(observation["payload"])["rows"])
        }
        for job in jobs:
            if job["kind"] != "financial_indicator" or job["status"] != "done":
                continue
            group_key = json.loads(job["payload"]).get("group")
            if group_key not in missing_clock_groups:
                continue
            for row in json.loads(job["result"]):
                key = (group_key, str(row.get("ts_code")), _ymd(row.get("end_date")),
                       _ymd(row.get("ann_date")))
                raw_clock = row.get("_data_observed_at") or job["completed_at"]
                if raw_clock is None:
                    continue
                clock = _stamp(raw_clock)
                legacy_source_clocks[key] = max(clock, legacy_source_clocks.get(key, clock))
        baseline = self._normal_financial_rows(base["financials"], [])
        for row in baseline:
            if row["ann_date"] <= self.baseline_date.strftime("%Y%m%d") and row["ann_date"] < day_ymd:
                key = (row["ts_code"], row["end_date"], row["source_ann_date"])
                financial[key] = {**row, "first_seen_at": None, "provenance": "legacy_baseline",
                                  "revision_hash": _hash(row)}
        industries = base["industries"]
        industry_observation = None
        industry_version = None
        empty_annual_refreshes = set()
        future_announcement_rows = 0
        for observation in observations:
            payload = json.loads(observation["payload"])
            if observation["kind"] == "industry":
                # Finishing an older week's retry must not displace a newer
                # completed scan that was already available to this signal day.
                version = (payload["cycle"], _stamp(observation["observed_at"]))
                if industry_version is None or version > industry_version:
                    industries = payload["rows"]
                    industry_observation = observation
                    industry_version = version
                continue
            future_announcement_rows += payload.get("future_announcement_rows_ignored", 0)
            if payload["empty_annual_response"]:
                empty_annual_refreshes.add(observation["asset"])
            else:
                empty_annual_refreshes.discard(observation["asset"])
            for row in payload["rows"]:
                # End-of-day data can support this date's next-session signal.
                # Announcements on the formation day remain unavailable.
                if row["ann_date"] >= day_ymd:
                    continue
                revision = _financial_revision(row)
                first = revision_first_seen.setdefault(revision, observation["observed_at"])
                key = (row["ts_code"], row["end_date"], row["source_ann_date"])
                legacy_key = (observation["group_key"], *key)
                source_clock = (_stamp(row["data_observed_at"]) if row.get("data_observed_at")
                                else legacy_source_clocks.get(legacy_key))
                if source_clock is None:
                    # Legacy records normally have retained job evidence. If
                    # externally imported without it, preserve availability but
                    # never give them precedence over a proven data fetch.
                    precedence = (False, _stamp(observation["observed_at"]))
                else:
                    precedence = (True, source_clock)
                previous = financial_source_clocks.get(key)
                if previous is not None and precedence < previous:
                    continue
                if (previous == precedence and key in financial and
                        financial[key]["revision_hash"] != revision):
                    row = {**row, "roe": None, "conflicting_values": True}
                    revision = _financial_revision(row)
                financial_source_clocks[key] = precedence
                financial[key] = {**row, "first_seen_at": first, "provenance": "observed",
                                  "revision_hash": revision,
                                  "data_observed_at": source_clock.isoformat() if source_clock is not None else None,
                                  "available_at": observation["observed_at"]}
        financial_frame = pd.DataFrame(list(financial.values()))
        annual_selection_assets = set()
        if financial_frame.empty:
            financial_frame = pd.DataFrame(columns=[
                "ts_code", "ann_date", "end_date", "roe", "source_ann_date",
                "first_seen_at", "provenance", "revision_hash"])
        else:
            # In the new forward route, a late restatement of an old year must
            # not displace an already-known newer annual report. Keep all public
            # identities within the latest period, so same-identity conflicts
            # and subsequent same-year amendments remain conservatively handled.
            # Pure legacy dates retain their original research selection rule.
            annual_selection_assets = set(financial_frame.loc[
                financial_frame.provenance.eq("observed"), "ts_code"])
            latest = financial_frame.groupby("ts_code").end_date.transform("max")
            financial_frame = financial_frame.loc[
                ~financial_frame.ts_code.isin(annual_selection_assets) |
                financial_frame.end_date.eq(latest)].copy()
        industry_frame = pd.DataFrame(industries)
        if industry_frame.empty:
            industry_frame = pd.DataFrame(columns=["ts_code", "l1_code", "in_date", "out_date"])
        else:
            starts = industry_frame.in_date.map(_ymd)
            industry_frame = industry_frame.loc[starts.notna() & starts.le(day_ymd)].copy()
        eligible_ids = {row["group_key"] for row in observations}
        maintained_ids = {row["group_key"] for row in all_observations}
        month = f"monthly:{day.strftime('%Y-%m')}"
        current_financial = [row for row in groups if row["kind"] == "financial" and row["cycle"] == month]
        financial_snapshot_complete = (len(current_financial) == len(self.assets) and
                                       all(row["key"] in eligible_ids for row in current_financial))
        financial_complete = (len(current_financial) == len(self.assets) and
                              all(row["key"] in maintained_ids for row in current_financial))
        year, week, _ = day.isocalendar()
        industry_snapshot_complete = (
            industry_observation is not None and
            json.loads(industry_observation["payload"])["cycle"] == f"{year}-W{week:02d}")
        maintained_industry = next((row for row in reversed(all_observations)
                                   if row["kind"] == "industry" and
                                   json.loads(row["payload"])["cycle"] == f"{year}-W{week:02d}"), None)
        industry_complete = maintained_industry is not None

        def completed_asof(job):
            return (job["status"] in {"done", "split"} and
                    job["completed_at"] is not None and
                    _stamp(job["completed_at"]) <= stamp)

        pending = [job for job in jobs if not completed_asof(job)]
        disclosure_days = {json.loads(job["payload"])["date"] for job in jobs
                           if job["kind"] == "disclosure" and completed_asof(job)}
        needed_days = {d.strftime("%Y%m%d") for d in pd.date_range(
            self.baseline_date + pd.Timedelta(days=1), day, freq="D")}
        disclosure_complete = needed_days.issubset(disclosure_days)
        all_blocked_groups = [row["key"] for row in groups if row["status"] == "blocked"]
        current_industry_pending = any(
            row["kind"] == "industry" and row["cycle"] == f"{year}-W{week:02d}" and
            row["status"] == "pending" for row in groups)
        blocked_groups = [
            row["key"] for row in groups if row["status"] == "blocked" and
            (row["kind"] != "industry" or
             (row["cycle"] == f"{year}-W{week:02d}" and not industry_complete and
              not current_industry_pending))]
        financial_last = max(
            (_stamp(row["completed_at"]).isoformat() for row in current_financial
             if row["key"] in maintained_ids), default=None)
        complete = financial_complete and industry_complete
        maintenance_ready = complete and disclosure_complete and not pending and not blocked_groups
        uses_prior_versions = len(observations) != len(all_observations)
        metadata = {
            "ready": maintenance_ready, "maintenance_ready": maintenance_ready,
            "signal_date": day_text, "decision_observed_at": stamp.isoformat(),
            "backlog_tasks": len(pending), "blocked_groups": blocked_groups,
            "historical_blocked_groups": [key for key in all_blocked_groups if key not in blocked_groups],
            "financial_last_success": financial_last if financial_complete else None,
            "industry_last_success": maintained_industry["observed_at"] if maintained_industry else None,
            "financial_full_scan_complete": financial_complete,
            "industry_full_scan_complete": industry_complete, "full_scan_complete": complete,
            "snapshot_full_scan_complete": financial_snapshot_complete and industry_snapshot_complete,
            "uses_prior_reference_versions": uses_prior_versions,
            "financial_reference_asof": max(
                (row["observed_at"] for row in observations if row["kind"] == "financial"), default=None),
            "industry_reference_asof": industry_observation["observed_at"] if industry_observation else None,
            "disclosure_scan_complete": disclosure_complete,
            "empty_annual_refresh_assets": sorted(empty_annual_refreshes),
            "future_announcement_rows_ignored": future_announcement_rows,
            "observed_financial_rows": int(financial_frame.provenance.eq("observed").sum()),
            "inherited_financial_rows": int(financial_frame.provenance.eq("legacy_baseline").sum()),
            "industry_source": "observed" if industry_observation is not None else "legacy_baseline",
            "universe_size": len(self.assets), "universe_hash": _hash(self.assets),
            "baseline_date": self.baseline_date.strftime("%Y-%m-%d"),
            "snapshot_for_signal_date_only": True,
            "reference_note": (
                "补跑使用形成日当时已观察的参考版本；本次新抓修订从实际观察日开始生效。"
                if uses_prior_versions else
                "新参考版本按公告日与实际观察日期门限加入，历史形成日不倒灌修订。"),
            "historical_vintages_complete": False,
            "annual_selection_policy": (
                "latest_available_report_period_for_observed_assets; inherited_only_assets_keep_legacy_rule"),
            "annual_period_selected_assets": len(annual_selection_assets),
            "financial_hash": _hash(_records(financial_frame)),
            "industry_hash": _hash(_records(industry_frame)),
            "limitations": [
                "Legacy references inherit the research snapshot; complete vendor vintages are not guaranteed.",
                "New revisions require both announcement and Shanghai first-observation date gates.",
                "Empty financial refreshes retain prior values; stale annual ROE remains subject to the feature-age filter.",
                "Industry exit dates use the platform's exclusive-end convention.",
            ],
        }
        return financial_frame.reset_index(drop=True), industry_frame.reset_index(drop=True), metadata
