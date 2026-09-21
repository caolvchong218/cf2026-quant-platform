"""Canonical data contract and Tushare acquisition. No credentials are persisted."""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = {"date", "asset", "open", "high", "low", "close", "volume",
            "raw_open", "raw_close", "adj_factor", "up_limit", "down_limit"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(path)


def load_market(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"], dtype={"asset": str})
    missing = REQUIRED - set(frame)
    if missing:
        raise ValueError(f"Missing market columns: {sorted(missing)}")
    if frame.duplicated(["date", "asset"]).any():
        raise ValueError("Duplicate date/asset keys")
    if frame[["date", "asset"]].isna().any().any():
        raise ValueError("Missing date/asset keys")
    numeric = list(REQUIRED - {"date", "asset"})
    for col in numeric:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if np.isinf(frame[numeric].to_numpy()).any():
        raise ValueError("Infinite market values")
    for col in ["open", "high", "low", "close", "raw_open", "raw_close", "adj_factor"]:
        if (frame[col].dropna() <= 0).any():
            raise ValueError(f"Nonpositive price/factor: {col}")
    if (frame.volume.dropna() < 0).any():
        raise ValueError("Negative volume")
    return frame.sort_values(["date", "asset"]).reset_index(drop=True)


def load_calendar(path: str | Path) -> pd.DatetimeIndex:
    dates = pd.to_datetime(pd.read_csv(path)["date"])
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("Invalid calendar")
    return pd.DatetimeIndex(dates.sort_values())


class TushareProvider:
    """HTTPS JSON API with deterministic cache keys, bounded retries and pacing."""

    def __init__(self, cache: Path, token_file: str | None = None, delay: float = 1.3):
        self.cache = cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self._token = os.environ.get("TUSHARE_TOKEN", "").strip()
        if not self._token and token_file:
            self._token = Path(token_file).read_text(encoding="utf-8-sig").strip()
        if not self._token:
            raise ValueError("Set TUSHARE_TOKEN or supply --token-file; never put it in YAML.")
        self.delay = delay
        self.last_call = 0.0
        self._pace_lock = threading.Lock()

    def query(self, api: str, params: dict, fields: str) -> pd.DataFrame:
        identity = {"api_name": api, "params": params, "fields": fields}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        path = self.cache / f"{api}_{key}.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
        else:
            for attempt in range(4):
                with self._pace_lock:
                    time.sleep(max(0, self.delay - (time.monotonic() - self.last_call)))
                    self.last_call = time.monotonic()
                req = urllib.request.Request(
                    "https://api.tushare.pro",
                    data=json.dumps({**identity, "token": self._token}).encode(),
                    headers={"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(req, timeout=45) as response:
                        body = json.load(response)
                    if body.get("code") != 0:
                        # Deliberately avoid logging the response or request: no secret echoes.
                        raise RuntimeError(f"Tushare {api}: API code {body.get('code')}")
                    record = {"request": identity, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                              "data": body["data"]}
                    write_json(path, record)
                    break
                except Exception as exc:
                    if attempt == 3:
                        raise RuntimeError(f"Acquisition failed: {api} ({type(exc).__name__}); "
                                           "check access/network; cached successes retained.") from None
                    time.sleep(3 * (attempt + 1))
        data = record["data"]
        return pd.DataFrame(data["items"], columns=data["fields"])


def download_project(root: Path, token_file: str | None, size: int = 60, *,
                     start: str = "20221001", end: str = "20251231",
                     selection_date: str = "20221230", study_start: str = "20230103",
                     delay: float = 1.3, max_gib: float = 3.0, workers: int = 1) -> dict:
    """Select before the study starts, without current constituent/listing filters."""
    start, end, selection_date, study_start = [
        pd.Timestamp(x).strftime("%Y%m%d") for x in (start, end, selection_date, study_start)]
    if not (start <= selection_date < study_start <= end):
        raise ValueError("Require data start <= selection date < study start <= end")
    if size < 1 or not np.isfinite(delay) or delay < 0.3 or not np.isfinite(max_gib) or max_gib <= 0 or not 1 <= workers <= 8:
        raise ValueError("Require positive size/budget, delay >= 0.3 seconds, and 1 <= workers <= 8")
    # Each asset request stays below the smallest endpoint row cap. Longer ranges
    # should be split into separate snapshots instead of silently truncating.
    if (pd.Timestamp(end) - pd.Timestamp(start)).days > 3653:
        raise ValueError("Use snapshots of at most ten years")
    raw = root / "data/raw"
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    plan = {"size": size, "start": start, "end": end,
            "selection_date": selection_date, "study_start": study_start}
    plan_path = processed / "download_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Snapshot parameters differ; choose a new --root to preserve the existing data")
    write_json(plan_path, plan)
    provider = TushareProvider(raw, token_file, delay=delay)
    selection = provider.query("daily", {"trade_date": selection_date},
                               "ts_code,trade_date,open,high,low,close,pre_close,vol,amount")
    # Main-board prefixes; eligibility uses only selection-day information.
    pool = selection[selection.ts_code.str.match(r"^(000|001|002|003|600|601|603|605)\d{3}\.(SZ|SH)$")]
    pool = pool[(pool.close >= 3) & (pool.vol > 0)]
    pool = pool.sort_values(["amount", "ts_code"], ascending=[False, True]).head(size)
    if len(pool) != size:
        raise ValueError("Insufficient selection-day universe")
    pool.to_csv(processed / "universe.csv", index=False)
    calendar = provider.query("trade_cal", {"exchange": "SSE", "start_date": start,
                              "end_date": end}, "cal_date,is_open")
    calendar = pd.DataFrame({"date": pd.to_datetime(
        calendar.loc[calendar.is_open == 1, "cal_date"].astype(str), format="%Y%m%d")})
    calendar.sort_values("date").to_csv(processed / "calendar.csv", index=False)
    if calendar.empty:
        raise ValueError("No trading sessions returned")
    frames, asset_reports = [], []
    budget = int(max_gib * 1024**3)
    def acquire_asset(code):
        # Leave room for processed CSV and a validation read; raw successes remain
        # cached if interrupted. Never overwrite a previously accepted snapshot.
        raw_bytes = sum(p.stat().st_size for p in raw.glob("*.json"))
        if raw_bytes > budget * 0.65:
            raise ValueError("Storage budget reserve reached; cached successes retained")
        params = {"ts_code": code, "start_date": start, "end_date": end}
        daily = provider.query("daily", params, "ts_code,trade_date,open,high,low,close,pre_close,vol,amount")
        adj = provider.query("adj_factor", params, "ts_code,trade_date,adj_factor")
        limits = provider.query("stk_limit", params, "ts_code,trade_date,up_limit,down_limit")
        if any(len(frame) >= 5800 for frame in (daily, adj, limits)):
            raise ValueError(f"Possible endpoint truncation for {code}; split date range")
        before = len(daily)
        source_sorted = bool(daily.trade_date.is_monotonic_increasing)
        source_missing = {col:int(daily[col].isna().sum()) for col in daily.columns}
        duplicates = int(daily.duplicated(["ts_code", "trade_date"]).sum())
        if duplicates or adj.duplicated(["ts_code", "trade_date"]).any() or limits.duplicated(["ts_code", "trade_date"]).any():
            raise ValueError(f"Ambiguous duplicate source keys for {code}")
        df = daily.merge(adj, on=["ts_code", "trade_date"], how="left", validate="one_to_one")
        df = df.merge(limits, on=["ts_code", "trade_date"], how="left", validate="one_to_one")
        df = df.sort_values("trade_date")
        columns = ["open", "high", "low", "close", "adj_factor", "vol"]
        for col in columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        invalid = (~np.isfinite(df[columns]).all(axis=1) |
                   (df[["open", "high", "low", "close", "adj_factor"]] <= 0).any(axis=1) |
                   (df.vol < 0) |
                   (df.high < df[["open", "close", "low"]].max(axis=1)) |
                   (df.low > df[["open", "close", "high"]].min(axis=1)))
        bad_count = int(invalid.sum())
        df = df.loc[~invalid].copy()
        if df.empty:
            raise ValueError(f"No valid prices for {code}")
        anchor = float(df.adj_factor.iloc[0])
        for col in ["open", "high", "low", "close"]:
            df[f"raw_{col}"] = df[col]
            df[col] = df[col] * df.adj_factor / anchor
        df["adjustment_reference"] = anchor
        df["volume"] = df.vol * 100  # Tushare A-share vol is in 100-share lots.
        df["date"] = pd.to_datetime(df.trade_date.astype(str), format="%Y%m%d")
        df["asset"] = code
        frame = df[["date", "asset", "open", "high", "low", "close", "volume",
                          "raw_open", "raw_high", "raw_low", "raw_close", "adj_factor",
                          "adjustment_reference", "up_limit", "down_limit"]]
        report = {"asset": code, "raw_rows": before, "valid_rows": len(df),
                              "duplicate_keys": duplicates, "invalid_rows_excluded": bad_count,
                              "source_date_ascending": source_sorted,
                              "clean_date_ascending": bool(df.date.is_monotonic_increasing),
                              "source_missing_by_field": source_missing,
                              "calendar_coverage": len(df)/len(calendar),
                              "missing_limit_rows": int(df.up_limit.isna().sum()),
                              "calendar_missing_rows": len(calendar) - len(df),
                              "first": str(df.date.min().date()), "last": str(df.date.max().date())}
        return frame, report
    # One provider enforces a global request rate across all worker threads.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        try:
            for i, (frame, report) in enumerate(executor.map(acquire_asset, pool.ts_code)):
                frames.append(frame)
                asset_reports.append(report)
                print(f"Downloaded {i+1}/{size}: {report['asset']}, {len(frame)} valid rows", flush=True)
                write_json(processed / "progress.json", {
                    "completed_assets": i+1, "target_assets": size, "rows": sum(len(x) for x in frames),
                    "last_asset": report["asset"], "requested_end": end, "status": "downloading",
                    "updated_utc": datetime.now(timezone.utc).isoformat()})
        except Exception:
            executor.shutdown(wait=True, cancel_futures=True)
            write_json(processed / "progress.json", {
                "completed_assets": len(frames), "target_assets": size, "status": "interrupted",
                "updated_utc": datetime.now(timezone.utc).isoformat()})
            raise
    market = pd.concat(frames).sort_values(["date", "asset"])
    market.to_csv(processed / "market.csv", index=False, float_format="%.12g")
    load_market(processed / "market.csv")
    quality = {"assets": asset_reports, "rows": len(market), "asset_count": size,
               "duplicate_keys": int(market.duplicated(["date", "asset"]).sum()),
               "raw_rows": sum(x["raw_rows"] for x in asset_reports),
               "excluded_invalid_rows": sum(x["invalid_rows_excluded"] for x in asset_reports),
               "missing_policy": "No blanket fill. Invalid source rows excluded and counted. Missing bars remain missing; valuation alone may carry the last close.",
               "price_convention": "raw_price * adj_factor / first observed adj_factor per asset",
               "volume_unit": "shares (source vol multiplied by 100)",
               "calendar": "SSE open sessions, used as mainland-equity research calendar"}
    write_json(processed / "quality.json", quality)
    files = sorted(raw.glob("*.json")) + sorted(processed.glob("*.csv")) + [processed / "quality.json"]
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "provider": "Tushare Pro",
                "selection_date": str(pd.Timestamp(selection_date).date()),
                "study_period": [str(pd.Timestamp(study_start).date()), str(pd.Timestamp(end).date())],
                "requested_period": [start, end],
                "actual_period": [str(market.date.min().date()), str(market.date.max().date())],
                "calendar_sessions": len(calendar), "max_gib": max_gib,
                "selection_rule": f"Top {size} amount on {selection_date}, mainland main-board prefixes, close >= 3, vol > 0; no current-listing or future-performance filter.",
                "limitations": ["Fixed liquid universe; not all A shares or a historical index.",
                               "Retrospectively downloaded vendor data may contain corrections.",
                               "Adjustment factors approximate reinvested total-return units, not actual shareholder cash events."],
                "files": {str(p.relative_to(root)).replace("\\", "/"): digest(p) for p in files}}
    write_json(processed / "manifest.json", manifest)
    stored_bytes = sum(p.stat().st_size for folder in (raw, processed) for p in folder.iterdir() if p.is_file())
    if stored_bytes > budget:
        raise ValueError("Final snapshot exceeds storage budget; increase limit to accept")
    write_json(processed / "progress.json", {
        "completed_assets": size, "target_assets": size, "rows": len(market),
        "status": "complete", "actual_end": str(market.date.max().date()),
        "updated_utc": datetime.now(timezone.utc).isoformat()})
    return manifest
