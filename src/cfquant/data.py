"""Canonical data contract and Tushare acquisition. No credentials are persisted."""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


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

    def query(self, api: str, params: dict, fields: str) -> pd.DataFrame:
        identity = {"api_name": api, "params": params, "fields": fields}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        path = self.cache / f"{api}_{key}.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
        else:
            for attempt in range(4):
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


def download_project(root: Path, token_file: str | None, size: int = 60) -> dict:
    """Select before the study starts, without current constituent/listing filters."""
    raw = root / "data/raw"
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    provider = TushareProvider(raw, token_file)
    selection = provider.query("daily", {"trade_date": "20221230"},
                               "ts_code,trade_date,open,high,low,close,pre_close,vol,amount")
    # Main-board prefixes; eligibility uses only data known before 2023.
    pool = selection[selection.ts_code.str.match(r"^(000|001|002|003|600|601|603|605)\d{3}\.(SZ|SH)$")]
    pool = pool[(pool.close >= 3) & (pool.vol > 0)]
    pool = pool.sort_values(["amount", "ts_code"], ascending=[False, True]).head(size)
    if len(pool) != size:
        raise ValueError("Insufficient selection-day universe")
    pool.to_csv(processed / "universe.csv", index=False)
    calendar = provider.query("trade_cal", {"exchange": "SSE", "start_date": "20221001",
                              "end_date": "20251231"}, "cal_date,is_open")
    calendar = pd.DataFrame({"date": pd.to_datetime(
        calendar.loc[calendar.is_open == 1, "cal_date"].astype(str), format="%Y%m%d")})
    calendar.sort_values("date").to_csv(processed / "calendar.csv", index=False)
    frames, asset_reports = [], []
    for i, code in enumerate(pool.ts_code):
        params = {"ts_code": code, "start_date": "20221001", "end_date": "20251231"}
        daily = provider.query("daily", params, "ts_code,trade_date,open,high,low,close,pre_close,vol,amount")
        adj = provider.query("adj_factor", params, "ts_code,trade_date,adj_factor")
        limits = provider.query("stk_limit", params, "ts_code,trade_date,up_limit,down_limit")
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
        frames.append(df[["date", "asset", "open", "high", "low", "close", "volume",
                          "raw_open", "raw_high", "raw_low", "raw_close", "adj_factor",
                          "adjustment_reference", "up_limit", "down_limit"]])
        asset_reports.append({"asset": code, "raw_rows": before, "valid_rows": len(df),
                              "duplicate_keys": duplicates, "invalid_rows_excluded": bad_count,
                              "source_date_ascending": source_sorted,
                              "clean_date_ascending": bool(df.date.is_monotonic_increasing),
                              "source_missing_by_field": source_missing,
                              "calendar_coverage": len(df)/len(calendar),
                              "missing_limit_rows": int(df.up_limit.isna().sum()),
                              "calendar_missing_rows": len(calendar) - len(df),
                              "first": str(df.date.min().date()), "last": str(df.date.max().date())})
        print(f"Downloaded {i+1}/{size}: {code}, {len(df)} valid rows", flush=True)
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
                "selection_date": "2022-12-30", "study_period": ["2023-01-03", "2025-12-31"],
                "selection_rule": f"Top {size} amount on 2022-12-30, mainland main-board prefixes, close >= 3, vol > 0; no current-listing or future-performance filter.",
                "limitations": ["Fixed liquid universe; not all A shares or a historical index.",
                               "Retrospectively downloaded vendor data may contain corrections.",
                               "Adjustment factors approximate reinvested total-return units, not actual shareholder cash events."],
                "files": {str(p.relative_to(root)).replace("\\", "/"): digest(p) for p in files}}
    write_json(processed / "manifest.json", manifest)
    return manifest
