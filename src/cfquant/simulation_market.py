"""Incremental homogeneous index history and raw ETF close verification.

Calendar rows must come from the separately verified official calendar store.
Only complete year-sized index chunks are checkpointed; partial provider
responses are never treated as holidays. This module performs no trading.
"""
from __future__ import annotations

from datetime import datetime, time, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import digest, write_json
from .simulation_signals import (
    EXECUTION_ASSET, HISTORY_START, MAPPING_NOTICE, SIGNAL_ASSET, STRATEGY_IDS,
    MarketSnapshot, _now, _prices, normalize_calendar, prepare_signal_bundle, session_dates,
)

HOMOGENEOUS_SOURCE = "Tushare Pro index_daily, incremental homogeneous fixed 2019 anchor"
INITIAL_HOMOGENEOUS_SOURCE = "Tushare Pro index_daily, homogeneous fixed 2019 anchor"
LEGACY_PRELOAD_SOURCE = (
    "Historical preload plus Tushare index_daily; overlapping OHLC agree within 0.02 points"
)


def signal_clock(now):
    """Keep the previous completed session until the daily 17:05 publication gate."""
    local = _now(now)
    if time(15, 30) <= local.time().replace(tzinfo=None) < time(17, 5):
        return local.replace(hour=15, minute=0, second=0, microsecond=0)
    return local


def latest_verified_market(root):
    paths = sorted(Path(root).glob("*/manifest.json"))
    if not paths:
        return None
    path = paths[-1]
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(receipt, dict) or receipt.get("ready") is not True
                or receipt.get("schema_version") not in (1, 2)
                or receipt.get("signal_asset") != SIGNAL_ASSET
                or receipt.get("execution_asset") != EXECUTION_ASSET):
            raise ValueError("Latest market snapshot is incomplete or unsupported")
        for name in ("index_daily.csv", "etf_daily_raw.csv", "trade_cal.csv"):
            if digest(path.parent / name) != receipt["files"][name]:
                raise ValueError("Market snapshot identity mismatch")
        history = _prices(pd.read_csv(path.parent / "index_daily.csv"), asset=SIGNAL_ASSET)
        etf = _prices(pd.read_csv(path.parent / "etf_daily_raw.csv"), asset=EXECUTION_ASSET)
        cal = normalize_calendar(pd.read_csv(path.parent / "trade_cal.csv"))
        signal_day = history.index[-1]
        expected = _expected_sessions(cal, pd.Timestamp(HISTORY_START), signal_day)
        following = cal.loc[cal.is_open.eq(1) & cal.cal_date.gt(signal_day), "cal_date"]
        if (cal.cal_date.iloc[0].date() > HISTORY_START or not history.index.equals(expected)
                or len(history) < 620 or following.empty
                or etf.index[-1] != signal_day or not etf.index.isin(expected).all()
                or receipt.get("signal_asof") != str(signal_day.date())
                or receipt.get("execution_date") != str(following.iloc[0].date())
                or type(receipt.get("index_rows")) is not int or receipt["index_rows"] != len(history)
                or type(receipt.get("etf_rows")) is not int or receipt["etf_rows"] != len(etf)):
            raise ValueError("Market snapshot dates or counts differ from official sessions")
        return MarketSnapshot(path.parent, history, etf, cal, receipt)
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        raise ValueError("Market snapshot metadata or files are unreadable") from None


def _expected_sessions(calendar, start, end):
    return pd.DatetimeIndex(calendar.loc[
        calendar.is_open.eq(1) & calendar.cal_date.between(start, end), "cal_date"])


def _query_prices(provider, api, *, asset, start, end):
    try:
        raw = provider.query(
            api, {"ts_code": asset, "start_date": start.strftime("%Y%m%d"),
                  "end_date": end.strftime("%Y%m%d")},
            "ts_code,trade_date,open,high,low,close")
    except Exception:
        raise ValueError("Market provider request failed") from None
    return _prices(raw, asset=asset)


def _load_checkpoint(receipt_path, data_path, expected):
    try:
        record = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or digest(data_path) != record.get("sha256"):
            raise ValueError("Index acquisition checkpoint corrupted")
        acquired = _prices(pd.read_parquet(data_path), asset=SIGNAL_ASSET)
        if type(record.get("rows")) is not int or len(acquired) != record["rows"]:
            raise ValueError("Index acquisition checkpoint count mismatch")
        if not acquired.index.equals(expected):
            raise ValueError("Index checkpoint differs from official sessions")
        return acquired
    except (OSError, TypeError, json.JSONDecodeError):
        raise ValueError("Index acquisition checkpoint unreadable") from None


def _check_overlap(history, acquired):
    if history is None:
        return
    overlap = history.index.intersection(acquired.index)
    if len(overlap) and not np.allclose(
            history.loc[overlap], acquired.loc[overlap], atol=1e-9, rtol=0):
        raise ValueError("Published historical index data changed; refusing silent splice")


def refresh_incremental_market(output_root, *, provider, calendar, now):
    """Append at most one year per request, or explicitly rebuild known legacy data.

    The schema-1 mixed-preload snapshot can bootstrap a pure Tushare series only
    by reacquiring the whole anchored index history. Its prices are never
    spliced into the rebuilt series. Old snapshots remain unchanged, and the
    transition receipt persists into subsequent incremental snapshots.
    """
    output_root = Path(output_root)
    local = _now(now)
    cal = normalize_calendar(calendar)
    if cal.cal_date.iloc[0].date() > HISTORY_START:
        raise ValueError("Calendar must cover the fixed 2019 history anchor")
    formation_clock = signal_clock(local)
    signal_day, execution_day = session_dates(cal, now=formation_clock)
    previous = latest_verified_market(output_root)
    transition = None
    if previous is not None:
        source = previous.manifest.get("index_source")
        migrate = (previous.manifest.get("schema_version") == 1 and source == LEGACY_PRELOAD_SOURCE)
        if not migrate and source not in (HOMOGENEOUS_SOURCE, INITIAL_HOMOGENEOUS_SOURCE):
            raise ValueError("Continuous history requires a homogeneous Tushare bootstrap")
        history = previous.index_prices
        if history.index[-1].date() > signal_day:
            raise ValueError("Cannot roll the active market history backwards")
        old_calendar = previous.calendar.set_index("cal_date")
        new_calendar = cal.set_index("cal_date")
        frozen = old_calendar.index[old_calendar.index <= pd.Timestamp(local.date())]
        if (not frozen.isin(new_calendar.index).all()
                or not old_calendar.loc[frozen, "is_open"].equals(new_calendar.loc[frozen, "is_open"])):
            raise ValueError("Published historical calendar changed; refusing silent splice")
        if migrate:
            transition = {
                "kind": "legacy_mixed_preload_to_homogeneous_tushare",
                "history_start": str(HISTORY_START),
                "previous_manifest_sha256": digest(previous.directory / "manifest.json"),
                "previous_index_source": source,
                "previous_signal_asof": previous.manifest["signal_asof"],
                "previous_index_rows": len(history),
                "rebuild_policy": "full_reacquisition_no_splice",
            }
            history = None
            start = pd.Timestamp(HISTORY_START)
        else:
            transition = previous.manifest.get("bootstrap_transition")
            if transition is not None and not isinstance(transition, dict):
                raise ValueError("Market bootstrap transition metadata is invalid")
            start = history.index[-1]
    else:
        history = None
        start = pd.Timestamp(HISTORY_START)
    checkpoint_root = output_root / "_index_checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    acquired_rows = 0
    while start <= pd.Timestamp(signal_day) and (history is None or history.index[-1].date() < signal_day):
        end = min(pd.Timestamp(signal_day), pd.Timestamp(start.year, 12, 31))
        expected = _expected_sessions(cal, start, end)
        key = f"{start:%Y%m%d}_{end:%Y%m%d}"
        receipt_path = checkpoint_root / (key + ".json")
        data_path = checkpoint_root / (key + ".parquet")
        # A complete official calendar can prove an empty interval. Do not call
        # a price endpoint or turn an empty price response into that proof.
        if expected.empty:
            if receipt_path.exists():
                _load_checkpoint(receipt_path, data_path, expected)
            start = end + pd.Timedelta(days=1)
            continue
        if receipt_path.exists():
            acquired = _load_checkpoint(receipt_path, data_path, expected)
            _check_overlap(history, acquired)
        else:
            acquired = _query_prices(provider, "index_daily", asset=SIGNAL_ASSET, start=start, end=end)
            if not acquired.index.equals(expected):
                raise ValueError("Index chunk has missing or extra official sessions")
            _check_overlap(history, acquired)
            temp = data_path.with_suffix(".pending")
            acquired.to_parquet(temp)
            temp.replace(data_path)
            write_json(receipt_path, {"sha256": digest(data_path), "rows": len(acquired)})
            acquired_rows += len(acquired)
        if history is None:
            history = acquired
        else:
            history = pd.concat([history, acquired.loc[acquired.index > history.index[-1]]])
        start = end + pd.Timedelta(days=1)
    if history is None:
        raise ValueError("Index history unavailable")
    if previous is not None and previous.etf_daily.index[-1].date() == signal_day:
        etf = previous.etf_daily
    else:
        etf = _query_prices(provider, "fund_daily", asset=EXECUTION_ASSET,
                            start=signal_day, end=signal_day)
    for strategy_id in STRATEGY_IDS:
        prepare_signal_bundle(strategy_id, history, cal, now=formation_clock, etf_daily=etf)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = output_root / stamp
    folder.mkdir(parents=True, exist_ok=False)
    history.to_csv(folder / "index_daily.csv", index=True)
    etf.to_csv(folder / "etf_daily_raw.csv", index=True)
    cal.to_csv(folder / "trade_cal.csv", index=False, date_format="%Y-%m-%d")
    manifest = {
        "schema_version": 2, "ready": True, "created_at": local.astimezone(timezone.utc).isoformat(),
        "signal_asof": str(signal_day), "execution_date": str(execution_day),
        "signal_asset": SIGNAL_ASSET, "execution_asset": EXECUTION_ASSET,
        "index_source": HOMOGENEOUS_SOURCE,
        "index_rows": len(history), "acquired_index_rows": acquired_rows, "etf_rows": len(etf),
        "mapping_notice": MAPPING_NOTICE, "files": {
            name: digest(folder / name) for name in ("index_daily.csv", "etf_daily_raw.csv", "trade_cal.csv")},
    }
    if transition is not None:
        manifest["bootstrap_transition"] = transition
    write_json(folder / "manifest.json", manifest)
    return MarketSnapshot(folder, history, etf, cal, manifest)
