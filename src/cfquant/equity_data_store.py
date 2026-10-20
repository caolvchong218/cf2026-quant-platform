"""Checkpointed equity market ingestion; committed sessions are immutable.

The historical research snapshot is a read-only bootstrap. Each new session is
published only after all four provider tables and their joins pass validation.
An interrupted invocation therefore resumes from its last complete session.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .data import digest, write_json

DAILY_FIELDS = {
    "daily": "ts_code,trade_date,open,high,low,close,vol",
    "adj_factor": "ts_code,trade_date,adj_factor",
    "stk_limit": "ts_code,trade_date,up_limit,down_limit",
    "daily_basic": "ts_code,trade_date,total_mv,pe_ttm,pb,dv_ttm,turnover_rate",
}


def query_pages(provider, api, params, fields, *, page_size=4000, max_pages=100):
    """Request explicit pages, rejecting duplicate/ignored offsets and truncation."""
    parts, seen = [], set()
    for number in range(max_pages):
        result = provider.query(api, {**params, "limit": page_size,
                                      "offset": number * page_size}, fields)
        if not set(fields.split(",")) <= set(result):
            raise ValueError(f"{api}: missing response columns")
        if result.empty:
            break
        if len(result) > page_size:
            raise ValueError(f"{api}: provider ignored requested page size")
        keys = list(zip(result.ts_code.astype(str), result.trade_date.astype(str)))
        if len(set(keys)) != len(keys) or seen.intersection(keys):
            raise ValueError(f"{api}: duplicate rows or ignored pagination offset")
        seen.update(keys)
        parts.append(result)
        if len(result) < page_size:
            break
    else:
        raise ValueError(f"{api}: acquisition page budget exhausted")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=fields.split(","))


class EquityDataStore:
    def __init__(self, root, *, source_identity, base_end, assets):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.assets = set(assets)
        self.base_end = pd.Timestamp(base_end).normalize()
        identity = {"schema": 1, "source_identity": source_identity,
                    "base_end": str(self.base_end.date()), "assets": sorted(self.assets)}
        path = self.root / "identity.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != identity:
                raise ValueError("Incremental market store bootstrap identity changed")
        else:
            write_json(path, identity)

    def _read(self, day):
        folder = self.root / pd.Timestamp(day).strftime("%Y%m%d")
        path = folder / "commit.json"
        if not path.exists():
            return None
        commit = json.loads(path.read_text(encoding="utf-8"))
        if commit["date"] != str(pd.Timestamp(day).date()):
            raise ValueError("Market checkpoint date mismatch")
        for name, expected in commit["files"].items():
            if name not in {"market.parquet", "basic.parquet"} or digest(folder / name) != expected:
                raise ValueError("Market checkpoint identity mismatch")
        if set(commit["files"]) != {"market.parquet", "basic.parquet"}:
            raise ValueError("Incomplete market checkpoint")
        return pd.read_parquet(folder / "market.parquet"), pd.read_parquet(folder / "basic.parquet")

    def _commit(self, day, market, basic, *, origin):
        day = pd.Timestamp(day)
        if day <= self.base_end or market.empty or basic.empty:
            raise ValueError("Empty or non-forward daily market checkpoint")
        if (not pd.to_datetime(market.date).eq(day).all()
                or market.asset.duplicated().any()
                or not set(market.asset) <= self.assets):
            raise ValueError("Invalid checkpoint market keys")
        if (set(basic.trade_date.astype(str)) != {day.strftime("%Y%m%d")}
                or basic.ts_code.duplicated().any()
                or not set(basic.ts_code) <= self.assets
                or not set(market.asset) <= set(basic.ts_code)):
            raise ValueError("Daily valuation data do not cover traded equities")
        existing = self._read(day)
        if existing is not None:
            pd.testing.assert_frame_equal(existing[0], market.reset_index(drop=True))
            pd.testing.assert_frame_equal(existing[1], basic.reset_index(drop=True))
            return
        folder = self.root / day.strftime("%Y%m%d")
        folder.mkdir(parents=True, exist_ok=True)
        for name, frame in (("market.parquet", market), ("basic.parquet", basic)):
            temp = folder / (name + ".tmp")
            frame.reset_index(drop=True).to_parquet(temp, index=False)
            temp.replace(folder / name)
        write_json(folder / "commit.json", {
            "schema": 1, "date": str(day.date()), "origin": origin,
            "rows": len(market), "files": {
                name: digest(folder / name) for name in ("market.parquet", "basic.parquet")},
        })

    def import_legacy(self, bundle_folder):
        """Seed verified v2.5 extensions without another provider download."""
        folder = Path(bundle_folder)
        receipt = json.loads((folder / "receipt.json").read_text(encoding="utf-8"))
        names = ("market_extension.parquet", "basic_extension.parquet")
        if receipt.get("scores_ready") is not True:
            raise ValueError("Legacy migration requires a verified completed score bundle")
        for name in names:
            if digest(folder / name) != receipt["files_sha256"][name]:
                raise ValueError("Legacy market extension hash mismatch")
        market, basic = [pd.read_parquet(folder / name) for name in names]
        for day, piece in market.groupby("date", sort=True):
            if self._read(day) is None:
                day_basic = basic[basic.trade_date.astype(str).eq(pd.Timestamp(day).strftime("%Y%m%d"))]
                self._commit(day, piece, day_basic, origin="verified_v2.5_extension")

    def load(self, base_market, base_basic, calendar, target):
        dates = pd.DatetimeIndex(calendar)
        dates = dates[(dates > self.base_end) & (dates <= pd.Timestamp(target))]
        market, basic, missing = [base_market], [base_basic], []
        for day in dates:
            part = self._read(day)
            if part is None:
                missing.append(day)
            else:
                market.append(part[0])
                basic.append(part[1])
        return (pd.concat(market, ignore_index=True), pd.concat(basic, ignore_index=True),
                pd.DatetimeIndex(missing))

    def refresh(self, provider, base_market, base_basic, calendar, target, *,
                append_bars, max_sessions=20):
        """Bound work per invocation, not the total history the store supports."""
        if type(max_sessions) is not int or max_sessions < 1:
            raise ValueError("Session budget must be a positive integer")
        market, basic, missing = self.load(base_market, base_basic, calendar, target)
        # Anchors are constant; pass only their last rows to the pure converter.
        anchors = base_market.sort_values("date").groupby("asset", sort=False).tail(1)
        completed = []
        for day in missing[:max_sessions]:
            fetched = {}
            for api, fields in DAILY_FIELDS.items():
                result = query_pages(provider, api, {"trade_date": day.strftime("%Y%m%d")}, fields)
                if result.empty or set(result.trade_date.astype(str)) != {day.strftime("%Y%m%d")}:
                    raise ValueError(f"{api}: empty or mismatched daily response")
                result = result[result.ts_code.isin(self.assets)].copy()
                if result.empty:
                    raise ValueError(f"{api}: no configured assets returned")
                fetched[api] = result
            _, extension = append_bars(anchors, fetched["daily"], fetched["adj_factor"], fetched["stk_limit"])
            self._commit(day, extension, fetched["daily_basic"], origin="tushare_daily_increment")
            completed.append(day)
        market, basic, remaining = self.load(base_market, base_basic, calendar, target)
        return market, basic, {
            "target_date": str(pd.Timestamp(target).date()),
            "completed_this_run": len(completed), "remaining_sessions": len(remaining),
            "ready": len(remaining) == 0,
            "next_missing_date": str(remaining[0].date()) if len(remaining) else None,
            "total_extension_sessions": len(market.loc[market.date > self.base_end, "date"].unique()),
        }


def save_feature_day(root, frame, *, reference_identity):
    """Freeze causal daily features separately from all future-return labels."""
    root = Path(root)
    if frame.empty or pd.to_datetime(frame.date).nunique() != 1 or frame.asset.duplicated().any():
        raise ValueError("Features must contain one date and unique assets")
    if {"label", "label_exit", "label_entry"}.intersection(frame):
        raise ValueError("Forward feature inputs must never contain labels")
    day = pd.Timestamp(frame.date.iloc[0])
    folder = root / day.strftime("%Y%m%d")
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path = folder / "commit.json"
    if receipt_path.exists():
        saved = json.loads(receipt_path.read_text(encoding="utf-8"))
        if digest(folder / "features.parquet") != saved["sha256"]:
            raise ValueError("Frozen daily feature identity mismatch")
        return pd.read_parquet(folder / "features.parquet")
    temp = folder / "features.parquet.tmp"
    frame.to_parquet(temp, index=False)
    temp.replace(folder / "features.parquet")
    write_json(receipt_path, {"date": str(day.date()), "rows": len(frame),
                             "reference_identity": reference_identity,
                             "sha256": digest(folder / "features.parquet")})
    return frame


def load_feature_days(root, *, through=None):
    frames, identities = [], {}
    for path in sorted(Path(root).glob("*/commit.json")):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if through is not None and pd.Timestamp(receipt["date"]) >= pd.Timestamp(through):
            continue
        source = path.parent / "features.parquet"
        if digest(source) != receipt["sha256"]:
            raise ValueError("Daily feature checkpoint corrupted")
        frames.append(pd.read_parquet(source))
        identities[receipt["date"]] = receipt["sha256"]
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), identities
