"""Offline replay of >40 actual historical sessions, explicitly not live acquisition."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from cfquant.data import load_market, load_calendar
from cfquant.equity_forward import append_daily_bars
from cfquant.equity_data_store import EquityDataStore

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-root', type=Path, required=True, help='Existing Qingxu private research checkout')
parser.add_argument('--split', default='2026-06-30', help='Retain data through this date and replay all later sessions')
parser.add_argument('--summary', type=Path, default=ROOT / 'runs/continuous_offline_replay/last_acceptance.json')
args = parser.parse_args()
desktop = args.data_root.resolve()
source = desktop / "data/private/mainboard1000_20260918/data/processed"
full = load_market(source / "market.csv")
basic = pd.read_parquet(desktop / "data/private/research_v2/daily_basic.parquet")
calendar = load_calendar(source / "calendar.csv")
split = pd.Timestamp(args.split)
target = full.date.max()
base = full[full.date <= split].copy()
base_basic = basic[basic.trade_date.astype(str) <= split.strftime("%Y%m%d")].copy()
days = calendar[(calendar > split) & (calendar <= target)]
assert len(days) > 40
prices = {day.strftime("%Y%m%d"): rows for day, rows in full[full.date > split].groupby("date")}
basics = {str(day): rows for day, rows in basic[basic.trade_date.astype(str) > split.strftime("%Y%m%d")].groupby("trade_date")}
class ReplayProvider:
    def __init__(self):
        self.calls = 0
        self.fail_once = True
    def query(self, api, params, fields):
        self.calls += 1
        day = params["trade_date"]
        if self.fail_once and day == days[24].strftime("%Y%m%d") and api == "daily_basic":
            self.fail_once = False
            raise RuntimeError("deliberate offline replay interruption")
        if api == "daily_basic":
            rows = basics[day].copy()
        else:
            part = prices[day]
            rows = pd.DataFrame({
                "ts_code": part.asset, "trade_date": day,
                **{name: part["raw_" + name] for name in ("open", "high", "low", "close")},
                "vol": part.volume / 100, "adj_factor": part.adj_factor,
                "up_limit": part.up_limit, "down_limit": part.down_limit,
            })
        return rows.loc[:, fields.split(",")].iloc[params["offset"]:params["offset"] + params["limit"]].reset_index(drop=True)

started = time.monotonic()
provider = ReplayProvider()
folder = ROOT / "runs/continuous_offline_replay" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
store = EquityDataStore(folder, source_identity={"kind": "offline_real_history_replay"},
                        base_end=split, assets=sorted(full.asset.unique()))
interrupted = False
for batch in range(10):
    try:
        market, _, progress = store.refresh(provider, base, base_basic, calendar, target,
            append_bars=append_daily_bars, max_sessions=20)
        if progress["ready"]:
            break
    except RuntimeError:
        interrupted = True
        # A new store object exercises persistence rather than retaining a memory cursor.
        store = EquityDataStore(folder, source_identity={"kind": "offline_real_history_replay"},
                                base_end=split, assets=sorted(full.asset.unique()))
else:
    raise RuntimeError("Offline replay budget exhausted")
assert interrupted
expected = full.sort_values(["date", "asset"]).reset_index(drop=True)
actual = market.sort_values(["date", "asset"]).reset_index(drop=True)
# Tushare reports volume in lots as decimals; converting back to shares leaves
# a floating dtype even when this historical CSV stored integral share counts.
expected["volume"] = expected["volume"].astype(float)
# The immutable source CSV was exported with %.12g; recomputing from its raw
# price/factor columns legitimately differs at its last significant digit.
pd.testing.assert_frame_equal(actual, expected, check_exact=False, atol=1e-8, rtol=6e-12)
price_difference = float((actual[["open", "high", "low", "close"]]
                          - expected[["open", "high", "low", "close"]]).abs().max().max())
before = provider.calls
store.refresh(provider, base, base_basic, calendar, target, append_bars=append_daily_bars)
assert provider.calls == before
result = {
    "mode": "offline_replay_of_real_historical_snapshot",
    "live_api_requests": 0, "original_snapshot_modified": False,
    "start_exclusive": str(split.date()), "through": str(target.date()),
    "sessions_replayed": len(days), "stock_rows_replayed": int((full.date > split).sum()),
    "deliberate_mid_session_interruption_recovered": True,
    "reconstructed_market_matches_original_export_precision": True,
    "maximum_adjusted_price_difference": price_difference,
    "source_csv_numeric_format": "%.12g",
    "complete_rerun_provider_calls": provider.calls - before,
    "elapsed_seconds": round(time.monotonic() - started, 2),
}
args.summary.parent.mkdir(parents=True, exist_ok=True)
args.summary.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result))
