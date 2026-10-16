"""Fetch a small public index snapshot, following AKShare's documented Tencent API.

No credentials needed. Cached raw responses and checksums make the experiment
auditable. Endpoint/column contract reference: akshare/index/index_stock_zh.py,
stock_zh_index_daily_tx; implementation here is independent.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import time
import pandas as pd
import requests

ENDPOINT = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"


def fetch(config_path: Path, output: Path) -> dict:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "raw"
    raw.mkdir(exist_ok=True)
    manifest = {"retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "provider": "Tencent Finance public daily index quotes",
                "endpoint": ENDPOINT, "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "volume_unit": "vendor amount column, hands according to AKShare; unused in core study",
                "files": {}}
    with requests.Session() as session:
        for symbol in cfg["symbols"]:
            rows = []
            for year in range(int(cfg["download_start"][:4]), int(cfg["download_end"][:4]) + 1, 2):
                cache = raw / f"{symbol}_{year}.json"
                if cache.exists():
                    payload = json.loads(cache.read_text(encoding="utf-8"))
                else:
                    params = {"param": f"{symbol},day,{year}-01-01,{year+1}-12-31,640,qfq"}
                    response = session.get(ENDPOINT, params=params, timeout=35)
                    response.raise_for_status()
                    payload = response.json()
                    if payload.get("code") != 0:
                        raise ValueError(f"Provider error for {symbol}/{year}")
                    cache.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                    time.sleep(0.2)
                rows.extend([r[:6] for r in payload["data"][symbol].get("day", [])])
            df = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "volume"])
            for col in df.columns[1:]:
                df[col] = pd.to_numeric(df[col], errors="raise")
            df = df.drop_duplicates().sort_values("date")
            df = df[df.date.between(cfg["download_start"], cfg["download_end"])].reset_index(drop=True)
            if df.empty or df.date.duplicated().any() or df.isna().any().any():
                raise ValueError(f"Missing/duplicate observations for {symbol}")
            if not ((df.low <= df[["open", "close"]].min(axis=1)) &
                    (df.high >= df[["open", "close"]].max(axis=1)) & (df.low > 0)).all():
                raise ValueError(f"Invalid OHLC for {symbol}")
            target = output / f"{symbol}.csv"
            df.to_csv(target, index=False, lineterminator="\n")
            manifest["files"][target.name] = {"rows": len(df), "start": df.date.iloc[0], "end": df.date.iloc[-1],
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
            print(symbol, len(df), df.date.iloc[0], df.date.iloc[-1], flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fetch(args.config, args.output)
