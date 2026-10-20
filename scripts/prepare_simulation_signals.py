"""Refresh read-only Tushare inputs and freeze next-session simulation targets."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from cfquant.simulation_signals import (
    STRATEGY_IDS, prepare_signal_bundle, refresh_market_data, save_signal_bundle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/simulation_auto"))
    parser.add_argument("--token-file", default="D:/CodexData/private/qingxu/tushare_token.txt")
    parser.add_argument("--preload", default=
                        "D:/Desktop/cf2026-quant-platform/data/private/project2_rsrs/sh000300.csv")
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    try:
        market = refresh_market_data(args.output / "market", token_file=args.token_file,
                                     now=now, preload_path=args.preload or None)
        bundles = []
        for strategy_id in STRATEGY_IDS:
            bundle = prepare_signal_bundle(strategy_id, market.index_prices, market.calendar,
                                            now=now, etf_daily=market.etf_daily)
            path = save_signal_bundle(bundle, args.output / "signals")
            bundles.append({"strategy_id": strategy_id, "signal_asof": str(bundle.signal_asof),
                            "execution_date": str(bundle.execution_date),
                            "target_weights": bundle.target_weights, "path": str(path.resolve())})
        print(json.dumps({"ready": True, "orders_submitted": 0,
                          "market": str(market.directory.resolve()), "signals": bundles},
                         ensure_ascii=False, indent=2))
    except (ValueError, RuntimeError, OSError) as exc:
        # Provider errors are already sanitized; never print requests or credentials.
        print(json.dumps({"ready": False, "orders_submitted": 0, "reason": str(exc)},
                         ensure_ascii=False))
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
