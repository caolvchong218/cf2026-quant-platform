"""Prepare private, dated forward-equity research inputs; never submit orders."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cfquant.equity_forward import prepare_equity_forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Existing Qingxu checkout with private research inputs")
    parser.add_argument("--output", type=Path, help="Private output directory, default runs/simulation_auto/equities")
    parser.add_argument("--asof", required=True, help="Latest requested closed signal session, YYYY-MM-DD")
    parser.add_argument("--token-file", default="D:/CodexData/private/qingxu/tushare_token.txt",
                        help="Local Tushare token file; never place the token on the command line")
    parser.add_argument("--offline", action="store_true", help="Use saved data without network requests; not authorized for automatic orders")
    args = parser.parse_args()
    output = args.output or args.data_root / "runs/simulation_auto/equities"
    result = prepare_equity_forward(args.data_root, output, args.asof,
                                    token_file=args.token_file, refresh=not args.offline)
    print(json.dumps({name: result.get(name) for name in (
        "status", "snapshot_directory", "signal_date", "extension_sessions",
        "extension_rows", "eligible_assets", "score_coverage", "scores_ready",
        "ready_for_auto_orders", "blockers", "market_progress", "reference_status", "progress_count")},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
