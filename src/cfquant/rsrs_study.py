"""Frozen Project 2 research matrix; public aggregates, private full ledgers.

No parameter search or winning-candidate replacement occurs here. Input hashes
are checked before computation. Resumption reuses only outputs with the same
protocol AND calculation-source fingerprint. Every new run reconciles the P1
cash/units ledger; binary baselines also face an independent legacy oracle.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from .rsrs import SPECS, run_rsrs

INDEX_NAMES = {"sh000300": "沪深300", "sh000016": "上证50", "sh000905": "中证500"}
PERIOD_NAMES = {"historical_overlap": "历史重叠期", "post_publication": "发表后时期", "recent": "近年时期"}
STRATEGY_NAMES = {"buy_hold": "买入持有", "ma20": "单独MA20", "slope": "原始斜率",
                  "standard": "标准分", "corrected": "修正标准分", "right": "右偏标准分",
                  "standard_ma": "标准分+MA20", "corrected_ma": "修正分+MA20",
                  "right_ma": "右偏分+MA20", "right_ma_vol15": "右偏+15%波动目标",
                  "right_ma_cash75": "右偏+固定75%仓位", "right_ma_vol10": "右偏+10%波动目标",
                  "right_ma_vol20": "右偏+20%波动目标"}
STRATEGIES = ["buy_hold", "ma20"] + [s.name for s in SPECS] + ["right_ma_vol15", "right_ma_cash75"]
SOURCE_FILES = ["src/cfquant/rsrs.py", "src/cfquant/rsrs_study.py", "src/cfquant/rsrs_diagnostics.py",
                "src/cfquant/engine.py", "src/cfquant/analytics.py", "projects/project2_rsrs/rsrs.py"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, payload):
    def convert(value):
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        if isinstance(value, (Path, pd.Timestamp)):
            return str(value)
        raise TypeError(type(value).__name__)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=convert) + "\n", encoding="utf-8")


def load_data(path):
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    if frame.index.has_duplicates or frame.index.hasnans or frame.empty:
        raise ValueError("Invalid calendar")
    fields = frame[["open", "high", "low", "close"]]
    if not np.isfinite(fields).all().all() or (fields <= 0).any().any():
        raise ValueError("Invalid OHLC values")
    if ((frame.high < fields.max(axis=1) - 1e-8) | (frame.low > fields.min(axis=1) + 1e-8)).any():
        raise ValueError("OHLC ordering failed")
    return frame


def daily_public(run, benchmark):
    daily = run.result.daily.copy()
    initial = run.metadata["initial_cash"]
    nav = daily.nav.to_numpy() / initial
    dates = pd.DatetimeIndex(daily.date)
    b = benchmark.result.daily.set_index("date").nav.reindex(dates).to_numpy() / initial
    return pd.DataFrame({"date": dates, "nav": nav, "benchmark_nav": b,
                         "exposure": daily.market_value.to_numpy() / daily.nav.to_numpy(),
                         "drawdown": 1 - nav / np.maximum.accumulate(np.r_[1., nav])[1:],
                         "capital_nav": daily.nav.to_numpy()})


def _legacy(root):
    spec = importlib.util.spec_from_file_location("_p2_independent_legacy", root / "projects/project2_rsrs/rsrs.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _evaluate(data, strategy, start, end, cost, initial, asset):
    options = {}
    base = strategy
    if strategy.startswith("right_ma_vol"):
        base, options = "right_ma", {"vol_target": int(strategy.rsplit("vol", 1)[1]) / 100}
    elif strategy == "right_ma_cash75":
        base, options = "right_ma", {"exposure_scale": .75}
    return run_rsrs(data, base, start, end, cost_bps=cost, initial_cash=initial, asset=asset, **options)


def _index_worker(root_name, data_dir_name, output_name, private_name, index_id, config, run_id):
    root, output, private = Path(root_name), Path(output_name), Path(private_name)
    data = load_data(Path(data_dir_name) / f"{index_id}.csv")
    oracle = _legacy(root)
    rows, validations = [], []
    for period_id, (start, end) in config["periods"].items():
        for cost in config["cost_bps_each_side"]:
            benchmark = _evaluate(data, "buy_hold", start, end, cost, config["initial_cash"], index_id)
            for strategy in STRATEGIES:
                key = f"{index_id}__{period_id}__{strategy}__{cost}bps"
                checkpoint = private / key / "record.json"
                if checkpoint.exists():
                    cached = json.loads(checkpoint.read_text(encoding="utf-8"))
                    if cached["run_id"] != run_id:
                        raise ValueError("Checkpoint identity mismatch")
                    if digest(output / cached["row"]["daily_file"]) != cached["daily_sha256"]:
                        raise ValueError("Checkpoint daily evidence changed")
                    rows.append(cached["row"])
                    validations.append(cached["validation"])
                    continue
                run = benchmark if strategy == "buy_hold" else _evaluate(data, strategy, start, end, cost, config["initial_cash"], index_id)
                day = daily_public(run, benchmark)
                daily_file = f"daily/{key}.csv"
                (output / "daily").mkdir(parents=True, exist_ok=True)
                day.to_csv(output / daily_file, index=False, float_format="%.12g")
                local = private / key
                local.mkdir(parents=True, exist_ok=True)
                for name in ("daily", "trades", "positions", "orders"):
                    getattr(run.result, name).to_csv(local / f"{name}.csv", index=False)
                run.factors.to_csv(local / "factors.csv", index_label="date")
                run.targets.to_csv(local / "targets.csv", index_label="date")
                json_write(local / "metadata.json", run.metadata)
                metric = run.metrics
                row = {"index_id": index_id, "index_name": INDEX_NAMES[index_id],
                       "period_id": period_id, "period_name": PERIOD_NAMES[period_id],
                       "strategy_id": strategy, "strategy_name": STRATEGY_NAMES[strategy],
                       "symbol": index_id, "period": period_id, "strategy": strategy,
                       "cost_bps": cost, "daily_file": daily_file, **metric,
                       "cagr": metric["annualized_return"], "annual_turnover": metric["mean_turnover"] * 252}
                validation = {"key": key, **run.checks, "legacy_checked": strategy not in {"right_ma_vol15", "right_ma_cash75"}}
                if validation["legacy_checked"]:
                    old = oracle.backtest(data, run.decision, start, end, cost_bps=cost)
                    errors = {
                        "nav": np.max(np.abs(old.equity.to_numpy() - run.result.daily.nav.to_numpy() / config["initial_cash"])),
                        "cash": np.max(np.abs(old.cash.to_numpy() - run.result.daily.cash.to_numpy() / config["initial_cash"])),
                        "cost": np.max(np.abs(old.cost.to_numpy() - run.result.daily.cost.to_numpy() / config["initial_cash"])),
                        "returns": np.max(np.abs(old["return"].to_numpy() - run.result.daily["return"].to_numpy())),
                    }
                    validation["legacy_max_absolute_errors"] = errors
                    if max(errors.values()) > 1e-10:
                        raise ArithmeticError(f"Independent oracle mismatch: {key}: {errors}")
                rows.append(row)
                validations.append(validation)
                json_write(checkpoint, {"run_id": run_id, "row": row, "validation": validation,
                                        "daily_sha256": digest(output / daily_file)})
                print(f"{index_id}: {len(rows)}/66 {period_id} {strategy} {cost}bp", flush=True)
    return rows, validations


def paired_table(performance):
    rows = []
    for (idx, per, cost), group in performance.groupby(["index_id", "period_id", "cost_bps"]):
        table = group.set_index("strategy_id")
        row = {"index_id": idx, "index_name": INDEX_NAMES[idx], "period_id": per,
               "period_name": PERIOD_NAMES[per], "cost_bps": cost}
        for prefix, strategy in [("baseline", "right_ma"), ("improved", "right_ma_vol15"), ("cash75", "right_ma_cash75")]:
            for metric in ["total_return", "cagr", "sharpe", "max_drawdown", "annual_turnover", "exposure"]:
                row[f"{prefix}_{metric}"] = table.loc[strategy, metric]
        for metric in ["cagr", "sharpe", "max_drawdown"]:
            row[f"delta_{metric}"] = row[f"improved_{metric}"] - row[f"baseline_{metric}"]
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap(output, config):
    params = config["bootstrap"]
    rng = np.random.default_rng(params["seed"])
    def returns(strategy):
        p = output / f"daily/sh000300__post_publication__{strategy}__10bps.csv"
        nav = pd.read_csv(p).nav.to_numpy()
        return np.diff(np.r_[1., nav]) / np.r_[1., nav][:-1]
    improved = returns("right_ma_vol15")
    n, block = len(improved), params["block_sessions"]
    comparisons = []
    for base in ["right_ma", "right_ma_cash75"]:
        difference = improved - returns(base)
        samples = np.empty(params["repetitions"])
        for i in range(len(samples)):
            starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
            selected = (starts[:, None] + np.arange(block)).ravel()[:n]
            samples[i] = difference[selected].mean() * 252
        lo, hi = np.quantile(samples, [.025, .975])
        comparisons.append({"improvement": "right_ma_vol15", "baseline": base,
                            "annualized_mean_daily_return_difference": difference.mean() * 252,
                            "ci_low": lo, "ci_high": hi, "sessions": n})
    return {**params, "method": "paired non-circular moving-block bootstrap, percentile 95% interval",
            "metric": "252 times arithmetic mean of paired daily return differences; NOT CAGR difference",
            "limitations": "Dependent overlapping histories and previously observed sample; not independent confirmation or a multiple-testing-adjusted significance claim",
            "comparisons": comparisons}


def robustness(data_dir, output, config):
    rows = []
    start, end = config["periods"]["post_publication"]
    for index_id in config["symbols"]:
        data = load_data(data_dir / f"{index_id}.csv")
        for strategy, cost in [("right_ma_vol10", 10), ("right_ma_vol20", 10), ("right_ma", 20), ("right_ma_vol15", 20)]:
            run = _evaluate(data, strategy, start, end, cost, config["initial_cash"], index_id)
            rows.append({"index_id": index_id, "index_name": INDEX_NAMES[index_id], "period_id": "post_publication",
                         "strategy_id": strategy, "strategy_name": STRATEGY_NAMES[strategy], "cost_bps": cost,
                         **run.metrics, "cagr": run.metrics["annualized_return"],
                         "annual_turnover": run.metrics["mean_turnover"] * 252, "ledger_passed": run.checks["passed"]})
            print(f"Robustness: {index_id} {strategy} {cost}bp", flush=True)
    pd.DataFrame(rows).to_csv(output / "robustness.csv", index=False)


def reproduce_table(output, performance):
    source = pd.read_csv(output / "source_compatible.csv")
    rows = []
    original = {"standard_ma": (.2737, 1.76, .1548), "corrected_ma": (.2613, 1.64, .1576), "right_ma": (.2819, 1.56, .2231)}
    for strategy, values in original.items():
        common = performance.query("index_id=='sh000300' and period_id=='historical_overlap' and cost_bps==0 and strategy_id==@strategy").iloc[0]
        compatible = source.query("warmup=='partial_prior_M_min2' and end_assumption=='march_end' and cost_bps==0 and strategy_id==@strategy").iloc[0]
        row = {"strategy_id": strategy, "strategy_name": STRATEGY_NAMES[strategy], "direct_error_allowed": False,
               "original_cagr": values[0], "original_sharpe": values[1], "original_max_drawdown": values[2],
               "source_compatible_start": compatible.start, "source_compatible_end": compatible.end,
               "common_window_start": common.start, "common_window_end": common.end,
               "comparison_note": "Original Wind sample/execution/cost/annualization unresolved; approximation and common-window mechanism checks, not exact numeric replication errors"}
        for prefix, obs in [("source_compatible", compatible), ("common_window", common)]:
            for metric in ["cagr", "sharpe", "max_drawdown"]:
                row[f"{prefix}_{metric}"] = obs[metric]
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "reproduction_table.csv", index=False)


def run_study(root, data_dir, output, private, config_path, workers=3):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    identity = {"config_sha256": digest(config_path), "source_sha256": {p: digest(root / p) for p in SOURCE_FILES}}
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8"))["run_id"] != run_id:
        raise ValueError("Existing evidence belongs to different code/protocol. Preserve it and use a new output directory.")
    quality = []
    for index_id in config["symbols"]:
        name = f"{index_id}.csv"
        if digest(data_dir / name) != config["source_snapshot_sha256"][name]:
            raise ValueError(f"Input snapshot mismatch: {name}")
        data = load_data(data_dir / name)
        quality.append({"index_id": index_id, "rows": len(data), "first": str(data.index[0].date()),
                        "last": str(data.index[-1].date()), "missing_ohlc": 0, "duplicate_dates": 0,
                        "ohlc_ordering_passed": True, "sha256": digest(data_dir / name)})
    json_write(identity_path, {"run_id": run_id, **identity})
    json_write(output / "protocol.json", config)
    json_write(output / "data_quality.json", quality)
    all_rows, all_checks = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_index_worker, str(root), str(data_dir), str(output), str(private), idx, config, run_id)
                   for idx in config["symbols"]]
        for future in as_completed(futures):
            rows, checks = future.result()
            all_rows.extend(rows)
            all_checks.extend(checks)
    performance = pd.DataFrame(all_rows).sort_values(["index_id", "period_id", "cost_bps", "strategy_id"])
    performance.to_csv(output / "performance.csv", index=False)
    paired_table(performance).to_csv(output / "paired_comparison.csv", index=False)
    json_write(output / "bootstrap.json", paired_bootstrap(output, config))
    from .rsrs_diagnostics import run_diagnostics
    diagnostic = run_diagnostics(load_data(data_dir / "sh000300.csv"), "sh000300", output)
    robustness(data_dir, output, config)
    reproduce_table(output, performance)
    json_write(output / "validation.json", {"all_ledgers_passed": all(v["passed"] for v in all_checks),
               "main_runs": len(all_rows), "legacy_oracle_runs": sum(v["legacy_checked"] for v in all_checks),
               "main_checked_sessions": sum(v["checked_sessions"] for v in all_checks),
               "legacy_worst_absolute_error": max(max(v.get("legacy_max_absolute_errors", {"n": 0}).values()) for v in all_checks),
               "diagnostics": diagnostic, "runs": all_checks})
    try:
        git_revision = subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_revision = "unavailable; exact calculation identities recorded separately"
    files = {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    json_write(output / "manifest.json", {"schema": "qingxu.project2.rsrs.evidence.v1", "run_id": run_id,
              "created_utc": datetime.now(timezone.utc).isoformat(), "git_revision_at_run": git_revision,
              "data_end": config["data_end"], "main_runs": len(all_rows), "historical_results_already_observed": True,
              "independent_blind_test": False, "index_proxy_not_live_instrument": True,
              "data_snapshot_sha256": config["source_snapshot_sha256"], **identity, "files": files})
    print(f"Formal study complete: {len(all_rows)} main runs; evidence {output}", flush=True)
    return performance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    root = args.root.resolve()
    run_study(root, args.data or root / "data/private/project2_rsrs", args.output or root / "evidence/project2_rsrs",
              args.private_output or root / "runs/project2_rsrs_formal", args.config or root / "configs/project2_rsrs.json", args.workers)


if __name__ == "__main__":
    main()
