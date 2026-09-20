from __future__ import annotations
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import pandas as pd
import yaml
from .config import Config
from .data import load_market, load_calendar, digest, write_json
from .factors import REGISTRY, compute, to_long, causal_check
from .engine import run_backtest
from .analytics import forward_returns, diagnostics, performance, reconcile


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def execute(root: Path, config: Config, output: Path | None = None):
    begin = time.monotonic()
    market = load_market(root/config.data_path)
    calendar = load_calendar(root/config.calendar_path)
    scores = compute(market, calendar, config.factor, config.factor_params)
    result = run_backtest(market, calendar, scores, config)
    metrics = performance(result.daily, config.initial_cash, config.annualization, config.risk_free_annual, result.trades)
    checks = reconcile(result, config.initial_cash, config.buy_cost, config.sell_cost)
    if not checks["passed"]:
        raise AssertionError(f"Accounting verification failed: {checks}")
    code_files = sorted((root/"src").rglob("*.py"))
    code_hashes = {str(p.relative_to(root)).replace("\\", "/"): digest(p) for p in code_files}
    identity = {"config":config.to_dict(), "data":digest(root/config.data_path),
                "calendar":digest(root/config.calendar_path), "code":code_hashes}
    signature = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
    output = output or root / "runs" / f"{config.name}_{signature}"
    output.mkdir(parents=True, exist_ok=True)
    for name in ["daily", "trades", "positions", "orders"]:
        getattr(result, name).to_csv(output/f"{name}.csv", index=False, float_format="%.12g")
    to_long(scores, config.factor).to_csv(output/"factors.csv", index=False, float_format="%.12g")
    (output/"config.yaml").write_text(yaml.safe_dump(config.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_json(output/"metrics.json", json_safe(metrics))
    write_json(output/"checks.json", checks)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--",
                     "src", "configs", "app.py", "scripts", "pyproject.toml"], cwd=root, text=True).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision, dirty = "unavailable", True
    write_json(output/"provenance.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
               "duration_seconds": time.monotonic()-begin, "git_revision": revision, "git_dirty": dirty,
               "git_dirty_scope": "research source, configs, app, scripts and pyproject; excludes generated reports/evidence",
               "data_sha256": digest(root/config.data_path), "calendar_sha256": digest(root/config.calendar_path),
               "code_sha256": code_hashes,
               "config_signature": signature,
               "gross_convention": "Same actual fills; cumulative costs restored into idle cash; not a separately reinvested frictionless portfolio."})
    return result, metrics, output


def study(root: Path, base: Config):
    """Predefined comparison; no performance-based parameter search."""
    output = root/"runs/study"
    output.mkdir(parents=True, exist_ok=True)
    market = load_market(root/base.data_path)
    calendar = load_calendar(root/base.calendar_path)
    # Deliberately restrict labels to the experiment end: no results after study cutoff.
    analysis_calendar = calendar[calendar <= base.end]
    analysis_market = market[market.date <= base.end]
    labels = forward_returns(analysis_market, analysis_calendar, base.diagnostic_horizon)
    summaries, checks, rows = [], [], []
    run_index = {}
    for name, spec in REGISTRY.items():
        config = replace(base, name=f"{name}_weekly", factor=name, factor_params={"window": spec.default_window})
        result, metrics, folder = execute(root, config)
        run_index[config.name] = folder.relative_to(root).as_posix()
        rows.append({"experiment": config.name, "factor": name, "rebalance_every": config.rebalance_every, **metrics})
        scores = compute(analysis_market, analysis_calendar, name)
        scores = scores.loc[(scores.index >= base.start) & (scores.index <= base.end)]
        diag = diagnostics(scores, labels, base.groups, base.min_assets)
        for key in ["daily", "groups", "spread", "assignments"]:
            diag[key].to_csv(output/f"{name}_{key}.csv", index=False, float_format="%.12g")
        summaries.append({"factor": name, **diag["summary"]})
        checks.append(causal_check(market, calendar, name))
    # One primary factor changed: 5-session versus 20-session rebalance.
    monthly = replace(base, name="momentum_monthly", rebalance_every=20)
    _, metrics, monthly_folder = execute(root, monthly)
    run_index[monthly.name] = monthly_folder.relative_to(root).as_posix()
    rows.append({"experiment": monthly.name, "factor": monthly.factor, "rebalance_every": 20, **metrics})
    # Repeat the primary baseline numerically, with the same data/config.
    first, _, _ = execute(root, base, output/"repeat_a")
    second, _, _ = execute(root, base, output/"repeat_b")
    repeatable = first.daily.equals(second.daily) and first.trades.equals(second.trades)
    checks.append({"factor": "reproducibility", "passed": repeatable, "errors": [] if repeatable else ["not identical"]})
    pd.DataFrame(rows).to_csv(output/"comparison.csv", index=False, float_format="%.12g")
    pd.DataFrame(summaries).to_csv(output/"factor_summary.csv", index=False, float_format="%.12g")
    write_json(output/"validation.json", {"passed": all(x["passed"] for x in checks), "checks": checks})
    write_json(output/"run_index.json",run_index)
    return output
