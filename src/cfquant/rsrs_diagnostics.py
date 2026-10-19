"""Prespecified RSRS interpretation and source-compatibility diagnostics.

These experiments explain implementation differences; they do not select a
strategy or alter the strict-warmup main study. Every portfolio uses the shared
Qingxu engine, actual next-session fills, fees and independent reconciliation.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .rsrs import SPECS, Specification, decisions, factors, run_signal


PERIODS = (
    ("historical_overlap", "历史重叠期", "2007-08-01", "2017-03-31"),
    ("post_publication", "发表后历史期", "2017-05-02", "2026-10-15"),
)
SOURCE_START = "2005-03-01"
SOURCE_ENDS = (
    ("march_end", "2017-03-31"),
    ("last_session_before_publication", "2017-04-28"),
)
FORMULA_EXECUTION_DATE = "2017-05-02"
BASELINE_CASE = "prior_M_ddof0_lag2_next_open_10bps"

ASSUMPTIONS = {
    "purpose": "Controlled implementation diagnostics, not parameter selection or a change to the main study.",
    "source_partial_warmup": "Original report p.22 permits an insufficient M history; partial_prior_M uses at least two available prior slopes, never zero-fill.",
    "source_ambiguities": "The original report does not uniquely identify population/sample standard deviation or whether the current slope enters M; these are explicit assumptions.",
    "strict_warmup": "The main study retains a full, strictly prior M window with ddof=0.",
    "end_assumptions": "2017-03-31 and 2017-04-28 are displayed as alternative end-date assumptions, not asserted as the report's exact sample end.",
    "next_close_control": "Signal remains based on close t. Only execution is changed to close t+1 by replacing the execution-open field with the original close. All price returns and fees are recomputed.",
    "benchmark": "Buy and hold, same evaluation window, fee rate and execution convention; newly funded at the first evaluation fill with no terminal liquidation.",
    "exposure": "Formation-date target shifted by one observed session; realized_close_exposure is reported separately.",
    "index_proxy": "CSI 300 price index is a non-executable research proxy; no ETF tracking, financing, market impact or exchange-lot claims.",
    "no_blind_test": "The post-publication interval is historical and already observable; it is not a prospectively blinded test.",
}


def _json_write(path: Path, value: dict) -> None:
    def encode(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, pd.Timestamp):
            return item.isoformat()
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=encode, allow_nan=False) + "\n",
                    encoding="utf-8")


def _variant_signal(data: pd.DataFrame, spec: Specification, *, include_current=False,
                    ddof=0, partial=False) -> tuple[pd.DataFrame, pd.Series]:
    """Change exactly the declared normalization convention; no backfilling."""
    f = factors(data, spec.n, spec.m)
    source = f.beta if include_current else f.beta.shift(1)
    history = source.rolling(spec.m, min_periods=2 if partial else spec.m)
    deviation = history.std(ddof=ddof).where(lambda x: x > 1e-12)
    f["z"] = (f.beta - history.mean()) / deviation
    f["corrected"] = f.z * f.r2
    f["right"] = f.corrected * f.beta
    permitted = f.ma20 > f.ma20.shift(spec.trend_lag) if spec.trend_filter else None
    return f, decisions(f[spec.score], spec.upper, spec.lower, permitted)


def _metric_row(run) -> dict:
    measured = run.metrics
    return {"start": measured["start"], "end": measured["end"],
            "total_return": float(measured["total_return"]),
            "cagr": float(measured["annualized_return"]),
            "sharpe": float(measured["sharpe"]),
            "max_drawdown": float(measured["max_drawdown"]),
            "annual_turnover": float(measured["total_turnover"] * 252 / measured["sessions"])}


def _daily_file(output: Path, name: str, run, benchmark) -> str:
    daily = run.result.daily.set_index("date")
    reference = benchmark.result.daily.set_index("date")
    if not daily.index.equals(reference.index):
        raise ValueError("Diagnostic and benchmark sessions differ")
    capital = float(run.metadata["initial_cash"])
    nav = daily.nav / capital
    peak = np.maximum.accumulate(np.r_[1., nav.to_numpy()])[1:]
    frame = pd.DataFrame({
        "date": daily.index, "nav": nav.to_numpy(),
        "benchmark_nav": (reference.nav / float(benchmark.metadata["initial_cash"])).to_numpy(),
        "exposure": run.targets.iloc[:, 0].shift(1).reindex(daily.index).to_numpy(),
        "drawdown": 1 - nav.to_numpy() / peak,
        "capital_nav": daily.nav.to_numpy(),
        "realized_close_exposure": (daily.market_value / daily.nav).to_numpy(),
        "cost": daily.cost.to_numpy(), "return": daily["return"].to_numpy(),
    })
    relative = Path("diagnostic_daily") / f"{name}.csv"
    frame.to_csv(output / relative, index=False, float_format="%.15g", lineterminator="\n")
    return relative.as_posix()


def _formula_check(data: pd.DataFrame, output: Path) -> dict:
    """Independently regress 16 points and each of the preceding 300 windows."""
    n, m = 16, 300
    f = factors(data, n, m)
    before = np.flatnonzero(data.index < pd.Timestamp(FORMULA_EXECUTION_DATE))
    if not len(before):
        raise ValueError("No formation session before the requested formula check")
    t = int(before[-1])
    if t < n + m - 1:
        raise ValueError("Formula check needs a complete 16-point regression and 300 prior slopes")

    def fit(end: int):
        window = data.iloc[end - n + 1:end + 1]
        x, y = window.low.to_numpy(dtype=float), window.high.to_numpy(dtype=float)
        design = np.column_stack([np.ones(n), x])
        intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
        fitted = intercept + slope * x
        residual = y - fitted
        denominator = float(np.sum((y - np.mean(y))**2))
        r2 = 1 - float(np.sum(residual**2)) / denominator if denominator > 0 else np.nan
        return float(intercept), float(slope), float(r2), fitted, residual

    intercept, slope, r2, fitted, residual = fit(t)
    history = np.array([fit(j)[1] for j in range(t - m, t)])
    mean, std = float(np.mean(history)), float(np.std(history, ddof=0))
    z = (slope - mean) / std if std > 1e-12 else np.nan
    manual = {"beta": slope, "r2": r2, "prior_mean": mean, "prior_std_ddof0": std,
              "z": z, "corrected": z * r2, "right": z * r2 * slope}
    vector_history = f.beta.iloc[t - m:t]
    vector = {key: float(f.iloc[t][key]) for key in ("beta", "r2", "z", "corrected", "right")}
    vector.update(prior_mean=float(vector_history.mean()), prior_std_ddof0=float(vector_history.std(ddof=0)))
    if not np.isfinite(list(manual.values()) + list(vector.values())).all():
        raise ValueError("Formula check chosen window is degenerate or unavailable")
    errors = {key: abs(manual[key] - vector[key]) for key in manual}
    tolerance = 1e-8
    checked = {
        "passed": bool(max(errors.values()) <= tolerance), "absolute_tolerance": tolerance,
        "requested_execution_date": FORMULA_EXECUTION_DATE,
        "formation_date": str(data.index[t].date()),
        "regression_first_date": str(data.index[t - n + 1].date()),
        "regression_last_date": str(data.index[t].date()),
        "prior_slope_first_date": str(data.index[t - m].date()),
        "prior_slope_last_date": str(data.index[t - 1].date()),
        "N": n, "M": m, "regression_intercept": intercept,
        "independent_method": "numpy.linalg.lstsq with explicit intercept, repeated for all 300 prior OLS windows",
        "manual": manual, "vectorized": vector, "absolute_errors": errors,
        "prior_slope_count": len(history), "prior_contains_current": False,
        "signal_date_precedes_execution": bool(data.index[t] < pd.Timestamp(FORMULA_EXECUTION_DATE)),
    }
    window = data.iloc[t - n + 1:t + 1]
    pd.DataFrame({"date": window.index, "low": window.low.to_numpy(), "high": window.high.to_numpy(),
                  "fitted_high": fitted, "residual": residual,
                  "low_minus_mean": window.low.to_numpy() - window.low.mean(),
                  "high_minus_mean": window.high.to_numpy() - window.high.mean()}).to_csv(
        output / "formula_ols_points.csv", index=False, float_format="%.15g", lineterminator="\n")
    _json_write(output / "formula_checks.json", checked)
    if not checked["passed"]:
        raise ArithmeticError(f"Independent RSRS OLS check failed: {errors}")
    return checked


def run_diagnostics(data: pd.DataFrame, index_id: str, output: Path) -> dict:
    """Write fixed CSI 300 controls and return their validation/file receipt.

    Inputs are a chronological, complete OHLC snapshot. No data is fetched,
    overwritten, filled or optimized. The main strict-warmup research is intact.
    """
    if index_id != "sh000300":
        raise ValueError("These source-comparison diagnostics are defined only for sh000300")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Diagnostics require a DatetimeIndex")
    if (data.index.min() >= pd.Timestamp(SOURCE_START)
            or data.index.max() < max(pd.Timestamp(p[3]) for p in PERIODS)
            or data.index.max() < max(pd.Timestamp(p[1]) for p in SOURCE_ENDS)):
        raise ValueError("Snapshot must include a pre-start session and cover all fixed diagnostic end dates")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "diagnostic_daily").mkdir(exist_ok=True)
    formula = _formula_check(data, output)
    _json_write(output / "diagnostic_assumptions.json", ASSUMPTIONS)
    base_spec = next(spec for spec in SPECS if spec.name == "right_ma")
    _, base_signal = _variant_signal(data, base_spec)
    _, current_signal = _variant_signal(data, base_spec, include_current=True)
    _, sample_signal = _variant_signal(data, base_spec, ddof=1)
    _, lag_signal = _variant_signal(data, replace(base_spec, trend_lag=3))
    close_execution = data.copy()
    close_execution["open"] = close_execution["close"]
    variants = (
        (BASELINE_CASE, base_signal, data, 10., "基线：此前300斜率、总体标准差、MA滞后2日、次日开盘、单边10bp。"),
        ("include_current_M", current_signal, data, 10., "只将当前斜率纳入300个标准化样本；不是未来数据。"),
        ("sample_std_ddof1", sample_signal, data, 10., "只将此前300斜率标准差改为样本标准差。"),
        ("trend_lag3", lag_signal, data, 10., "只将入场均线比较从MA[t-2]改为MA[t-3]，不强制平仓。"),
        ("next_close_execution", base_signal, close_execution, 10., "信号仍在t收盘形成，只将成交延至t+1收盘；同时重算买持基准。"),
        ("cost_0bps", base_signal, data, 0., "只移除交易成本；无终端强制清仓。"),
        ("cost_20bps", base_signal, data, 20., "只将单边成本提高至20bp。"),
        ("cost_30bps", base_signal, data, 30., "只将单边成本提高至30bp。"),
    )
    benchmark_cache = {}
    checks = []
    files = []

    def benchmark(prices, start, end, cost, mode="next_open"):
        key = (str(start), str(end), float(cost), mode)
        if key not in benchmark_cache:
            run = run_signal(prices, pd.Series(1, index=prices.index, dtype=int), start, end,
                             cost_bps=cost, asset=index_id)
            benchmark_cache[key] = run
            checks.append({"run": "benchmark_" + "_".join(map(str, key)), **run.checks})
        return benchmark_cache[key]

    records = []
    for period_id, period_name, start, end in PERIODS:
        base_metrics = None
        for case, signal, prices, cost, interpretation in variants:
            run = run_signal(prices, signal, start, end, cost_bps=cost, asset=index_id)
            row = _metric_row(run)
            if case == BASELINE_CASE:
                base_metrics = row.copy()
            mode = "next_close" if case == "next_close_execution" else "next_open"
            compare_id = f"{index_id}_{period_id}_{case}"
            daily = _daily_file(output, compare_id, run, benchmark(prices, start, end, cost, mode))
            files.append(daily)
            checks.append({"run": compare_id, **run.checks})
            records.append({"compare_id": compare_id, "period_id": period_id,
                            "period_name": period_name, "case": case, "baseline_case": BASELINE_CASE,
                            "strategy_id": "right_ma", "cost_bps": cost, **row,
                            "delta_cagr": row["cagr"] - base_metrics["cagr"],
                            "delta_sharpe": row["sharpe"] - base_metrics["sharpe"],
                            "delta_max_drawdown": row["max_drawdown"] - base_metrics["max_drawdown"],
                            "interpretation": interpretation, "daily_file": daily,
                            "index_id": index_id})
    pd.DataFrame(records).to_csv(output / "diagnostics.csv", index=False, float_format="%.15g", lineterminator="\n")

    compatible = []
    for strategy_id in ("standard_ma", "corrected_ma", "right_ma"):
        spec = next(s for s in SPECS if s.name == strategy_id)
        for partial in (False, True):
            warmup = "partial_prior_M_min2" if partial else "strict_prior_M"
            _, signal = _variant_signal(data, spec, partial=partial)
            for end_assumption, end in SOURCE_ENDS:
                for cost in (0., 10.):
                    run = run_signal(data, signal, SOURCE_START, end, cost_bps=cost, asset=index_id)
                    name = f"{index_id}_{strategy_id}_{warmup}_{end_assumption}_{int(cost)}bps"
                    daily = _daily_file(output, name, run, benchmark(data, SOURCE_START, end, cost))
                    files.append(daily)
                    checks.append({"run": name, **run.checks})
                    compatible.append({"index_id": index_id, "strategy_id": strategy_id,
                                       "warmup": warmup, "end_assumption": end_assumption,
                                       "cost_bps": cost, **_metric_row(run), "daily_file": daily})
    pd.DataFrame(compatible).to_csv(output / "source_compatible.csv", index=False,
                                   float_format="%.15g", lineterminator="\n")
    receipt = {"passed": all(check["passed"] for check in checks) and formula["passed"],
               "index_id": index_id, "diagnostic_runs": len(records),
               "source_compatible_runs": len(compatible), "benchmark_runs": len(benchmark_cache),
               "reconciliation": checks, "formula_check_passed": formula["passed"],
               "assumptions_file": "diagnostic_assumptions.json",
               "written_files": ["diagnostics.csv", "source_compatible.csv", "formula_checks.json",
                                 "formula_ols_points.csv", "diagnostic_assumptions.json"] + files}
    _json_write(output / "diagnostic_checks.json", receipt)
    return receipt
