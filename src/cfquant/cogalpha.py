"""Bounded factor evaluation building blocks for an independent CogAlpha study.

This module does not call a model, search for expressions, select a test winner,
or claim to implement the unpublished reference evaluator. The MI estimator is
an explicit local substitute: daily rank-quantile plug-in MI in natural units.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real
import time

import numpy as np
import pandas as pd

from .factor_lab import DEFAULT_FIELDS, evaluate_expression, validate_expression


METRIC_NAMES = ("ic", "rank_ic", "ic_ir", "rank_ic_ir", "mi")
QUALIFIED_FLOORS = dict(zip(METRIC_NAMES, (.005, .005, .05, .05, .02)))
ELITE_FLOORS = dict(zip(METRIC_NAMES, (.01, .01, .1, .1, .02)))


def _integer(value, name, lower, upper):
    if isinstance(value, bool) or not isinstance(value, Integral) or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return int(value)


def _fraction(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must lie in [0, 1]")
    return float(value)


def _numeric_panel(frame, name):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{name} must be a nonempty DataFrame")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans:
        raise ValueError(f"{name} requires a nonmissing DatetimeIndex")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing or not frame.columns.is_unique:
        raise ValueError(f"{name} requires unique, increasing dates and unique assets")
    if any(not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype) for dtype in frame.dtypes):
        raise ValueError(f"{name} must be numeric, not Boolean")


def _eligible_mask(eligible, reference):
    if (not isinstance(eligible, pd.DataFrame) or not eligible.index.equals(reference.index)
            or not eligible.columns.equals(reference.columns)):
        raise ValueError("eligible must have exactly the score date and asset axes")
    if eligible.isna().any().any() or any(not pd.api.types.is_bool_dtype(dtype) for dtype in eligible.dtypes):
        raise ValueError("eligible must contain nonmissing Boolean values")
    return eligible.astype(bool)


def quantile_mutual_information(x, y, *, bins=5, min_pairs=30):
    """Uncorrected rank-quantile plug-in MI (nats), with tied values kept together.

    Average percentile ranks are assigned to ceil(rank * bins) - 1. Therefore
    ties never receive arbitrary distinct buckets, even when this leaves some
    buckets empty. Constant marginals and insufficient finite pairs are unknown
    (NaN), not fabricated information. This finite-sample estimator is biased
    upward; its values are not claimed comparable to an unspecified estimator.
    """
    bins = _integer(bins, "bins", 2, 100)
    min_pairs = _integer(min_pairs, "min_pairs", 30, 1_000_000)
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("MI requires one-dimensional arrays with equal lengths")
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < min_pairs or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")

    def categories(values):
        ranks = pd.Series(values).rank(method="average", pct=True).to_numpy()
        return np.clip(np.ceil(ranks * bins).astype(int) - 1, 0, bins - 1)

    counts = np.zeros((bins, bins), dtype=float)
    np.add.at(counts, (categories(x), categories(y)), 1.)
    probability = counts / counts.sum()
    independent = probability.sum(axis=1)[:, None] * probability.sum(axis=0)[None, :]
    used = probability > 0
    value = np.sum(probability[used] * np.log(probability[used] / independent[used]))
    return float(max(0., value))


def factor_metrics(scores, labels, eligible, *, date_start, date_end, label_exit,
                   split_end_nextday=None, min_pairs=30, mi_bins=5):
    """Evaluate one factor on realized labels within one explicitly bounded split.

    All input axes must already match; this evaluator never silently reindexes.
    A formation date is used only if its label_exit is nonmissing and strictly
    earlier than split_end_nextday (default: the day after date_end). Coverage
    counts every eligible cell on retained dates, including rolling warmup NaNs.
    Daily correlations and MI share the same finite-pair mask, minimum 30 by
    default. ICIR is mean / sample std (ddof=1), without annualization.
    """
    _numeric_panel(scores, "scores")
    _numeric_panel(labels, "labels")
    if not scores.index.equals(labels.index) or not scores.columns.equals(labels.columns):
        raise ValueError("labels must have exactly the score date and asset axes")
    eligible = _eligible_mask(eligible, scores)
    min_pairs = _integer(min_pairs, "min_pairs", 30, 1_000_000)
    mi_bins = _integer(mi_bins, "mi_bins", 2, 100)
    if not isinstance(label_exit, pd.Series) or not label_exit.index.equals(scores.index):
        raise ValueError("label_exit must have exactly the score date axis")
    start, end = pd.Timestamp(date_start), pd.Timestamp(date_end)
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError("date_start and date_end must define an ordered, nonmissing interval")
    cutoff = pd.Timestamp(split_end_nextday) if split_end_nextday is not None else end.normalize() + pd.Timedelta(days=1)
    if pd.isna(cutoff) or cutoff <= start:
        raise ValueError("split_end_nextday must be later than date_start")
    exits = pd.to_datetime(label_exit, errors="raise")
    requested = (scores.index >= start) & (scores.index <= end)
    realized = exits.notna() & (exits < cutoff)
    dates = scores.index[requested & realized.to_numpy()]
    rows = []
    for date in dates:
        active = eligible.loc[date].to_numpy(dtype=bool)
        x, y = scores.loc[date].to_numpy(dtype=float), labels.loc[date].to_numpy(dtype=float)
        valid = active & np.isfinite(x) & np.isfinite(y)
        pairs, population = int(valid.sum()), int(active.sum())
        ic = rank_ic = mi = float("nan")
        if pairs >= min_pairs and np.unique(x[valid]).size > 1 and np.unique(y[valid]).size > 1:
            a, b = pd.Series(x[valid]), pd.Series(y[valid])
            ic = float(a.corr(b))
            rank_ic = float(a.rank(method="average").corr(b.rank(method="average")))
            mi = quantile_mutual_information(a, b, bins=mi_bins, min_pairs=min_pairs)
        rows.append({"date": date, "ic": ic, "rank_ic": rank_ic, "mi": mi,
                     "n_pairs": pairs, "eligible_count": population,
                     "coverage": pairs / population if population else float("nan"),
                     "factor_coverage": float((active & np.isfinite(x)).sum() / population) if population else float("nan"),
                     "label_coverage": float((active & np.isfinite(y)).sum() / population) if population else float("nan")})
    daily = pd.DataFrame(rows, columns=["date", "ic", "rank_ic", "mi", "n_pairs", "eligible_count",
                                       "coverage", "factor_coverage", "label_coverage"])

    def moments(name):
        values = pd.to_numeric(daily[name], errors="coerce").astype(float).dropna()
        values = values[np.isfinite(values)]
        mean = float(values.mean()) if len(values) else float("nan")
        deviation = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        ratio = mean / deviation if np.isfinite(deviation) and deviation > 1e-12 else float("nan")
        return mean, ratio, len(values)

    ic, ic_ir, ic_days = moments("ic")
    rank_ic, rank_ic_ir, rank_days = moments("rank_ic")
    mi, _, mi_days = moments("mi")
    count = int(daily.eligible_count.sum())
    summary = {"ic": ic, "rank_ic": rank_ic, "ic_ir": ic_ir, "rank_ic_ir": rank_ic_ir, "mi": mi,
               "coverage": float(daily.n_pairs.sum() / count) if count else float("nan"),
               "valid_days": int((daily.ic.notna() & daily.rank_ic.notna()).sum()),
               "ic_days": ic_days, "rank_ic_days": rank_days, "mi_days": mi_days,
               "n_dates": len(daily), "observations": int(daily.n_pairs.sum()),
               "dropped_unrealized_dates": int((requested & ~realized.to_numpy()).sum())}
    assumptions = {"date_start": start.isoformat(), "date_end": end.isoformat(),
                   "label_exit_strictly_before": cutoff.isoformat(), "minimum_pairs": min_pairs,
                   "ic_ir_ddof": 1, "ic_ir_annualized": False,
                   "coverage_denominator": "all eligible cells on realized formation dates; warmup NaN retained",
                   "mi_estimator": "daily average-rank quantile discrete plug-in MI, then daily arithmetic mean",
                   "mi_bins": mi_bins, "mi_units": "nats", "mi_bias_correction": False,
                   "mi_matches_unpublished_original_estimator": False}
    return {"daily": daily, "summary": summary, "assumptions": assumptions}


def _metric_table(metrics, id_column, required):
    if not isinstance(metrics, pd.DataFrame) or not {id_column, *required}.issubset(metrics):
        raise ValueError("metrics must contain factor IDs and every required metric")
    identifiers = metrics[id_column]
    if identifiers.isna().any() or identifiers.duplicated().any() or not identifiers.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Factor IDs must be unique, nonempty strings")
    result = metrics.copy(deep=True)
    for key in required:
        result[key] = pd.to_numeric(result[key], errors="raise").astype(float)
    return result


def qualify_factors(metrics, *, id_column="id", qualified_quantile=.65, elite_quantile=.80):
    """Apply five positive-direction tests using thresholds from this same pool.

    Every threshold is max(finite-pool quantile, absolute floor). All five must
    pass; no absolute-value sign rescue is used. Missing/infinite metrics fail.
    Returns an annotated table and complete threshold derivation for freezing.
    """
    q = _fraction(qualified_quantile, "qualified_quantile")
    e = _fraction(elite_quantile, "elite_quantile")
    if e < q:
        raise ValueError("elite_quantile cannot be lower than qualified_quantile")
    result = _metric_table(metrics, id_column, METRIC_NAMES)
    thresholds = {"pool_size": len(result), "quantile_interpolation": "linear",
                  "direction": "positive metrics; no absolute values", "qualified": {}, "elite": {}}
    for stage, quantile, floors in [("qualified", q, QUALIFIED_FLOORS), ("elite", e, ELITE_FLOORS)]:
        reasons = [[] for _ in range(len(result))]
        for name in METRIC_NAMES:
            values = result[name]
            finite = values[np.isfinite(values)]
            percentile = float(finite.quantile(quantile, interpolation="linear")) if len(finite) else float("nan")
            threshold = max(floors[name], percentile) if np.isfinite(percentile) else float("nan")
            thresholds[stage][name] = {"quantile": quantile, "quantile_value": percentile,
                                       "absolute_floor": floors[name], "threshold": threshold,
                                       "finite_pool_count": len(finite)}
            result[stage + "_threshold_" + name] = threshold
            for i, value in enumerate(values):
                if not np.isfinite(value):
                    reasons[i].append(name + ": nonfinite")
                elif not np.isfinite(threshold):
                    reasons[i].append(name + ": no finite pool threshold")
                elif value < threshold:
                    reasons[i].append(name + ": below threshold")
        result[stage] = pd.Series([not rejected for rejected in reasons], index=result.index, dtype=bool)
        result[stage + "_reasons"] = reasons
    return result, thresholds


def _same_values(left, right):
    return (left.index.equals(right.index) and left.columns.equals(right.columns)
            and bool(np.allclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float),
                                 atol=1e-10, rtol=1e-10, equal_nan=True)))


def causal_quality_check(expression, panels, eligible, train_dates, *, max_missing_rate=.30,
                         max_runtime_seconds=None):
    """Check execution, eligible training coverage, and two causal perturbations.

    Only the eligibility mask and formation dates enter quality selection; no
    label is accepted. Both training midpoint and training end are checked by
    prefix evaluation and by changing subsequent price rows. If no later row
    exists, two synthetic future rows are appended so the end test is not empty.
    DSL sanitizes arithmetic infinities to NaN; they count as missing coverage.
    """
    limit = _fraction(max_missing_rate, "max_missing_rate")
    if max_runtime_seconds is not None and (isinstance(max_runtime_seconds, bool)
            or not isinstance(max_runtime_seconds, Real) or not np.isfinite(max_runtime_seconds)
            or max_runtime_seconds <= 0):
        raise ValueError("max_runtime_seconds must be positive and finite")
    started = time.perf_counter()
    result = {"passed": False, "reasons": [], "checks": [], "runtime_seconds": None,
              "missing_rate": None, "nan_rate": None, "inf_rate": None,
              "coverage": None, "max_missing_rate": limit, "uses_labels": False,
              "inf_handling": "DSL converts arithmetic infinities to NaN; missing rate counts these outputs"}
    try:
        if not isinstance(panels, Mapping) or not panels:
            raise ValueError("panels must be a nonempty mapping")
        if not set(panels).issubset(DEFAULT_FIELDS):
            raise ValueError("quality panels may contain only OHLCV fields, never labels")
        validate_expression(expression, panels.keys())
        reference = next(iter(panels.values()))
        _numeric_panel(reference, "panels")
        active = _eligible_mask(eligible, reference)
        dates = pd.DatetimeIndex(train_dates)
        if len(dates) < 2 or dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing or not dates.isin(reference.index).all():
            raise ValueError("train_dates must contain at least two unique ordered dates present in the panels")
        full = evaluate_expression(expression, dict(panels))
        mask = active.loc[dates].to_numpy(dtype=bool)
        count = int(mask.sum())
        values = full.loc[dates].to_numpy(dtype=float)[mask]
        if not count:
            result["reasons"].append("no_eligible_training_cells")
        else:
            result.update(eligible_cells=count, nan_rate=float(np.isnan(values).mean()),
                          inf_rate=float(np.isinf(values).mean()),
                          missing_rate=float((~np.isfinite(values)).mean()),
                          coverage=float(np.isfinite(values).mean()))
            if result["missing_rate"] > limit:
                result["reasons"].append("nonfinite_rate_exceeds_limit")
        for cutoff in [dates[(len(dates) - 1) // 2], dates[-1]]:
            before = reference.index <= cutoff
            prefix = evaluate_expression(expression, {name: frame.loc[before] for name, frame in panels.items()})
            prefix_passed = _same_values(prefix, full.loc[before])
            changed = {name: frame.copy(deep=True) for name, frame in panels.items()}
            after = reference.index > cutoff
            synthetic = not after.any()
            if synthetic:
                extra_dates = pd.date_range(reference.index[-1] + pd.Timedelta(days=1), periods=2)
                for name, frame in changed.items():
                    last = frame.iloc[-1].to_numpy(dtype=float)
                    extension = np.vstack([last * np.arange(2, len(last) + 2) + 7., last * 3. + 13.])
                    changed[name] = pd.concat([frame, pd.DataFrame(extension, index=extra_dates, columns=frame.columns)])
            else:
                rows = int(after.sum())
                scales = np.linspace(1.7, 4.1, len(reference.columns))[None, :]
                trend = np.arange(1, rows + 1)[:, None]
                for frame in changed.values():
                    frame.loc[after] = frame.loc[after].to_numpy(dtype=float) * scales + trend
            perturbed = evaluate_expression(expression, changed)
            perturbation_passed = _same_values(perturbed.loc[full.index[before]], full.loc[before])
            result["checks"].append({"cutoff": cutoff.isoformat(), "prefix_passed": prefix_passed,
                                      "future_perturbation_passed": perturbation_passed,
                                      "future_rows_changed": 2 if synthetic else int(after.sum()),
                                      "synthetic_future_added": synthetic})
            if not prefix_passed:
                result["reasons"].append("prefix_mismatch_at_" + cutoff.isoformat())
            if not perturbation_passed:
                result["reasons"].append("future_dependence_at_" + cutoff.isoformat())
    except (ValueError, TypeError, KeyError, ArithmeticError, RuntimeError) as exc:
        result["reasons"].append("evaluation_error: " + type(exc).__name__ + ": " + str(exc))
    result["runtime_seconds"] = time.perf_counter() - started
    if max_runtime_seconds is not None and result["runtime_seconds"] > max_runtime_seconds:
        result["reasons"].append("runtime_budget_exceeded")
    result["passed"] = not result["reasons"]
    return result


def select_diverse(metrics, rank_ic_daily=None, *, id_column="id", max_factors=20,
                   max_abs_correlation=.8, min_correlation_days=2):
    """Training-positive RankIC top-k, optionally rejecting correlated IC paths.

    Without daily IC data this is the literal top-20 rule. With a training-only
    date-by-factor RankIC frame it becomes a separately labeled stability
    improvement. Correlation uses jointly finite dates; undefined/insufficient
    comparisons are rejected, never treated as zero. Absolute correlation also
    excludes mirror images. Test-period data must not be supplied by callers.
    """
    max_factors = _integer(max_factors, "max_factors", 1, 1000)
    minimum = _integer(min_correlation_days, "min_correlation_days", 2, 1_000_000)
    cap = _fraction(max_abs_correlation, "max_abs_correlation")
    ranked = _metric_table(metrics, id_column, ("rank_ic",))
    ranked = ranked[np.isfinite(ranked.rank_ic) & (ranked.rank_ic > 0)]
    ranked = ranked.sort_values(["rank_ic", id_column], ascending=[False, True], kind="stable")
    if ranked.empty:
        return []
    if rank_ic_daily is None:
        return ranked[id_column].head(max_factors).tolist()
    _numeric_panel(rank_ic_daily, "rank_ic_daily")
    if not set(ranked[id_column]).issubset(rank_ic_daily.columns):
        raise ValueError("rank_ic_daily must include every positive candidate ID")
    selected = []
    for candidate in ranked[id_column]:
        current = rank_ic_daily[candidate].replace([np.inf, -np.inf], np.nan)
        if current.notna().sum() < minimum or current.nunique() < 2:
            continue
        redundant = False
        for previous in selected:
            pair = pd.concat([current, rank_ic_daily[previous]], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
            correlation = pair.iloc[:, 0].corr(pair.iloc[:, 1]) if len(pair) >= minimum else float("nan")
            if not np.isfinite(correlation) or abs(correlation) > cap:
                redundant = True
                break
        if not redundant:
            selected.append(candidate)
            if len(selected) == max_factors:
                break
    return selected


def paper_literal_candidates():
    """Three translated Listing expressions; callers must provide volume in shares."""
    definitions = [
        ("paper_listing1", "原文 Listing 1 · 高点至收盘距离", "safe_divide(high-close,volume+1e-8)"),
        ("paper_listing2", "原文 Listing 2 · 振幅/成交量", "safe_divide(high-low,volume+1e-8)"),
        ("paper_listing3", "原文 Listing 3 · 实体金额比", "tanh(safe_divide(abs(close-open),volume*close+1e-8))"),
    ]
    return [{"id": name, "label": label, "expression": expression,
             "source": "arXiv:2511.18850v4 literal Listing translation",
             "volume_unit": "shares", "denominator_epsilon": 1e-8,
             "direction": "positive; no retrospective sign reversal",
             "hypothesis": "保留原文公式与正向定义，其实测有效性须由独立评价确定。"}
            for name, label, expression in definitions]
