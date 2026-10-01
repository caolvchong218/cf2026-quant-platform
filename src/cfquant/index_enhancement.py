"""As-of index snapshots and benchmark-relative target allocation.

This module constructs targets, not trades or backtest results. Active stock
and industry weights are measured against the benchmark scaled to the same
equity exposure. Historical membership is never inferred from a later snapshot.
Blocked dates retain NaN targets; they are not successful all-cash allocations.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np
import pandas as pd


def validate_index_weights(frame: pd.DataFrame, *, weights_in_percent: bool = False,
                           sum_tolerance: float = 1e-6) -> pd.DataFrame:
    """Validate complete snapshots without silently repairing membership.

    Required columns are date, asset, weight, known_at. Weights are fractions
    unless weights_in_percent=True explicitly divides them by 100. Every dated
    snapshot must sum to one within tolerance. Mixed known_at timestamps are
    permitted, but a snapshot is usable only after its last row is known.
    Timestamps are exact, timezone-naive formation timestamps; callers must
    express publication and formation times using the same clock.
    """
    columns = ["date", "asset", "weight", "known_at"]
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Index weights must be a nonempty DataFrame of complete snapshots")
    if not set(columns).issubset(frame.columns):
        raise ValueError("Index weights require date, asset, weight, known_at columns")
    if type(weights_in_percent) is not bool:
        raise ValueError("weights_in_percent must be explicitly True or False")
    if not isinstance(sum_tolerance, Real) or isinstance(sum_tolerance, bool) or not 0 < sum_tolerance <= .001:
        raise ValueError("sum_tolerance must be in (0, .001]")
    result = frame[columns].copy()
    for name in ("date", "known_at"):
        try:
            result[name] = pd.to_datetime(result[name], errors="raise")
        except (ValueError, TypeError, OverflowError):
            raise ValueError(f"Index {name} must contain valid timestamps") from None
        if result[name].isna().any() or not pd.api.types.is_datetime64_ns_dtype(result[name].dtype):
            raise ValueError(f"Index {name} must contain timezone-naive, nonmissing timestamps")
    if any(not isinstance(asset, str) or not asset.strip() for asset in result.asset):
        raise ValueError("Index asset identifiers must be nonempty strings; preserve leading zeros when importing")
    result["asset"] = result.asset.str.strip()
    if result.duplicated(["date", "asset"]).any():
        raise ValueError("Index snapshot has duplicate date/asset rows")
    if result.weight.map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise ValueError("Index weights must be numeric, not boolean")
    try:
        result["weight"] = pd.to_numeric(result.weight, errors="raise").astype(float)
    except (ValueError, TypeError):
        raise ValueError("Index weights must be numeric") from None
    if weights_in_percent:
        result["weight"] = result.weight / 100.
    if not np.isfinite(result.weight).all() or not result.weight.between(0, 1).all():
        raise ValueError("Index weights must be finite fractions in [0, 1]; percent conversion must be explicit")
    sums = result.groupby("date", sort=True).weight.sum()
    if ((sums - 1).abs() > sum_tolerance).any():
        raise ValueError("Each complete index snapshot must sum to 1; missing members are never renormalized")
    result = result.sort_values(["date", "asset"], kind="stable").reset_index(drop=True)
    result.attrs["weights_in_percent_input"] = weights_in_percent
    result.attrs["weight_sums_repaired"] = False
    return result


def _aligned_inputs(scores, close, industry):
    if not isinstance(scores, pd.DataFrame) or not isinstance(close, pd.DataFrame) or scores.empty:
        raise ValueError("scores and close must be nonempty aligned DataFrames")
    if not isinstance(scores.index, pd.DatetimeIndex) or scores.index.tz is not None:
        raise ValueError("Formation dates must be a timezone-naive DatetimeIndex")
    if not scores.index.is_unique or not scores.columns.is_unique or not scores.index.is_monotonic_increasing:
        raise ValueError("Formation dates/assets must be unique and dates increasing")
    if not scores.index.equals(close.index) or not scores.columns.equals(close.columns):
        raise ValueError("scores and close must have identical date and asset axes")
    if any(not isinstance(asset, str) for asset in scores.columns):
        raise ValueError("Panel asset identifiers must be strings matching the imported snapshot")
    for panel in (scores, close):
        if any(not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype) for dtype in panel.dtypes):
            raise ValueError("Scores/prices must be numeric")
    if isinstance(industry, pd.DataFrame):
        if not industry.index.equals(scores.index) or not industry.columns.equals(scores.columns):
            raise ValueError("Industry panel must have identical date and asset axes")
        source = "dated_panel_supplied_by_caller"
    elif isinstance(industry, pd.Series):
        if not industry.index.is_unique:
            raise ValueError("Industry labels must have unique assets")
        source = "static_labels_supplied_by_caller_not_historical_membership"
    else:
        raise ValueError("industry must be an aligned DataFrame or asset-indexed Series")
    return scores.astype(float).replace([np.inf, -np.inf], np.nan), close.astype(float).replace([np.inf, -np.inf], np.nan), source


def _project_sum(desired: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                 total: float) -> np.ndarray:
    """Euclidean box projection with a required sum, solved by water filling."""
    if total < lower.sum() - 1e-10 or total > upper.sum() + 1e-10:
        raise ValueError("Infeasible group sum")
    if abs(total - lower.sum()) < 1e-13:
        return lower.copy()
    if abs(total - upper.sum()) < 1e-13:
        return upper.copy()
    left, right = float(np.min(desired - upper)), float(np.max(desired - lower))
    for _ in range(70):
        midpoint = (left + right) / 2
        value = np.clip(desired - midpoint, lower, upper)
        if abs(float(value.sum()) - total) < 1e-13:
            return value
        if value.sum() > total:
            left = midpoint
        else:
            right = midpoint
    return np.clip(desired - (left + right) / 2, lower, upper)


def _project(desired: np.ndarray, benchmark: np.ndarray, groups: np.ndarray,
             stock_cap: float, industry_cap: float) -> np.ndarray:
    """Projection onto stock boxes, industry bands, and the equity budget."""
    lower = np.maximum(0., benchmark - stock_cap)
    upper = benchmark + stock_cap
    labels = sorted(set(groups))
    grouped = []
    for label in labels:
        positions = np.flatnonzero(groups == label)
        base_sum = float(benchmark[positions].sum())
        floor = max(float(lower[positions].sum()), base_sum - industry_cap)
        ceiling = min(float(upper[positions].sum()), base_sum + industry_cap)
        grouped.append((positions, floor, ceiling))

    def at(multiplier):
        shifted = desired - multiplier
        result = np.clip(shifted, lower, upper)
        for positions, floor, ceiling in grouped:
            group_sum = float(result[positions].sum())
            if group_sum < floor:
                result[positions] = _project_sum(shifted[positions], lower[positions], upper[positions], floor)
            elif group_sum > ceiling:
                result[positions] = _project_sum(shifted[positions], lower[positions], upper[positions], ceiling)
        return result

    left, right = float(np.min(desired - upper)), float(np.max(desired - lower))
    total = float(benchmark.sum())
    for _ in range(70):
        midpoint = (left + right) / 2
        value = at(midpoint)
        if abs(float(value.sum()) - total) < 1e-12:
            return value
        if value.sum() > total:
            left = midpoint
        else:
            right = midpoint
    return at((left + right) / 2)


def index_enhanced_targets(scores: pd.DataFrame, close: pd.DataFrame,
                           industry: pd.DataFrame | pd.Series, weights: pd.DataFrame,
                           max_active_stock: float = .01, max_active_industry: float = .03,
                           max_exposure: float = .95, lookback: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct daily as-of targets and explicit readiness/constraint diagnostics.

    Only positive-weight members of a complete eligible snapshot may be held.
    Missing formation prices or classifications block allocation. Missing
    scores or full trailing return history leave those members at zero alpha.
    Scores are standardized, scaled by observed trailing volatility, then
    standardized again. Their requested tilt is half the stock active cap per
    score standard deviation, clipped to three standard deviations. Projection
    enforces long-only stock/sector bands and the exposure-scaled benchmark sum.
    During warmup/all-missing scores the result is explicitly baseline-only.
    Industry labels are caller-supplied; a static Series is not historical data.
    """
    for name, value in (("max_active_stock", max_active_stock), ("max_active_industry", max_active_industry)):
        if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if not isinstance(max_exposure, Real) or isinstance(max_exposure, bool) or not math.isfinite(max_exposure) or not 0 < max_exposure <= 1:
        raise ValueError("max_exposure must be in (0, 1]")
    if type(lookback) is not int or not 2 <= lookback <= 2520:
        raise ValueError("lookback must be an integer in [2, 2520]")
    scores, close, industry_source = _aligned_inputs(scores, close, industry)
    snapshots = validate_index_weights(weights)
    known = snapshots.groupby("date", sort=True).known_at.max()
    snapshot_dates = known.index
    returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    returns = returns.where((close > 0) & (close.shift(1) > 0))
    risk_count = returns.rolling(lookback, min_periods=lookback).count()
    volatility = returns.rolling(lookback, min_periods=lookback).std(ddof=1)
    targets = pd.DataFrame(np.nan, index=scores.index, columns=scores.columns)
    records = []
    for date in scores.index:
        diagnostic = {"date": date, "status": "blocked", "block_reason": None,
                      "snapshot_date": pd.NaT, "snapshot_known_at": pd.NaT,
                      "member_count": 0, "member_coverage": 0., "price_coverage": None,
                      "score_coverage": None, "history_coverage": None,
                      "target_exposure": None, "cash_target": None,
                      "max_active_stock": None, "max_active_industry": None,
                      "constraints_passed": False, "alpha_members": 0,
                      "active_reference": "exposure_scaled_benchmark",
                      "industry_source": industry_source, "allocation_is_trade": False}
        eligible = snapshot_dates[(snapshot_dates <= date) & (known <= date).to_numpy()]
        if not len(eligible):
            diagnostic["block_reason"] = "no_complete_snapshot_known_by_formation"
            records.append(diagnostic)
            continue
        snapshot_date = eligible[-1]
        snapshot = snapshots.loc[(snapshots.date == snapshot_date) & (snapshots.weight > 0)].set_index("asset")
        members = snapshot.index
        diagnostic.update({"snapshot_date": snapshot_date, "snapshot_known_at": known.loc[snapshot_date],
                           "member_count": len(members), "snapshot_weight_sum": float(snapshot.weight.sum()),
                           "member_coverage": float(members.isin(close.columns).mean())})
        missing = members.difference(close.columns)
        if len(missing):
            diagnostic["block_reason"] = "benchmark_members_outside_price_panel"
            diagnostic["missing_members"] = ",".join(missing)
            records.append(diagnostic)
            continue
        prices = close.loc[date, members]
        valid_price = prices.notna() & (prices > 0)
        diagnostic["price_coverage"] = float(valid_price.mean())
        diagnostic["price_weight_coverage"] = float(snapshot.weight.loc[valid_price].sum())
        if not valid_price.all():
            diagnostic["block_reason"] = "benchmark_member_formation_price_unavailable"
            diagnostic["missing_members"] = ",".join(members[~valid_price])
            records.append(diagnostic)
            continue
        labels = (industry.loc[date].reindex(members) if isinstance(industry, pd.DataFrame)
                  else industry.reindex(members))
        valid_labels = labels.map(lambda value: isinstance(value, str) and bool(value.strip())
                                  and value.strip().lower() not in {"unknown", "nan", "none", "未知", "未分类"})
        if not valid_labels.all():
            diagnostic["block_reason"] = "benchmark_member_industry_unavailable"
            diagnostic["missing_members"] = ",".join(members[~valid_labels])
            records.append(diagnostic)
            continue
        groups = labels.str.strip().to_numpy()
        row = scores.loc[date, members]
        risk = volatility.loc[date, members]
        history_valid = (risk_count.loc[date, members] == lookback) & risk.notna() & (risk > 0)
        alpha_valid = row.notna() & history_valid
        diagnostic.update({"score_coverage": float(row.notna().mean()),
                           "history_coverage": float(history_valid.mean()),
                           "alpha_members": int(alpha_valid.sum()), "lookback": lookback})
        alpha = pd.Series(0., index=members)
        if alpha_valid.sum() >= 2:
            observed = row.loc[alpha_valid]
            scale = float(observed.std(ddof=0))
            if scale > 1e-12:
                standardized = (observed - observed.mean()) / scale
                adjusted = standardized / risk.loc[alpha_valid].clip(lower=1e-6)
                adjusted_scale = float(adjusted.std(ddof=0))
                if adjusted_scale > 1e-12 and math.isfinite(adjusted_scale):
                    alpha.loc[alpha_valid] = ((adjusted - adjusted.mean()) / adjusted_scale).clip(-3, 3)
        base = snapshot.weight.to_numpy() * max_exposure
        desired = base + .5 * max_active_stock * alpha.to_numpy()
        # A missing score/history never becomes an indirect alpha position.
        stock_caps = np.where(alpha_valid.to_numpy(), max_active_stock, 0.)
        result = _project(desired, base, groups, stock_caps, max_active_industry)
        active = result - base
        sector_active = pd.Series(active).groupby(groups).sum()
        stock_deviation = float(np.abs(active).max())
        sector_deviation = float(sector_active.abs().max())
        exposure = float(result.sum())
        passed = (np.isfinite(result).all() and result.min() >= -1e-10
                  and abs(exposure - base.sum()) <= 1e-9
                  and exposure <= max_exposure + 1e-6
                  and stock_deviation <= max_active_stock + 1e-9
                  and sector_deviation <= max_active_industry + 1e-9)
        if not passed:
            diagnostic["block_reason"] = "allocation_constraint_verification_failed"
            records.append(diagnostic)
            continue
        # Every usable row explicitly assigns zero to non-benchmark assets.
        targets.loc[date] = 0.
        targets.loc[date, members] = result
        active_signal = bool(np.abs(active).max() > 1e-10)
        diagnostic.update({"status": "ready" if active_signal else "ready_baseline_only",
                           "block_reason": None, "target_exposure": exposure,
                           "cash_target": 1 - exposure, "max_active_stock": stock_deviation,
                           "max_active_industry": sector_deviation, "constraints_passed": True,
                           "allocation_method": "deterministic_euclidean_projection",
                           "missing_score_alpha": 0., "risk_model": "trailing_full_window_daily_volatility"})
        records.append(diagnostic)
    diagnostics = pd.DataFrame(records)
    targets.attrs.update({"active_reference": "exposure_scaled_benchmark", "allocation_is_trade": False,
                          "blocked_dates_are_nan": True, "weight_sums_repaired": False})
    return targets, diagnostics
