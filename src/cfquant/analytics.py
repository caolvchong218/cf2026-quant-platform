"""Assignment metric definitions; diagnostics are never compounded as NAV."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .factors import panel


def forward_returns(market, calendar, horizon):
    opens = panel(market, calendar, "open")
    # Close t signal -> open t+1 entry -> open t+1+h exit.
    return opens.shift(-(horizon + 1)) / opens.shift(-1) - 1


def diagnostics(scores, labels, groups=5, min_assets=10):
    daily, grouped, assignments = [], [], []
    for date, row in scores.iterrows():
        valid_scores = row.dropna()
        y = labels.loc[date].reindex(row.index)
        pairs = pd.concat([row.rename("factor"), y.rename("label")], axis=1).dropna()
        ic = rank_ic = np.nan
        if len(pairs) >= min_assets and pairs.factor.nunique() > 1 and pairs.label.nunique() > 1:
            ic = pairs.factor.corr(pairs.label)
            rank_ic = float(spearmanr(pairs.factor, pairs.label).statistic)
        daily.append({"date": date, "assets_total": len(row), "factor_count": len(valid_scores),
                      "factor_coverage": len(valid_scores) / len(row), "valid_pairs": len(pairs),
                      "ic": ic, "rank_ic": rank_ic,
                      "factor_mean": valid_scores.mean(), "factor_std": valid_scores.std(ddof=1),
                      "factor_min": valid_scores.min(), "factor_max": valid_scores.max()})
        if len(valid_scores) < max(groups, min_assets) or valid_scores.nunique() < 2:
            continue
        ordered = valid_scores.rename("factor").rename_axis("asset").reset_index().sort_values(["factor", "asset"])
        ordered["group"] = np.arange(len(ordered)) * groups // len(ordered) + 1
        # Only after group formation do labels enter: no re-ranking on future availability.
        ordered["label"] = ordered.asset.map(y)
        ordered["date"] = date
        assignments.append(ordered)
        for g in range(1, groups+1):
            part = ordered[ordered.group == g]
            grouped.append({"date": date, "group": g, "formed_count": len(part),
                            "valid_count": int(part.label.notna().sum()),
                            "mean_forward_return": part.label.mean()})
    d = pd.DataFrame(daily)
    g = pd.DataFrame(grouped, columns=["date", "group", "formed_count", "valid_count", "mean_forward_return"])
    if not g.empty:
        wide = g.pivot(index="date", columns="group", values="mean_forward_return")
        spread = (wide.get(groups) - wide.get(1)).rename("spread").reset_index()
    else:
        spread = pd.DataFrame(columns=["date", "spread"])
    assignment = pd.concat(assignments, ignore_index=True) if assignments else pd.DataFrame()
    summary = {}
    for key in ["ic", "rank_ic"]:
        s = d[key].dropna()
        summary.update({f"{key}_mean": s.mean(), f"{key}_std": s.std(ddof=1),
                        f"{key}_valid_days": len(s)})
    summary["mean_factor_coverage"] = d.factor_coverage.mean()
    summary["mean_spread"] = spread.spread.mean()
    summary["spread_valid_days"] = int(spread.spread.notna().sum())
    return {"daily": d, "groups": g, "spread": spread, "assignments": assignment, "summary": summary}


def performance(daily, initial_cash, annualization=252, risk_free_annual=0.0, trades=None):
    nav = daily.nav.to_numpy(dtype=float)
    if not len(nav):
        raise ValueError("Cannot evaluate an empty NAV")
    returns = np.diff(np.r_[initial_cash, nav]) / np.r_[initial_cash, nav][:-1]
    rf = (1 + risk_free_annual) ** (1/annualization) - 1
    excess = returns - rf
    vol = np.std(returns, ddof=1) if len(nav) > 1 else np.nan
    excess_vol = np.std(excess, ddof=1) if len(nav) > 1 else np.nan
    highs = np.maximum.accumulate(np.r_[initial_cash, nav])[1:]
    dd = 1 - nav / highs
    return {
        "sessions": len(nav), "total_return": nav[-1]/initial_cash-1,
        "annualized_return": (nav[-1]/initial_cash)**(annualization/len(nav))-1,
        "annualized_volatility": np.sqrt(annualization)*vol,
        "sharpe": np.sqrt(annualization)*np.mean(excess)/excess_vol if excess_vol > 1e-15 else np.nan,
        "max_drawdown": np.max(dd), "total_cost": daily.cost.sum(),
        "mean_turnover": daily.turnover.mean(), "total_turnover": daily.turnover.sum(),
        "max_asset_weight": daily.max_weight.max(), "mean_holdings": daily.holdings_count.mean(),
        "trade_count": len(trades) if trades is not None else 0,
        "blocked_orders": int(daily.blocked_orders.sum()),
        "stale_holding_days": int(daily.stale_holdings.sum()),
        "gross_attribution_total_return": daily.gross_attribution_nav.iloc[-1]/initial_cash-1,
    }


def reconcile(result, initial_cash, buy_cost, sell_cost):
    """Rebuild cash and unit balances from fills, without calling engine logic."""
    errors, cash = [], initial_cash
    balances = {}
    for _, day in result.daily.iterrows():
        fills = result.trades[result.trades.date == day.date]
        for _, fill in fills.iterrows():
            sign = 1 if fill.side == "buy" else -1
            cash -= sign * fill.notional + fill.cost
            balances[fill.asset] = balances.get(fill.asset, 0.0) + sign*fill.units
            rate = buy_cost if sign == 1 else sell_cost
            if not np.isclose(fill.cost, rate * fill.notional, atol=1e-8):
                errors.append("fee")
            if not fill.signal_date < fill.date:
                errors.append("signal timing")
        pos = result.positions[result.positions.date == day.date].set_index("asset")
        reported = pos.units.to_dict()
        if any(not np.isclose(units, reported.get(asset, 0), atol=1e-7, rtol=1e-10) for asset, units in balances.items()):
            errors.append("units")
        if not np.isclose(cash, day.cash, atol=1e-6, rtol=1e-10):
            errors.append("cash")
        if not np.isclose(cash + pos.value.sum(), day.nav, atol=1e-6, rtol=1e-10):
            errors.append("nav")
        if not np.isclose(fills.cost.sum(), day.cost, atol=1e-8):
            errors.append("daily cost")
        if not np.isclose((day.buy_notional+day.sell_notional)/day.opening_nav, day.turnover):
            errors.append("turnover")
    final_multiple = np.prod(1+result.daily["return"])
    if not np.isclose(final_multiple, result.daily.nav.iloc[-1]/initial_cash, atol=1e-10):
        errors.append("compounding")
    return {"passed": not errors, "errors": sorted(set(errors)), "checked_sessions": len(result.daily),
            "cash_tolerance": 1e-6, "relative_tolerance": 1e-10}


# The workbench below accepts aggregate, public daily ledgers. It never calls the
# backtest engine, changes a portfolio or estimates unavailable stock exposures.
def _dated_series(values, name="series", positive=False):
    """Validate an observed daily series without filling or dropping missing data."""
    if not isinstance(values, pd.Series) or not isinstance(values.index, pd.DatetimeIndex):
        raise ValueError(f"{name}: expected a Series with a DatetimeIndex")
    result = values.copy().astype(float).sort_index()
    if result.empty or result.index.has_duplicates or result.index.hasnans:
        raise ValueError(f"{name}: empty or duplicate/invalid dates")
    if result.index.tz is not None or not (result.index == result.index.normalize()).all():
        raise ValueError(f"{name}: expected timezone-naive daily dates")
    if not np.isfinite(result).all() or (positive and (result <= 0).any()):
        raise ValueError(f"{name}: observations must be finite" + (" and positive" if positive else ""))
    result.index.name = "date"
    return result.rename(name)


def monthly_returns(returns):
    """Compound observed returns by calendar month, retaining partial months.

    Input is a Series or wide DataFrame of simple daily returns, indexed by date.
    Output has a (year, month) MultiIndex and one column per series. There is no
    fabricated zero for unobserved months, and the first observation is included.
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_frame(returns.name or "strategy")
    if returns.empty or not len(returns.columns):
        raise ValueError("Cannot evaluate empty returns")
    checked = pd.concat([_dated_series(returns[c], str(c)) for c in returns], axis=1)
    if (checked <= -1).any().any():
        raise ValueError("Simple returns must exceed -100%")
    groups = [checked.index.year.rename("year"), checked.index.month.rename("month")]
    return (1 + checked).groupby(groups).prod() - 1


def rolling_risk(returns, window=60, annualization=252, risk_free_annual=0.0):
    """Trailing, full-window daily-return diagnostics; no look-ahead or backfill.

    Volatility uses sample standard deviation. Downside deviation is the square
    root of the mean squared negative excess returns, including zero for positive
    observations (the denominator is all window observations). Annual risk-free
    rate is converted geometrically to a daily minimum acceptable return.
    """
    if not isinstance(window, (int, np.integer)) or window < 2:
        raise ValueError("window must be an integer of at least 2 observations")
    if annualization <= 0 or risk_free_annual <= -1:
        raise ValueError("Invalid annualization or annual risk-free rate")
    r = _dated_series(returns, returns.name or "strategy")
    if (r <= -1).any():
        raise ValueError("Simple returns must exceed -100%")
    rf = (1 + risk_free_annual) ** (1 / annualization) - 1
    excess = r - rf
    cumulative = (1 + r).rolling(window, min_periods=window).apply(np.prod, raw=True)
    volatility = r.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(annualization)
    downside = np.sqrt(excess.clip(upper=0).pow(2).rolling(window, min_periods=window).mean() * annualization)
    return pd.DataFrame({
        "rolling_return": cumulative - 1,
        "annualized_return": cumulative.pow(annualization / window) - 1,
        "annualized_volatility": volatility,
        "downside_deviation": downside,
        "sharpe": excess.rolling(window, min_periods=window).mean() * annualization / volatility.where(volatility > 1e-15),
    })


def drawdown_episodes(wealth, initial_value=1.0, baseline_date=None, min_depth=0.0):
    """List completed and open peak-to-recovery episodes, deepest first.

    Depth is a positive loss fraction. Sessions count *observed* daily intervals;
    recovery_sessions is trough-to-recovery, duration_sessions is peak-to-end.
    An open episode has NaT recovery_date and NaN recovery_sessions. If the first
    loss starts from initial capital with no dated baseline, peak_date and calendar
    duration remain missing instead of inventing a previous trading session.
    """
    if initial_value <= 0 or not np.isfinite(initial_value) or not 0 <= min_depth <= 1:
        raise ValueError("Invalid initial value or minimum drawdown depth")
    s = _dated_series(wealth, wealth.name or "strategy", positive=True)
    baseline = pd.Timestamp(baseline_date) if baseline_date is not None else pd.NaT
    if pd.notna(baseline) and baseline >= s.index[0]:
        raise ValueError("baseline_date must precede the first observation")
    fields = ["peak_date", "start_date", "trough_date", "recovery_date", "end_date", "depth",
              "decline_sessions", "recovery_sessions", "duration_sessions", "underwater_sessions",
              "calendar_days", "recovered"]
    peak, peak_index, peak_date = float(initial_value), -1, baseline
    current, rows = None, []

    def finish(end_index, end_date, recovered):
        depth = 1 - current["trough_value"] / peak
        if depth + 1e-15 < min_depth:
            return
        rows.append({
            "peak_date": peak_date, "start_date": current["start_date"],
            "trough_date": current["trough_date"], "recovery_date": end_date if recovered else pd.NaT,
            "end_date": end_date, "depth": depth,
            "decline_sessions": current["trough_index"] - peak_index,
            "recovery_sessions": end_index - current["trough_index"] if recovered else np.nan,
            "duration_sessions": end_index - peak_index,
            "underwater_sessions": current["underwater_sessions"],
            "calendar_days": (end_date - peak_date).days if pd.notna(peak_date) else np.nan,
            "recovered": recovered,
        })

    for i, (date, value) in enumerate(s.items()):
        if value >= peak * (1 - 1e-12):
            if current is not None:
                finish(i, date, True)
                current = None
            peak, peak_index, peak_date = max(peak, float(value)), i, date
        else:
            if current is None:
                current = {"start_date": date, "trough_date": date, "trough_index": i,
                           "trough_value": float(value), "underwater_sessions": 0}
            current["underwater_sessions"] += 1
            if value < current["trough_value"]:
                current.update(trough_value=float(value), trough_date=date, trough_index=i)
    if current is not None:
        finish(len(s) - 1, s.index[-1], False)
    return pd.DataFrame(rows, columns=fields).sort_values("depth", ascending=False, kind="stable").reset_index(drop=True)


def _risk_summary(returns, wealth, annualization, risk_free_annual):
    rf = (1 + risk_free_annual) ** (1 / annualization) - 1
    excess = returns - rf
    volatility = returns.std(ddof=1) * np.sqrt(annualization)
    downside = np.sqrt(np.mean(np.minimum(excess, 0) ** 2) * annualization)
    drawdown = 1 - wealth / np.maximum.accumulate(np.r_[1.0, wealth.to_numpy()])[1:]
    annual_return = wealth.iloc[-1] ** (annualization / len(wealth)) - 1
    quantile = returns.quantile(.05)
    max_drawdown = float(drawdown.max())
    return {
        "sessions": len(returns), "total_return": float(wealth.iloc[-1] - 1),
        "annualized_return": float(annual_return), "annualized_volatility": float(volatility),
        "sharpe": float(excess.mean() * annualization / volatility) if volatility > 1e-15 else np.nan,
        "downside_deviation": float(downside),
        "sortino": float(excess.mean() * annualization / downside) if downside > 1e-15 else np.nan,
        "max_drawdown": max_drawdown,
        "calmar": float(annual_return / max_drawdown) if max_drawdown > 1e-15 else np.nan,
        "daily_var_95": float(-quantile),
        "daily_expected_shortfall_95": float(-returns[returns <= quantile].mean()),
        "best_day": float(returns.max()), "worst_day": float(returns.min()),
        "positive_day_fraction": float((returns > 0).mean()),
    }


def compare_strategies(ledgers, benchmarks=None, initial_cash=1e6, start=None, end=None,
                       benchmark_initial_values=None, window=60, min_drawdown_depth=.01,
                       annualization=252, risk_free_annual=0.0):
    """Fair, read-only comparison of existing daily strategy and benchmark paths.

    ``ledgers`` maps display names to DataFrames with date/nav; cost, cash,
    turnover and other engine aggregate columns are optional. ``initial_cash``
    can be one number or a name-to-capital mapping. ``benchmarks`` maps names to
    dated Series. Their default initial value is 1, suitable for published
    normalized curves; use ``benchmark_initial_values`` for raw price indices.

    All paths use the exact same observed sessions. A selected subperiod begins
    at the previous close (or explicit initial capital), retaining its first-day
    return and fee. We reject internal observation gaps and inconsistent dated
    baselines instead of interpolating or treating multi-day changes as daily.
    A subperiod slices an existing portfolio path; it is not a new cash-start
    backtest. Annualization assumes the supplied observations are trading days.

    Output: summary (series index); wealth/returns/drawdowns/correlation (wide);
    monthly ((year, month) index); rolling and episodes (long, series column);
    relative (one strategy/benchmark pair per row); assumptions (JSON-safe dict).
    Missing ledger fields stay NaN, including unknown fees for benchmark curves.
    """
    if not ledgers:
        raise ValueError("At least one strategy ledger is required")
    if annualization <= 0 or risk_free_annual <= -1:
        raise ValueError("Invalid annualization or annual risk-free rate")
    if not isinstance(window, (int, np.integer)) or window < 2:
        raise ValueError("window must be an integer of at least 2 observations")
    benchmarks = benchmarks or {}
    benchmark_initial_values = benchmark_initial_values or {}
    if set(ledgers) & set(benchmarks):
        raise ValueError("Strategy and benchmark names must be distinct")
    paths, frames, initial = {}, {}, {}
    for name, raw in ledgers.items():
        if not {"date", "nav"}.issubset(raw.columns):
            raise ValueError(f"{name}: ledger requires date and nav")
        frame = raw.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.set_index("date").sort_index()
        paths[name] = _dated_series(frame.nav, name, positive=True)
        frames[name] = frame
        initial[name] = float(initial_cash[name] if isinstance(initial_cash, dict) else initial_cash)
    for name, values in benchmarks.items():
        paths[name] = _dated_series(values, name, positive=True)
        initial[name] = float(benchmark_initial_values.get(name, 1.0))
    if any(not np.isfinite(v) or v <= 0 for v in initial.values()):
        raise ValueError("All initial values must be finite and positive")

    shared = next(iter(paths.values())).index
    union = shared
    for values in paths.values():
        shared = shared.intersection(values.index)
        union = union.union(values.index)
    if start is not None:
        shared = shared[shared >= pd.Timestamp(start)]
    if end is not None:
        shared = shared[shared <= pd.Timestamp(end)]
    shared = shared.sort_values()
    if shared.empty:
        raise ValueError("No overlapping observed dates in the requested interval")
    missing = union[(union >= shared[0]) & (union <= shared[-1])].difference(shared)
    if len(missing):
        raise ValueError("Internal observation gaps prevent a daily comparison; no interpolation is applied")

    bases, base_dates = {}, {}
    for name, values in paths.items():
        previous = values[values.index < shared[0]]
        bases[name] = float(previous.iloc[-1]) if len(previous) else initial[name]
        base_dates[name] = previous.index[-1] if len(previous) else None
    observed_baselines = {d for d in base_dates.values() if d is not None}
    if len(observed_baselines) > 1:
        raise ValueError("Series have inconsistent previous-close dates at the comparison boundary")
    baseline_date = next(iter(observed_baselines), None)
    wealth = pd.DataFrame({name: values.reindex(shared) / bases[name] for name, values in paths.items()})
    wealth.index.name = "date"
    returns = wealth.pct_change(fill_method=None)
    returns.iloc[0] = wealth.iloc[0] - 1
    highs = wealth.cummax().clip(lower=1.)
    drawdowns = 1 - wealth / highs
    summaries, rolling_tables, episode_tables, pairs = [], [], [], []

    for name in wealth:
        row = {"series": name, "kind": "strategy" if name in ledgers else "benchmark",
               "start": shared[0], "end": shared[-1], "baseline_nav": bases[name],
               **_risk_summary(returns[name], wealth[name], annualization, risk_free_annual)}
        row.update({k: np.nan for k in ["total_cost", "cost_over_starting_nav", "mean_cash_weight",
                    "ending_cash_weight", "mean_turnover", "total_turnover", "mean_holdings",
                    "max_asset_weight", "blocked_orders", "stale_holding_days"]})
        if name in frames:
            part = frames[name].reindex(shared)
            for col, target, operation in [
                ("cost", "total_cost", "sum"), ("turnover", "mean_turnover", "mean"),
                ("turnover", "total_turnover", "sum"), ("holdings_count", "mean_holdings", "mean"),
                ("max_weight", "max_asset_weight", "max"), ("blocked_orders", "blocked_orders", "sum"),
                ("stale_holdings", "stale_holding_days", "sum"),
            ]:
                if col in part and np.isfinite(pd.to_numeric(part[col], errors="coerce")).all():
                    row[target] = float(getattr(part[col], operation)())
            if np.isfinite(row["total_cost"]):
                row["cost_over_starting_nav"] = row["total_cost"] / bases[name]
            if "cash" in part and np.isfinite(pd.to_numeric(part.cash, errors="coerce")).all():
                cash_weights = part.cash / part.nav
                row["mean_cash_weight"] = float(cash_weights.mean())
                row["ending_cash_weight"] = float(cash_weights.iloc[-1])
        summaries.append(row)
        rolling = rolling_risk(returns[name], window, annualization, risk_free_annual).reset_index()
        rolling.insert(1, "series", name)
        rolling_tables.append(rolling)
        episodes = drawdown_episodes(wealth[name], baseline_date=baseline_date, min_depth=min_drawdown_depth)
        episodes.insert(0, "series", name)
        episode_tables.append(episodes)

    for name in ledgers:
        for benchmark in benchmarks:
            active = returns[name] - returns[benchmark]
            tracking = active.std(ddof=1) * np.sqrt(annualization)
            benchmark_variance = returns[benchmark].var(ddof=1)
            pairs.append({"series": name, "benchmark": benchmark,
                          "benchmark_return": float(wealth[benchmark].iloc[-1] - 1),
                          "excess_return_pp": float(wealth[name].iloc[-1] - wealth[benchmark].iloc[-1]),
                          "relative_wealth_return": float(wealth[name].iloc[-1] / wealth[benchmark].iloc[-1] - 1),
                          "beta": float(returns[name].cov(returns[benchmark]) / benchmark_variance)
                          if benchmark_variance > 1e-15 else np.nan,
                          "annualized_tracking_error": float(tracking),
                          "information_ratio": float(active.mean() * annualization / tracking)
                          if tracking > 1e-15 else np.nan})

    return {
        "summary": pd.DataFrame(summaries).set_index("series"), "wealth": wealth,
        "returns": returns, "drawdowns": drawdowns, "correlation": returns.corr(min_periods=2),
        "monthly": monthly_returns(returns), "rolling": pd.concat(rolling_tables, ignore_index=True),
        "episodes": pd.concat(episode_tables, ignore_index=True),
        "relative": pd.DataFrame(pairs, columns=["series", "benchmark", "benchmark_return",
                    "excess_return_pp", "relative_wealth_return", "beta", "annualized_tracking_error", "information_ratio"]),
        "assumptions": {
            "annualization": int(annualization), "risk_free_annual": float(risk_free_annual),
            "start": shared[0].date().isoformat(), "end": shared[-1].date().isoformat(),
            "sessions": len(shared), "window": int(window),
            "baseline_date": baseline_date.date().isoformat() if baseline_date is not None else None,
            "first_day_return_included": True, "internal_gaps_interpolated": False,
            "period_is_existing_path_slice": True,
            "benchmark_initial_values": {name: initial[name] for name in benchmarks},
            "notes": [
                "同一交易日交集；内部缺日拒绝计算，不插值。", "区间起点使用前收盘或显式初始净值，包含首日损益。",
                "日期筛选是现有持仓路径的区间切片，不是从现金重新回测。", "年化按交易日252天（或指定参数）；不足一年亦机械年化。",
                "波动率为样本标准差；下行偏差以全部观察数为分母。", "历史VaR/ES为日频95%损失度量，正值表示损失，不能预测极端损失上限。",
                "月度表包含首尾不完整月份；缺失月份不填零。", "换手为买卖名义金额之和除以当日开盘权益，即双边换手。",
                "基准曲线没有交易费或持仓明细时保留缺失；现金不假定利息。", "费用占比为观察期费用/区间期初净值，非无费用策略的收益差。",
            ],
        },
    }
