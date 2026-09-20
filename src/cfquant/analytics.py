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

