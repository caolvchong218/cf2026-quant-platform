"""Daily long-only research accounting, independent of factor formulas and UI.

Positions are fractional adjusted total-return units (NOT exchange share lots).
Missing quotes cannot execute. Missing closes use last known value for valuation
only; every such holding is counted. Orders expire at the rebalance session end.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .config import Config
from .factors import panel
from .portfolio import top_equal_weights


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    positions: pd.DataFrame
    orders: pd.DataFrame


def run_backtest(market: pd.DataFrame, calendar: pd.DatetimeIndex,
                 scores: pd.DataFrame, config: Config) -> BacktestResult:
    assets = scores.columns
    if len(assets) == 0 or not scores.index.equals(calendar):
        raise ValueError("Scores must use the complete supplied calendar")
    fields = ["open", "close", "raw_open", "up_limit", "down_limit"]
    matrices = {f: panel(market, calendar, f).reindex(columns=assets).to_numpy() for f in fields}
    signals = scores.to_numpy()
    dates = calendar[(calendar >= config.start) & (calendar <= config.end)]
    if dates.empty:
        raise ValueError("No sessions in configured date range")
    if dates[0] == calendar[0]:
        raise ValueError("Supply at least one pre-start session for next-open execution")
    units = np.zeros(len(assets))
    last_close = np.full(len(assets), np.nan)
    stale_age = np.zeros(len(assets), dtype=int)
    previous_stale = np.zeros(len(assets), dtype=bool)
    cash = config.initial_cash
    prior_nav = config.initial_cash
    cumulative_cost = 0.0
    daily, trades, positions, orders = [], [], [], []
    trade_session = 0
    for j, date in enumerate(calendar):
        if date > dates[-1]:
            break
        close = matrices["close"][j]
        stale_age = np.where(np.isfinite(close), 0, stale_age + 1)
        if date < dates[0]:
            last_close = np.where(np.isfinite(close), close, last_close)
            previous_stale = stale_age > config.max_stale_sessions
            continue
        opens = matrices["open"][j]
        raw_open = matrices["raw_open"][j]
        # Only prior-session staleness is known at today's open.
        prior_stale = np.zeros(len(assets), dtype=bool) if j == 0 else previous_stale
        opening_mark = np.where(np.isfinite(opens), opens, np.where(prior_stale, 0, last_close))
        if np.any((units > 1e-12) & ~np.isfinite(opening_mark)):
            raise ValueError("Held position has no valuation")
        opening_nav = cash + np.nansum(units * opening_mark)
        if opening_nav <= 0:
            raise ValueError("Nonpositive portfolio NAV")
        bought = sold = fee = 0.0
        blocked = 0
        rebalance = trade_session % config.rebalance_every == 0
        if rebalance:
            weights = top_equal_weights(pd.Series(signals[j-1], index=assets), config.holdings).to_numpy()
            # Reserve an explicit cost buffer. No implicit borrowing when all weights sum to one.
            budget = opening_nav / (1 + config.buy_cost + config.sell_cost)
            target_value = budget * weights
            desired = np.divide(target_value, opening_mark, out=np.zeros_like(units),
                                where=np.isfinite(opening_mark) & (opening_mark > 0))
            delta = desired - units
            can_quote = np.isfinite(opens) & (opens > 0)
            for side, indices in [("sell", np.flatnonzero(delta < -1e-10)),
                                  ("buy", np.flatnonzero(delta > 1e-10))]:
                allowed = []
                for k in indices:
                    reason = ""
                    if not can_quote[k]:
                        reason = "missing_open"
                    elif config.respect_price_limits:
                        limit = matrices["up_limit" if side == "buy" else "down_limit"][j, k]
                        if not np.isfinite(limit):
                            reason = "missing_limit"
                        elif side == "buy" and raw_open[k] >= limit - 1e-8:
                            reason = "upper_limit"
                        elif side == "sell" and raw_open[k] <= limit + 1e-8:
                            reason = "lower_limit"
                    if reason:
                        blocked += 1
                        orders.append({"date": date, "signal_date": calendar[j-1], "asset": assets[k],
                                       "side": side, "status": "blocked", "reason": reason,
                                       "requested_units": float(abs(delta[k])), "filled_units": 0.0})
                    else:
                        allowed.append(k)
                # Pro-rate buys if halted/limit-blocked sells leave insufficient cash.
                requested = sum(delta[k] * opens[k] * (1 + config.buy_cost) for k in allowed) if side == "buy" else 0
                scale = min(1.0, max(0.0, cash) / requested) if requested > 0 else 1.0
                for k in allowed:
                    quantity = abs(delta[k]) * (scale if side == "buy" else 1)
                    notional = quantity * opens[k]
                    cost = notional * (config.buy_cost if side == "buy" else config.sell_cost)
                    if notional > 1e-8:
                        if side == "buy":
                            cash -= notional + cost
                            units[k] += quantity
                            bought += notional
                        else:
                            cash += notional - cost
                            units[k] -= quantity
                            sold += notional
                        fee += cost
                        trades.append({"date": date, "signal_date": calendar[j-1], "asset": assets[k],
                                       "side": side, "units": quantity, "adjusted_price": opens[k],
                                       "raw_price": raw_open[k], "notional": notional, "cost": cost})
                    orders.append({"date": date, "signal_date": calendar[j-1], "asset": assets[k],
                                   "side": side, "status": "filled" if scale == 1 or side == "sell" else "cash_scaled",
                                   "reason": "", "requested_units": float(abs(delta[k])), "filled_units": quantity})
        if cash < -1e-6 or np.min(units) < -1e-8:
            raise AssertionError("Cash / long-only invariant violated")
        cash = max(0.0, cash)
        units[np.abs(units) < 1e-10] = 0
        stale = (units > 0) & ~np.isfinite(close)
        last_close = np.where(np.isfinite(close), close, last_close)
        previous_stale = stale_age > config.max_stale_sessions
        marks = np.where(previous_stale, 0, last_close)
        values = units * marks
        nav = cash + np.nansum(values)
        cumulative_cost += fee
        daily.append({"date": date, "nav": nav, "return": nav / prior_nav - 1,
                      "cash": cash, "market_value": np.nansum(values), "opening_nav": opening_nav,
                      "buy_notional": bought, "sell_notional": sold, "cost": fee,
                      "cumulative_cost": cumulative_cost, "turnover": (bought + sold) / opening_nav,
                      "holdings_count": int((units > 0).sum()),
                      "max_weight": float(np.nanmax(values / nav)) if np.any(units > 0) else 0.0,
                      "stale_holdings": int(stale.sum()), "blocked_orders": blocked, "rebalance": rebalance,
                      "written_down_holdings": int(((units > 0) & previous_stale).sum()),
                      # Same actual fills, restore paid costs into idle zero-interest cash.
                      "gross_attribution_nav": nav + cumulative_cost})
        for k in np.flatnonzero(units > 0):
            positions.append({"date": date, "asset": assets[k], "units": units[k],
                              "mark_price": marks[k], "value": values[k],
                              "weight": values[k] / nav, "stale_mark": bool(stale[k])})
        prior_nav = nav
        trade_session += 1
    return BacktestResult(pd.DataFrame(daily),
                          pd.DataFrame(trades, columns=["date", "signal_date", "asset", "side", "units", "adjusted_price", "raw_price", "notional", "cost"]),
                          pd.DataFrame(positions, columns=["date", "asset", "units", "mark_price", "value", "weight", "stale_mark"]),
                          pd.DataFrame(orders, columns=["date", "signal_date", "asset", "side", "status", "reason", "requested_units", "filled_units"]))
