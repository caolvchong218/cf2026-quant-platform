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


def _affordable_notional(cash_budget: float, rate: float, minimum: float) -> float:
    """Solve x + max(rate*x, minimum) <= budget for a nonzero buy fill."""
    return max(0.0, min(cash_budget / (1 + rate), cash_budget - minimum))


def _minimum_fee_buy_scale(notionals, cash, rate, minimum):
    """Largest common scale whose fills and per-order minimum fees fit cash."""
    values = np.asarray(notionals, dtype=float)
    if not len(values):
        return 1.0

    def required(scale):
        scaled = values * scale
        active = scaled > 1e-8
        return float(scaled[active].sum() + np.maximum(scaled[active] * rate, minimum).sum())

    available = max(0., cash)
    if required(1.) <= available:
        return 1.0
    if available <= minimum * int((values > 1e-8).sum()):
        return 0.0
    low, high = 0., 1.
    for _ in range(60):
        midpoint = (low + high) / 2
        if required(midpoint) <= available:
            low = midpoint
        else:
            high = midpoint
    return low


def _dropout_rebalance(*, units, cash, scores, assets, opens, raw_open, up_limit,
                       down_limit, config, dropout, minimum_commission, liquidity,
                       date, signal_date):
    """Qlib-inspired top/bottom selection with fills based on actual holdings.

    Finite-score entrants only; missing held scores rank last. When candidates
    are scarce, only genuinely needed slots are freed. Blocked or partial sells
    continue to occupy a slot, so failed exits cannot increase holdings beyond K.
    Retained units are never resized; all new buys share available cash equally,
    including actual fees. This uses fractional adjusted units, not exchange lots.
    """
    held = set(np.flatnonzero(units > 1e-10))

    def ranking(k):
        return (-float(scores[k]) if np.isfinite(scores[k]) else np.inf, str(assets[k]))

    available = sorted((k for k in range(len(assets)) if k not in held and np.isfinite(scores[k])), key=ranking)
    today = available[:dropout + max(0, config.holdings - len(held))]
    combined = sorted(held.union(today), key=ranking)
    replace_count = min(dropout, max(0, len(combined) - config.holdings))
    sell = [k for k in reversed(combined[-replace_count:]) if k in held] if replace_count else []
    trade_rows, order_rows = [], []
    bought = sold = fees = 0.
    blocked = 0

    def reason_for(k, side):
        if not np.isfinite(opens[k]) or opens[k] <= 0:
            return "missing_open"
        if config.respect_price_limits:
            if not np.isfinite(raw_open[k]):
                return "missing_raw_open"
            limit = up_limit[k] if side == "buy" else down_limit[k]
            if not np.isfinite(limit):
                return "missing_limit"
            if side == "buy" and raw_open[k] >= limit - 1e-8:
                return "upper_limit"
            if side == "sell" and raw_open[k] <= limit + 1e-8:
                return "lower_limit"
        return ""

    def order(k, side, status, reason, requested, filled):
        order_rows.append({"date": date, "signal_date": signal_date, "asset": assets[k],
                           "side": side, "status": status, "reason": reason,
                           "requested_units": float(requested), "filled_units": float(filled)})

    def limited(k, requested):
        if liquidity is None:
            return requested
        cap = liquidity[k] if np.isfinite(liquidity[k]) else 0.
        return min(requested, max(0., cap) / opens[k])

    def trade(k, side, quantity, notional, cost):
        trade_rows.append({"date": date, "signal_date": signal_date, "asset": assets[k],
                           "side": side, "units": quantity, "adjusted_price": opens[k],
                           "raw_price": raw_open[k], "notional": notional, "cost": cost})

    for k in sell:
        requested = float(units[k])
        reason = reason_for(k, "sell")
        if reason:
            blocked += 1
            order(k, "sell", "blocked", reason, requested, 0.)
            continue
        quantity = limited(k, requested)
        notional = quantity * opens[k]
        if notional <= 1e-8:
            order(k, "sell", "liquidity_limited", "", requested, 0.)
            continue
        cost = max(config.sell_cost * notional, minimum_commission)
        if cash + notional < cost - 1e-8:
            blocked += 1
            order(k, "sell", "blocked", "insufficient_cash_for_fee", requested, 0.)
            continue
        cash += notional - cost
        units[k] -= quantity
        sold += notional
        fees += cost
        trade(k, "sell", quantity, notional, cost)
        order(k, "sell", "liquidity_limited" if quantity < requested - 1e-12 else "filled", "", requested, quantity)

    # Plan only against slots actually released by fills, not intended sells.
    vacancies = max(0, config.holdings - int((units > 1e-10).sum()))
    buy = today[:vacancies]
    per_order_budget = max(0., cash) / len(buy) if buy else 0.
    requested_notional = _affordable_notional(per_order_budget, config.buy_cost, minimum_commission)
    for k in buy:
        requested = requested_notional / opens[k] if np.isfinite(opens[k]) and opens[k] > 0 else 0.
        reason = reason_for(k, "buy")
        if reason:
            blocked += 1
            order(k, "buy", "blocked", reason, requested, 0.)
            continue
        if requested_notional <= 1e-8:
            blocked += 1
            order(k, "buy", "blocked", "insufficient_cash_for_fee", requested, 0.)
            continue
        quantity = limited(k, requested)
        notional = quantity * opens[k]
        if notional <= 1e-8:
            order(k, "buy", "liquidity_limited", "", requested, 0.)
            continue
        cost = max(config.buy_cost * notional, minimum_commission)
        # The exact per-order inversion includes min fees; this check also
        # prevents numerical or future execution changes from borrowing cash.
        if notional + cost > cash + 1e-8:
            blocked += 1
            order(k, "buy", "blocked", "insufficient_cash_for_fee", requested, 0.)
            continue
        cash -= notional + cost
        units[k] += quantity
        bought += notional
        fees += cost
        trade(k, "buy", quantity, notional, cost)
        order(k, "buy", "liquidity_limited" if quantity < requested - 1e-12 else "filled", "", requested, quantity)
    return cash, bought, sold, fees, blocked, trade_rows, order_rows


def run_backtest(market: pd.DataFrame, calendar: pd.DatetimeIndex,
                 scores: pd.DataFrame, config: Config,
                 target_weights: pd.DataFrame | None = None,
                 participation_limit: float | None = None,
                 rebalance_mask: pd.Series | None = None,
                 reserve_cost_buffer: bool = True,
                 dropout: int | None = None,
                 minimum_commission: float = 0.0) -> BacktestResult:
    """Execute close-formed targets at the following open.

    An optional Boolean mask is indexed by *formation* session, like scores and
    targets. Its value at t therefore controls execution at t+1. The first
    evaluation open always establishes the supplied prior-close target. Without
    a mask the existing periodic schedule is unchanged. Disabling the optional
    cost reserve sizes targets from opening NAV; the cash-scaled buy mechanism
    still pays actual fees and prevents borrowing.

    Optional dropout replaces at most that many existing holdings per rebalance,
    without resizing retained holdings. It cannot be combined with target
    weights; it allocates actual available cash including fees and does not use
    the ordinary target-weight cost buffer. The minimum commission applies to
    each nonzero actual fill; defaults preserve all existing P1/RSRS behavior.
    """
    assets = scores.columns
    if len(assets) == 0 or not scores.index.equals(calendar):
        raise ValueError("Scores must use the complete supplied calendar")
    if dropout is not None:
        if type(dropout) is not int or not 0 <= dropout <= config.holdings:
            raise ValueError("dropout must be an integer between zero and holdings")
        if target_weights is not None:
            raise ValueError("dropout cannot be combined with target_weights")
        if not assets.is_unique:
            raise ValueError("dropout requires unique asset codes")
    if (isinstance(minimum_commission, bool)
            or not isinstance(minimum_commission, (int, float, np.integer, np.floating))
            or not np.isfinite(minimum_commission) or minimum_commission < 0):
        raise ValueError("minimum_commission must be finite and nonnegative")
    if target_weights is not None:
        if not target_weights.index.equals(calendar) or not target_weights.columns.equals(assets):
            raise ValueError("Target weights must preserve the date/asset axes")
        weights_array=target_weights.to_numpy(dtype=float)
        if not np.isfinite(weights_array).all() or (weights_array<0).any() or (weights_array.sum(axis=1)>1+1e-10).any():
            raise ValueError("Weights must be finite, long-only, and sum to at most one")
    if participation_limit is not None and not 0<participation_limit<=1:
        raise ValueError("Participation limit must lie in (0,1]")
    if rebalance_mask is not None:
        if (not isinstance(rebalance_mask, pd.Series)
                or not rebalance_mask.index.equals(calendar)
                or not pd.api.types.is_bool_dtype(rebalance_mask.dtype)
                or rebalance_mask.isna().any()):
            raise ValueError("Rebalance mask must be a Boolean Series on the complete supplied calendar")
        mask_array = rebalance_mask.to_numpy(dtype=bool)
    if not isinstance(reserve_cost_buffer, bool):
        raise ValueError("reserve_cost_buffer must be Boolean")
    fields = ["open", "close", "raw_open", "up_limit", "down_limit"]
    matrices = {f: panel(market, calendar, f).reindex(columns=assets).to_numpy() for f in fields}
    # Budget based on prior-session rolling turnover, available before execution.
    liquidity = None
    if participation_limit is not None:
        raw_turnover=panel(market,calendar,'raw_close')*panel(market,calendar,'volume')
        liquidity=raw_turnover.rolling(20,min_periods=20).mean().shift(1).reindex(columns=assets).to_numpy()*participation_limit
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
        rebalance = ((trade_session == 0 or bool(mask_array[j-1])) if rebalance_mask is not None
                     else trade_session % config.rebalance_every == 0)
        if rebalance and dropout is not None:
            cash, bought, sold, fee, blocked, new_trades, new_orders = _dropout_rebalance(
                units=units, cash=cash, scores=signals[j-1], assets=assets,
                opens=opens, raw_open=raw_open, up_limit=matrices["up_limit"][j],
                down_limit=matrices["down_limit"][j], config=config, dropout=dropout,
                minimum_commission=minimum_commission,
                liquidity=liquidity[j] if liquidity is not None else None,
                date=date, signal_date=calendar[j-1])
            trades.extend(new_trades)
            orders.extend(new_orders)
        if rebalance and dropout is None:
            weights = (weights_array[j-1] if target_weights is not None else
                       top_equal_weights(pd.Series(signals[j-1], index=assets), config.holdings).to_numpy())
            # Reserve an explicit cost buffer. No implicit borrowing when all weights sum to one.
            budget = (opening_nav / (1 + config.buy_cost + config.sell_cost)
                      if reserve_cost_buffer else opening_nav)
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
                if side == "buy" and minimum_commission > 0:
                    scale = _minimum_fee_buy_scale([delta[k] * opens[k] for k in allowed],
                                                   cash, config.buy_cost, minimum_commission)
                for k in allowed:
                    requested_quantity = abs(delta[k]) * (scale if side == "buy" else 1)
                    quantity = requested_quantity
                    liquidity_limited = False
                    if liquidity is not None:
                        max_notional=liquidity[j,k] if np.isfinite(liquidity[j,k]) else 0.
                        quantity=min(quantity,max_notional/opens[k])
                        liquidity_limited=quantity<requested_quantity-1e-12
                    notional = quantity * opens[k]
                    cost = notional * (config.buy_cost if side == "buy" else config.sell_cost)
                    if minimum_commission > 0 and notional > 1e-8:
                        cost = max(cost, minimum_commission)
                    if side == "sell" and notional > 1e-8 and cash + notional < cost - 1e-8:
                        blocked += 1
                        orders.append({"date": date, "signal_date": calendar[j-1], "asset": assets[k],
                                       "side": side, "status": "blocked", "reason": "insufficient_cash_for_fee",
                                       "requested_units": float(abs(delta[k])), "filled_units": 0.0})
                        continue
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
                                   "side": side, "status": "liquidity_limited" if liquidity_limited else ("filled" if scale == 1 or side == "sell" else "cash_scaled"),
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
        if nav <= 0:
            raise ValueError("Nonpositive portfolio NAV after valuation; stop rather than emit undefined returns.")
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
