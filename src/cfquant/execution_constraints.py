"""Execution stress for adjusted-return research, with raw-equivalent lots.

Accounting remains in the supplied adjusted-price units. The observed
``open / raw_open`` ratio converts each fill to raw-equivalent shares; it does
not generate dividends, splits, custody balances, or settlement cash. This is
not a corporate-action-complete exchange simulator. Fees and adverse slippage
are separate cash costs on the opening reference notional (not double-counted
in the reference fill price). Canonical market volume must be in raw shares.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd

from .config import Config
from .engine import BacktestResult
from .factors import panel
from .portfolio import top_equal_weights


@dataclass(frozen=True)
class ExecutionPolicy:
    lot_size: int = 100
    min_commission: float = 5.0
    buy_commission: float = .0003
    sell_commission: float = .0003
    # Explicit scenario inputs, editable by the user; not an inferred tax rule.
    stamp_tax_schedule: tuple = (("1900-01-01", .001), ("2023-08-28", .0005))
    slippage: float = .0005
    participation: float = .01
    liquidity_window: int = 20

    def __post_init__(self):
        for key in ("lot_size", "liquidity_window"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not math.isfinite(self.min_commission) or self.min_commission < 0:
            raise ValueError("min_commission must be finite and nonnegative")
        for key in ("buy_commission", "sell_commission", "slippage"):
            value = getattr(self, key)
            if not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{key} must lie in [0, 1)")
        if not math.isfinite(self.participation) or not 0 < self.participation <= 1:
            raise ValueError("participation must lie in (0, 1]")
        schedule = tuple((pd.Timestamp(date).normalize(), float(rate)) for date, rate in self.stamp_tax_schedule)
        if not schedule or any(pd.isna(date) or date.tz is not None or not math.isfinite(rate) or not 0 <= rate < 1
                               for date, rate in schedule):
            raise ValueError("stamp_tax_schedule requires valid dates and rates in [0, 1)")
        if any(schedule[i][0] >= schedule[i + 1][0] for i in range(len(schedule) - 1)):
            raise ValueError("stamp_tax_schedule dates must increase strictly")
        object.__setattr__(self, "stamp_tax_schedule", tuple((date.date().isoformat(), rate) for date, rate in schedule))

    def stamp_tax_rate(self, date) -> float:
        known = [rate for effective, rate in self.stamp_tax_schedule if pd.Timestamp(effective) <= pd.Timestamp(date)]
        if not known:
            raise ValueError("stamp_tax_schedule has no effective rate for this session")
        return known[-1]

    def to_dict(self) -> dict:
        return asdict(self)


def _costs(notional: float, side: str, date, policy: ExecutionPolicy):
    commission = max(policy.min_commission, notional * (policy.buy_commission if side == "buy" else policy.sell_commission))
    stamp_tax = notional * policy.stamp_tax_rate(date) if side == "sell" else 0.0
    slippage_cost = notional * policy.slippage
    return commission, stamp_tax, slippage_cost


def run_constrained_backtest(market: pd.DataFrame, calendar: pd.DatetimeIndex,
                             scores: pd.DataFrame, config: Config,
                             target_weights: pd.DataFrame | None = None,
                             policy: ExecutionPolicy = ExecutionPolicy()) -> BacktestResult:
    """Execute prior-session targets with lot, T+1, cost and lagged caps.

    Buy fills are multiples of ``lot_size`` raw-equivalent shares. Partial
    sells also use lots; full liquidation may sell an odd remainder. Opening
    eligible units are frozen before today's buys, enforcing T+1 in adjusted
    research units. Limits require both a positive raw open and the side's
    supplied limit; absent quotes never become executable through valuation.
    ``config.buy_cost/sell_cost`` are superseded by this explicit policy.
    """
    assets = scores.columns
    if (len(assets) == 0 or assets.has_duplicates or not scores.index.equals(calendar)
            or calendar.empty or calendar.has_duplicates or calendar.hasnans
            or not calendar.is_monotonic_increasing or calendar.tz is not None):
        raise ValueError("Scores must use a complete, unique, increasing calendar and asset axes")
    if not {"date", "asset", "open", "close"}.issubset(market.columns):
        raise ValueError("Market requires date, asset, open and close")
    if market.duplicated(["date", "asset"]).any():
        raise ValueError("Duplicate market date/asset keys")
    weights_array = None
    if target_weights is not None:
        if not target_weights.index.equals(calendar) or not target_weights.columns.equals(assets):
            raise ValueError("Target weights must preserve the date/asset axes")
        weights_array = target_weights.to_numpy(dtype=float)
        if (not np.isfinite(weights_array).all() or (weights_array < 0).any()
                or (weights_array.sum(axis=1) > 1 + 1e-10).any()):
            raise ValueError("Weights must be finite, long-only, and sum to at most one")
    fields = ("open", "close", "raw_open", "raw_close", "volume", "up_limit", "down_limit")
    matrices = {field: (panel(market, calendar, field).reindex(columns=assets).to_numpy(dtype=float)
                       if field in market else np.full((len(calendar), len(assets)), np.nan)) for field in fields}
    for field in ("open", "close", "raw_open", "raw_close"):
        values = matrices[field]
        if np.isinf(values).any() or (values[np.isfinite(values)] <= 0).any():
            raise ValueError(f"{field} must contain positive prices or missing values")
    if np.isinf(matrices["volume"]).any() or (matrices["volume"][np.isfinite(matrices["volume"])] < 0).any():
        raise ValueError("volume must be raw shares, nonnegative or missing")
    turnover = pd.DataFrame(matrices["raw_close"] * matrices["volume"], index=calendar, columns=assets)
    liquidity = turnover.rolling(policy.liquidity_window, min_periods=policy.liquidity_window).mean().shift(1).to_numpy() * policy.participation
    dates = calendar[(calendar >= config.start) & (calendar <= config.end)]
    if dates.empty:
        raise ValueError("No sessions in configured date range")
    if dates[0] == calendar[0]:
        raise ValueError("Supply at least one pre-start session for next-open execution")
    signals = scores.to_numpy(dtype=float)
    units = np.zeros(len(assets))
    last_close = np.full(len(assets), np.nan)
    stale_age = np.zeros(len(assets), dtype=int)
    previous_stale = np.zeros(len(assets), dtype=bool)
    cash, prior_nav, cumulative_cost = config.initial_cash, config.initial_cash, 0.0
    daily, trades, positions, orders = [], [], [], []
    trade_session = 0
    for j, date in enumerate(calendar):
        if date > dates[-1]:
            break
        close, opens, raw_open = matrices["close"][j], matrices["open"][j], matrices["raw_open"][j]
        stale_age = np.where(np.isfinite(close), 0, stale_age + 1)
        if date < dates[0]:
            last_close = np.where(np.isfinite(close), close, last_close)
            previous_stale = stale_age > config.max_stale_sessions
            continue
        opening_mark = np.where(np.isfinite(opens), opens, np.where(previous_stale, 0.0, last_close))
        if np.any((units > 1e-12) & ~np.isfinite(opening_mark)):
            raise ValueError("Held position has no valuation")
        opening_nav = cash + np.nansum(units * opening_mark)
        if not np.isfinite(opening_nav) or opening_nav <= 0:
            raise ValueError("Nonpositive portfolio NAV")
        eligible_units = units.copy()
        bought = sold = fee = commission_total = stamp_total = slip_total = 0.0
        blocked = 0
        rebalance = trade_session % config.rebalance_every == 0
        if rebalance:
            weights = (weights_array[j - 1] if weights_array is not None else
                       top_equal_weights(pd.Series(signals[j - 1], index=assets), config.holdings).to_numpy())
            budget = opening_nav / (1 + policy.buy_commission + policy.slippage)
            target_value = budget * weights
            desired = np.divide(target_value, opening_mark, out=np.zeros_like(units),
                                where=np.isfinite(opening_mark) & (opening_mark > 0))
            delta = desired - units
            for side, indices in (("sell", np.flatnonzero(delta < -1e-10)), ("buy", np.flatnonzero(delta > 1e-10))):
                allowed = []
                for k in indices:
                    reason = ""
                    if not np.isfinite(opens[k]) or opens[k] <= 0:
                        reason = "missing_open"
                    elif not np.isfinite(raw_open[k]) or raw_open[k] <= 0:
                        reason = "missing_raw_open"
                    elif config.respect_price_limits:
                        limit = matrices["up_limit" if side == "buy" else "down_limit"][j, k]
                        if not np.isfinite(limit) or limit <= 0:
                            reason = "missing_limit"
                        elif side == "buy" and raw_open[k] >= limit - 1e-8:
                            reason = "upper_limit"
                        elif side == "sell" and raw_open[k] <= limit + 1e-8:
                            reason = "lower_limit"
                    if reason:
                        blocked += 1
                        orders.append({"date": date, "signal_date": calendar[j - 1], "asset": assets[k], "side": side,
                                       "status": "blocked", "reason": reason, "requested_units": float(abs(delta[k])), "filled_units": 0.0})
                    else:
                        allowed.append(k)
                requested_cash = sum(delta[k] * opens[k] + sum(_costs(delta[k] * opens[k], "buy", date, policy))
                                     for k in allowed) if side == "buy" else 0.0
                scale = min(1.0, max(cash, 0.0) / requested_cash) if requested_cash > 0 else 1.0
                for k in allowed:
                    requested_units = float(abs(delta[k]))
                    ratio = opens[k] / raw_open[k]
                    cap = float(liquidity[j, k]) if np.isfinite(liquidity[j, k]) else 0.0
                    quantity = min(requested_units * (scale if side == "buy" else 1.0), cap / opens[k])
                    full_exit = side == "sell" and desired[k] <= 1e-10
                    if side == "sell":
                        quantity = min(quantity, eligible_units[k])
                    # Only a genuinely complete liquidation can dispose of odd lots.
                    odd_lot_exit = full_exit and quantity >= units[k] - 1e-10
                    if odd_lot_exit:
                        quantity = float(units[k])
                    else:
                        quantity = math.floor((quantity * ratio + 1e-9) / policy.lot_size) * policy.lot_size / ratio
                    if side == "buy":
                        affordable_notional = max(0.0, min(cash / (1 + policy.buy_commission + policy.slippage),
                                                           (cash - policy.min_commission) / (1 + policy.slippage)))
                        affordable_lots = math.floor((affordable_notional / raw_open[k] + 1e-9) / policy.lot_size)
                        quantity = min(quantity, affordable_lots * policy.lot_size / ratio)
                    notional = quantity * opens[k]
                    commission = stamp_tax = slippage_cost = 0.0
                    reason = ""
                    if notional <= 1e-8:
                        reason = "liquidity" if cap <= 0 else "lot_or_cash"
                    else:
                        commission, stamp_tax, slippage_cost = _costs(notional, side, date, policy)
                        if side == "sell" and cash + notional < commission + stamp_tax + slippage_cost - 1e-8:
                            quantity = notional = 0.0
                            commission = stamp_tax = slippage_cost = 0.0
                            reason = "sell_fee_cash"
                    cost = commission + stamp_tax + slippage_cost
                    if notional > 1e-8:
                        if side == "buy":
                            cash -= notional + cost
                            units[k] += quantity
                            bought += notional
                        else:
                            cash += notional - cost
                            units[k] -= quantity
                            eligible_units[k] -= quantity
                            sold += notional
                        fee += cost
                        commission_total += commission
                        stamp_total += stamp_tax
                        slip_total += slippage_cost
                        trades.append({"date": date, "signal_date": calendar[j - 1], "asset": assets[k], "side": side,
                                       "units": quantity, "adjusted_price": opens[k], "raw_price": raw_open[k],
                                       "notional": notional, "cost": cost, "raw_equivalent_shares": quantity * ratio,
                                       "adjusted_raw_ratio": ratio, "commission": commission, "stamp_tax": stamp_tax,
                                       "slippage_cost": slippage_cost, "liquidity_budget": cap,
                                       "liquidation": bool(odd_lot_exit),
                                       "execution_adjusted_price": opens[k] * (1 + policy.slippage if side == "buy" else 1 - policy.slippage)})
                    limited = quantity < requested_units - 1e-8
                    status = "blocked" if reason else ("partial" if limited else "filled")
                    if reason:
                        blocked += 1
                    elif limited:
                        reason = "liquidity" if cap < requested_units * opens[k] - 1e-8 else "lot_or_cash"
                    orders.append({"date": date, "signal_date": calendar[j - 1], "asset": assets[k], "side": side,
                                   "status": status, "reason": reason, "requested_units": requested_units, "filled_units": quantity})
        if cash < -1e-6 or np.min(units) < -1e-8:
            raise AssertionError("Cash / long-only invariant violated")
        cash = max(0.0, cash)
        units[np.abs(units) < 1e-10] = 0.0
        stale = (units > 0) & ~np.isfinite(close)
        last_close = np.where(np.isfinite(close), close, last_close)
        previous_stale = stale_age > config.max_stale_sessions
        marks = np.where(previous_stale, 0.0, last_close)
        values = units * marks
        nav = cash + np.nansum(values)
        if not np.isfinite(nav) or nav <= 0:
            raise ValueError("Nonpositive portfolio NAV after valuation")
        cumulative_cost += fee
        daily.append({"date": date, "nav": nav, "return": nav / prior_nav - 1, "cash": cash,
                      "market_value": np.nansum(values), "opening_nav": opening_nav,
                      "buy_notional": bought, "sell_notional": sold, "cost": fee,
                      "commission": commission_total, "stamp_tax": stamp_total, "slippage_cost": slip_total,
                      "cumulative_cost": cumulative_cost, "turnover": (bought + sold) / opening_nav,
                      "holdings_count": int((units > 0).sum()),
                      "max_weight": float(np.nanmax(values / nav)) if np.any(units > 0) else 0.0,
                      "stale_holdings": int(stale.sum()), "blocked_orders": blocked, "rebalance": rebalance,
                      "written_down_holdings": int(((units > 0) & previous_stale).sum()),
                      "gross_attribution_nav": nav + cumulative_cost})
        for k in np.flatnonzero(units > 0):
            positions.append({"date": date, "asset": assets[k], "units": units[k], "mark_price": marks[k],
                              "value": values[k], "weight": values[k] / nav, "stale_mark": bool(stale[k])})
        prior_nav = nav
        trade_session += 1
    trade_columns = ["date", "signal_date", "asset", "side", "units", "adjusted_price", "raw_price", "notional", "cost",
                     "raw_equivalent_shares", "adjusted_raw_ratio", "commission", "stamp_tax", "slippage_cost",
                     "liquidity_budget", "liquidation", "execution_adjusted_price"]
    result = BacktestResult(pd.DataFrame(daily), pd.DataFrame(trades, columns=trade_columns),
                            pd.DataFrame(positions, columns=["date", "asset", "units", "mark_price", "value", "weight", "stale_mark"]),
                            pd.DataFrame(orders, columns=["date", "signal_date", "asset", "side", "status", "reason", "requested_units", "filled_units"]))
    result.daily.attrs["execution_policy"] = policy.to_dict()
    result.daily.attrs["accounting_scope"] = "adjusted-return research with raw-equivalent lot constraints; no corporate-action cash or settlement ledger"
    return result


def audit_constrained_result(result: BacktestResult, initial_cash: float,
                             policy: ExecutionPolicy = ExecutionPolicy()) -> dict:
    """Independently reconstruct actual cash, units, costs, lots and T+1.

    Uses the actual recorded ``trade.cost`` for accounting, rather than the
    baseline engine's flat config rates. Component checks use the policy.
    """
    errors, balances = [], {}
    cash, total_cost, previous_nav = float(initial_cash), 0.0, float(initial_cash)
    if not np.isfinite(cash) or cash <= 0 or result.daily.empty:
        raise ValueError("Positive initial_cash and a nonempty result are required")
    daily = result.daily.copy()
    daily["date"] = pd.to_datetime(daily.date)
    trade_ledger = result.trades.copy()
    position_ledger = result.positions.copy()
    trade_ledger["date"] = pd.to_datetime(trade_ledger.date)
    position_ledger["date"] = pd.to_datetime(position_ledger.date)
    fills_by_date = dict(tuple(trade_ledger.groupby("date", sort=False)))
    positions_by_date = dict(tuple(position_ledger.groupby("date", sort=False)))
    if daily.date.duplicated().any() or not daily.date.is_monotonic_increasing:
        errors.append("daily dates")
    known_dates = set(daily.date)
    if any(d not in known_dates for d in trade_ledger.date):
        errors.append("trade date outside ledger")
    if any(d not in known_dates for d in position_ledger.date):
        errors.append("position date outside ledger")
    for _, day in daily.iterrows():
        fills = fills_by_date.get(day.date, trade_ledger.iloc[:0])
        eligible = balances.copy()
        day_cost = buy_notional = sell_notional = 0.0
        used_liquidity = {}
        for _, fill in fills.iterrows():
            if fill.side not in ("buy", "sell"):
                errors.append("trade side")
                continue
            sign = 1 if fill.side == "buy" else -1
            numbers = [fill.units, fill.adjusted_price, fill.raw_price, fill.notional, fill.cost]
            if not all(np.isfinite(numbers)) or any(value <= 0 for value in numbers[:4]) or fill.cost < 0:
                errors.append("invalid fill")
                continue
            if not pd.Timestamp(fill.signal_date) < day.date:
                errors.append("signal timing")
            if not np.isclose(fill.units * fill.adjusted_price, fill.notional, atol=1e-7, rtol=1e-10):
                errors.append("notional")
            expected_shares = fill.units * fill.adjusted_price / fill.raw_price
            shares = fill.get("raw_equivalent_shares", np.nan)
            if not np.isclose(shares, expected_shares, atol=1e-7, rtol=1e-10):
                errors.append("raw conversion")
            before = balances.get(fill.asset, 0.0)
            is_full_sell = fill.side == "sell" and np.isclose(fill.units, before, atol=1e-8, rtol=1e-10)
            if not np.isfinite(shares):
                errors.append("lot size")
            elif fill.side == "buy" or not is_full_sell:
                if not np.isclose(shares / policy.lot_size, round(shares / policy.lot_size), atol=1e-8, rtol=0):
                    errors.append("lot size")
            if fill.side == "sell":
                if fill.units > eligible.get(fill.asset, 0.0) + 1e-8:
                    errors.append("T+1")
                eligible[fill.asset] = eligible.get(fill.asset, 0.0) - fill.units
            expected = _costs(fill.notional, fill.side, day.date, policy)
            actual = tuple(fill.get(field, np.nan) for field in ("commission", "stamp_tax", "slippage_cost"))
            if not np.allclose(actual, expected, atol=1e-8, rtol=1e-10) or not np.isclose(sum(actual), fill.cost, atol=1e-8):
                errors.append("cost components")
            used_liquidity[fill.asset] = used_liquidity.get(fill.asset, 0.0) + fill.notional
            if used_liquidity[fill.asset] > fill.get("liquidity_budget", -np.inf) + 1e-7:
                errors.append("liquidity cap")
            cash -= sign * fill.notional + fill.cost
            balances[fill.asset] = before + sign * fill.units
            day_cost += fill.cost
            if sign == 1:
                buy_notional += fill.notional
            else:
                sell_notional += fill.notional
            if cash < -1e-6 or balances[fill.asset] < -1e-8:
                errors.append("nonnegative balances")
        total_cost += day_cost
        pos = positions_by_date.get(day.date, position_ledger.iloc[:0]).set_index("asset")
        if pos.index.has_duplicates:
            errors.append("duplicate position")
            continue
        reported = pos.units.to_dict()
        if any(not np.isclose(value, reported.get(asset, 0.0), atol=1e-7, rtol=1e-10)
               for asset, value in balances.items()) or any(asset not in balances for asset in reported):
            errors.append("units")
        if not np.allclose((pos.units * pos.mark_price).to_numpy(dtype=float), pos.value.to_numpy(dtype=float), atol=1e-7, rtol=1e-10):
            errors.append("position value")
        for label, actual, expected in (("cash", day.cash, cash), ("nav", day.nav, cash + pos.value.sum()),
                                         ("daily cost", day.cost, day_cost), ("cumulative cost", day.cumulative_cost, total_cost),
                                         ("buy notional", day.buy_notional, buy_notional), ("sell notional", day.sell_notional, sell_notional),
                                         ("return", day["return"], day.nav / previous_nav - 1),
                                         ("gross attribution", day.gross_attribution_nav, day.nav + total_cost),
                                         ("turnover", day.turnover, (buy_notional + sell_notional) / day.opening_nav)):
            if not np.isclose(actual, expected, atol=1e-6, rtol=1e-10):
                errors.append(label)
        previous_nav = day.nav
    return {"passed": not errors, "errors": sorted(set(errors)), "checked_sessions": len(daily),
            "checked_trades": len(result.trades), "cash_tolerance": 1e-6, "relative_tolerance": 1e-10,
            "accounting_scope": "adjusted-return research with raw-equivalent lots; corporate actions and settlement cash are not modeled"}
