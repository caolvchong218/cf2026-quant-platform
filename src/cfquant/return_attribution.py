"""Descriptive stock/industry P&L from actual cashflows and final marks.

This is accounting attribution, not causal attribution to investment factors.
Adjusted-unit ledgers retain their original research/corporate-action scope.
Industry labels are an explicitly supplied grouping, not inferred exposures.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .engine import BacktestResult


@dataclass
class AttributionResult:
    stocks: pd.DataFrame
    industries: pd.DataFrame
    reconciliation: dict


def _industry_labels(industry_mapping):
    if industry_mapping is None:
        return None
    if isinstance(industry_mapping, pd.DataFrame):
        if not {"asset", "industry"}.issubset(industry_mapping.columns) or industry_mapping.asset.duplicated().any():
            raise ValueError("Industry mapping requires one explicit asset/industry label per asset")
        return industry_mapping.set_index("asset").industry.to_dict()
    if isinstance(industry_mapping, pd.Series):
        if industry_mapping.index.has_duplicates:
            raise ValueError("Industry mapping has duplicate assets")
        return industry_mapping.to_dict()
    if isinstance(industry_mapping, Mapping):
        return dict(industry_mapping)
    raise ValueError("industry_mapping must be a mapping, Series, or static asset/industry table")


def attribute_result(result: BacktestResult, initial_cash: float,
                     industry_mapping=None) -> AttributionResult:
    """Compute stock P&L = sells - buys - paid costs + final marked value.

    ``return_contribution`` is that P&L divided by initial capital, so additive
    stock contributions reconcile to portfolio total return. No FIFO choice or
    fabricated realized/unrealized split is needed. Commission/tax/slippage
    components stay missing if the input engine did not record them.
    """
    if not np.isfinite(initial_cash) or initial_cash <= 0 or result.daily.empty:
        raise ValueError("Positive initial_cash and a nonempty daily ledger are required")
    if not {"date", "nav", "cash"}.issubset(result.daily.columns):
        raise ValueError("Daily ledger requires date/nav/cash")
    daily = result.daily.copy()
    daily["date"] = pd.to_datetime(daily.date)
    if daily.date.hasnans or daily.date.duplicated().any() or not daily.date.is_monotonic_increasing:
        raise ValueError("Daily ledger dates must be unique and increasing")
    trades = result.trades.copy()
    positions = result.positions.copy()
    trade_fields = {"date", "asset", "side", "units", "notional", "cost"}
    position_fields = {"date", "asset", "units", "mark_price", "value"}
    if not trade_fields.issubset(trades.columns) or not position_fields.issubset(positions.columns):
        raise ValueError("Trade/position ledger lacks cashflow or marked holding fields")
    trades["date"] = pd.to_datetime(trades.date)
    positions["date"] = pd.to_datetime(positions.date)
    if trades.date.hasnans or positions.date.hasnans or not trades.side.isin(["buy", "sell"]).all():
        raise ValueError("Invalid trade/position dates or trade side")
    if (not np.isfinite(trades[["units", "notional", "cost"]].to_numpy(dtype=float)).all()
            or (trades[["units", "notional"]] <= 0).any().any() or (trades.cost < 0).any()):
        raise ValueError("Trade cashflows must be finite, positive, with nonnegative costs")
    if (not np.isfinite(positions[["units", "mark_price", "value"]].to_numpy(dtype=float)).all()
            or (positions[["units", "mark_price", "value"]] < 0).any().any()
            or positions.duplicated(["date", "asset"]).any()):
        raise ValueError("Position marks must be finite, nonnegative, with unique date/asset keys")
    errors = []
    ledger_dates = set(daily.date)
    if any(value not in ledger_dates for value in trades.date) or any(value not in ledger_dates for value in positions.date):
        errors.append("record date outside daily ledger")
    if not np.allclose((positions.units * positions.mark_price).to_numpy(dtype=float), positions.value.to_numpy(dtype=float), atol=1e-7, rtol=1e-10):
        errors.append("position value")
    fills_by_date = dict(tuple(trades.groupby("date", sort=False)))
    positions_by_date = dict(tuple(positions.groupby("date", sort=False)))
    cash, units = float(initial_cash), {}
    for _, day in daily.iterrows():
        fills = fills_by_date.get(day.date, trades.iloc[:0])
        for _, fill in fills.iterrows():
            sign = 1 if fill.side == "buy" else -1
            cash -= sign * fill.notional + fill.cost
            units[fill.asset] = units.get(fill.asset, 0.0) + sign * fill.units
            if units[fill.asset] < -1e-8 or cash < -1e-6:
                errors.append("negative cash or units")
        holdings = positions_by_date.get(day.date, positions.iloc[:0]).set_index("asset")
        reported = holdings.units.to_dict()
        if any(not np.isclose(value, reported.get(asset, 0.0), atol=1e-7, rtol=1e-10)
               for asset, value in units.items()) or any(asset not in units for asset in reported):
            errors.append("units")
        if not np.isclose(cash, day.cash, atol=1e-6, rtol=1e-10):
            errors.append("cash")
        if not np.isclose(cash + holdings.value.sum(), day.nav, atol=1e-6, rtol=1e-10):
            errors.append("nav")
        if "cost" in daily and not np.isclose(fills.cost.sum(), day.cost, atol=1e-8, rtol=1e-10):
            errors.append("daily cost")
    final_date = daily.date.iloc[-1]
    ending = positions[positions.date == final_date].set_index("asset")
    all_assets = sorted(set(trades.asset) | set(ending.index), key=str)
    labels = _industry_labels(industry_mapping)
    rows = []
    for asset in all_assets:
        fills = trades[trades.asset == asset]
        buys = float(fills.loc[fills.side == "buy", "notional"].sum())
        sells = float(fills.loc[fills.side == "sell", "notional"].sum())
        cost = float(fills.cost.sum())
        end_value = float(ending.loc[asset, "value"]) if asset in ending.index else 0.0
        end_units = float(ending.loc[asset, "units"]) if asset in ending.index else 0.0
        gross_pnl = sells - buys + end_value
        row = {"asset": asset, "buy_notional": buys, "sell_notional": sells,
               "net_trade_cashflow": sells - buys - cost, "ending_units": end_units,
               "ending_value": end_value, "gross_pnl": gross_pnl, "cost": cost,
               "net_pnl": gross_pnl - cost, "return_contribution": (gross_pnl - cost) / initial_cash,
               "trade_count": len(fills)}
        for component in ("commission", "stamp_tax", "slippage_cost"):
            row[component] = float(fills[component].sum()) if component in fills and fills[component].notna().all() else np.nan
        if labels is not None:
            label = labels.get(asset)
            row["industry"] = "UNKNOWN" if label is None or pd.isna(label) else str(label)
        rows.append(row)
    columns = ["asset", "buy_notional", "sell_notional", "net_trade_cashflow", "ending_units", "ending_value",
               "gross_pnl", "cost", "net_pnl", "return_contribution", "trade_count", "commission", "stamp_tax", "slippage_cost"]
    if labels is not None:
        columns.append("industry")
    stocks = pd.DataFrame(rows, columns=columns)
    aggregate_columns = ["buy_notional", "sell_notional", "net_trade_cashflow", "ending_value", "gross_pnl", "cost",
                         "net_pnl", "return_contribution", "trade_count", "commission", "stamp_tax", "slippage_cost"]
    industries = pd.DataFrame(columns=["industry", *aggregate_columns, "asset_count"])
    if labels is not None and not stocks.empty:
        industries = stocks.groupby("industry", sort=True)[aggregate_columns].sum(min_count=1).reset_index()
        industries["asset_count"] = industries.industry.map(stocks.groupby("industry").size())
    attributed_pnl = float(stocks.net_pnl.sum())
    portfolio_pnl = float(daily.nav.iloc[-1] - initial_cash)
    difference = attributed_pnl - portfolio_pnl
    if not np.isclose(difference, 0, atol=1e-6, rtol=0):
        errors.append("portfolio pnl")
    ending_cash = float(initial_cash + stocks.net_trade_cashflow.sum())
    if not np.isclose(ending_cash, daily.cash.iloc[-1], atol=1e-6, rtol=1e-10):
        errors.append("ending cash")
    if labels is not None and not np.isclose(industries.net_pnl.sum(), attributed_pnl, atol=1e-6, rtol=1e-10):
        errors.append("industry aggregation")
    reconciliation = {"passed": not errors, "errors": sorted(set(errors)), "initial_cash": float(initial_cash),
                      "ending_nav": float(daily.nav.iloc[-1]), "ending_cash": ending_cash,
                      "portfolio_pnl": portfolio_pnl, "attributed_pnl": attributed_pnl, "pnl_difference": difference,
                      "total_cost": float(trades.cost.sum()), "checked_sessions": len(daily),
                      "cash_tolerance": 1e-6, "relative_tolerance": 1e-10,
                      "scope": "descriptive cashflow and ending-mark P&L; no causal factor attribution",
                      "industry_grouping": "explicit static labels" if labels is not None else "not supplied"}
    return AttributionResult(stocks, industries, reconciliation)
