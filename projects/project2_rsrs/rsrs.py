"""Independent RSRS formulas and an audited long/cash research ledger.

Source: Everbright Securities, Liu Junwei / Zhou Xiaoxiao, 2017-05-01,
pp. 8-10, 13-20. Uses NumPy/Pandas, not third-party notebook trading code.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

pd.set_option("compute.use_numexpr", False)


@dataclass(frozen=True)
class Specification:
    name: str
    n: int = 18
    m: int = 600
    score: str = "z"
    upper: float = 0.7
    lower: float = -0.7
    trend_filter: bool = False
    trend_lag: int = 2  # at t+1 open: MA[t] versus MA[t-2]


SPECS = [
    Specification("slope", score="beta", upper=1.0, lower=0.8),
    Specification("standard"),
    Specification("corrected", score="corrected"),
    Specification("right", n=16, m=300, score="right"),
    Specification("standard_ma", trend_filter=True),
    Specification("corrected_ma", score="corrected", trend_filter=True),
    Specification("right_ma", n=16, m=300, score="right", trend_filter=True),
]


def factors(data: pd.DataFrame, n: int = 18, m: int = 600) -> pd.DataFrame:
    """t-close available OLS; z uses strictly prior M slopes, no warmup fill."""
    if n < 3 or m < 2:
        raise ValueError("Need n >= 3, m >= 2")
    out = pd.DataFrame(index=data.index)
    out["beta"] = np.nan
    out["r2"] = np.nan
    if len(data) >= n:
        x = np.lib.stride_tricks.sliding_window_view(data.low.to_numpy(dtype=float), n)
        y = np.lib.stride_tricks.sliding_window_view(data.high.to_numpy(dtype=float), n)
        xc = x - x.mean(axis=1, keepdims=True)
        yc = y - y.mean(axis=1, keepdims=True)
        xx = (xc * xc).sum(axis=1)
        yy = (yc * yc).sum(axis=1)
        xy = (xc * yc).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = np.where(xx > 1e-20, xy / xx, np.nan)
            r2 = np.where((xx > 1e-20) & (yy > 1e-20), xy**2 / (xx * yy), np.nan)
        out.iloc[n - 1:, 0] = beta
        out.iloc[n - 1:, 1] = np.clip(r2, 0, 1)
    past = out.beta.shift(1).rolling(m, min_periods=m)
    sd = past.std(ddof=0).where(lambda x: x > 1e-12)
    out["z"] = (out.beta - past.mean()) / sd
    out["corrected"] = out.z * out.r2
    out["right"] = out.corrected * out.beta
    out["ma20"] = data.close.rolling(20, min_periods=20).mean()
    return out


def decisions(score: pd.Series, upper: float, lower: float,
              entry_allowed: pd.Series | None = None) -> pd.Series:
    """Hysteresis; filter applies to entries only, never forced exits."""
    if upper <= lower:
        raise ValueError("upper must exceed lower")
    if entry_allowed is None:
        allowed = np.ones(len(score), dtype=bool)
    else:
        if not score.index.equals(entry_allowed.index):
            raise ValueError("Mismatched entry filter index")
        allowed = entry_allowed.fillna(False).to_numpy(dtype=bool)
    state, positions = 0, []
    for value, permit in zip(score.to_numpy(), allowed):
        if not np.isfinite(value):
            state = 0
        elif state and value < lower:
            state = 0
        elif not state and value > upper and permit:
            state = 1
        positions.append(state)
    return pd.Series(positions, index=score.index, name="decision", dtype=int)


def strategy(data: pd.DataFrame, spec: Specification) -> tuple[pd.DataFrame, pd.Series]:
    f = factors(data, spec.n, spec.m)
    gate = f.ma20 > f.ma20.shift(spec.trend_lag) if spec.trend_filter else None
    return f, decisions(f[spec.score], spec.upper, spec.lower, gate)


def backtest(data: pd.DataFrame, decision: pd.Series, start: str, end: str,
             cost_bps: float = 10.0) -> pd.DataFrame:
    """Signal t -> open t+1. Full investment or cash; fractional index units.

    New capital at first evaluation open; signal state is carried from warmup.
    No interest, leverage, synthetic pre-sample return or terminal liquidation.
    The index is a research proxy, not an executable instrument.
    """
    if not data.index.equals(decision.index):
        raise ValueError("Align decisions to prices by date")
    if not data.index.is_unique or not data.index.is_monotonic_increasing:
        raise ValueError("Dates must be unique and increasing")
    if cost_bps < 0 or cost_bps >= 10000:
        raise ValueError("Invalid cost")
    frame = data.loc[start:end].copy()
    if frame.empty:
        raise ValueError("Empty evaluation period")
    target = decision.shift(1).fillna(0).reindex(frame.index)
    if not target.isin([0, 1]).all():
        raise ValueError("Long/cash decisions must be binary")
    cash, units, previous, fee = 1.0, 0.0, 1.0, cost_bps / 10000
    records = []
    for (date, row), want in zip(frame.iterrows(), target):
        if not np.isfinite([row.open, row.close]).all() or min(row.open, row.close) <= 0:
            raise ValueError("Missing or nonpositive execution price")
        side, cost, turnover, delta = "", 0.0, 0.0, 0.0
        at_open = cash + units * row.open
        if want and units == 0:
            delta = cash / (row.open * (1 + fee))
            turnover = delta * row.open / at_open
            cost = delta * row.open * fee
            cash -= delta * row.open + cost
            units += delta
            side = "BUY"
        elif not want and units > 0:
            delta = -units
            turnover = units * row.open / at_open
            cost = units * row.open * fee
            cash += units * row.open - cost
            units = 0.0
            side = "SELL"
        equity = cash + units * row.close
        if cash < -1e-10 or units < 0 or equity <= 0:
            raise ArithmeticError("Invalid portfolio ledger")
        records.append({"date": date, "open": row.open, "close": row.close,
                        "target": int(want), "cash": cash, "units": units,
                        "delta_units": delta, "cost": cost, "turnover": turnover,
                        "equity": equity, "return": equity / previous - 1, "side": side})
        previous = equity
    return pd.DataFrame(records).set_index("date")


def metrics(ledger: pd.DataFrame, annual_sessions: int = 252) -> dict:
    r = ledger["return"]
    nav = ledger.equity
    peaks = np.maximum.accumulate(np.r_[1.0, nav.to_numpy()])[1:]
    sd = r.std(ddof=1)
    return {"start": str(ledger.index[0].date()), "end": str(ledger.index[-1].date()),
            "sessions": len(ledger), "total_return": nav.iloc[-1] - 1,
            "cagr": nav.iloc[-1] ** (annual_sessions / len(ledger)) - 1,
            "sharpe": r.mean() / sd * np.sqrt(annual_sessions) if sd > 0 else np.nan,
            "volatility": sd * np.sqrt(annual_sessions),
            "max_drawdown": float(-(nav.to_numpy() / peaks - 1).min()),
            "exposure": ledger.target.mean(), "buys": int((ledger.side == "BUY").sum()),
            "sells": int((ledger.side == "SELL").sum()),
            "turnover": ledger.turnover.sum(), "cost_initial_capital": ledger.cost.sum()}


def audit_ledger(ledger: pd.DataFrame, cost_bps: float) -> dict:
    """Independent balance reconstruction from signed fills."""
    implied_units = ledger.delta_units.cumsum()
    implied_cash = 1.0 - (ledger.delta_units * ledger.open + ledger.cost).cumsum()
    fee_expected = ledger.delta_units.abs() * ledger.open * cost_bps / 10000
    errors = {"cash": float((implied_cash-ledger.cash).abs().max()),
              "units": float((implied_units-ledger.units).abs().max()),
              "equity": float((implied_cash+implied_units*ledger.close-ledger.equity).abs().max()),
              "cost": float((fee_expected-ledger.cost).abs().max()),
              "returns": float(((1+ledger["return"]).cumprod()-ledger.equity).abs().max())}
    if max(errors.values()) > 1e-9:
        raise ArithmeticError(errors)
    return errors
