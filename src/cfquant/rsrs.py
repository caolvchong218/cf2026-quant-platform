"""Causal RSRS signals routed through the shared Qingxu accounting engine.

Source: Everbright Securities, Liu Junwei / Zhou Xiaoxiao, 2017-05-01,
pp. 8-10 and 13-20. The legacy implementation in projects/project2_rsrs is an
independent ledger reference. Here, all fills, NAV, metrics and reconciliation
come from the Project 1 engine. Price-index units are a research proxy; they do
not represent executable ETF or futures positions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .analytics import performance, reconcile
from .config import Config
from .engine import BacktestResult, run_backtest


@dataclass(frozen=True)
class Specification:
    name: str
    n: int = 18
    m: int = 600
    score: str = "z"
    upper: float = 0.7
    lower: float = -0.7
    trend_filter: bool = False
    trend_lag: int = 2

    def __post_init__(self):
        for key, minimum in (("n", 3), ("m", 2), ("trend_lag", 1)):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        if self.score not in {"beta", "z", "corrected", "right"}:
            raise ValueError("Unknown RSRS score")
        if not np.isfinite([self.upper, self.lower]).all() or self.upper <= self.lower:
            raise ValueError("Finite upper must exceed lower")
        if not isinstance(self.trend_filter, bool):
            raise ValueError("trend_filter must be Boolean")


SPECS = [
    Specification("slope", score="beta", upper=1.0, lower=0.8),
    Specification("standard"),
    Specification("corrected", score="corrected"),
    Specification("right", n=16, m=300, score="right"),
    Specification("standard_ma", trend_filter=True),
    Specification("corrected_ma", score="corrected", trend_filter=True),
    Specification("right_ma", n=16, m=300, score="right", trend_filter=True),
]
CONTROL_NAMES = ("buy_hold", "ma20")


@dataclass
class RSRSRun:
    result: BacktestResult
    metrics: dict
    checks: dict
    factors: pd.DataFrame
    decision: pd.Series
    targets: pd.DataFrame
    volatility: pd.Series
    metadata: dict


def _check_index(index: pd.Index) -> None:
    if (not isinstance(index, pd.DatetimeIndex) or index.empty or index.hasnans
            or not index.is_unique or not index.is_monotonic_increasing
            or index.tz is not None or not index.equals(index.normalize())):
        raise ValueError("Use nonempty, unique, increasing, timezone-naive daily dates")


def _check_prices(data: pd.DataFrame, required: tuple[str, ...]) -> None:
    _check_index(data.index)
    if any(column not in data for column in required):
        raise ValueError(f"Prices require columns {required}")
    values = data.loc[:, list(required)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Index prices must be finite and positive; missing sessions must not be filled")
    if "low" in required and "high" in required and (data.high < data.low).any():
        raise ValueError("High cannot be lower than low")


def factors(data: pd.DataFrame, n: int = 18, m: int = 600) -> pd.DataFrame:
    """OLS available at close t; z uses strictly prior M slopes, ddof=0.

    A full history is required. Degenerate regressions and unavailable warmup
    remain NaN. Moving-average gating uses closing prices through t only.
    """
    _check_prices(data, ("low", "high", "close"))
    for value, minimum in ((n, 3), (m, 2)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError("Need integer n >= 3, m >= 2")
    out = pd.DataFrame(index=data.index)
    out["beta"] = np.nan
    out["r2"] = np.nan
    if len(data) >= n:
        x = np.lib.stride_tricks.sliding_window_view(data.low.to_numpy(dtype=float), n)
        y = np.lib.stride_tricks.sliding_window_view(data.high.to_numpy(dtype=float), n)
        xc, yc = x - x.mean(axis=1, keepdims=True), y - y.mean(axis=1, keepdims=True)
        xx, yy, xy = (xc * xc).sum(axis=1), (yc * yc).sum(axis=1), (xc * yc).sum(axis=1)
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
    """Binary hysteresis; moving-average filter controls entries, never exits."""
    if not np.isfinite([upper, lower]).all() or upper <= lower:
        raise ValueError("Finite upper must exceed lower")
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


def _specification(value: Specification | str) -> Specification:
    if isinstance(value, Specification):
        return value
    for spec in SPECS:
        if spec.name == value:
            return spec
    raise ValueError(f"Unknown RSRS specification: {value}")


def strategy(data: pd.DataFrame, spec: Specification | str) -> tuple[pd.DataFrame, pd.Series]:
    if isinstance(spec, str) and spec in CONTROL_NAMES:
        _check_prices(data, ("close",))
        f = pd.DataFrame({"ma20": data.close.rolling(20, min_periods=20).mean()}, index=data.index)
        decision = (pd.Series(1, index=data.index, dtype=int) if spec == "buy_hold"
                    else data.close.gt(f.ma20).astype(int))
        return f, decision.rename("decision")
    spec = _specification(spec)
    f = factors(data, spec.n, spec.m)
    gate = f.ma20 > f.ma20.shift(spec.trend_lag) if spec.trend_filter else None
    return f, decisions(f[spec.score], spec.upper, spec.lower, gate)


def _realized_volatility(close: pd.Series, window: int, annualization: int) -> pd.Series:
    _check_index(close.index)
    for value, minimum in ((window, 2), (annualization, 1)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError("Need integer window >= 2 and annualization >= 1")
    if not np.isfinite(close.to_numpy(dtype=float)).all() or (close <= 0).any():
        raise ValueError("Closes must be finite and positive")
    return (close.pct_change(fill_method=None).rolling(window, min_periods=window)
            .std(ddof=1).mul(np.sqrt(annualization)).rename("annualized_volatility"))


def volatility_weights(decision: pd.Series, close: pd.Series, target: float = 0.15,
                       window: int = 20, annualization: int = 252) -> pd.Series:
    """Close-formed unlevered risk scaling, with cash throughout vol warmup.

    This scales exposure only; it never changes the underlying RSRS decision.
    A fully observed zero-volatility window receives scale 1, not infinity.
    """
    if not decision.index.equals(close.index) or not decision.isin([0, 1]).all():
        raise ValueError("Binary decisions must align exactly with closes")
    if not np.isfinite(target) or target <= 0:
        raise ValueError("Volatility target must be positive and finite")
    volatility = _realized_volatility(close, window, annualization)
    scale = (target / volatility.where(volatility > 0)).clip(upper=1)
    scale = scale.mask(volatility == 0, 1.0).fillna(0.0)
    return decision.astype(float).mul(scale).rename("target_weight")


def index_to_market(data: pd.DataFrame, asset: str = "000300.SH") -> pd.DataFrame:
    """Adapt an OHLC price-index series to P1's long market-table schema.

    No exchange limits or liquidity are fabricated. run_rsrs explicitly disables
    price-limit checks because an index value is not a traded security.
    """
    _check_prices(data, ("open", "high", "low", "close"))
    if not isinstance(asset, str) or not asset.strip():
        raise ValueError("Provide a nonempty index identifier")
    market = data.loc[:, ["open", "high", "low", "close"]].copy()
    market["date"] = data.index
    market["asset"] = asset
    market["raw_open"] = market.open
    market["raw_close"] = market.close
    market["up_limit"] = np.nan
    market["down_limit"] = np.nan
    return market.reset_index(drop=True)


def run_signal(data: pd.DataFrame, decision: pd.Series, start: str, end: str,
               *, cost_bps: float = 10.0, initial_cash: float = 1_000_000.0,
               annualization: int = 252, asset: str = "000300.SH",
               risk_free_annual: float = 0.0, rebalance_daily: bool = False,
               target_weights: pd.Series | None = None) -> RSRSRun:
    """Audit an aligned binary research signal, optionally with scaled exposure.

    Supplied targets are formation-date weights, not execution-date holdings.
    All signals and targets must use precisely the complete input-price calendar.
    This helper lets controlled formula variants share the same P1 accounting.
    """
    if not np.isfinite(cost_bps) or not 0 <= cost_bps < 10000:
        raise ValueError("cost_bps must lie in [0, 10000)")
    market = index_to_market(data, asset)
    if (not isinstance(decision, pd.Series) or not decision.index.equals(data.index)
            or not decision.isin([0, 1]).all()):
        raise ValueError("Binary decisions must align exactly with prices")
    if not isinstance(rebalance_daily, bool):
        raise ValueError("rebalance_daily must be Boolean")
    weight = decision.astype(float) if target_weights is None else target_weights.copy()
    if (not isinstance(weight, pd.Series) or not weight.index.equals(data.index)
            or not weight.between(0, 1).all() or not np.isfinite(weight).all()
            or (weight > decision).any()):
        raise ValueError("Targets must align exactly with prices and lie between zero and the binary signal")
    if not rebalance_daily and not np.array_equal(weight.to_numpy(), decision.to_numpy()):
        raise ValueError("Scaled targets require rebalance_daily=True")
    mask = (pd.Series(True, index=data.index, dtype=bool) if rebalance_daily
            else decision.ne(decision.shift(1)))
    targets = pd.DataFrame({asset: weight}, index=data.index)
    scores = pd.DataFrame({asset: decision}, index=data.index)
    cfg = Config(name="rsrs_supplied_signal",
                 start=str(start), end=str(end), factor="rsrs", factor_params={},
                 holdings=1, rebalance_every=1, initial_cash=initial_cash,
                 buy_cost=cost_bps / 10000, sell_cost=cost_bps / 10000,
                 annualization=annualization, risk_free_annual=risk_free_annual,
                 respect_price_limits=False)
    result = run_backtest(market, data.index, scores, cfg, target_weights=targets,
                          rebalance_mask=mask, reserve_cost_buffer=False)
    checks = reconcile(result, cfg.initial_cash, cfg.buy_cost, cfg.sell_cost)
    if not checks["passed"]:
        raise ArithmeticError(f"RSRS shared-engine ledger failed reconciliation: {checks}")
    measured = performance(result.daily, cfg.initial_cash, cfg.annualization,
                           cfg.risk_free_annual, result.trades)
    execution_dates = pd.DatetimeIndex(result.daily.date)
    measured.update({
        "start": str(execution_dates[0].date()), "end": str(execution_dates[-1].date()),
        "exposure": float(weight.shift(1).reindex(execution_dates).mean()),
        "mean_realized_close_exposure": float((result.daily.market_value / result.daily.nav).mean()),
        "buys": int(result.trades.side.eq("buy").sum()),
        "sells": int(result.trades.side.eq("sell").sum()),
        "cost_initial_capital": float(result.daily.cost.sum() / initial_cash),
    })
    metadata = {
        "specification": {"name": "supplied_signal"},
        "asset": asset,
        "instrument_type": "price_index_research_proxy", "is_executable_instrument": False,
        "signal_timing": "close t", "execution_timing": "open t+1",
        "respect_price_limits": False, "reserve_cost_buffer": False,
        "cost_bps_each_side": cost_bps, "initial_cash": initial_cash,
        "annualization": annualization, "risk_free_annual": risk_free_annual,
        "rebalance": "daily" if rebalance_daily else "state_change",
        "cash_interest": 0.0, "terminal_liquidation": False,
    }
    return RSRSRun(result, measured, checks, pd.DataFrame(index=data.index), decision.copy(), targets,
                   pd.Series(np.nan, index=data.index, name="annualized_volatility"), metadata)


def run_rsrs(data: pd.DataFrame, spec_or_name: Specification | str, start: str, end: str,
             *, cost_bps: float = 10.0, vol_target: float | None = None,
             vol_window: int = 20, initial_cash: float = 1_000_000.0,
             annualization: int = 252, asset: str = "000300.SH",
             risk_free_annual: float = 0.0, exposure_scale: float = 1.0) -> RSRSRun:
    """Evaluate a fixed RSRS specification using prior-close/next-open fills.

    Baseline invests fully or holds cash, trading only at state changes and the
    first evaluation open. The optional improvement resizes risk daily with a
    predeclared annual volatility target (15% by research protocol). No interest,
    leverage, synthetic presample return or terminal liquidation is assumed.
    """
    if not np.isfinite(exposure_scale) or not 0 <= exposure_scale <= 1:
        raise ValueError("exposure_scale must lie in [0, 1]")
    is_control = isinstance(spec_or_name, str) and spec_or_name in CONTROL_NAMES
    spec = None if is_control else _specification(spec_or_name)
    f, decision = strategy(data, spec_or_name)
    volatility = _realized_volatility(data.close, vol_window, annualization)
    weight = (decision.astype(float) if vol_target is None else
              volatility_weights(decision, data.close, vol_target, vol_window, annualization))
    run = run_signal(data, decision, start, end, cost_bps=cost_bps, initial_cash=initial_cash,
                     annualization=annualization, asset=asset, risk_free_annual=risk_free_annual,
                     rebalance_daily=vol_target is not None or exposure_scale < 1,
                     target_weights=weight * exposure_scale)
    run.factors, run.volatility = f, volatility
    run.metadata.update({
        "specification": ({"name": spec_or_name, "control": True} if is_control else spec.__dict__.copy()),
        "standardization": "strictly prior M slopes; population standard deviation",
        "vol_target": vol_target, "vol_window": vol_window, "exposure_scale": exposure_scale,
    })
    return run
