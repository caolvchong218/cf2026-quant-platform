"""Factor plugins consume a calendar-aligned panel and return date x asset scores."""
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorSpec:
    name: str
    label: str
    formula: str
    hypothesis: str
    default_window: int
    failure_mode: str
    function: Callable


REGISTRY: dict[str, FactorSpec] = {}


def register(spec: FactorSpec) -> None:
    if spec.name in REGISTRY:
        raise ValueError(f"Factor already registered: {spec.name}")
    REGISTRY[spec.name] = spec


def panel(market: pd.DataFrame, calendar: pd.DatetimeIndex, field: str) -> pd.DataFrame:
    return market.pivot(index="date", columns="asset", values=field).reindex(calendar).sort_index(axis=1)


def momentum(close, window):
    # Requiring the whole window prevents treating gaps as adjacent sessions.
    valid = close.rolling(window + 1, min_periods=window + 1).count() == window + 1
    return (close / close.shift(window) - 1).where(valid)


def reversal(close, window):
    return -momentum(close, window)


def low_volatility(close, window):
    return -close.pct_change(fill_method=None).rolling(window, min_periods=window).std(ddof=1)


register(FactorSpec("momentum", "20日动量", "C(t)/C(t-w)-1",
                    "近期相对强势可能持续；高分优先。", 20, "震荡反转和趋势突变可能使动量失效。", momentum))
register(FactorSpec("reversal", "5日反转", "-(C(t)/C(t-w)-1)",
                    "短期下跌后可能反弹；高分优先。", 5, "持续趋势或基本面恶化可能使反转失效。", reversal))
register(FactorSpec("low_volatility", "20日低波动", "-std(r[t-w+1:t], ddof=1)",
                    "负波动率将低风险暴露统一到高分方向。", 20, "快速上涨阶段可能落后，行业暴露可能混淆结论。", low_volatility))


def compute(market, calendar, name, params=None) -> pd.DataFrame:
    if name not in REGISTRY:
        raise ValueError(f"Unknown factor {name}; available: {list(REGISTRY)}")
    spec = REGISTRY[name]
    params = dict(params or {})
    window = params.pop("window", spec.default_window)
    if params:
        raise ValueError(f"Unknown factor parameters: {list(params)}")
    if not isinstance(window, int) or isinstance(window, bool) or window < 2:
        raise ValueError("window must be an integer >= 2")
    close = panel(market, calendar, "close")
    scores = spec.function(close.copy(), window)
    if not isinstance(scores, pd.DataFrame) or not scores.index.equals(close.index) or not scores.columns.equals(close.columns):
        raise ValueError("Factor must preserve the date/asset axes")
    return scores.astype(float).replace([np.inf, -np.inf], np.nan)


def to_long(scores, name):
    # Retain warm-up / unavailable rows as explicit NaN.
    result = scores.rename_axis(index="date", columns="asset").stack(future_stack=True).rename("value").reset_index()
    result["factor"] = name
    return result[["date", "asset", "factor", "value"]]


def causal_check(market, calendar, name, params=None, cutoffs=None) -> dict:
    """Prefix invariance and future perturbation catch implemented look-ahead."""
    full = compute(market, calendar, name, params)
    cutoffs = cutoffs or [calendar[len(calendar)//3], calendar[2*len(calendar)//3]]
    errors = []
    for date in cutoffs:
        prefix = compute(market.loc[market.date <= date], calendar[calendar <= date], name, params)
        expected = full.loc[prefix.index]
        if not np.allclose(prefix, expected, equal_nan=True, atol=1e-12, rtol=1e-12):
            errors.append(f"prefix:{date.date()}")
        perturbed = market.copy()
        perturbed.loc[perturbed.date > date, "close"] *= 1.987
        changed = compute(perturbed, calendar, name, params).loc[:date]
        if not np.allclose(changed, full.loc[:date], equal_nan=True, atol=1e-12, rtol=1e-12):
            errors.append(f"future:{date.date()}")
    return {"factor": name, "passed": not errors, "errors": errors,
            "scope": "Empirical checks at selected cutoffs, not a proof for arbitrary plugins."}

