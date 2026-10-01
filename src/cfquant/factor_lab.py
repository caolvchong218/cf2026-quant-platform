"""Small, auditable factor expressions over aligned date-by-asset panels.

Expressions are parsed, checked, and interpreted as a DSL. Python code is never
executed. All time operators use the current row and historical rows; rank and
zscore operate across assets in the same row. A signal using today's close is
only available after that close, so execution timing remains a caller concern.
"""
from __future__ import annotations

import ast
import math
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


MAX_EXPRESSION_LENGTH = 2000
MAX_AST_NODES = 160
MAX_AST_DEPTH = 24
MAX_WINDOW = 2520
DEFAULT_FIELDS = ("open", "high", "low", "close", "volume")
OPERATORS = {
    "delay": 2, "returns": (1, 2), "rolling_mean": 2, "rolling_std": 2,
    "rolling_min": 2, "rolling_max": 2, "rolling_corr": 3,
    "rank": 1, "zscore": 1, "clip": 3, "safe_divide": 2, "abs": 1,
}


def _number(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        value = node.value
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _number(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
    else:
        raise ValueError("Operator parameters must be literal numbers")
    if not math.isfinite(value) or abs(value) > 1_000_000:
        raise ValueError("Numeric literals must be finite and bounded")
    return value


def _integer(node: ast.AST, minimum: int, label: str) -> int:
    value = _number(node)
    if int(value) != value or not minimum <= value <= MAX_WINDOW:
        raise ValueError(f"{label} must be an integer in [{minimum}, {MAX_WINDOW}]")
    return int(value)


def _parse(expression: str, field_names: Iterable[str]) -> tuple[ast.Expression, set[str]]:
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Expression must be a nonempty string")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError("Expression is too long")
    names = set(field_names)
    if any(not isinstance(name, str) or not name.isidentifier() or name.startswith("_")
           or name in OPERATORS for name in names):
        raise ValueError("Field names must be public identifiers distinct from operators")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, RecursionError, MemoryError) as exc:
        raise ValueError("Invalid expression syntax") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise ValueError("Expression is too complex")
    used: set[str] = set()

    def check(node: ast.AST, depth: int = 0) -> bool:
        if depth > MAX_AST_DEPTH:
            raise ValueError("Expression nesting is too deep")
        if isinstance(node, ast.Name):
            if node.id not in names:
                raise ValueError(f"Unknown field: {node.id}")
            used.add(node.id)
            return True
        if isinstance(node, ast.Constant):
            _number(node)
            return False
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return check(node.operand, depth + 1)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left = check(node.left, depth + 1)
            right = check(node.right, depth + 1)
            return left or right
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name not in OPERATORS or node.keywords:
                raise ValueError("Only positional calls to whitelisted operators are allowed")
            count = OPERATORS[name]
            counts = count if isinstance(count, tuple) else (count,)
            if len(node.args) not in counts:
                raise ValueError(f"Wrong number of arguments for {name}")
            first = check(node.args[0], depth + 1)
            if not first and name != "safe_divide":
                raise ValueError(f"{name} requires a panel as its first argument")
            if name == "delay":
                _integer(node.args[1], 0, "lag")
            elif name == "returns":
                if len(node.args) == 2:
                    _integer(node.args[1], 1, "return lag")
            elif name.startswith("rolling_"):
                if name == "rolling_corr":
                    if not check(node.args[1], depth + 1):
                        raise ValueError("rolling_corr requires two panels")
                _integer(node.args[-1], 1, "window")
            elif name == "clip":
                if _number(node.args[1]) > _number(node.args[2]):
                    raise ValueError("clip lower bound must not exceed upper bound")
            elif name == "safe_divide":
                second = check(node.args[1], depth + 1)
                if not first and not second:
                    raise ValueError("safe_divide requires at least one panel")
            return True
        raise ValueError(f"Forbidden expression syntax: {type(node).__name__}")

    if not check(tree.body):
        raise ValueError("Expression must produce a panel, not a constant")
    return tree, used


def validate_expression(expression: str, field_names: Iterable[str] = DEFAULT_FIELDS) -> dict:
    """Check the whole expression without data, returning its required fields."""
    _, used = _parse(expression, field_names)
    return {"expression": expression, "fields": sorted(used), "causal_operators_only": True,
            "signal_timing": "Current-row data must be observed before the signal is used."}


def _panels(fields: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if not isinstance(fields, Mapping) or not fields:
        raise ValueError("fields must be a nonempty mapping of aligned DataFrames")
    result = {}
    first = None
    for name, frame in fields.items():
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError("Each field must be a nonempty DataFrame")
        if not frame.index.is_unique or not frame.columns.is_unique:
            raise ValueError("Panels must have unique dates and assets")
        if not frame.index.is_monotonic_increasing:
            raise ValueError("Panel dates must be in increasing order")
        if first is not None and (not frame.index.equals(first.index)
                                  or not frame.columns.equals(first.columns)):
            raise ValueError("All field panels must have identical date and asset axes")
        if any(not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
               for dtype in frame.dtypes):
            raise ValueError("Panel values must be numeric")
        first = frame
        result[name] = frame.astype(float).replace([np.inf, -np.inf], np.nan)
    return result


def _clean(value):
    if isinstance(value, pd.DataFrame):
        return value.replace([np.inf, -np.inf], np.nan)
    return value if math.isfinite(value) else np.nan


def _divide(left, right):
    if isinstance(right, pd.DataFrame):
        right = right.where(right != 0)
    elif right == 0:
        return left * np.nan
    return _clean(left / right)


def evaluate_expression(expression: str, fields: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Evaluate the checked DSL; missing/zero-denominator results remain NaN."""
    if not isinstance(fields, Mapping) or not fields:
        raise ValueError("fields must be a nonempty mapping of aligned DataFrames")
    tree, _ = _parse(expression, fields.keys())
    data = _panels(fields)

    def run(node: ast.AST):
        if isinstance(node, ast.Name):
            return data[node.id]
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.UnaryOp):
            return -run(node.operand) if isinstance(node.op, ast.USub) else run(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = run(node.left), run(node.right)
            if isinstance(node.op, ast.Add):
                return _clean(left + right)
            if isinstance(node.op, ast.Sub):
                return _clean(left - right)
            if isinstance(node.op, ast.Mult):
                return _clean(left * right)
            return _divide(left, right)
        name = node.func.id
        values = [run(arg) for arg in node.args]
        value = values[0]
        if name == "delay":
            return value.shift(int(values[1]))
        if name == "returns":
            lag = int(values[1]) if len(values) == 2 else 1
            # Gaps never become adjacent observations or disappear via filling.
            valid = value.rolling(lag + 1, min_periods=lag + 1).count() == lag + 1
            return (_divide(value, value.shift(lag)) - 1).where(valid)
        if name.startswith("rolling_"):
            window = int(values[-1])
            roll = value.rolling(window, min_periods=window)
            if name == "rolling_corr":
                return _clean(roll.corr(values[1], ddof=1))
            if name == "rolling_std":
                return _clean(roll.std(ddof=1))
            return _clean(getattr(roll, name.removeprefix("rolling_"))())
        if name == "rank":
            return value.rank(axis=1, method="average", pct=True, na_option="keep")
        if name == "zscore":
            centered = value.sub(value.mean(axis=1), axis=0)
            scale = value.std(axis=1, ddof=0).replace(0, np.nan)
            return _clean(centered.div(scale, axis=0))
        if name == "clip":
            return value.clip(lower=values[1], upper=values[2])
        if name == "abs":
            return value.abs()
        return _divide(value, values[1])

    return _clean(run(tree.body)).copy()


def builtin_candidates() -> list[dict]:
    """Readable hypotheses inspired by bundled CogAlpha OHLCV families.

    These are research seeds, with no performance claim or fitted direction.
    The bundled reference's Python code and optimizer are not imported.
    """
    seeds = [
        ("momentum_20", "20日动量", "returns(close, 20)", "中期相对强势可能延续。"),
        ("momentum_60", "60日动量", "returns(close, 60)", "较长趋势可能具有持续性。"),
        ("reversal_5", "5日反转", "-returns(close, 5)", "短期下跌可能伴随价格修复。"),
        ("mean_reversion_10", "10日均价偏离反转", "1 - safe_divide(close, rolling_mean(close, 10))", "偏离近期均价可能回归。"),
        ("low_volatility_20", "20日低波动", "-rolling_std(returns(close), 20)", "较低历史波动可能带来不同的风险收益结构。"),
        ("volatility_change", "波动状态变化", "rolling_std(returns(close), 5) - rolling_std(returns(close), 20)", "短期波动相对长期波动的变化可能包含状态信息。"),
        ("range_20", "20日振幅", "rolling_mean(safe_divide(high - low, close), 20)", "历史振幅可能反映交易分歧，方向需要验证。"),
        ("volume_surprise", "相对成交量", "safe_divide(volume, rolling_mean(volume, 20)) - 1", "相对成交量可能反映关注度变化，方向需要验证。"),
        ("volume_trend", "成交量趋势", "safe_divide(rolling_mean(volume, 5), rolling_mean(volume, 20)) - 1", "持续放量可能与价格趋势状态相关。"),
        ("bar_pressure_5", "5日实体压力", "rolling_mean(safe_divide(close - open, high - low), 5)", "收盘相对开盘的实体位置可能反映持续买卖压力。"),
        ("close_position_10", "10日收盘区间位置", "rolling_mean(safe_divide(close - low, high - low), 10)", "收盘靠近日内高点的状态可能持续。"),
        ("price_volume_corr", "20日价量相关", "rolling_corr(returns(close), returns(volume), 20)", "收益与成交量变化的关系可能刻画趋势可靠性。"),
        ("trend_volume", "趋势与成交量组合", "rank(returns(close, 20)) * rank(safe_divide(volume, rolling_mean(volume, 20)))", "趋势和成交量同时较强可能提供不同于单因子的信号。"),
        ("risk_adjusted_trend", "波动调整趋势", "safe_divide(returns(close, 20), rolling_std(returns(close), 20))", "同等波动下的趋势强度可能具有比较价值。"),
        ("distance_high_20", "距20日高点", "safe_divide(close, rolling_max(high, 20)) - 1", "接近历史高点的价格可能具有趋势或阻力信息。"),
        ("distance_low_20", "距20日低点", "safe_divide(close, rolling_min(low, 20)) - 1", "脱离历史低点的幅度可能刻画恢复状态。"),
    ]
    return [{"id": key, "label": label, "expression": expression, "hypothesis": hypothesis,
             "expected_direction": "unvalidated", "source": "Bundled CogAlpha OHLCV research principles; rewritten in causal DSL"}
            for key, label, expression, hypothesis in seeds]


def summarize_factor(values: pd.DataFrame, forward_returns: pd.DataFrame,
                     start=None, end=None) -> dict:
    """Daily cross-sectional Pearson IC and Spearman RankIC on supplied labels.

    Future labels are deliberately separate from expression inputs. The caller
    owns horizon alignment and removal of labels crossing a training cutoff.
    No direction fitting or significance claim is made here. IR is the daily
    mean divided by sample standard deviation, without annualization.
    """
    panels = _panels({"values": values, "forward_returns": forward_returns})
    x, y = panels["values"], panels["forward_returns"]
    if not isinstance(x.index, pd.DatetimeIndex):
        raise ValueError("Factor summaries require a DatetimeIndex")
    first = pd.Timestamp(start) if start is not None else x.index[0]
    last = pd.Timestamp(end) if end is not None else x.index[-1]
    if start is not None and end is not None and first > last:
        raise ValueError("start must not exceed end")
    selected = (x.index >= first) & (x.index <= last)
    x, y = x.loc[selected], y.loc[selected]
    records = []
    for date in x.index:
        valid = x.loc[date].notna() & y.loc[date].notna()
        n = int(valid.sum())
        a, b = x.loc[date, valid], y.loc[date, valid]
        ic = rank_ic = np.nan
        if n >= 3 and a.nunique() > 1 and b.nunique() > 1:
            ic = float(a.corr(b))
            # pandas rank + Pearson avoids an optional scipy dependency here.
            rank_ic = float(a.rank(method="average").corr(b.rank(method="average")))
        records.append({"date": date, "ic": ic, "rank_ic": rank_ic, "n_pairs": n,
                        "coverage": n / len(x.columns),
                        "factor_coverage": float(x.loc[date].notna().mean()),
                        "label_coverage": float(y.loc[date].notna().mean())})
    daily = pd.DataFrame(records, columns=["date", "ic", "rank_ic", "n_pairs", "coverage",
                                          "factor_coverage", "label_coverage"])

    def moment(name):
        observations = daily[name].dropna()
        mean = float(observations.mean()) if len(observations) else None
        std = float(observations.std(ddof=1)) if len(observations) >= 2 else None
        ir = mean / std if std is not None and std > 0 else None
        return mean, ir, int(len(observations))

    ic, ic_ir, n_ic = moment("ic")
    rank_ic, rank_ic_ir, n_rank_ic = moment("rank_ic")
    return {"daily": daily, "ic_mean": ic, "rank_ic_mean": rank_ic,
            "ic_ir": ic_ir, "rank_ic_ir": rank_ic_ir,
            "coverage": float(daily.coverage.mean()) if len(daily) else None,
            "factor_coverage": float(daily.factor_coverage.mean()) if len(daily) else None,
            "observations": int(daily.n_pairs.sum()), "n_dates": len(daily),
            "ic_dates": n_ic, "rank_ic_dates": n_rank_ic,
            "start": x.index[0].isoformat() if len(x) else None,
            "end": x.index[-1].isoformat() if len(x) else None,
            "assumptions": {"minimum_pairs": 3, "ir_ddof": 1, "ir_annualized": False,
                            "label_alignment": "caller supplied; exclude labels crossing the split cutoff"}}
