"""Dated equity score preparation; this module never submits an order.

An append-only price extension is kept separate from the accepted research
snapshot. V3's formula and CogAlpha's seed-731 selection remain versioned; the
continuous module maintains reference vintages and yearly training caches.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .benchmark_research import training_mask
from .data import TushareProvider, digest, load_calendar, load_market, write_json
from .factor_lab import evaluate_expression
from .features import FEATURES, financial_asof, industry_history, neutralize, standardize

SHANGHAI = ZoneInfo("Asia/Shanghai")
PRICE_COLUMNS = ["open", "high", "low", "close"]
LGB_PARAMETERS = dict(
    n_estimators=250, learning_rate=.03, num_leaves=15, max_depth=5,
    min_child_samples=500, colsample_bytree=.8, reg_lambda=10.,
    random_state=20260921, n_jobs=4, verbosity=-1, deterministic=True,
    force_col_wise=True,
)
SCORE_COLUMNS = ("v3_lightgbm_frozen_reference", "cogalpha_731_rank_frozen_reference")


def assess_score_readiness(scores, signal_date, *, refreshed, minimum_assets=30):
    """Score availability is separate from execution authorization and fidelity."""
    date = pd.Timestamp(signal_date)
    if scores.empty or scores.asset.duplicated().any():
        raise ValueError("A score bundle must contain unique asset rows")
    if not (pd.to_datetime(scores.date) == date).all():
        raise ValueError("Every score must share the declared signal date")
    usable_price = np.isfinite(scores.raw_close) & (scores.raw_close > 0)
    counts = {name: int((np.isfinite(scores[name]) & scores.eligible & usable_price).sum())
              for name in SCORE_COLUMNS}
    return {
        "scores_ready": bool(refreshed and min(counts.values()) >= minimum_assets),
        "score_coverage": counts,
        "score_readiness_scope": "Dated forward adaptations; execution independently checks next-session freshness and account constraints",
    }


def completed_session(calendar, requested, *, now=None):
    """Reject future dates and same-day bars before the daily-data safety cutoff."""
    today = datetime.now(SHANGHAI) if now is None else now
    if today.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    today = today.astimezone(SHANGHAI)
    requested = pd.Timestamp(requested).normalize()
    if requested.tzinfo is not None:
        raise ValueError("requested date must be a timezone-free date")
    if requested > pd.Timestamp(today.date()):
        raise ValueError("Future signal dates are forbidden")
    last_day = pd.Timestamp(today.date())
    if (today.hour, today.minute) < (17, 5):
        last_day -= pd.Timedelta(days=1)
    candidates = pd.DatetimeIndex(calendar)
    candidates = candidates[(candidates <= requested) & (candidates <= last_day)]
    if not len(candidates):
        raise ValueError("No completed trading session")
    return candidates.max()


def _unique(frame, columns, name):
    if frame.duplicated(columns).any():
        raise ValueError(f"Duplicate {name} keys")


def append_daily_bars(base, daily, adjustment, limits):
    """Convert Tushare lots/raw prices using each asset's original price anchor."""
    for name, frame in (("daily", daily), ("adjustment", adjustment), ("limits", limits)):
        _unique(frame, ["ts_code", "trade_date"], name)
    anchors = base.groupby("asset").adjustment_reference.agg(["min", "max"])
    if not np.allclose(anchors["min"], anchors["max"]):
        raise ValueError("Original adjustment anchor changed within an asset")
    part = daily[daily.ts_code.isin(anchors.index)].copy()
    part = part.merge(adjustment, on=["ts_code", "trade_date"], how="left", validate="one_to_one")
    part = part.merge(limits, on=["ts_code", "trade_date"], how="left", validate="one_to_one")
    part["date"] = pd.to_datetime(part.trade_date.astype(str), format="%Y%m%d")
    if len(part) and part.date.min() <= base.date.max():
        raise ValueError("A forward extension must not rewrite historical sessions")
    numeric = PRICE_COLUMNS + ["adj_factor", "vol", "up_limit", "down_limit"]
    part[numeric] = part[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(part[numeric].to_numpy()).all():
        raise ValueError("Incomplete daily price/adjustment/limit fields")
    if (part[PRICE_COLUMNS + ["adj_factor", "up_limit", "down_limit"]] <= 0).any().any():
        raise ValueError("Nonpositive price or adjustment")
    if (part.vol < 0).any() or (part.high < part[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("Invalid daily OHLCV")
    if (part.low > part[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Invalid daily low")
    part["asset"] = part.ts_code
    part["adjustment_reference"] = part.asset.map(anchors["min"])
    for column in PRICE_COLUMNS:
        part["raw_" + column] = part[column]
        part[column] = part[column] * part.adj_factor / part.adjustment_reference
    part["volume"] = part.vol * 100
    extension = part.reindex(columns=base.columns).sort_values(["date", "asset"])
    result = pd.concat([base, extension], ignore_index=True).sort_values(["date", "asset"])
    _unique(result, ["date", "asset"], "market")
    return result.reset_index(drop=True), extension.reset_index(drop=True)


def latest_features(market, calendar, basic, financials, industries, signal_date):
    """Same twelve causal formulas as V3, with no future labels or future bars."""
    signal_date = pd.Timestamp(signal_date)
    dates = pd.DatetimeIndex(calendar)
    dates = dates[dates <= signal_date][-82:]
    if not len(dates) or dates[-1] != signal_date:
        raise ValueError("Signal date must be an explicit calendar session")
    source = market[market.date.isin(dates)]
    assets = pd.Index(sorted(market.asset.unique()), name="asset")
    def panel(column):
        return source.pivot(index="date", columns="asset", values=column).reindex(index=dates, columns=assets)
    close, volume, raw_close = panel("close"), panel("volume"), panel("raw_close")
    returns = close.pct_change(fill_method=None)
    raw = {
        "momentum_60_20": (close.shift(20) / close.shift(60) - 1).where(close.rolling(61).count() == 61),
        "reversal_5": -(close / close.shift(5) - 1).where(close.rolling(6).count() == 6),
        "low_volatility_20": -returns.rolling(20, min_periods=20).std(),
        "range_20": -((panel("high") - panel("low")) / close).rolling(20, min_periods=20).mean(),
        "trend_20": close / close.rolling(20, min_periods=20).mean() - 1,
        "volume_trend": np.log(volume.rolling(5, min_periods=5).mean() / volume.rolling(20, min_periods=20).mean()),
        "liquidity_20": -(returns.abs() / (raw_close * volume / 1e6)).rolling(20, min_periods=20).mean(),
    }
    day_basic = basic[pd.to_datetime(basic.trade_date.astype(str), format="%Y%m%d") == signal_date]
    _unique(day_basic, ["ts_code"], "daily basic")
    day_basic = day_basic.set_index("ts_code").reindex(assets)
    pe, pb = day_basic.pe_ttm, day_basic.pb
    frame = pd.DataFrame({name: value.loc[signal_date] for name, value in raw.items()})
    frame["turnover"] = -day_basic.turnover_rate
    frame["earnings_yield"] = 1 / pe.where(pe > 0)
    frame["book_yield"] = 1 / pb.where(pb > 0)
    frame["dividend_yield"] = day_basic.dv_ttm / 100
    sectors = pd.Series("UNKNOWN", index=assets)
    day_index = pd.DatetimeIndex([signal_date])
    for asset in assets:
        frame.loc[asset, "quality_roe"] = financial_asof(financials[financials.ts_code == asset], day_index).iloc[0]
        sectors.loc[asset] = industry_history(industries[industries.ts_code == asset], day_index).iloc[0]
    cap = day_basic.total_mv
    limit_ratio = panel("up_limit").loc[signal_date] / panel("down_limit").loc[signal_date]
    amount20 = (raw_close * volume).rolling(20, min_periods=20).mean().loc[signal_date]
    eligible = ((raw_close.loc[signal_date] >= 3) & (amount20 >= 20_000_000) &
                (limit_ratio > 1.15) & (cap > 0) & (frame[FEATURES].notna().sum(axis=1) >= 10))
    z = standardize(frame.loc[eligible, FEATURES])
    neutral = neutralize(z, np.log(cap.loc[eligible]), sectors.loc[eligible])
    for name in FEATURES:
        frame["z_" + name] = z[name].reindex(assets)
        frame["n_" + name] = neutral[name].reindex(assets)
    frame["date"], frame["asset"] = signal_date, assets
    frame["eligible"], frame["industry"], frame["log_cap"] = eligible, sectors, np.log(cap)
    frame["volatility"] = -frame.low_volatility_20
    for name in ("raw_close", "up_limit", "down_limit", "volume"):
        frame[name] = panel(name).loc[signal_date]
    return frame.reset_index(drop=True)


def v3_training_rows(frame, cutoff):
    """Explicit label embargo matching the original annual V3 fit."""
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame.date)
    frame["label_exit"] = pd.to_datetime(frame.label_exit)
    eligible = frame.eligible & (frame[["n_" + n for n in FEATURES]].notna().sum(axis=1) >= 10)
    mask = training_mask(frame, eligible, pd.Timestamp(cutoff))
    selected = frame.loc[mask].copy()
    if selected.empty:
        raise ValueError("No fully realized training rows")
    if selected.label_exit.max() >= pd.Timestamp(cutoff):
        raise AssertionError("Training embargo violated")
    return selected


def fit_v3_model(history, current, cutoff, model_path):
    """Persist a text model, never a pickle, with frozen historical parameters."""
    from lightgbm import LGBMRegressor
    selected = v3_training_rows(history, cutoff)
    fields = ["z_" + n for n in FEATURES] + ["n_" + n for n in FEATURES]
    x = selected[fields].fillna(0).to_numpy(dtype=np.float32)
    y = (selected.groupby("date").label.rank(pct=True) - .5).to_numpy(dtype=np.float32)
    identity = hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()
    model = LGBMRegressor(**LGB_PARAMETERS).fit(x, y)
    model.booster_.save_model(str(model_path))
    valid = current.eligible & (current[["n_" + n for n in FEATURES]].notna().sum(axis=1) >= 10)
    scores = pd.Series(np.nan, index=current.index)
    if valid.any():
        scores.loc[valid] = model.predict(current.loc[valid, fields].fillna(0).to_numpy(dtype=np.float32))
    return scores, {
        "cutoff_exclusive": str(pd.Timestamp(cutoff).date()), "training_rows": len(selected),
        "latest_training_label_exit": str(selected.label_exit.max().date()),
        "training_matrix_sha256": identity, "parameters": LGB_PARAMETERS,
        "fields": fields, "model_sha256": digest(model_path),
        "prediction_rows": int(valid.sum()),
    }


def cog_rank_scores(market, calendar, eligible, selected_cards, signal_date):
    """Fixed selected formulas; a missing factor blocks that asset's rank score."""
    dates = pd.DatetimeIndex(calendar)
    dates = dates[dates <= pd.Timestamp(signal_date)]
    assets = pd.Index(sorted(market.asset.unique()), name="asset")
    source = market[market.date <= pd.Timestamp(signal_date)]
    def panel(column):
        return source.pivot(index="date", columns="asset", values=column).reindex(index=dates, columns=assets)
    panels = {n: panel(n) for n in ("open", "high", "low", "close", "volume")}
    panels["volume"] = panels["volume"] * panel("adjustment_reference") / panel("adj_factor")
    eligible = eligible.reindex(assets).fillna(False).astype(bool)
    columns = {}
    for card in selected_cards:
        values = evaluate_expression(card["expression"], panels).astype("float32").loc[signal_date]
        columns[card["id"]] = values.replace([np.inf, -np.inf], np.nan).where(eligible).rank(pct=True) - .5
    if not columns:
        raise ValueError("Empty CogAlpha selection")
    ranks = pd.DataFrame(columns)
    return ranks.mean(axis=1).where(ranks.notna().all(axis=1) & eligible), ranks


def _selected_cards(private, evaluator_root):
    selection_path = private / "selection_731_llm_evolved49.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))["selected"]
    if len(selection) != 20 or len(set(selection)) != 20:
        raise ValueError("Expected the frozen seed-731 twenty-factor selection")
    cards, identities = [], {"selection_sha256": digest(selection_path)}
    for factor_id in selection:
        # IDs from receipts are not allowed to become arbitrary paths.
        if not factor_id.replace("_", "").isalnum():
            raise ValueError("Invalid factor ID")
        path = private / "factors" / (factor_id + ".json")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        card = receipt["card"]
        if card["id"] != factor_id or hashlib.sha256(card["expression"].encode()).hexdigest() != receipt["expression_sha256"]:
            raise ValueError("Frozen factor expression identity changed")
        if not receipt["evaluation"]["quality_passed"]:
            raise ValueError("Unapproved factor in frozen selection")
        for name in ("factor_lab.py", "factors.py"):
            if digest(evaluator_root / name) != receipt["evaluator_sha256"][name]:
                raise ValueError("Frozen factor evaluator changed")
        cards.append(card)
        identities[factor_id] = digest(path)
    return cards, identities


def prepare_equity_forward(*args, **kwargs):
    """Compatibility entry point for the continuous, checkpointed preparation."""
    from .equity_continuous import prepare_equity_forward as prepare
    return prepare(*args, **kwargs)
