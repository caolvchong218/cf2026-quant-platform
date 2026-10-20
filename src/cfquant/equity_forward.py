"""Dated equity score preparation; this module never submits an order.

An append-only price extension is kept separate from the accepted research
snapshot. V3's existing 2026 model definition and CogAlpha's seed-731 selection
are frozen. Reference-data limitations remain explicit readiness blockers.
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
        "score_readiness_scope": "Dated frozen-reference adaptations; execution must independently check next-session freshness and account constraints",
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
    if (today.hour, today.minute) < (17, 0):
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


def prepare_equity_forward(data_root, output_root, asof, *, token_file=None,
                           refresh=True, provider=None, now=None):
    """Produce reviewable current scores with fail-closed readiness metadata.

    This first forward variant freezes annual statements and industry intervals
    at the old research snapshot. It is useful for research comparison but is
    explicitly *not* certified as an unchanged original strategy for auto-orders.
    """
    data_root, output_root = Path(data_root), Path(output_root)
    stamp = datetime.now(SHANGHAI).strftime("%Y%m%dT%H%M%S%f")
    folder = output_root / stamp
    folder.mkdir(parents=True, exist_ok=False)
    source = data_root / "data/private/mainboard1000_20260918/data/processed"
    extra = data_root / "data/private/research_v2"
    paths = {name: path for name, path in (
        ("market", source / "market.csv"), ("calendar", source / "calendar.csv"),
        ("features", extra / "features.parquet"), ("basic", extra / "daily_basic.parquet"),
        ("financials", extra / "financials.parquet"), ("industries", extra / "industries.parquet"))}
    receipt = {
        "schema": 1, "status": "preparing", "requested_date": str(pd.Timestamp(asof).date()),
        "order_submission": False, "snapshot_directory": str(folder),
        "scores_ready": False, "ready_for_auto_orders": False, "blockers": [],
        "code_sha256": {name: digest(Path(__file__).parent / name) for name in (
            "equity_forward.py", "features.py", "benchmark_research.py", "factor_lab.py", "data.py")},
    }
    write_json(folder / "receipt.json", receipt)
    try:
        market = load_market(paths["market"])
        original_last = market.date.max()
        calendar = load_calendar(paths["calendar"])
        basic = pd.read_parquet(paths["basic"])
        assets = set(market.asset)
        receipt["source_sha256"] = {name: digest(path) for name, path in paths.items()}
        cache_path = extra / "features_cache.json"
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if cache["output"] != receipt["source_sha256"]["features"]:
            raise ValueError("Original feature cache content changed")
        for name, expected in cache["inputs"].items():
            if name.startswith("data/") and digest(data_root / name) != expected:
                raise ValueError("Original feature-cache data input changed")
        receipt["feature_cache_sha256"] = digest(cache_path)
        if refresh:
            # Separate acquisition per immutable run: an empty, early vendor
            # response from one attempt cannot poison all subsequent attempts.
            provider = provider or TushareProvider(folder / "raw", token_file, delay=.5)
            cal = provider.query("trade_cal", {"exchange": "SSE",
                                 "start_date": original_last.strftime("%Y%m%d"),
                                 "end_date": pd.Timestamp(asof).strftime("%Y%m%d")}, "cal_date,is_open")
            added = pd.to_datetime(cal.loc[cal.is_open.astype(int) == 1, "cal_date"].astype(str), format="%Y%m%d")
            calendar = calendar.union(pd.DatetimeIndex(added)).sort_values()
        signal_date = completed_session(calendar, asof, now=now)
        extension_dates = calendar[(calendar > original_last) & (calendar <= signal_date)]
        if len(extension_dates) > 40:
            raise ValueError("More than 40 missing sessions; use a separately reviewed larger refresh")
        extension_parts, basic_parts = [], []
        for day in extension_dates:
            fetched = {}
            for api, fields in (
                ("daily", "ts_code,trade_date,open,high,low,close,vol"),
                ("adj_factor", "ts_code,trade_date,adj_factor"),
                ("stk_limit", "ts_code,trade_date,up_limit,down_limit"),
                ("daily_basic", "ts_code,trade_date,total_mv,pe_ttm,pb,dv_ttm,turnover_rate"),
            ):
                result = provider.query(api, {"trade_date": day.strftime("%Y%m%d")}, fields)
                if result.empty or len(result) >= 6000:
                    raise ValueError(f"{api}: empty or potentially truncated daily response")
                if set(result.trade_date.astype(str)) != {day.strftime("%Y%m%d")}:
                    raise ValueError(f"{api}: response date mismatch")
                result = result[result.ts_code.isin(assets)].copy()
                _unique(result, ["ts_code", "trade_date"], api)
                fetched[api] = result
            market, extension = append_daily_bars(market, fetched["daily"], fetched["adj_factor"], fetched["stk_limit"])
            extension_parts.append(extension)
            basic_parts.append(fetched["daily_basic"])
            print(f"Equity extension: {day.date()}, {len(extension)} stock bars", flush=True)
        if basic_parts:
            basic = pd.concat([basic, *basic_parts], ignore_index=True)
        _unique(basic, ["ts_code", "trade_date"], "combined daily basic")
        calendar = calendar[calendar <= signal_date]
        if not refresh and pd.Timestamp(asof) > original_last:
            receipt["blockers"].append("No daily refresh: only the historical snapshot is available")
        if extension_parts:
            pd.concat(extension_parts, ignore_index=True).to_parquet(folder / "market_extension.parquet", index=False)
            pd.concat(basic_parts, ignore_index=True).to_parquet(folder / "basic_extension.parquet", index=False)
        financials, industries = pd.read_parquet(paths["financials"]), pd.read_parquet(paths["industries"])
        current = latest_features(market, calendar, basic, financials, industries, signal_date)
        current.to_parquet(folder / "current_features.parquet", index=False)
        history = pd.read_parquet(paths["features"])
        cutoff_candidates = calendar[calendar < pd.Timestamp(signal_date.year, 1, 1)]
        cutoff = cutoff_candidates.max()
        # The unchanged legacy labels stop in 2026; do not imply a 2027 refresh.
        if signal_date.year != 2026:
            raise ValueError("This frozen deployment supports only the reviewed 2026 V3 fit")
        current["v3_lightgbm_frozen_reference"] , model = fit_v3_model(
            history, current, cutoff, folder / "v3_2026_lightgbm.txt")
        cards, factor_identity = _selected_cards(
            data_root / "runs/project2_cogalpha_formal", Path(__file__).parent)
        scores, ranks = cog_rank_scores(market, calendar, current.set_index("asset").eligible, cards, signal_date)
        current["cogalpha_731_rank_frozen_reference"] = current.asset.map(scores)
        ranks.to_parquet(folder / "cogalpha_current_factor_ranks.parquet")
        fields = ["date", "asset", "eligible", "raw_close", "up_limit", "down_limit", "volume",
                  "volatility", "industry", "v3_lightgbm_frozen_reference",
                  "cogalpha_731_rank_frozen_reference"]
        current[fields].to_parquet(folder / "current_scores.parquet", index=False)
        market[market.date.isin(calendar[-160:])].to_parquet(folder / "market_tail.parquet", index=False)
        pd.DataFrame({"date": calendar}).to_csv(folder / "calendar.csv", index=False)
        receipt.update(
            status="research_scores_prepared", signal_date=str(signal_date.date()),
            original_data_end=str(original_last.date()), extension_sessions=len(extension_dates),
            extension_rows=sum(map(len, extension_parts)), assets=len(assets),
            eligible_assets=int(current.eligible.sum()), model=model,
            cogalpha_selection=factor_identity,
            **assess_score_readiness(current, signal_date, refreshed=refresh),
            reference_policy="Annual statements and industry intervals frozen at the original snapshot",
            reference_snapshot_date=str(original_last.date()),
            variant_labels={
                "v3_lightgbm_frozen_reference": "V3 前向适配：财务与行业参考冻结截至 " + str(original_last.date()),
                "cogalpha_731_rank_frozen_reference": "CogAlpha 731 前向适配：财务与行业参考冻结截至 " + str(original_last.date()),
            },
            original_strategy_fidelity=False,
            simulated_execution_requires="Separate new plan, current quotes, lot/T+1/limit checks, and account reconciliation",
        )
        receipt["blockers"].append(
            "Financial announcements and industry changes after the original snapshot are not refreshed; "
            "these are explicitly frozen-reference research variants, not certified original V3/Cog deployment")
        receipt["files_sha256"] = {
            p.relative_to(folder).as_posix(): digest(p) for p in folder.rglob("*")
            if p.is_file() and p.name != "receipt.json"}
        write_json(folder / "receipt.json", receipt)
    except Exception as exc:
        # Provider internals may contain request details. Persist only the class.
        receipt.update(status="blocked", error_type=type(exc).__name__)
        receipt["blockers"].append("Preparation failed; inspect local input availability or provider access")
        write_json(folder / "receipt.json", receipt)
        raise
    return receipt
