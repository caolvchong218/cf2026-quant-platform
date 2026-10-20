"""Continuous equity feature preparation over an immutable research bootstrap."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

from .data import TushareProvider, digest, load_calendar, load_market, write_json
from .equity_data_store import EquityDataStore, load_feature_days, save_feature_day
from .equity_forward import (
    SHANGHAI, _selected_cards, append_daily_bars, assess_score_readiness,
    cog_rank_scores, completed_session, latest_features,
)
from .equity_training import (
    extend_training_history, load_or_fit_annual_v3, predict_current, resolve_annual_window,
)


def _verified_feature_day(path):
    receipt_path = path / "commit.json"
    if not receipt_path.exists():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if digest(path / "features.parquet") != receipt["sha256"]:
        raise ValueError("Frozen feature identity mismatch")
    return pd.read_parquet(path / "features.parquet")


def prepare_equity_forward(data_root, output_root, asof, *, token_file=None,
                           refresh=True, provider=None, now=None, official_calendar=None,
                           max_sessions=20, max_reference_requests=120):
    """Resume data/reference checkpoints, freeze daily features, and cache annual fits.

    A partial receipt explicitly remains unready. Older accepted research files
    are never changed. The fixed stock universe and factor definitions remain
    versioned; this is a forward implementation, not a new historical backtest.
    """
    from .equity_reference import ReferenceStore
    from .simulation_calendar import load_calendar_cache, refresh_calendar
    data_root, output_root = Path(data_root), Path(output_root)
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("A timezone-aware observation time is required")
    # v2.5 equity directories used Shanghai timestamps. Preserve that ordering
    # so a same-evening upgrade cannot accidentally keep selecting the old run.
    folder = output_root / datetime.now(SHANGHAI).strftime("%Y%m%dT%H%M%S%f")
    folder.mkdir(parents=True, exist_ok=False)
    state_root = output_root.parent
    committed_before = len(list((state_root / "equity_market_store").glob("*/commit.json")))
    source = data_root / "data/private/mainboard1000_20260918/data/processed"
    extra = data_root / "data/private/research_v2"
    paths = {
        "market": source / "market.csv", "calendar": source / "calendar.csv",
        "features": extra / "features.parquet", "basic": extra / "daily_basic.parquet",
        "financials": extra / "financials.parquet", "industries": extra / "industries.parquet",
    }
    receipt = {
        "schema": 2, "status": "preparing", "requested_date": str(pd.Timestamp(asof).date()),
        "order_submission": False, "snapshot_directory": str(folder), "scores_ready": False,
        "ready_for_auto_orders": False, "blockers": [], "progress_count": 0,
        "reference_policy": "Observed versions; announcement lag; immutable daily features",
        "code_sha256": {name: digest(Path(__file__).parent / name) for name in (
            "equity_continuous.py", "equity_forward.py", "equity_data_store.py", "equity_reference.py",
            "equity_training.py", "features.py", "factor_lab.py")},
    }
    write_json(folder / "receipt.json", receipt)
    try:
        base_market, base_basic = load_market(paths["market"]), pd.read_parquet(paths["basic"])
        original_last = base_market.date.max()
        assets = sorted(base_market.asset.unique())
        receipt["source_sha256"] = {name: digest(path) for name, path in paths.items()}
        cache_path = extra / "features_cache.json"
        original_cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if original_cache["output"] != receipt["source_sha256"]["features"]:
            raise ValueError("Original feature cache content changed")
        for name, expected in original_cache["inputs"].items():
            if name.startswith("data/") and digest(data_root / name) != expected:
                raise ValueError("Original feature-cache data input changed")
        receipt["feature_cache_sha256"] = digest(cache_path)
        provider = provider or (TushareProvider(folder / "raw", token_file, delay=1.3) if refresh else None)
        if official_calendar is None:
            # The worker supplies its separately maintained calendar. A standalone
            # preparation must also extend an existing cache, not only create one.
            official_calendar = (
                refresh_calendar(state_root / "calendar_store", provider, now=now)
                if refresh else load_calendar_cache(state_root / "calendar_store"))
        if official_calendar is None:
            days = load_calendar(paths["calendar"])
        else:
            days = pd.DatetimeIndex(pd.to_datetime(
                official_calendar.loc[official_calendar.is_open.eq(1), "cal_date"])).sort_values()
        signal_day = completed_session(days, asof, now=now)
        receipt["signal_date"] = str(signal_day.date())
        next_days = days[days > signal_day]
        if next_days.empty:
            raise ValueError("Next official execution session is not published")
        window = resolve_annual_window(days, signal_asof=signal_day, execution_date=next_days[0])
        receipt["execution_date"] = str(next_days[0].date())
        store = EquityDataStore(state_root / "equity_market_store",
            source_identity={key: receipt["source_sha256"][key] for key in ("market", "basic")},
            base_end=original_last, assets=assets)
        migration = store.root / "legacy_import.json"
        if not migration.exists():
            candidates = sorted((data_root / "runs/simulation_auto/equities").glob("*/receipt.json"), reverse=True)
            imported = None
            for path in candidates:
                old = json.loads(path.read_text(encoding="utf-8"))
                if (old.get("schema") == 1 and old.get("scores_ready") is True
                        and (path.parent / "market_extension.parquet").is_file()):
                    store.import_legacy(path.parent)
                    imported = digest(path)
                    break
            write_json(migration, {"receipt_sha256": imported})
        if refresh:
            market, basic, progress = store.refresh(provider, base_market, base_basic, days, signal_day,
                append_bars=append_daily_bars, max_sessions=max_sessions)
        else:
            market, basic, missing = store.load(base_market, base_basic, days, signal_day)
            progress = {"ready": not len(missing), "remaining_sessions": len(missing), "completed_this_run": 0}
        receipt["market_progress"] = progress
        receipt["progress_count"] += progress["completed_this_run"]
        references = ReferenceStore(state_root / "reference_store",
            pd.read_parquet(paths["financials"]), pd.read_parquet(paths["industries"]),
            assets, original_last)
        if refresh:
            reference_status = references.refresh(provider, signal_day,
                                                   max_requests=max_reference_requests)
        else:
            _, _, reference_status = references.snapshot_asof(signal_day, observed_at=now)
        receipt["reference_status"] = reference_status
        receipt["progress_count"] += reference_status.get("completed_this_run", 0)
        if not progress["ready"] or not reference_status.get("ready", False):
            receipt.update(status="catching_up", blockers=["Incremental market/reference work remains"])
            write_json(folder / "receipt.json", receipt)
            return receipt
        # Each missing feature date uses the reference version available then.
        # Catch-up never applies today's corrected industry/financial tables to September.
        feature_root = state_root / "feature_store"
        reference_observed_at = datetime.now(timezone.utc)
        extension_days = days[(days > original_last) & (days <= signal_day)]
        current = None
        for day in extension_days:
            frozen = _verified_feature_day(feature_root / day.strftime("%Y%m%d"))
            if frozen is None:
                financials, industries, reference = references.snapshot_asof(day, observed_at=reference_observed_at)
                features = latest_features(market, days, basic, financials, industries, day)
                frozen = save_feature_day(feature_root, features, reference_identity=reference)
                receipt["progress_count"] += 1
            if day == signal_day:
                current = frozen
        if current is None:
            financials, industries, _ = references.snapshot_asof(signal_day, observed_at=reference_observed_at)
            current = latest_features(market, days, basic, financials, industries, signal_day)
        # Training identity contains only input partitions which can affect this year's fit.
        cutoff = pd.Timestamp(window.cutoff_exclusive)
        _, feature_ids = load_feature_days(feature_root, through=cutoff)
        price_ids = {}
        for path in sorted(store.root.glob("*/commit.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if pd.Timestamp(record["date"]) < cutoff:
                price_ids[record["date"]] = record["files"]["market.parquet"]
        identity = {"seed_features": receipt["source_sha256"]["features"],
                    "seed_market": receipt["source_sha256"]["market"],
                    "feature_days": feature_ids, "label_price_days": price_ids,
                    "training_calendar_sha256": hashlib.sha256(
                        "\n".join(days[days <= cutoff].strftime("%Y-%m-%d")).encode()).hexdigest()}

        def training_history():
            fresh_features, _ = load_feature_days(feature_root, through=cutoff)
            reference_by_day = {}
            for path in sorted(feature_root.glob("*/commit.json")):
                item = json.loads(path.read_text(encoding="utf-8"))
                if pd.Timestamp(item["date"]) < cutoff:
                    # The per-day feature is frozen using snapshot_asof(day), never a later vintage.
                    reference_by_day[item["date"]] = {"available_by": item["date"],
                        "commit_sha256": digest(path), "snapshot": item["reference_identity"]}
            return extend_training_history(
                pd.read_parquet(paths["features"]), fresh_features, market, days,
                observed_asof=signal_day, seed_identity={"features_sha256": receipt["source_sha256"]["features"]},
                reference_identity={"by_date": reference_by_day})

        model = load_or_fit_annual_v3(training_history, window,
            cache_root=state_root / "annual_models", feature_identity=identity)
        current = current.copy()
        current.to_parquet(folder / "current_features.parquet", index=False)
        current["v3_lightgbm_frozen_reference"] = predict_current(model, current, execution_date=next_days[0])
        cards, factor_ids = _selected_cards(data_root / "runs/project2_cogalpha_formal", Path(__file__).parent)
        # CogAlpha contains rolling formulas; preserve its full fixed-anchor history.
        rank_scores, ranks = cog_rank_scores(
            market, days[days <= signal_day], current.set_index("asset").eligible, cards, signal_day)
        current["cogalpha_731_rank_frozen_reference"] = current.asset.map(rank_scores)
        ranks.to_parquet(folder / "cogalpha_current_factor_ranks.parquet")
        fields = ["date", "asset", "eligible", "raw_close", "up_limit", "down_limit", "volume",
                  "volatility", "industry", "v3_lightgbm_frozen_reference", "cogalpha_731_rank_frozen_reference"]
        current[fields].to_parquet(folder / "current_scores.parquet", index=False)
        market[market.date.isin(days[days <= signal_day][-160:])].to_parquet(folder / "market_tail.parquet", index=False)
        pd.DataFrame({"date": days[days <= signal_day]}).to_csv(folder / "calendar.csv", index=False)
        shutil.copy2(model.model_path, folder / f"v3_{window.model_year}_lightgbm.txt")
        readiness = assess_score_readiness(current, signal_day, refreshed=refresh)
        receipt.update(
            status="continuous_scores_prepared", **readiness,
            ready_for_auto_orders=readiness["scores_ready"] and reference_status["ready"],
            original_data_end=str(original_last.date()), extension_sessions=len(extension_days),
            extension_rows=int((market.date > original_last).sum()), assets=len(assets),
            eligible_assets=int(current.eligible.sum()),
            model={**model.metadata, "model_year": window.model_year, "cache_hit": model.cache_hit},
            cogalpha_selection=factor_ids, reference_snapshot_date=str(signal_day.date()),
            variant_labels={"v3_lightgbm_frozen_reference": "V3 · 年度模型 / 观测时点参考数据",
                            "cogalpha_731_rank_frozen_reference": "CogAlpha 731 · 冻结公式 / 持续参考数据"},
            original_strategy_fidelity=False,
            simulated_execution_requires="Current quotes, lot/T+1/limit checks and account reconciliation",
        )
        receipt["files_sha256"] = {path.relative_to(folder).as_posix(): digest(path)
            for path in folder.iterdir() if path.is_file() and path.name != "receipt.json"}
        write_json(folder / "receipt.json", receipt)
        return receipt
    except Exception as exc:
        committed_after = len(list((state_root / "equity_market_store").glob("*/commit.json")))
        receipt["progress_count"] = max(receipt["progress_count"], committed_after - committed_before)
        receipt.update(status="blocked", error_type=type(exc).__name__,
                       blockers=["Data preparation failed; no stale score execution"])
        if receipt["progress_count"] > 0:
            receipt["status"] = "catching_up"
            write_json(folder / "receipt.json", receipt)
            return receipt
        write_json(folder / "receipt.json", receipt)
        raise
