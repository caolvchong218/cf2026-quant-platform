"""Causal annual V3 training, append-only feature history and verified model cache.

The legacy feature snapshot is an immutable seed. New formation-date features
must already use announcement-lagged/reference-vintage inputs; this module
additionally refuses a reference vintage observed after its formation date.
Forward-return labels are a separate table, never prediction inputs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
import hashlib
from importlib import metadata as package_metadata
import json
from pathlib import Path
from typing import Callable, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd

from .benchmark_research import training_mask
from .data import digest
from .features import FEATURES


SCHEMA_VERSION = 1
INPUT_FIELDS = tuple(["z_" + name for name in FEATURES] + ["n_" + name for name in FEATURES])
LGB_PARAMETERS = {
    "n_estimators": 250, "learning_rate": .03, "num_leaves": 15, "max_depth": 5,
    "min_child_samples": 500, "colsample_bytree": .8, "reg_lambda": 10.,
    "random_state": 20260921, "n_jobs": 4, "verbosity": -1, "deterministic": True,
    "force_col_wise": True,
}
LABEL_DEFINITION = {
    "entry_sessions_after_formation": 1, "exit_sessions_after_formation": 21,
    "holding_sessions": 20, "price": "fixed-anchor adjusted open",
    "formula": "open[t+21]/open[t+1]-1", "maturity": "exit session <= observed_asof",
    "training_embargo": "label_exit strictly before annual cutoff",
    "target": "within-formation-date percentile rank of eligible training labels minus 0.5",
}
_LABEL_COLUMNS = ("label", "label_entry", "label_exit", "label_observed_asof")


class CacheIntegrityError(ValueError):
    """A partial/corrupt/differently identified cache is never silently replaced."""


def _json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _sha(value) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _day(value) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is not None or result != result.normalize():
        raise ValueError("Use timezone-naive formation/session dates, without a time of day.")
    return result


def _calendar(calendar) -> pd.DatetimeIndex:
    result = pd.DatetimeIndex(pd.to_datetime(calendar))
    if (result.empty or result.hasnans or result.tz is not None or not result.is_unique
            or not result.is_monotonic_increasing or not result.equals(result.normalize())):
        raise ValueError("An ordered, unique official daily trading calendar is required.")
    return result


@dataclass(frozen=True)
class AnnualWindow:
    model_year: int
    cutoff_exclusive: date
    training_start: date
    first_execution_session: date

    def __post_init__(self):
        if (type(self.model_year) is not int
                or any(type(value) is not date for value in
                       (self.cutoff_exclusive, self.training_start, self.first_execution_session))
                or self.first_execution_session.year != self.model_year
                or self.cutoff_exclusive.year != self.model_year - 1
                or self.cutoff_exclusive >= self.first_execution_session
                or self.training_start != (pd.Timestamp(self.cutoff_exclusive)
                                            - pd.DateOffset(years=3)).date()):
            raise ValueError("Invalid three-year annual training window.")

    def to_dict(self) -> dict:
        return {key: str(value) if isinstance(value, date) else value
                for key, value in asdict(self).items()}


def resolve_annual_window(calendar, *, signal_asof, execution_date) -> AnnualWindow:
    """Execution year owns the model, including signals formed on New Year's Eve."""
    days = _calendar(calendar)
    signal, execution = _day(signal_asof), _day(execution_date)
    if signal not in days or execution not in days or days.get_loc(execution) != days.get_loc(signal) + 1:
        raise ValueError("Execution must be the strictly next official session after formation.")
    year = execution.year
    first_candidates = days[days >= pd.Timestamp(year, 1, 1)]
    first = first_candidates[0]
    position = days.get_loc(first)
    if position == 0 or days[0] > pd.Timestamp(year - 1, 12, 1):
        raise ValueError("Calendar must cover the prior December and the first execution session.")
    cutoff = days[position - 1]
    return AnnualWindow(year, cutoff.date(), (cutoff - pd.DateOffset(years=3)).date(), first.date())


def _feature_rows(frame: pd.DataFrame, *, name: str, allow_empty: bool = False) -> pd.DataFrame:
    required = {"date", "asset", "eligible", *INPUT_FIELDS}
    if not isinstance(frame, pd.DataFrame) or not required <= set(frame):
        raise ValueError(f"{name} lacks dated assets, eligibility or the frozen 24 input fields.")
    result = frame.copy()
    result["date"] = pd.to_datetime(result.date, errors="raise")
    if (result.empty and not allow_empty) or result.date.isna().any():
        raise ValueError(f"{name} must contain valid formation dates.")
    if (result.date.dt.tz is not None or not result.date.eq(result.date.dt.normalize()).all()
            or result.duplicated(["date", "asset"]).any()
            or not result.asset.map(lambda value: isinstance(value, str) and bool(value)).all()):
        raise ValueError(f"{name} has invalid or duplicate date/asset keys.")
    if not result.eligible.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError(f"{name} eligibility must be explicit Booleans.")
    for field in INPUT_FIELDS:
        result[field] = pd.to_numeric(result[field], errors="raise")
    if np.isinf(result.loc[:, INPUT_FIELDS].to_numpy(dtype=float)).any():
        raise ValueError(f"{name} contains infinite model features.")
    return result


def _reference_dates(identity: Mapping, formation_dates: pd.DatetimeIndex) -> dict[str, str]:
    if not isinstance(identity, Mapping) or not identity:
        raise ValueError("Reference-vintage identity is required.")
    result = {}
    for day in formation_dates:
        item = identity.get("by_date", {}).get(str(day.date()), identity)
        if not isinstance(item, Mapping) or "available_by" not in item:
            raise ValueError("Reference identity needs available_by, globally or for every new formation date.")
        available = _day(item["available_by"])
        if available > day:
            raise ValueError("Reference vintage postdates formation; do not backfill later observations.")
        result[str(day.date())] = str(available.date())
    return result


@dataclass(frozen=True)
class TrainingHistory:
    features: pd.DataFrame
    labels: pd.DataFrame
    receipt: dict

    def training_frame(self) -> pd.DataFrame:
        return self.features.merge(self.labels, on=["date", "asset"], how="left", validate="one_to_one")


def extend_training_history(seed_features: pd.DataFrame, new_feature_rows: pd.DataFrame,
                            market: pd.DataFrame, calendar, *, observed_asof,
                            seed_identity: Mapping, reference_identity: Mapping) -> TrainingHistory:
    """Append dated features and separately mature t+1/t+21 open labels.

    Already-known seed labels remain usable when ``market`` only contains a
    recent tail. When both original endpoints are available, their recomputed
    return must agree with the seed. No feature value is rewritten. Missing
    endpoint prices remain unlabeled; intervening missing prices do not alter
    the original endpoint-return definition.

    ``reference_identity`` must identify a vintage available by each new
    formation date, using either a global ``available_by`` or ``by_date`` map.
    This complements, rather than replaces, the upstream announcement lag.
    """
    days, observed = _calendar(calendar), _day(observed_asof)
    if observed not in days:
        raise ValueError("Observation cutoff must be an official trading session.")
    if not isinstance(seed_identity, Mapping) or not seed_identity:
        raise ValueError("The immutable seed feature identity is required.")
    _json_bytes(dict(seed_identity))
    _json_bytes(dict(reference_identity))
    seed = _feature_rows(seed_features, name="seed")
    if isinstance(new_feature_rows, pd.DataFrame) and new_feature_rows.empty and not len(new_feature_rows.columns):
        added = seed.iloc[:0].drop(columns=list(_LABEL_COLUMNS), errors="ignore")
    else:
        added = _feature_rows(new_feature_rows, name="new features", allow_empty=True)
    if seed.date.max() > observed or (not added.empty and added.date.max() > observed):
        raise ValueError("Features from unobserved future sessions are forbidden.")
    if not added.empty and added.date.min() <= seed.date.max():
        raise ValueError("New features must append after the seed; historical feature replacement is forbidden.")
    vintages = _reference_dates(reference_identity, pd.DatetimeIndex(added.date.unique()).sort_values())
    seed_keys = pd.MultiIndex.from_frame(seed[["date", "asset"]])
    features = pd.concat([
        seed.drop(columns=list(_LABEL_COLUMNS), errors="ignore"),
        added.drop(columns=list(_LABEL_COLUMNS), errors="ignore"),
    ], ignore_index=True).sort_values(["date", "asset"]).reset_index(drop=True)
    if not features.date.isin(days).all():
        raise ValueError("Feature formation date is absent from the official calendar.")
    if not {"date", "asset", "open"} <= set(market):
        raise ValueError("Adjusted open prices need date, asset and open fields.")
    prices = market.loc[:, ["date", "asset", "open"]].copy()
    prices["date"] = pd.to_datetime(prices.date, errors="raise")
    prices["open"] = pd.to_numeric(prices.open, errors="raise")
    if (prices.date.isna().any() or prices.date.dt.tz is not None
            or not prices.date.eq(prices.date.dt.normalize()).all()
            or not prices.date.isin(days).all() or (prices.date > observed).any()
            or prices.duplicated(["date", "asset"]).any()
            or np.isinf(prices.open).any() or (prices.open.dropna() <= 0).any()):
        raise ValueError("Open history has future/invalid/missing-calendar dates, duplicates or invalid prices.")
    locations = days.get_indexer(features.date)

    def shifted(offset):
        result = np.full(len(features), np.datetime64("NaT"), dtype="datetime64[ns]")
        valid = locations + offset < len(days)
        result[valid] = days.to_numpy()[locations[valid] + offset]
        return pd.Series(result)

    labels = features[["date", "asset"]].copy()
    labels["label_entry"], labels["label_exit"] = shifted(1), shifted(21)
    indexed = prices.set_index(["date", "asset"]).open
    entry_keys = pd.MultiIndex.from_arrays([labels.label_entry, labels.asset])
    exit_keys = pd.MultiIndex.from_arrays([labels.label_exit, labels.asset])
    entry, exit_ = indexed.reindex(entry_keys).to_numpy(), indexed.reindex(exit_keys).to_numpy()
    complete = (np.isfinite(entry) & np.isfinite(exit_)
                & labels.label_exit.le(observed).to_numpy())
    calculated = np.full(len(labels), np.nan)
    calculated[complete] = exit_[complete] / entry[complete] - 1
    if np.isinf(calculated).any():
        raise ValueError("Nonfinite forward return.")
    labels["label"] = calculated
    retained = 0
    if "label" in seed and "label_exit" in seed:
        original = seed.set_index(["date", "asset"])[["label", "label_exit"]].copy()
        original["label_exit"] = pd.to_datetime(original.label_exit, errors="raise")
        original["label"] = pd.to_numeric(original.label, errors="raise")
        if np.isinf(original.label).any():
            raise ValueError("Seed labels cannot be infinite.")
        aligned = original.reindex(pd.MultiIndex.from_frame(labels[["date", "asset"]]))
        known = aligned.label.notna().to_numpy()
        scheduled = labels.label_exit.to_numpy()
        known_exit = aligned.label_exit.to_numpy()
        if (known & (pd.isna(known_exit) | (known_exit != scheduled))).any():
            raise ValueError("Seed label exit differs from the original 21-session label contract.")
        observed_known = known & (known_exit <= observed.to_datetime64())
        overlap = observed_known & complete
        if not np.allclose(aligned.label.to_numpy()[overlap], calculated[overlap], rtol=1e-10, atol=1e-12):
            raise ValueError("Historical label endpoints changed relative to the immutable seed.")
        labels.loc[observed_known, "label"] = aligned.label.to_numpy()[observed_known]
        retained = int(observed_known.sum())
    labels["label_observed_asof"] = pd.NaT
    labels.loc[labels.label.notna(), "label_observed_asof"] = observed
    is_seed = pd.MultiIndex.from_frame(labels[["date", "asset"]]).isin(seed_keys)
    receipt = {
        "schema_version": SCHEMA_VERSION, "observed_asof": str(observed.date()),
        "seed_identity": dict(seed_identity), "reference_identity": dict(reference_identity),
        "new_feature_reference_available_by": vintages,
        "seed_rows": len(seed), "new_feature_rows": len(added), "feature_rows": len(features),
        "mature_label_rows": int(labels.label.notna().sum()), "retained_seed_labels": retained,
        "newly_mature_seed_labels": int((is_seed & labels.label.notna().to_numpy()).sum()) - retained,
        "label_definition": LABEL_DEFINITION, "original_features_modified": False,
    }
    return TrainingHistory(features, labels, receipt)


def _training_selection(history, window: AnnualWindow):
    frame = history.training_frame() if isinstance(history, TrainingHistory) else history
    data = _feature_rows(frame, name="training history")
    if not {"label", "label_exit"} <= set(data):
        raise ValueError("Training history requires separately matured labels.")
    data["label_exit"] = pd.to_datetime(data.label_exit, errors="raise")
    data["label"] = pd.to_numeric(data.label, errors="raise")
    if data.label_exit.dt.tz is not None or np.isinf(data.label).any():
        raise ValueError("Invalid training label values or dates.")
    eligible = data.eligible & (data[["n_" + name for name in FEATURES]].notna().sum(axis=1) >= 10)
    selected = data.loc[training_mask(data, eligible, pd.Timestamp(window.cutoff_exclusive))].copy()
    if selected.empty:
        raise ValueError("No fully matured, eligible labels in the rolling three-year window.")
    if (selected.label_exit <= selected.date).any():
        raise ValueError("Training labels must exit strictly after their formation date.")
    selected = selected.sort_values(["date", "asset"]).reset_index(drop=True)
    x = selected.loc[:, INPUT_FIELDS].fillna(0).to_numpy(dtype=np.float32)
    y = (selected.groupby("date").label.rank(pct=True) - .5).to_numpy(dtype=np.float32)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Training matrix and ranked targets must be finite.")
    matrix_sha = hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()
    key_text = "\n".join(
        f"{day:%Y-%m-%d}|{asset}|{exit_day:%Y-%m-%d}"
        for day, asset, exit_day in selected[["date", "asset", "label_exit"]].itertuples(index=False, name=None)
    )
    keys_sha = hashlib.sha256(key_text.encode()).hexdigest()
    return selected, x, y, matrix_sha, keys_sha


def _backend_version() -> str:
    try:
        return package_metadata.version("lightgbm")
    except package_metadata.PackageNotFoundError:
        return "unavailable"


def _input_values_sha(selected) -> str:
    """Also detect source changes that preserve float32 inputs or label ranks."""
    values = selected.loc[:, [*INPUT_FIELDS, "label"]].to_numpy(dtype=np.float64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _fit_text_model(x, y, parameters, path):
    from lightgbm import LGBMRegressor
    estimator = LGBMRegressor(**parameters).fit(x, y)
    estimator.booster_.save_model(str(path))


def _load_predictor(path):
    from lightgbm import Booster
    return Booster(model_file=str(path))


@dataclass(frozen=True)
class AnnualModel:
    model_path: Path
    metadata_path: Path
    training_path: Path
    metadata: dict
    cache_hit: bool


def _verified_cache(folder: Path, identity: dict) -> AnnualModel:
    metadata_path = folder / "metadata.json"
    try:
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
        if record["identity"] != identity or record["cache_key"] != _sha(identity):
            raise CacheIntegrityError("Annual model cache identity changed.")
        for name in ("model.txt", "training.parquet"):
            if digest(folder / name) != record["files_sha256"][name]:
                raise CacheIntegrityError("Annual model or frozen training snapshot is corrupt.")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise CacheIntegrityError("Annual model cache is incomplete; preserve it for inspection.") from exc
    return AnnualModel(folder / "model.txt", metadata_path, folder / "training.parquet", record, True)


def load_or_fit_annual_v3(history: TrainingHistory | pd.DataFrame | Callable,
                          window: AnnualWindow, *, cache_root,
                          feature_identity: Mapping, parameters: Mapping | None = None) -> AnnualModel:
    """Load a verified annual model, or fit once into a new immutable cache.

    A callable history loader is not invoked on cache hits. In that mode the
    caller MUST verify ``feature_identity`` against the relevant training input
    manifests first. It should identify only the annual training slice, not
    today's prediction inputs or later append-only observations.

    With an eager history object, a cache hit also recomputes its selected matrix
    identity. Changed rows under an unchanged declared input identity fail
    closed; pass a genuinely new input identity to create a separate cache.
    """
    if not isinstance(window, AnnualWindow) or not isinstance(feature_identity, Mapping) or not feature_identity:
        raise ValueError("An annual window and verified training-input identity are required.")
    params = dict(LGB_PARAMETERS if parameters is None else parameters)
    if not params:
        raise ValueError("Explicit model parameters cannot be empty.")
    identity = {
        "schema_version": SCHEMA_VERSION, "window": window.to_dict(),
        "input_fields": list(INPUT_FIELDS), "label_definition": LABEL_DEFINITION,
        "parameters": params, "feature_identity": dict(feature_identity),
        "training_code_sha256": digest(Path(__file__)),
        "feature_code_sha256": digest(Path(__file__).with_name("features.py")),
        "training_mask_code_sha256": digest(Path(__file__).with_name("benchmark_research.py")),
        "lightgbm_version": _backend_version(),
    }
    # Normalize JSON-compatible containers before comparing to a persisted record.
    identity = json.loads(_json_bytes(identity))
    key = _sha(identity)
    folder = Path(cache_root) / f"year_{window.model_year}" / key
    if folder.exists():
        cached = _verified_cache(folder, identity)
        if not callable(history):
            selected, _, _, matrix_sha, keys_sha = _training_selection(history, window)
            if (matrix_sha != cached.metadata["training_matrix_sha256"]
                    or keys_sha != cached.metadata["training_row_keys_sha256"]
                    or _input_values_sha(selected) != cached.metadata["training_values_sha256"]):
                raise CacheIntegrityError("Training inputs changed under an unchanged declared identity.")
        return cached
    loaded = history() if callable(history) else history
    selected, x, y, matrix_sha, keys_sha = _training_selection(loaded, window)
    # An interrupted fit leaves an inspectable pending directory. It never
    # publishes a partial cache that would prevent a later successful retry.
    pending = folder.with_name(f".{key}.pending-{uuid4().hex}")
    pending.mkdir(parents=True, exist_ok=False)
    training_path, model_path = pending / "training.parquet", pending / "model.txt"
    snapshot_fields = ["date", "asset", "eligible", *INPUT_FIELDS, "label", "label_exit"]
    if "label_entry" in selected:
        snapshot_fields.append("label_entry")
    selected[snapshot_fields].to_parquet(training_path, index=False)
    _fit_text_model(x, y, params, model_path)
    record = {
        "identity": identity, "cache_key": key, "training_rows": len(selected),
        "first_training_formation": str(selected.date.min().date()),
        "last_training_formation": str(selected.date.max().date()),
        "latest_training_label_exit": str(selected.label_exit.max().date()),
        "training_matrix_sha256": matrix_sha, "training_row_keys_sha256": keys_sha,
        "training_values_sha256": _input_values_sha(selected),
        "files_sha256": {"model.txt": digest(model_path), "training.parquet": digest(training_path)},
    }
    metadata_path = pending / "metadata.json"
    with metadata_path.open("xb") as stream:
        stream.write(_json_bytes(record) + b"\n")
    _verified_cache(pending, identity)
    try:
        pending.rename(folder)
    except FileExistsError:
        # Another worker published this exact identity first. Preserve our
        # completed pending attempt and use only the verified winning cache.
        return _verified_cache(folder, identity)
    return AnnualModel(folder / "model.txt", folder / "metadata.json",
                       folder / "training.parquet", record, False)


def predict_current(model: AnnualModel, current: pd.DataFrame, *, execution_date=None) -> pd.Series:
    """Predict one formation cross-section without fitting or using return labels."""
    if not isinstance(model, AnnualModel):
        raise ValueError("A verified annual model is required.")
    if digest(model.model_path) != model.metadata["files_sha256"]["model.txt"]:
        raise CacheIntegrityError("Model changed after cache validation.")
    frame = _feature_rows(current, name="prediction")
    if frame.date.nunique() != 1:
        raise ValueError("Prediction expects a single formation-date cross-section.")
    formation = frame.date.iloc[0]
    window = model.metadata["identity"]["window"]
    if formation < pd.Timestamp(window["cutoff_exclusive"]) or formation.year > window["model_year"]:
        raise ValueError("Prediction date is outside this annual model's formation range.")
    if execution_date is not None:
        execution = _day(execution_date)
        if execution <= formation or execution.year != window["model_year"]:
            raise ValueError("Prediction must use the model belonging to its execution year.")
    valid = frame.eligible & (frame[["n_" + name for name in FEATURES]].notna().sum(axis=1) >= 10)
    result = pd.Series(np.nan, index=frame.index, name="v3_prediction", dtype=float)
    if valid.any():
        x = frame.loc[valid, INPUT_FIELDS].fillna(0).to_numpy(dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError("Prediction inputs must be finite.")
        predicted = np.asarray(_load_predictor(model.model_path).predict(x), dtype=float)
        if predicted.shape != (int(valid.sum()),) or not np.isfinite(predicted).all():
            raise ValueError("Model returned invalid prediction values.")
        result.loc[valid] = predicted
    return result
