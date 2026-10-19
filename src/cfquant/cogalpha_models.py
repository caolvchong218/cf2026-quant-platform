"""One frozen training fit for rank, Ridge, and LightGBM factor combinations.

Candidate discovery, label construction, qualification, and evaluation belong
to the caller. These models use one pre-cutoff training set, with no validation
parameter search or annual refit. No model objects or pickles are persisted.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from numbers import Integral

import numpy as np
import pandas as pd


RIDGE_PARAMETERS = {"alpha": 100.0}
LIGHTGBM_PARAMETERS = {
    "n_estimators": 200,
    "num_leaves": 15,
    "max_depth": 5,
    "learning_rate": .05,
    "min_child_samples": 100,
    "reg_lambda": 1.0,
    "deterministic": True,
    "force_col_wise": True,
    "n_jobs": 4,
    "objective": "regression",
    "verbosity": -1,
}


def _panel(frame, name):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"{name} must be a nonempty DataFrame")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans:
        raise ValueError(f"{name} must have a nonmissing DatetimeIndex")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing or not frame.columns.is_unique:
        raise ValueError(f"{name} must have unique ordered dates and unique assets")
    if any(not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype) for dtype in frame.dtypes):
        raise ValueError(f"{name} must contain numeric values")


def _dates(value, name, calendar):
    if not isinstance(value, pd.DatetimeIndex):
        raise ValueError(f"{name} must be a DatetimeIndex")
    if value.hasnans or not value.is_unique or not value.is_monotonic_increasing or not value.isin(calendar).all():
        raise ValueError(f"{name} must have unique ordered dates from the input calendar")
    return value


def _date_range(dates):
    return {"start": dates[0].isoformat() if len(dates) else None,
            "end": dates[-1].isoformat() if len(dates) else None,
            "sessions": len(dates)}


def _training_identity(selected, assets, train_dates, cutoff, valid, x, y):
    """Hash actual training design, its row identities, and fixed preprocessing."""
    digest = hashlib.sha256()
    definition = {"selected": selected, "assets": list(map(str, assets)),
                  "dates": [date.isoformat() for date in train_dates], "cutoff": cutoff.isoformat(),
                  "preprocessing": "eligible cross-sectional average percentile rank minus 0.5"}
    digest.update(json.dumps(definition, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    digest.update(np.ascontiguousarray(valid, dtype=np.uint8).tobytes())
    digest.update(np.ascontiguousarray(x, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(y, dtype="<f8").tobytes())
    return digest.hexdigest()


def fit_predict(factors, selected, labels, eligible, train_dates, prediction_dates, *,
                label_exit, cutoff="2023-01-01", seed=20261019):
    """Fit once before cutoff and score only explicit post-cutoff dates.

    Each factor and the supplied label is percentile-ranked within eligible
    assets on its own formation day, using average ties, then shifted by -0.5.
    No cross-date scaler is fitted. Only eligible rows with all selected factor
    ranks and realized labels finite enter training; incomplete prediction rows
    remain NaN. Equal-weight rank scores preserve the positive factor direction.

    The study's caller supplies next-open 10-session returns as a local label
    assumption; this module ranks those supplied labels and checks exit dates,
    but cannot infer or verify their price-based construction.
    """
    _panel(labels, "labels")
    calendar, assets = labels.index, labels.columns
    if not isinstance(factors, Mapping):
        raise ValueError("factors must map factor IDs to aligned DataFrames")
    if not isinstance(selected, list) or any(not isinstance(name, str) or not name for name in selected):
        raise ValueError("selected must be a list of nonempty factor IDs")
    if len(selected) != len(set(selected)):
        raise ValueError("selected factor IDs must be unique")
    if not set(selected).issubset(factors):
        raise ValueError("selected contains an unavailable factor ID")
    if (not isinstance(eligible, pd.DataFrame) or not eligible.index.equals(calendar)
            or not eligible.columns.equals(assets)):
        raise ValueError("eligible must have exactly the label date and asset axes")
    if eligible.isna().any().any() or any(not pd.api.types.is_bool_dtype(dtype) for dtype in eligible.dtypes):
        raise ValueError("eligible must contain nonmissing Boolean values")
    for name in selected:
        _panel(factors[name], "factor " + name)
        if not factors[name].index.equals(calendar) or not factors[name].columns.equals(assets):
            raise ValueError("Every selected factor must have exactly the label date and asset axes")
    train_dates = _dates(train_dates, "train_dates", calendar)
    prediction_dates = _dates(prediction_dates, "prediction_dates", calendar)
    boundary = pd.Timestamp(cutoff)
    if pd.isna(boundary):
        raise ValueError("cutoff cannot be missing")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer in [0, 2**31)")
    if (train_dates >= boundary).any():
        raise ValueError("Every training date must be strictly before cutoff")
    if (prediction_dates < boundary).any():
        raise ValueError("Every prediction date must be at or after cutoff")
    if not isinstance(label_exit, pd.Series) or not label_exit.index.equals(calendar):
        raise ValueError("label_exit must have exactly the label date axis")
    exits = pd.to_datetime(label_exit, errors="raise").loc[train_dates]
    if exits.isna().any() or not (exits < boundary).all():
        raise ValueError("Every training label exit must be nonmissing and strictly before cutoff")

    scores = {name: pd.DataFrame(np.nan, index=calendar, columns=assets)
              for name in ("rank", "ridge", "lightgbm")}
    lgb_parameters = {**LIGHTGBM_PARAMETERS, "random_state": int(seed)}
    metadata = {
        "status": "empty_pool" if not selected else "prepared",
        "selected": list(selected), "seed": int(seed), "cutoff": boundary.isoformat(),
        "training": _date_range(train_dates), "prediction": _date_range(prediction_dates),
        "training_rows": 0, "prediction_rows": 0,
        "last_label_exit": exits.max().isoformat() if len(exits) else None,
        "preprocessing": "mask ineligible assets, then same-day average percentile rank minus 0.5",
        "complete_factor_rows_required": True, "rank_direction": "positive, never flipped from prediction results",
        "label_definition": "caller-supplied returns; local study assumption open[t+11]/open[t+1]-1",
        "label_horizon_sessions_assumption": 10, "label_construction_verified_here": False,
        "training_fit_count_per_model": 0, "annual_retraining": False,
        "validation_tuning": False, "historical_dates_already_observed": True,
        "parameters": {"rank": {"weighting": "equal", "direction": "positive"},
                       "ridge": dict(RIDGE_PARAMETERS), "lightgbm": lgb_parameters},
        "coefficients": {}, "intercept": None, "importances": {}, "training_sha256": None,
    }
    if not selected:
        return scores, metadata
    if not len(train_dates):
        raise ValueError("A nonempty factor pool requires training dates")

    # Slice each input before ranking; future labels never enter the design.
    train_eligible = eligible.loc[train_dates]
    prediction_eligible = eligible.loc[prediction_dates]
    x_parts, prediction_parts = [], []
    for name in selected:
        frame = factors[name].replace([np.inf, -np.inf], np.nan)
        x_parts.append((frame.loc[train_dates].where(train_eligible).rank(axis=1, method="average", pct=True) - .5).to_numpy().ravel())
        prediction_parts.append((frame.loc[prediction_dates].where(prediction_eligible).rank(axis=1, method="average", pct=True) - .5).to_numpy().ravel())
    x_all = np.column_stack(x_parts)
    x_prediction = np.column_stack(prediction_parts)
    label_ranks = (labels.loc[train_dates].replace([np.inf, -np.inf], np.nan)
                   .where(train_eligible).rank(axis=1, method="average", pct=True) - .5)
    y_all = label_ranks.to_numpy().ravel()
    valid = train_eligible.to_numpy(dtype=bool).ravel() & np.isfinite(y_all) & np.isfinite(x_all).all(axis=1)
    prediction_valid = prediction_eligible.to_numpy(dtype=bool).ravel() & np.isfinite(x_prediction).all(axis=1)
    if valid.sum() < 2:
        raise ValueError("At least two eligible, complete, realized training rows are required")
    x, y = x_all[valid], y_all[valid]
    metadata["training_sha256"] = _training_identity(selected, assets, train_dates, boundary, valid, x, y)

    from sklearn.linear_model import Ridge
    from lightgbm import LGBMRegressor

    ridge = Ridge(**RIDGE_PARAMETERS).fit(x, y)
    lightgbm = LGBMRegressor(**lgb_parameters).fit(x, y)
    if prediction_valid.any():
        prediction = x_prediction[prediction_valid]
        computed = {"rank": prediction.mean(axis=1), "ridge": ridge.predict(prediction),
                    "lightgbm": lightgbm.predict(prediction)}
        for name, values in computed.items():
            if not np.isfinite(values).all():
                raise ArithmeticError(name + " returned nonfinite predictions for complete rows")
            output = np.full(len(x_prediction), np.nan)
            output[prediction_valid] = values
            scores[name].loc[prediction_dates] = output.reshape(len(prediction_dates), len(assets))
    used_dates = train_dates[np.flatnonzero(valid.reshape(len(train_dates), len(assets)).any(axis=1))]
    metadata.update(status="complete", training_rows=int(valid.sum()), prediction_rows=int(prediction_valid.sum()),
                    used_training_dates=_date_range(used_dates),
                    last_used_label_exit=exits.loc[used_dates].max().isoformat(),
                    training_fit_count_per_model=1,
                    coefficients=dict(zip(selected, map(float, ridge.coef_))), intercept=float(ridge.intercept_),
                    importances=dict(zip(selected, map(int, lightgbm.feature_importances_))))
    return scores, metadata
