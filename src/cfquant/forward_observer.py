"""Immutable local research plans and append-only daily observation records.

Local timestamps and hashes establish reproducibility and detect accidental
rewrites. They cannot authenticate when a researcher learned information or
prove live execution. Historical post-cutoff replay is labelled separately.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


SCHEMA_VERSION = "cfquant.forward-observer.v1"
_PLAN_ID = re.compile(r"^[0-9a-f]{24}$")


def _json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported identity value: {type(value).__name__}")


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, default=_json_default)


def _clean(value):
    return json.loads(_canonical(value))


def _hash(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _day(value, name):
    parsed = pd.Timestamp(value)
    if pd.isna(parsed) or parsed.tz is not None or parsed != parsed.normalize():
        raise ValueError(f"{name} must be a timezone-naive calendar date")
    return parsed.date().isoformat()


def _instant(value=None):
    parsed = pd.Timestamp(datetime.now(timezone.utc) if value is None else value)
    if pd.isna(parsed) or parsed.tz is None:
        raise ValueError("Recording timestamps must include a timezone")
    return parsed.tz_convert("UTC").isoformat()


def _plan_directory(root, plan_id):
    if not isinstance(plan_id, str) or not _PLAN_ID.fullmatch(plan_id):
        raise ValueError("Invalid plan_id")
    return Path(root) / plan_id


def _write_once(path, value):
    # Exclusive creation preserves prior records; an existing file is never replaced.
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
        stream.write("\n")


def _default_code_identity():
    folder = Path(__file__).resolve().parent
    return {"kind": "package_source_sha256", "files": {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(folder.glob("*.py"))}}


def _default_data_identity(config):
    files = {}
    for key in ("data_path", "calendar_path"):
        value = config.get(key)
        if value:
            path = Path(value)
            if path.is_file():
                files[key] = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {"kind": "file_sha256", "files": files} if files else {"kind": "unspecified"}


def freeze_plan(root, strategy_definition, data_cutoff, *, config=None,
                code_identity=None, data_identity=None, last_known_trade_date=None,
                frozen_at=None) -> dict:
    """Create a new immutable plan manifest and return it, including its ID.

    Supply actual snapshot hashes via ``data_identity`` (or an existing data
    path in ``config``). Absent data identities are explicit and cannot support
    a reproducible forward claim. A timestamp supplied by the caller is marked.
    """
    cutoff = _day(data_cutoff, "data_cutoff")
    last_date = _day(last_known_trade_date if last_known_trade_date is not None else cutoff, "last_known_trade_date")
    if last_date > cutoff:
        raise ValueError("last_known_trade_date cannot exceed data_cutoff")
    definition = _clean(strategy_definition)
    if not definition:
        raise ValueError("A strategy_definition is required")
    configuration = _clean(config if config is not None else {})
    if not isinstance(configuration, dict):
        raise ValueError("config must be a mapping or configuration dataclass")
    body = {
        "schema_version": SCHEMA_VERSION, "strategy_definition": definition,
        "strategy_sha256": _hash(definition), "config": configuration,
        "config_sha256": _hash(configuration),
        "code_identity": _clean(code_identity if code_identity is not None else _default_code_identity()),
        "data_identity": _clean(data_identity if data_identity is not None else _default_data_identity(configuration)),
        "data_cutoff": cutoff, "last_known_trade_date": last_date,
        "frozen_at": _instant(frozen_at),
        "timestamp_source": "caller_supplied" if frozen_at is not None else "local_clock",
        "research_scope": "paper observations; timestamps do not prove live trading or information availability",
    }
    body["code_sha256"] = _hash(body["code_identity"])
    body["data_sha256"] = _hash(body["data_identity"])
    manifest = {"plan_id": _hash(body)[:24], **body}
    directory = _plan_directory(root, manifest["plan_id"])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "observations").mkdir(exist_ok=True)
    path = directory / "plan.json"
    if path.exists():
        existing = load_plan(root, manifest["plan_id"])
        if existing != manifest:
            raise ValueError("Frozen plan conflict; prior manifest is preserved")
        return existing
    try:
        _write_once(path, manifest)
    except FileExistsError:
        existing = load_plan(root, manifest["plan_id"])
        if existing != manifest:
            raise ValueError("Frozen plan conflict; prior manifest is preserved") from None
    return manifest


def load_plan(root, plan_id) -> dict:
    manifest = json.loads((_plan_directory(root, plan_id) / "plan.json").read_text(encoding="utf-8"))
    body = {key: value for key, value in manifest.items() if key != "plan_id"}
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("plan_id") != plan_id or _hash(body)[:24] != plan_id:
        raise ValueError("Frozen plan integrity mismatch")
    for key, value in (("strategy", manifest["strategy_definition"]), ("config", manifest["config"]),
                       ("code", manifest["code_identity"]), ("data", manifest["data_identity"])):
        if manifest[f"{key}_sha256"] != _hash(value):
            raise ValueError("Frozen plan identity mismatch")
    return manifest


def _validate_payload(payload, manifest):
    observation_date = _day(payload["observation_date"], "observation_date")
    signal_asof = _day(payload["signal_asof"], "signal_asof")
    if observation_date <= manifest["data_cutoff"]:
        raise ValueError("Observation date must be strictly after data_cutoff")
    if signal_asof >= observation_date:
        raise ValueError("signal_asof must strictly precede fills/observation date")
    if not isinstance(payload.get("fills"), list) or not isinstance(payload.get("metrics"), dict):
        raise ValueError("fills must be a list and metrics must be a mapping")
    for fill in payload["fills"]:
        if not isinstance(fill, dict) or "date" not in fill:
            raise ValueError("Each observed fill requires a date")
        fill_date = _day(fill["date"], "fill date")
        if fill_date != observation_date or signal_asof >= fill_date:
            raise ValueError("Fill date must equal observation date and follow signal_asof")
        if "signal_date" in fill and _day(fill["signal_date"], "fill signal_date") >= fill_date:
            raise ValueError("Fill signal_date must strictly precede fill date")


def check_observations(root, plan_id) -> dict:
    """Read and validate the manifest and all prior observations; writes nothing."""
    manifest = load_plan(root, plan_id)
    records, previous = [], plan_id
    prior_date = manifest["data_cutoff"]
    for path in sorted((_plan_directory(root, plan_id) / "observations").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        body = {key: value for key, value in record.items() if key != "record_sha256"}
        if (record.get("schema_version") != SCHEMA_VERSION or record.get("plan_id") != plan_id
                or record.get("record_sha256") != _hash(body) or record.get("previous_sha256") != previous):
            raise ValueError("Observation integrity mismatch; prior records must not be rewritten")
        payload = record["payload"]
        _validate_payload(payload, manifest)
        observation_date = payload["observation_date"]
        if observation_date <= prior_date or path.stem != observation_date:
            raise ValueError("Observation dates must increase and match their record filenames")
        recorded = _instant(record["observed_at"])
        if pd.Timestamp(recorded) < pd.Timestamp(manifest["frozen_at"]) or observation_date > pd.Timestamp(recorded).date().isoformat():
            raise ValueError("Observation recording time must follow the freeze and observed day")
        previous, prior_date = record["record_sha256"], observation_date
        records.append(record)
    return {"passed": True, "plan": manifest, "observations": records,
            "new_days": len(records), "last_observation_date": prior_date if records else None,
            "last_record_sha256": previous}


def append_observation(root, plan_id, observation_date, signal_asof, *, fills=None,
                       metrics=None, observed_at=None) -> dict:
    """Append one daily record; identical retries return the existing record.

    A retry may have a different recording timestamp, but its signal, fills and
    metrics must match exactly. Conflicts and insertion before the latest day
    are rejected. This local file store has one writer; readers verify its chain.
    """
    checked = check_observations(root, plan_id)
    payload = _clean({"observation_date": _day(observation_date, "observation_date"),
                      "signal_asof": _day(signal_asof, "signal_asof"),
                      "fills": [] if fills is None else fills, "metrics": {} if metrics is None else metrics})
    _validate_payload(payload, checked["plan"])
    path = _plan_directory(root, plan_id) / "observations" / f"{payload['observation_date']}.json"
    for record in checked["observations"]:
        if record["payload"]["observation_date"] == payload["observation_date"]:
            if record["payload"] != payload:
                raise ValueError("Observation conflict; the existing record cannot be rewritten")
            return record
    if checked["last_observation_date"] and payload["observation_date"] <= checked["last_observation_date"]:
        raise ValueError("New observations must follow the last recorded observation")
    recorded = _instant(observed_at)
    if pd.Timestamp(recorded) < pd.Timestamp(checked["plan"]["frozen_at"]):
        raise ValueError("observed_at cannot precede frozen_at")
    if payload["observation_date"] > pd.Timestamp(recorded).date().isoformat():
        raise ValueError("An unobserved future date cannot be appended")
    body = {"schema_version": SCHEMA_VERSION, "plan_id": plan_id,
            "previous_sha256": checked["last_record_sha256"], "payload": payload,
            "observed_at": recorded, "timestamp_source": "caller_supplied" if observed_at is not None else "local_clock"}
    record = {**body, "record_sha256": _hash(body)}
    try:
        _write_once(path, record)
    except FileExistsError:
        existing = check_observations(root, plan_id)
        for accepted in existing["observations"]:
            if accepted["payload"]["observation_date"] == payload["observation_date"] and accepted["payload"] == payload:
                return accepted
        raise ValueError("Observation conflict; the existing record cannot be rewritten") from None
    return record


def observation_summary(root, plan_id, *, available_dates=None) -> dict:
    """Read-only UI status, with historical replay and genuine-forward caveats."""
    checked = check_observations(root, plan_id)
    plan, records = checked["plan"], checked["observations"]
    # A same-day local freeze cannot prove a plan preceded that session's signal.
    frozen_day = pd.Timestamp(plan["frozen_at"]).tz_convert("Asia/Shanghai").date().isoformat()
    historical = sum(record["payload"]["observation_date"] <= frozen_day for record in records)
    candidates = len(records) - historical
    available = sorted({_day(value, "available date") for value in available_dates}) if available_dates is not None else []
    unseen = [value for value in available if value > (checked["last_observation_date"] or plan["data_cutoff"])]
    complete_identity = plan["data_identity"] != {"kind": "unspecified"}
    status = "awaiting_new_sessions" if not records else ("historical_post_freeze_replay" if not candidates else "forward_paper_observations")
    return {"plan_id": plan_id, "schema_version": SCHEMA_VERSION, "data_cutoff": plan["data_cutoff"],
            "last_known_trade_date": plan["last_known_trade_date"], "frozen_at": plan["frozen_at"],
            "status": status, "new_days": len(records), "observed_days": len(records),
            "historical_post_freeze_days": historical, "forward_candidate_days": candidates,
            "available_new_days": len(unseen), "last_observation_date": checked["last_observation_date"],
            "data_identity_supplied": complete_identity, "genuine_forward_verified": False,
            "caveat": "Local paper records and caller/local-clock timestamps cannot prove genuine prospective signals or live execution; historical replay stays research."}


def list_plans(root) -> list[dict]:
    """Return verified local manifests without changing them."""
    folder = Path(root)
    if not folder.exists():
        return []
    return [load_plan(folder, path.name) for path in sorted(folder.iterdir())
            if path.is_dir() and _PLAN_ID.fullmatch(path.name) and (path / "plan.json").is_file()]
