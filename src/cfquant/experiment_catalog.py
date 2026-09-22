"""Read-only experiment discovery, filtered ledgers and local research notes.

Discovery reads small metadata files only. Large CSV ledgers are opened lazily,
in chunks, for the selected experiment. Notes are independent of result files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import pandas as pd
import yaml


TABLES = {"daily", "positions", "trades", "orders"}
NOTE_STATES = ("待研究", "复核中", "已复核", "保留对照")
_ID = re.compile(r"[a-f0-9]{24}\Z")
_SECRET = re.compile(r"token|password|secret|credential|api.?key", re.I)


@dataclass(frozen=True)
class Experiment:
    id: str
    relative_path: str
    label: str
    strategy: str
    version: str
    stage: str
    start: str
    end: str
    source: str
    updated_at: str
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    checks: dict[str, Any] = field(default_factory=dict)
    warning: str = ""

    @property
    def has_details(self) -> bool:
        return self.source == "本地完整实验"


def _inside(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("路径超出项目目录，已拒绝读取或保存。")
    return resolved


def _small_json(path: Path) -> dict:
    if not path.exists():
        return {}
    if path.stat().st_size > 1_000_000:
        raise ValueError(f"元数据文件过大：{path.name}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"元数据格式应为对象：{path.name}")
    return value


def _clean(value: Any) -> Any:
    """Keep exports strict JSON and avoid accidentally exporting credentials."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items() if not _SECRET.search(str(k))}
    if isinstance(value, (list, tuple)):
        return [_clean(x) for x in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return _clean(value.item())
    return value


def _date(value: Any) -> str:
    if value is None or value == "":
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _identity(relative_path: str) -> str:
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:24]


def _kind(path: str, name: str, config: dict) -> tuple[str, str, str]:
    strategy = str(config.get("strategy_id", ""))
    version = "V3" if "research_v3" in path or strategy.startswith("v3/") else (
        "V2" if "research_v2" in path or strategy.startswith("v2/") else "V1 / 其他")
    stage = "交互重跑" if "interactive" in path else (
        "验证区间" if name.startswith("validation_") else (
            "连续区间" if "continuous" in name else (
                "压力测试" if any(x in name for x in ("stress", "double_cost", "delay_one_day")) else (
                    "最终历史区间" if name.startswith("test_") else "基础实验"))))
    if not strategy:
        strategy = re.sub(r"^(test_|validation_)", "", name)
        if stage == "基础实验":
            strategy = str(config.get("factor", strategy))
    return version, stage, strategy


def discover_experiments(root: Path, include_archives: bool = False) -> list[Experiment]:
    """Index local metadata plus public fallbacks, without loading private CSVs."""
    root = Path(root).resolve()
    records: list[Experiment] = []
    runs = root / "runs"
    if runs.exists():
        for current, directories, files in os.walk(runs, followlinks=False):
            directories[:] = [name for name in directories if name != "notes"
                              and (include_archives or "archive" not in name)]
            if "metrics.json" not in files or "daily.csv" not in files:
                continue
            folder = _inside(root, Path(current))
            relative = folder.relative_to(root).as_posix()
            warning = ""
            try:
                metrics = _small_json(_inside(root, folder / "metrics.json"))
                checks = _small_json(_inside(root, folder / "checks.json"))
                config = _small_json(_inside(root, folder / "config.json"))
                if not config and (folder / "config.yaml").exists():
                    cfg_path = _inside(root, folder / "config.yaml")
                    if cfg_path.stat().st_size > 1_000_000:
                        raise ValueError("配置文件超过读取限制")
                    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8-sig")) or {}
                if not isinstance(config, dict):
                    raise ValueError("配置文件格式无效")
                used = config.get("backtest", config)
                if not isinstance(used, dict):
                    raise ValueError("回测参数格式无效")
                start, end = _date(used.get("start")), _date(used.get("end"))
            except (ValueError, OSError, yaml.YAMLError) as exc:
                metrics, checks, config, used = {}, {}, {}, {}
                start = end = ""
                warning = f"元数据读取失败：{exc}"
            version, stage, strategy = _kind(relative, folder.name, used)
            label = str(used.get("strategy_label") or used.get("name") or folder.name)
            records.append(Experiment(
                _identity(relative), relative, label, strategy, version, stage,
                start, end, "本地完整实验",
                datetime.fromtimestamp((folder / "metrics.json").stat().st_mtime, timezone.utc).isoformat(),
                _clean(config), _clean(metrics), _clean(checks), warning))

    local_paths = {x.relative_path for x in records}
    for version in ("v3", "v2"):
        evidence = root / "evidence" / f"research_{version}"
        table_path = evidence / "test.csv"
        if not table_path.exists():
            continue
        try:
            table = pd.read_csv(_inside(root, table_path))
            decision = _small_json(_inside(root, evidence / "decision.json"))
            for row in table.to_dict("records"):
                name = str(row.get("candidate", row.get("model", "")))
                if not re.fullmatch(r"[a-zA-Z0-9_\-]+", name):
                    continue
                if f"runs/research_{version}/test_{name}" in local_paths:
                    continue
                daily = _inside(root, evidence / "daily" / f"{name}.csv")
                if not daily.exists():
                    continue
                # Public aggregate ledgers are small; private ledgers are never read here.
                dates = pd.read_csv(daily, usecols=["date"])["date"]
                if dates.empty:
                    continue
                first = pd.read_csv(daily, nrows=1)
                capital = float(first.iloc[0].get("opening_nav", 1_000_000))
                relative = daily.relative_to(root).as_posix()
                checks = (_small_json(_inside(root, evidence / "checks.json"))
                          if name == decision.get("selected") else {})
                records.append(Experiment(
                    _identity(relative), relative, f"{version.upper()} · {name}", name,
                    version.upper(), "最终历史区间", _date(dates.iloc[0]), _date(dates.iloc[-1]),
                    "公开聚合快照", datetime.fromtimestamp(daily.stat().st_mtime, timezone.utc).isoformat(),
                    {"initial_cash": capital, "start": _date(dates.iloc[0]), "end": _date(dates.iloc[-1])},
                    _clean(row), _clean(checks)))
        except (ValueError, OSError, pd.errors.ParserError):
            continue
    return sorted(records, key=lambda x: (
        not (x.version == "V3" and x.strategy.endswith("rolling_lightgbm__managed")
             and x.stage == "最终历史区间"), -int(x.version[1]) if x.version[:2] in ("V2", "V3") else 0,
        x.stage != "交互重跑", x.relative_path))


def filter_experiments(records: list[Experiment], *, query: str = "", versions: list[str] | None = None,
                       stages: list[str] | None = None, start: str | None = None,
                       end: str | None = None) -> list[Experiment]:
    """Date filtering keeps experiments that overlap the requested interval."""
    lower, upper = _date(start), _date(end)
    if lower and upper and lower > upper:
        raise ValueError("开始日期不能晚于结束日期。")
    words = query.casefold().split()
    return [record for record in records
            if (not versions or record.version in versions)
            and (not stages or record.stage in stages)
            and all(word in f"{record.label} {record.strategy} {record.relative_path}".casefold() for word in words)
            and (not lower or not record.end or record.end >= lower)
            and (not upper or not record.start or record.start <= upper)]


def read_ledger(root: Path, record: Experiment, table: str, *, start: str | None = None,
                end: str | None = None, asset: str = "", side: str = "all", status: str = "all",
                max_rows: int = 200_000, chunk_size: int = 25_000) -> tuple[pd.DataFrame, int]:
    """Return filtered rows and total match count; bounded memory, explicit truncation.

    ``status='unfilled'`` includes partial fills as well as blocked orders.
    The UI exports all returned rows and clearly reports any configured limit.
    """
    if table not in TABLES:
        raise ValueError("未知账本类型。")
    if max_rows < 1 or chunk_size < 1:
        raise ValueError("读取行数必须为正数。")
    lower, upper = _date(start), _date(end)
    if lower and upper and lower > upper:
        raise ValueError("开始日期不能晚于结束日期。")
    if side not in ("all", "buy", "sell"):
        raise ValueError("未知交易方向。")
    if status not in ("all", "unfilled", "filled", "partial", "blocked", "liquidity_limited", "cash_scaled"):
        raise ValueError("未知订单状态。")
    path = _inside(Path(root), Path(root) / record.relative_path)
    if record.has_details:
        path = _inside(Path(root), path / f"{table}.csv")
    elif table != "daily":
        return pd.DataFrame(), 0
    if not path.exists():
        return pd.DataFrame(), 0
    parts, count, kept = [], 0, 0
    columns: list[str] = []
    try:
        for chunk in pd.read_csv(path, chunksize=chunk_size, dtype={"asset": str}):
            columns = list(chunk.columns)
            if "date" in chunk:
                chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
                if lower:
                    chunk = chunk[chunk.date >= pd.Timestamp(lower)]
                if upper:
                    chunk = chunk[chunk.date <= pd.Timestamp(upper)]
            if asset.strip() and "asset" in chunk:
                chunk = chunk[chunk.asset.str.contains(asset.strip(), case=False, regex=False, na=False)]
            if side != "all" and "side" in chunk:
                chunk = chunk[chunk.side == side]
            if status != "all" and "status" in chunk:
                if status == "unfilled":
                    chunk = chunk[order_fill_state(chunk) != "filled"]
                elif status in ("partial", "blocked", "filled"):
                    chunk = chunk[order_fill_state(chunk) == status]
                else:
                    chunk = chunk[chunk.status == status]
            count += len(chunk)
            if kept < max_rows and not chunk.empty:
                selected = chunk.iloc[:max_rows - kept]
                parts.append(selected)
                kept += len(selected)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), 0
    frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    return frame, count


def order_fill_state(orders: pd.DataFrame) -> pd.Series:
    """Classify execution quantities; a liquidity-limited order may fill zero."""
    if {"requested_units", "filled_units"} <= set(orders.columns):
        requested = pd.to_numeric(orders.requested_units, errors="coerce").abs()
        filled = pd.to_numeric(orders.filled_units, errors="coerce").abs()
        state = pd.Series("filled", index=orders.index)
        state.loc[filled + 1e-8 < requested] = "partial"
        state.loc[(filled <= 1e-8) & (requested > 1e-8)] = "blocked"
        state.loc[requested.isna() | filled.isna()] = "unknown"
        return state
    return orders.get("status", pd.Series("unknown", index=orders.index)).replace(
        {"liquidity_limited": "partial", "cash_scaled": "partial"})


def research_summary(record: Experiment, note: dict | None = None) -> dict:
    return _clean({"schema_version": 1, "exported_at_utc": datetime.now(timezone.utc).isoformat(),
                   "experiment_id": record.id, "experiment": record.label,
                   "strategy": record.strategy, "version": record.version, "stage": record.stage,
                   "period": {"start": record.start, "end": record.end}, "source": record.source,
                   "source_path": record.relative_path, "config": record.config,
                   "metrics": record.metrics, "accounting_checks": record.checks,
                   "note": note or {}, "interpretation": "历史回测记录；账本通过不代表策略收益达标或未来获利。"})


def _note_path(root: Path, experiment_id: str) -> Path:
    if not _ID.fullmatch(experiment_id):
        raise ValueError("实验 ID 无效，不能用作笔记文件名。")
    root = Path(root).resolve()
    folder = _inside(root, root / "runs" / "notes")
    return _inside(folder, folder / f"{experiment_id}.json")


def load_note(root: Path, experiment_id: str) -> dict:
    return _small_json(_note_path(root, experiment_id))


def save_note(root: Path, experiment_id: str, title: str, body: str, state: str = "待研究") -> dict:
    """Atomically replace one note; never mutate experiment results."""
    path = _note_path(root, experiment_id)
    if state not in NOTE_STATES:
        raise ValueError("未知笔记状态。")
    if len(title) > 160 or len(body) > 50_000:
        raise ValueError("标题最多 160 字，笔记最多 50,000 字。")
    path.parent.mkdir(parents=True, exist_ok=True)
    note = {"schema_version": 1, "experiment_id": experiment_id, "title": title.strip(),
            "body": body, "state": state, "updated_at_utc": datetime.now(timezone.utc).isoformat()}
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{experiment_id}.", suffix=".tmp", delete=False) as temp:
        json.dump(note, temp, ensure_ascii=False, indent=2, allow_nan=False)
        temp.flush()
        os.fsync(temp.fileno())
        temporary = temp.name
    os.replace(temporary, path)
    return note
