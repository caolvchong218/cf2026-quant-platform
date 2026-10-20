"""Local worker orchestration for dedicated THS simulation portfolios.

Strategy decisions, submitted intents, and provider observations are separate
artifacts. A worker never replays a historical backtest order as a current one.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime, time, timezone
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .simulation_execution import ExecutionConfig, SimulationExecutor, SubmitReceipt
from .simulation_signals import (
    EXECUTION_ASSET, STRATEGY_IDS, fetch_raw_quote, load_signal_bundle,
    plan_order, prepare_signal_bundle, refresh_market_data, save_signal_bundle,
)
from .ths_simulation import THSSimulationBroker, THSSimulationClient, ProviderError
from .equity_simulation import (
    CogSelection, EquityTargets, SCORE_COLUMNS, build_cogalpha_selection,
    build_v3_targets, fetch_equity_quotes, plan_cogalpha_orders, plan_v3_orders,
)

SHANGHAI = ZoneInfo("Asia/Shanghai")
DEFAULT_TOKEN_FILE = "D:/CodexData/private/qingxu/tushare_token.txt"
LABELS = {
    "v3_managed": "V3 · 风控前向适配",
    "cogalpha_rank": "CogAlpha · 白箱前向适配",
    "rsrs_standard_ma": "RSRS · 标准分 + 趋势",
    "rsrs_right_ma": "RSRS · 右偏修正 + 趋势",
    "etf_buy_hold": "沪深300 ETF 对照",
}


def directory(root: Path) -> Path:
    return Path(root) / "runs/simulation_auto"


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_bytes((json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())
    os.replace(temp, path)


def load_settings(root: Path) -> dict:
    path = directory(root) / "settings.json"
    default = {
        "enabled": False, "strategies": list(LABELS),
        "token_file": DEFAULT_TOKEN_FILE, "quote_source": "tencent_public",
        "max_order_notional": 200_000.0, "max_daily_notional": 400_000.0,
        "max_stock_order_notional": 20_000.0,
        "execution_window": "09:35–09:45 Asia/Shanghai",
    }
    if path.exists():
        supplied = json.loads(path.read_text(encoding="utf-8"))
        default.update(supplied)
    if type(default["enabled"]) is not bool:
        raise ValueError("自动执行开关格式无效。")
    if (not isinstance(default["strategies"], list)
            or any(item not in LABELS for item in default["strategies"])):
        raise ValueError("只有已完成前向适配的策略可以启用自动执行。")
    if default["quote_source"] not in {"tencent_public", "rt_etf_k"}:
        raise ValueError("未知报价来源。")
    return default


def set_enabled(root: Path, enabled: bool) -> None:
    if type(enabled) is not bool:
        raise ValueError("需要明确的开关状态。")
    settings = load_settings(root)
    settings["enabled"] = enabled
    settings["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(directory(root) / "settings.json", settings)


def latest_signals(root: Path) -> dict:
    found = {}
    for path in sorted((directory(root) / "signals").glob("*.json")):
        bundle = load_signal_bundle(path)
        old = found.get(bundle.strategy_id)
        if old is None or (bundle.execution_date, bundle.created_at) > (old.execution_date, old.created_at):
            found[bundle.strategy_id] = bundle
    return found


def market_calendar(root: Path) -> pd.DataFrame | None:
    """Read the most recent successfully validated market acquisition."""
    manifests = sorted((directory(root) / "market").glob("*/manifest.json"))
    if not manifests:
        return None
    manifest = json.loads(manifests[-1].read_text(encoding="utf-8"))
    path = manifests[-1].parent / "trade_cal.csv"
    if (manifest.get("ready") is not True
            or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"]["trade_cal.csv"]):
        raise ValueError("模拟交易日历身份校验失败。")
    calendar = pd.read_csv(path, dtype={"cal_date": str})
    calendar["cal_date"] = pd.to_datetime(calendar.cal_date, format="mixed")
    return calendar


def refresh_signals(root: Path, *, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    settings = load_settings(root)
    market = refresh_market_data(directory(root) / "market", token_file=settings["token_file"],
                                 now=now, preload_path=None)
    result = {}
    for name in STRATEGY_IDS:
        bundle = prepare_signal_bundle(name, market.index_prices, market.calendar,
                                       now=now, etf_daily=market.etf_daily)
        save_signal_bundle(bundle, directory(root) / "signals")
        result[name] = bundle
    return result


def _open_days(calendar) -> tuple[date, ...]:
    opens = pd.to_datetime(calendar.loc[calendar.is_open.eq(1), "cal_date"].astype(str)).dt.date
    return tuple(sorted(set(opens)))


def _equity_bundle(root: Path):
    receipts = sorted((directory(root) / "equities").glob("*/receipt.json"))
    if not receipts:
        raise ValueError("No prepared equity signals")
    path = receipts[-1]
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("scores_ready") is not True:
        raise ValueError("Latest equity preparation did not finish")
    for name in ("current_scores.parquet", "market_tail.parquet"):
        if hashlib.sha256((path.parent / name).read_bytes()).hexdigest() != receipt["files_sha256"][name]:
            raise ValueError("Equity signal identity changed")
    return path.parent, receipt


def latest_equity_status(root: Path) -> dict:
    """Small UI summary; score preparation is distinct from permission to trade."""
    folder, receipt = _equity_bundle(root)
    calendar = market_calendar(root)
    if calendar is None:
        raise ValueError("No verified market calendar")
    days = _open_days(calendar)
    signal = date.fromisoformat(receipt["signal_date"])
    if signal not in days or days.index(signal) + 1 >= len(days):
        raise ValueError("No verified next session")
    return {"signal_asof": str(signal), "execution_date": str(days[days.index(signal) + 1]),
            "reference_snapshot_date": receipt["reference_snapshot_date"],
            "score_coverage": receipt["score_coverage"], "variant_labels": receipt["variant_labels"]}


def refresh_equities(root: Path, *, now: datetime | None = None):
    from .equity_forward import prepare_equity_forward
    now = now or datetime.now(timezone.utc)
    settings = load_settings(root)
    # The original data, model inputs and factor registry stay under this root.
    return prepare_equity_forward(root, directory(root) / "equities",
                                  now.astimezone(SHANGHAI).date(),
                                  token_file=settings["token_file"], now=now)


class _QuotedSimulationBroker(THSSimulationBroker):
    """Last local gate immediately before the provider's order request."""

    def __init__(self, client, root):
        super().__init__(client)
        self.root, self.quotes = Path(root), {}

    def submit(self, intent):
        current = datetime.now(timezone.utc).astimezone(SHANGHAI)
        settings = load_settings(self.root)
        quote = self.quotes.get(intent.asset)
        valid = (
            settings["enabled"] and intent.strategy_id in settings["strategies"]
            and current.date() == intent.execution_date
            and time(9, 35) <= current.time().replace(tzinfo=None) < time(9, 45)
            and quote is not None and quote.price == intent.limit_price
            and quote.price_basis == "raw"
            and quote.asof.astimezone(SHANGHAI).date() == current.date()
            and 0 <= (current - quote.asof).total_seconds() <= 30
        )
        if valid and intent.strategy_id in SCORE_COLUMNS:
            valid = (quote.down_limit <= quote.price <= quote.up_limit
                     and (intent.side != "buy" or quote.price < quote.up_limit)
                     and (intent.side != "sell" or quote.price > quote.down_limit))
        if not valid:
            # Definitively not sent; distinguish this from an HTTP timeout,
            # whose remote outcome is unknown and must retain reservations.
            return SubmitReceipt(self.client.account_binding, None, "rejected")
        return super().submit(intent)


def _executor(root, client, calendar, settings, allowed_assets=None):
    opens = pd.to_datetime(calendar.loc[calendar.is_open.eq(1), "cal_date"].astype(str)).dt.date
    config = ExecutionConfig(
        client.account_binding, frozenset(allowed_assets or {EXECUTION_ASSET}), tuple(sorted(set(opens))),
        enabled=settings["enabled"],
        max_order_notional=(settings["max_stock_order_notional"] if client.strategy_id in SCORE_COLUMNS
                            else settings["max_order_notional"]),
        max_daily_notional=settings["max_daily_notional"],
    )
    return SimulationExecutor(directory(root) / "orders.sqlite", _QuotedSimulationBroker(client, root), config)


def _index_close(root: Path) -> pd.Series:
    manifests = sorted((directory(root) / "market").glob("*/manifest.json"))
    if not manifests:
        raise ValueError("Missing benchmark input")
    manifest = json.loads(manifests[-1].read_text(encoding="utf-8"))
    path = manifests[-1].parent / "index_daily.csv"
    if manifest.get("ready") is not True or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"]["index_daily.csv"]:
        raise ValueError("Benchmark identity changed")
    return pd.read_csv(path, parse_dates=["date"]).set_index("date").close.sort_index()


def _freeze_equity_decision(root, name, client, folder, receipt, scores, state, calendar, execution_day):
    """Persist the day's selection before any order, including across restarts."""
    path = directory(root) / "decisions" / name / f"{execution_day}.json"
    inputs_hash = receipt["files_sha256"]["current_scores.parquet"]
    signal_day = date.fromisoformat(receipt["signal_date"])
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if (saved["account_binding"] != client.account_binding or saved["scores_sha256"] != inputs_hash
                or saved["strategy_id"] != name):
            raise ValueError("Frozen daily decision differs from current inputs/account")
    else:
        if name == "v3_managed":
            previous = []
            for old in sorted(path.parent.glob("*.json")):
                if old.name >= path.name:
                    continue
                record = json.loads(old.read_text(encoding="utf-8"))
                if record["decision"]["rebalance_due"]:
                    previous = list(record["decision"]["weights"])
            target = build_v3_targets(
                scores, pd.read_parquet(folder / "market_tail.parquet"), _index_close(root),
                signal_asof=signal_day, execution_date=execution_day,
                trading_days=_open_days(calendar), previous_target_constituents=previous)
            decision = asdict(target)
            decision.update(signal_asof=str(signal_day), execution_date=str(execution_day))
        else:
            decision = build_cogalpha_selection(
                scores, signal_asof=signal_day, execution_date=execution_day,
                trading_days=_open_days(calendar), holdings=state["positions"]).to_dict()
        saved = {"strategy_id": name, "account_binding": client.account_binding,
                 "scores_sha256": inputs_hash, "decision": decision}
        write_json(path, saved)
    decision = saved["decision"]
    if name == "v3_managed":
        return EquityTargets(name, date.fromisoformat(decision["signal_asof"]),
                             date.fromisoformat(decision["execution_date"]), decision["weights"],
                             decision["rebalance_due"], decision["metadata"])
    return CogSelection.from_dict(decision)


def _equity_cycle(root, name, client, executor, calendar, settings, item, *,
                  now, quote_fetcher=fetch_equity_quotes):
    folder, receipt = _equity_bundle(root)
    signal_day = date.fromisoformat(receipt["signal_date"])
    execution_day = now.astimezone(SHANGHAI).date()
    days = _open_days(calendar)
    if execution_day not in days or days.index(execution_day) == 0 or days[days.index(execution_day)-1] != signal_day:
        item.update(status="stale_signal", detail="股票信号不是前一官方交易日，已拦截。")
        return
    scores = pd.read_parquet(folder / "current_scores.parquet")
    state = client.inspect()
    frozen = _freeze_equity_decision(root, name, client, folder, receipt, scores, state, calendar, execution_day)
    if name == "v3_managed":
        assets = set(frozen.weights).union(state["positions"])
        if not frozen.rebalance_due:
            item.update(status="hold", detail="尚未到 20 个交易日调仓日，保持实际持仓。")
            return
    else:
        assets = set(frozen.buy_candidates).union(state["positions"])
    if not assets:
        item.update(status="hold", detail="当日目标为现金。")
        return
    # Batch quote endpoint caps at 100; a 50/5 dropout union can contain 105.
    quotes = {}
    ordered = sorted(assets)
    for offset in range(0, len(ordered), 100):
        quotes.update(quote_fetcher(ordered[offset:offset + 100]))
    kwargs = dict(account_binding=client.account_binding, account_nav=state["nav"],
                  available_cash=state["available_cash"], holdings=state["positions"],
                  available_shares=state["available_shares"], quotes=quotes,
                  max_order_notional=settings["max_stock_order_notional"])
    if name == "v3_managed":
        plan = plan_v3_orders(frozen, **kwargs)
    else:
        plan = plan_cogalpha_orders(scores, signal_asof=signal_day, execution_date=execution_day,
                                   trading_days=days, selection=frozen, **kwargs)
    already = {row.intent_id for row in executor.journal()}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    # Persist reviewable orders before sending, but never replay this file later.
    artifact = asdict(plan)
    for intent in artifact["intents"]:
        intent["signal_asof"], intent["execution_date"] = str(intent["signal_asof"]), str(intent["execution_date"])
    if artifact.get("selection"):
        artifact["selection"] = plan.selection.to_dict()
    write_json(directory(root) / "plans" / name / f"{stamp}.json", artifact)
    outcomes = []
    for intent in plan.intents:
        if intent.intent_id in already:
            continue
        current = datetime.now(timezone.utc).astimezone(SHANGHAI)
        latest = load_settings(root)
        if (not latest["enabled"] or name not in latest["strategies"]
                or not time(9, 35) <= current.time().replace(tzinfo=None) < time(9, 45)):
            break
        # A batch can take longer than quote freshness. New quotes may justify
        # a new plan next cycle; never silently alter an already frozen order.
        if (current - quotes[intent.asset].asof).total_seconds() > 20:
            fresh = quote_fetcher([intent.asset])[intent.asset]
            if fresh.price != intent.limit_price:
                break
            quotes[intent.asset] = fresh
        executor.broker.quotes[intent.asset] = quotes[intent.asset]
        outcome = executor.execute(intent)  # live clock advances after broker snapshot
        outcomes.append(outcome)
        if outcome.status in {"unknown", "submitting", "blocked", "rejected"}:
            break
    item.update(status=outcomes[-1].status if outcomes else "hold",
                detail="当日计划已检查；实际成交以服务回执为准。",
                sent_this_cycle=len(outcomes), blocked_candidates=len(plan.blocked))


def run_cycle(root: Path, *, now: datetime | None = None, refresh=False,
              client_factory=THSSimulationClient, quote_fetcher=fetch_raw_quote,
              equity_quote_fetcher=fetch_equity_quotes) -> dict:
    """Run bounded work once. No sleep, hidden account creation, or retries."""
    now = now or datetime.now(timezone.utc)
    local = now.astimezone(SHANGHAI)
    settings = load_settings(root)
    active_window = time(9, 35) <= local.time().replace(tzinfo=None) < time(9, 45)
    snapshots_due = (time(9, 30) <= local.time().replace(tzinfo=None) <= time(15, 35))
    status = {"asof": now.isoformat(), "pid": os.getpid(), "enabled": settings["enabled"],
              "mode": "dedicated_simulation", "mobile_account_linked": False, "strategies": []}
    try:
        signals = refresh_signals(root, now=now) if refresh else latest_signals(root)
        calendar = market_calendar(root)
    except (ValueError, RuntimeError, OSError, KeyError):
        status["data_error"] = "数据或信号未就绪；请更新行情并检查本地凭据。"
        signals, calendar = {}, None
    if refresh:
        try:
            status["phase"] = "refreshing_equities"
            write_json(directory(root) / "service_status.json", status)
            refresh_equities(root, now=now)
        except (ValueError, RuntimeError, OSError, KeyError):
            status["equity_data_error"] = "股票数据刷新未完成，过期分数不会用于下单。"
        status.pop("phase", None)
    try:
        equity_summary = latest_equity_status(root)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError):
        equity_summary = None
    is_session = (calendar is not None and
                  pd.to_datetime(calendar.loc[calendar.is_open.eq(1), "cal_date"].astype(str)).dt.date.eq(local.date()).any())
    quote = None
    for name in LABELS:
        item = {"strategy_id": name, "label": LABELS[name], "status": "waiting", "detail": ""}
        status["strategies"].append(item)
        try:
            client = client_factory(Path(root), name)
            if calendar is not None:
                allowed = None
                if name in SCORE_COLUMNS and equity_summary:
                    folder, _ = _equity_bundle(root)
                    allowed = set(pd.read_parquet(folder / "current_scores.parquet", columns=["asset"]).asset)
                executor = _executor(root, client, calendar, settings, allowed)
                journal = executor.journal()
            else:
                executor, journal = None, []
            if (snapshots_due and is_session) or refresh:
                client.save_observation()
                if executor is not None:
                    reconciliation = executor.reconcile()
                    if reconciliation.issues:
                        item["reconciliation_issues"] = list(reconciliation.issues)
                    journal = executor.journal()
            bundle = signals.get(name)
            if (name in STRATEGY_IDS and bundle is None) or (name in SCORE_COLUMNS and equity_summary is None):
                item.update(status="data_pending", detail="缺少经过校验的最新信号。")
                continue
            if name in STRATEGY_IDS:
                item.update(signal_asof=str(bundle.signal_asof), execution_date=str(bundle.execution_date),
                            target_weight=bundle.target_weight)
            else:
                item.update(**equity_summary, variant="frozen_reference_forward")
            if not settings["enabled"] or name not in settings["strategies"]:
                item.update(status="paused", detail="自动执行已暂停。")
                continue
            if not active_window or not is_session:
                item.update(status="waiting_session", detail="等待官方交易日 09:35–09:45 执行窗口。")
                continue
            if name in STRATEGY_IDS and bundle.execution_date != local.date():
                item.update(status="stale_signal", detail="信号不属于今天，拒绝旧信号下单。")
                continue
            if any(row.status in {"submitting", "unknown", "accepted", "open", "partial"} for row in journal):
                item.update(status="reconciliation_pending", detail="已有委托待确认或待成交，不再发送新订单。")
                continue
            if item.get("reconciliation_issues"):
                item.update(status="reconciliation_pending", detail="成交核验未通过，暂停新增委托。")
                continue
            if name in SCORE_COLUMNS:
                _equity_cycle(root, name, client, executor, calendar, settings, item,
                              now=now, quote_fetcher=equity_quote_fetcher)
                continue
            state = client.inspect()
            if set(state["positions"]) - {EXECUTION_ASSET}:
                item.update(status="unexpected_position", detail="专用 ETF 账户出现其他持仓，自动执行暂停。")
                continue
            held = state["positions"].get(EXECUTION_ASSET, 0)
            # These are binary entry/exit strategies, not daily 95% rebalancers.
            if (bundle.target_weight > 0 and held > 0) or (bundle.target_weight == 0 and held == 0):
                item.update(status="hold", detail="当前持仓符合信号，保持现有股数。")
                continue
            if quote is None:
                quote = quote_fetcher(token_file=settings["token_file"],
                                      quote_source=settings["quote_source"])
            intent = plan_order(
                bundle, account_binding=client.account_binding, total_shares=held,
                available_shares=state["available_shares"].get(EXECUTION_ASSET, 0),
                available_cash=state["available_cash"], quote=quote,
                max_order_notional=settings["max_order_notional"])
            if intent is None:
                item.update(status="hold", detail="不足一手或没有可执行的持仓差额。")
                continue
            latest = load_settings(root)
            current = datetime.now(timezone.utc).astimezone(SHANGHAI)
            if (not latest["enabled"] or name not in latest["strategies"]
                    or not time(9, 35) <= current.time().replace(tzinfo=None) < time(9, 45)):
                item.update(status="paused", detail="自动执行已暂停。")
                continue
            executor.broker.quotes[intent.asset] = quote
            outcome = executor.execute(intent)
            item.update(status=outcome.status, detail=outcome.reason_code or "模拟委托已记录。",
                        intent_id=outcome.intent_id, filled_quantity=outcome.filled_quantity)
        except (ValueError, RuntimeError, OSError, KeyError):
            # Errors may contain account IDs or vendor responses. Detailed
            # account values stay in private observations, never public logs.
            item.update(status="blocked", detail="账户、报价或接口核验未通过；未自动重试下单。")
    write_json(directory(root) / "service_status.json", status)
    return status
