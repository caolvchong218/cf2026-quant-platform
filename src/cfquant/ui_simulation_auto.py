"""Read-only monitoring and an explicit switch for isolated simulated accounts.

Rendering never provisions an account or submits an order.  Refresh buttons only
query account state.  All charts derive from saved observations; fixture data is
labelled and never silently presented as a connected live account.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, time, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from .ths_simulation import PROVIDER, THSSimulationClient, load_account


SHANGHAI = ZoneInfo("Asia/Shanghai")
STRATEGY_LABELS = {
    "v3_managed": "V3 · 风控前向适配",
    "cogalpha_rank": "CogAlpha · 白箱前向适配",
    "rsrs_standard_ma": "RSRS · 标准分 + 趋势",
    "rsrs_right_ma": "RSRS · 右偏修正 + 趋势",
    "etf_buy_hold": "沪深 300 ETF 对照",
}
STATUS_LABELS = {
    "submitting": "提交待核对", "unknown": "结果未知 · 禁止重试",
    "accepted": "已受理 · 未证实成交", "open": "委托未完成", "partial": "部分成交",
    "filled": "已核实全部成交", "cancelled": "已撤销", "rejected": "已拒绝", "blocked": "已拦截",
}
SERVICE_LABELS = {
    **STATUS_LABELS, "waiting": "等待检查", "adapter_pending": "策略执行适配待完成",
    "data_pending": "最新数据或信号待准备", "paused": "已暂停",
    "waiting_session": "等待交易时段", "stale_signal": "信号过期 · 已拦截",
    "reconciliation_pending": "已有委托待核对", "unexpected_position": "持仓异常 · 已暂停",
    "hold": "持仓符合信号 · 保持",
}


def _json(path: Path) -> dict:
    try:
        if path.stat().st_size > 2_000_000:
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _timestamp(value: object) -> datetime | None:
    try:
        instant = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if instant.tzinfo is None or instant.utcoffset() is None:
            return None
        return instant.astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError):
        return None


def _number(value: object) -> float | None:
    try:
        if isinstance(value, bool):
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _mask(value: object) -> str:
    value = str(value or "")
    return "••••" + value[-4:] if len(value) > 4 else "••••"


def _account_binding(account: dict) -> str:
    return hashlib.sha256((PROVIDER + ":" + str(account["account_id"])).encode()).hexdigest()[:24]


def read_accounts(root: Path) -> tuple[dict[str, dict], int]:
    """Only load the five supported, validated dedicated identities."""
    accounts, invalid = {}, 0
    for strategy_id in STRATEGY_LABELS:
        path = root / "runs/simulation_auto/accounts" / f"{strategy_id}.json"
        if not path.exists():
            continue
        try:
            accounts[strategy_id] = load_account(root, strategy_id)
        except (OSError, ValueError, TypeError, KeyError):
            invalid += 1
    return accounts, invalid


def read_observations(root: Path, strategy_id: str, binding: str,
                      *, now: datetime | None = None) -> tuple[list[dict], int]:
    now = now or datetime.now(timezone.utc)
    observations, invalid = [], 0
    for path in (root / "runs/simulation_auto/observations" / strategy_id).glob("*.json"):
        value = _json(path)
        instant, nav = _timestamp(value.get("asof")), _number(value.get("nav"))
        if (value.get("provider") != PROVIDER or value.get("simulation_only") is not True
                or value.get("strategy_id") != strategy_id or value.get("account_binding") != binding
                or instant is None or instant > now or nav is None or nav < 0):
            invalid += 1
            continue
        observations.append({**value, "_instant": instant, "nav": nav})
    observations.sort(key=lambda value: value["_instant"])
    return observations, invalid


def daily_observations(observations: list[dict], trading_days: set[date] | None = None) -> pd.DataFrame:
    """One last post-close observation per day, without filling missing dates.

    The earliest observation provides a baseline only when that day has no
    post-close observation.  A verified calendar, when supplied, excludes
    holidays; otherwise only weekdays are eligible and Sharpe remains withheld.
    """
    columns = ["date", "asof", "nav"]
    if not observations:
        return pd.DataFrame(columns=columns)
    ordered = sorted(observations, key=lambda value: value["_instant"])
    chosen: dict[date, dict] = {}
    first = ordered[0]
    for value in ordered:
        local = value["_instant"].astimezone(SHANGHAI)
        eligible = local.date() in trading_days if trading_days is not None else local.weekday() < 5
        if eligible and local.time().replace(tzinfo=None) >= time(15, 15):
            chosen[local.date()] = value
    first_day = first["_instant"].astimezone(SHANGHAI).date()
    baseline_eligible = first_day in trading_days if trading_days is not None else first_day.weekday() < 5
    if baseline_eligible:
        chosen.setdefault(first_day, first)
    return pd.DataFrame([
        {"date": day.isoformat(), "asof": value["_instant"], "nav": value["nav"]}
        for day, value in sorted(chosen.items())
    ], columns=columns)


def performance_summary(observations: list[dict], trading_days: set[date] | None = None) -> dict:
    frame = daily_observations(observations, trading_days)
    result = {
        "observations": len(observations), "daily_points": len(frame),
        "cumulative_change": None, "sharpe": None, "sharpe_reason": "观察期不足",
        "max_drawdown": None,
    }
    if not observations:
        return result
    ordered = sorted(observations, key=lambda value: value["_instant"])
    if ordered[0]["nav"] <= 0:
        result["sharpe_reason"] = "起始资产无法归一"
        return result
    result["cumulative_change"] = ordered[-1]["nav"] / ordered[0]["nav"] - 1
    if len(frame) >= 2:
        values = frame.nav.astype(float)
        result["max_drawdown"] = float((1 - values / values.cummax().clip(lower=ordered[0]["nav"])).max())
    if len(frame) < 21:
        return result
    if frame.nav.le(0).any():
        result["sharpe_reason"] = "资产触零，无法计算"
        return result
    if trading_days is None:
        result["sharpe_reason"] = "交易日覆盖未核验"
        return result
    recorded = {date.fromisoformat(value) for value in frame.date}
    expected = {day for day in trading_days if min(recorded) <= day <= max(recorded)}
    if recorded != expected:
        result["sharpe_reason"] = "缺少交易日收盘观察"
        return result
    returns = frame.nav.pct_change().dropna()
    std = float(returns.std(ddof=1))
    if std <= 0 or not math.isfinite(std):
        result["sharpe_reason"] = "波动不足，无法计算"
        return result
    result["sharpe"] = float(np.sqrt(252) * returns.mean() / std)
    result["sharpe_reason"] = ""
    return result


def common_period_comparison(series: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Inner join on exact observed dates and normalize from the common start."""
    eligible = {key: value for key, value in series.items() if not value.empty}
    if len(eligible) < 2:
        return pd.DataFrame(columns=["日期", "策略", "累计资产变化"])
    common = set.intersection(*(set(frame.date) for frame in eligible.values()))
    if len(common) < 2:
        return pd.DataFrame(columns=["日期", "策略", "累计资产变化"])
    days = sorted(common)
    records = []
    for strategy_id, frame in eligible.items():
        nav = frame.set_index("date").loc[days, "nav"].astype(float)
        if nav.iloc[0] <= 0:
            return pd.DataFrame(columns=["日期", "策略", "累计资产变化"])
        for day, change in (nav / nav.iloc[0] - 1).items():
            records.append({"日期": day, "策略": STRATEGY_LABELS.get(strategy_id, strategy_id),
                            "累计资产变化": float(change)})
    return pd.DataFrame(records)


def read_journal(root: Path, binding: str) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    path = root / "runs/simulation_auto/orders.sqlite"
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame(), False
    try:
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=2)) as conn:
            intents = pd.read_sql_query(
                """SELECT intent_id, strategy_id, asset, side, quantity, limit_price,
                   status, order_id, filled_quantity, reason_code, updated_at
                   FROM intents WHERE account=? ORDER BY updated_at DESC LIMIT 200""", conn, params=(binding,))
            fills = pd.read_sql_query(
                """SELECT fill_id, order_id, quantity, price, timestamp, fee
                   FROM fills WHERE account=? ORDER BY timestamp DESC LIMIT 200""", conn, params=(binding,))
        return intents, fills, False
    except (OSError, sqlite3.Error, pd.errors.DatabaseError):
        return pd.DataFrame(), pd.DataFrame(), True


def _settings(root: Path) -> dict:
    return _json(root / "runs/simulation_auto/settings.json")


def _calendar(root: Path) -> set[date] | None:
    # The service verifies the market snapshot manifest before returning data.
    try:
        from .simulation_service import market_calendar
        frame = market_calendar(root)
    except (ImportError, OSError, ValueError, TypeError, KeyError):
        return None
    if frame is None or not isinstance(frame, pd.DataFrame) or not {"cal_date", "is_open"} <= set(frame):
        return None
    try:
        values = frame.loc[pd.to_numeric(frame.is_open, errors="raise").eq(1), "cal_date"].astype(str)
        dates = pd.to_datetime(values, format="mixed", errors="raise")
        return set(dates.dt.date)
    except (ValueError, TypeError, AttributeError):
        return None


def _set_enabled(root: Path, enabled: bool) -> bool:
    try:
        from .simulation_service import set_enabled
        set_enabled(root, enabled)
        return True
    except (ImportError, OSError, ValueError, TypeError):
        return False


def _service_panel(root: Path, accounts: dict[str, dict]) -> None:
    settings = _settings(root)
    heartbeat = _json(root / "runs/simulation_auto/service_status.json")
    stamp = next((_timestamp(heartbeat.get(key)) for key in ("asof", "updated_at", "timestamp", "time")
                  if _timestamp(heartbeat.get(key)) is not None), None)
    age = (datetime.now(timezone.utc) - stamp).total_seconds() if stamp else None
    fresh = age is not None and 0 <= age < 180
    a, b, c = st.columns(3)
    a.metric("自动委托开关", "允许" if settings.get("enabled") is True else "暂停")
    b.metric("服务心跳", "在线" if fresh else "未检测到近期心跳")
    c.metric("专用模拟账户", f"{len(accounts)} / {len(STRATEGY_LABELS)}")
    if stamp:
        st.caption("最近心跳：" + stamp.astimezone(SHANGHAI).strftime("%Y-%m-%d %H:%M:%S"))
    if settings.get("enabled") is True and not fresh:
        st.warning("开关已允许，但没有近期服务心跳；不能据此判断自动下单正在运行。")
    if heartbeat.get("data_error"):
        st.warning("服务尚未取得通过核验的最新行情或信号，自动委托保持拦截。")
    refresh = heartbeat.get("data_refresh") or _json(root / "runs/simulation_auto/refresh_status.json")
    if isinstance(refresh, dict) and refresh:
        decision = refresh.get("decision", {})
        result = refresh.get("last_result", {})
        target = refresh.get("target_date") or decision.get("target_date") or "待核对"
        if refresh.get("running"):
            job = "官方交易日历" if refresh.get("job") == "calendar" else "行情、财务、行业与策略分数"
            st.info(f"后台正在更新{job} · 目标日期 {target}。账户查询和暂停开关仍可使用。")
        elif result.get("success") is False:
            if result.get("progress"):
                st.info("本批补数已保存，仍有历史数据或参考资料待补；服务会从检查点继续。")
            else:
                st.warning("最近一次数据更新未完成，服务会按重试间隔补跑；旧信号不能用于今天下单。")
        else:
            st.caption(f"自动更新目标：{target} · 交易日 17:05 后补齐，启动或恢复联网后也会检查漏更。")
        retry = _timestamp(decision.get("next_attempt_at"))
        if retry:
            st.caption("下次更新检查点：" + retry.astimezone(SHANGHAI).strftime("%m-%d %H:%M:%S"))
        with st.expander("持续运行与补数状态"):
            st.write("行情按交易日增量保存，财报按披露记录触发并定期回扫，行业数据定期刷新。"
                     "V3 在新执行年度使用新的滚动三年模型；当年模型缓存复用。"
                     "补跑只准备最新可执行信号，不补发错过交易日的订单。")
            reference = result.get("references", {})
            if isinstance(reference, dict) and reference:
                pending = reference.get("backlog_tasks", reference.get("pending_tasks"))
                if pending is not None:
                    st.caption(f"财务 / 行业待完成任务：{pending}")
            missing = result.get("remaining_sessions")
            if missing is not None:
                st.caption(f"行情待补交易日：{missing}")
            st.caption("需要电脑开机、登录 Windows 并保持联网。休眠或关机时不执行，恢复后按官方日历补数据。")
    a, b = st.columns(2)
    if a.button("允许后续自动模拟委托", key="auto_sim_enable", disabled=not accounts):
        if _set_enabled(root, True):
            st.success("已允许后续自动模拟委托；执行情况以服务心跳和委托回执为准。")
            st.rerun()
        else:
            st.error("未能保存开关，当前设置保持不变。")
    if b.button("暂停后续自动委托", key="auto_sim_pause", disabled=not accounts):
        if _set_enabled(root, False):
            st.success("已暂停后续自动委托。已经发出的委托仍需核对，不会自动撤单。")
            st.rerun()
        else:
            st.error("未能保存开关，当前设置保持不变。")
    st.caption("每条策略使用独立专用模拟账户。暂停阻止后续委托，不等于撤销已经发出的委托。")
    statuses = heartbeat.get("strategies")
    if isinstance(statuses, list):
        rows = [{"策略": STRATEGY_LABELS[item["strategy_id"]],
                 "当前状态": SERVICE_LABELS.get(item.get("status"), "待核对")}
                for item in statuses if isinstance(item, dict) and item.get("strategy_id") in STRATEGY_LABELS]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _signal_panel(root: Path, strategy_id: str) -> None:
    equity_columns = {
        "v3_managed": "v3_lightgbm_frozen_reference",
        "cogalpha_rank": "cogalpha_731_rank_frozen_reference",
    }
    if strategy_id in equity_columns:
        try:
            from .simulation_service import latest_equity_status
            status = latest_equity_status(root)
        except (ImportError, OSError, ValueError, KeyError, TypeError):
            status = None
        if not isinstance(status, dict):
            st.info("股票前向分数尚未通过准备核验；历史回测持仓不会自动当作今天的委托。")
            return
        signal = str(status.get("signal_asof", "待核对"))[:32]
        execution = str(status.get("execution_date", "待核对"))[:32]
        reference = str(status.get("reference_snapshot_date", "待核对"))[:32]
        coverage = status.get("score_coverage", {})
        count = _number(coverage.get(equity_columns[strategy_id])) if isinstance(coverage, dict) else None
        st.caption(f"股票分数截面：{signal} · 计划执行日：{execution}")
        st.caption("有效分数覆盖：" + (f"{int(count)} 只股票" if count is not None and count >= 0 else "待核对"))
        if status.get("continuous"):
            model_year = status.get("model_year", "待核对")
            st.caption(f"V3 执行年度模型：{model_year} · 财务 / 行业版本按 {reference} 的可用信息选取。")
            st.info("财务公告、行业变更和首次观测时间共同约束数据使用；已冻结的历史特征不会被后来修订覆盖。"
                    "策略公式仍按各自版本保存，前向收益从实际模拟成交之后开始积累。")
        else:
            st.warning(f"旧版前向适配：财务与行业参考冻结截至 {reference}。"
                       "这不是原历史策略在所有数据口径上的等价延伸。"
                       "持续更新版本尚未完成准备，分数准备完成不等于委托或收益已经验证。")
        return
    try:
        from .simulation_service import latest_signals
        bundle = latest_signals(root).get(strategy_id)
    except (ImportError, OSError, ValueError, KeyError, TypeError):
        bundle = None
    if is_dataclass(bundle) and not isinstance(bundle, type):
        bundle = asdict(bundle)
    if not isinstance(bundle, dict):
        st.info("尚无可执行的当前信号；历史回测持仓不会自动当作今天的委托。")
        return
    signal_day = bundle.get("signal_asof", bundle.get("asof", "待核对"))
    execution_day = bundle.get("execution_date", "待核对")
    st.caption(f"最近信号截面：{str(signal_day)[:32]} · 计划执行日：{str(execution_day)[:32]}")
    weight = _number(bundle.get("target_weight"))
    if weight is not None and 0 <= weight <= 1:
        st.caption(f"执行标的：{str(bundle.get('execution_asset', '待核对'))[:24]} · 目标仓位：{weight:.0%}。"
                   "目标仓位仍需通过实时价格、可用资金、手数与时段检查。")


def _positions(latest: dict) -> None:
    holdings, available = latest.get("positions", {}), latest.get("available_shares", {})
    if not isinstance(holdings, dict) or not holdings:
        st.info("最新查询没有持仓；现金仓位也是有效的策略状态。")
        return
    frame = pd.DataFrame([{"证券": asset, "总持仓（股）": quantity,
                           "当前可卖（股）": available.get(asset, "待核对")}
                          for asset, quantity in holdings.items()])
    st.dataframe(frame, hide_index=True, width="stretch")
    st.caption("可卖数量来自模拟服务，可能低于总持仓；自动执行按可卖数量检查 T+1。")


def _trades(latest: dict, fills: pd.DataFrame) -> None:
    rows = latest.get("trades_today", [])
    if isinstance(rows, list) and rows:
        displayed = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            # Never display a raw provider row: it can contain account IDs.
            displayed.append({
                "证券": str(row.get("zqdm", row.get("stock_code", ""))),
                "方向": str(row.get("mmlb", row.get("direction", ""))),
                "数量（股）": row.get("cjsl", row.get("volume", "")),
                "成交价格": row.get("cjg", row.get("cjj", row.get("price", ""))),
                "成交时间": str(row.get("cjsj", row.get("business_time", ""))),
                "委托尾号": _mask(row.get("entrust_no", row.get("wth"))),
                "成交尾号": _mask(row.get("business_no", row.get("cjbh"))),
            })
        st.dataframe(pd.DataFrame(displayed), hide_index=True, width="stretch")
        st.caption("以上为查询时的当日成交记录。缺少完整编号或日期的成交不会自动冲销未知委托。")
    else:
        st.info("最新查询尚无当日成交。已受理委托不等于已经成交。")
    if not fills.empty:
        display = fills.copy()
        display["fill_id"] = display.fill_id.map(_mask)
        display["order_id"] = display.order_id.map(_mask)
        st.write("已核验并留存的成交")
        st.dataframe(display.rename(columns={
            "fill_id": "成交尾号", "order_id": "委托尾号", "quantity": "成交数量",
            "price": "成交价格", "timestamp": "成交时间", "fee": "已知费用",
        }), hide_index=True, width="stretch")


def _journal(intents: pd.DataFrame, problem: bool) -> None:
    if problem:
        st.warning("本地执行账本暂时不可读，未据此判定委托不存在。")
    elif intents.empty:
        st.info("还没有自动委托记录。平台不会填入示例成交或回测利润。")
    else:
        frame = intents.drop(columns=["strategy_id"]).copy()
        frame["status"] = frame.status.map(lambda value: STATUS_LABELS.get(value, "待核对"))
        frame["order_id"] = frame.order_id.map(_mask)
        st.dataframe(frame.rename(columns={
            "intent_id": "信号决策编号", "asset": "证券", "side": "方向", "quantity": "委托数量",
            "limit_price": "限价", "status": "状态", "order_id": "委托尾号",
            "filled_quantity": "已核验成交数量", "reason_code": "状态原因", "updated_at": "更新时间",
        }), hide_index=True, width="stretch")
        if intents.status.isin(["submitting", "unknown"]).any():
            st.warning("存在结果未知的委托：资金或持仓仍被预约，平台禁止自动重发，须取得可核验回执。")


def _comparison(all_observations: dict[str, list[dict]], selected: str,
                calendar: set[date] | None) -> None:
    current = all_observations[selected]
    summary = performance_summary(current, calendar)
    a, b, c, d = st.columns(4)
    change = summary["cumulative_change"]
    a.metric("观察期资产变化", "暂无数据" if change is None else f"{change:.2%}")
    b.metric("年化 Sharpe", f"{summary['sharpe']:.3f}" if summary["sharpe"] is not None
             else summary["sharpe_reason"])
    c.metric("可用日度观察", str(summary["daily_points"]))
    drawdown = summary["max_drawdown"]
    d.metric("日度样本最大回撤", "观察期不足" if drawdown is None else f"{drawdown:.2%}")
    st.caption("资产变化从首次查询的总资产开始计算，未调整额外出入金。Sharpe 使用日收益、252 个交易日年化、"
               "零无风险利率；至少需要 20 个完整日度区间，且必须核验交易日覆盖。")
    series = {key: daily_observations(values, calendar) for key, values in all_observations.items()}
    comparison = common_period_comparison(series)
    if comparison.empty:
        st.info("收益对比需要至少两个账户、两个共同日度观察日期，以及可归一的正值起点；"
                "不会用回测曲线代替，也不会补齐缺失日期。")
    else:
        figure = px.line(comparison, x="日期", y="累计资产变化", color="策略",
                         color_discrete_sequence=["#168b80", "#375f9e", "#c48d2a", "#8c6bb1", "#617384"])
        figure.update_yaxes(tickformat=".1%")
        figure.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=15, b=10))
        st.plotly_chart(figure, width="stretch", key="auto_sim_common_period")
        st.caption(f"共同区间：{comparison['日期'].min()} 至 {comparison['日期'].max()}。"
                   "所有曲线从共同首日重新归一，只比较共同观察日期。")
    with st.expander("查看日度观察口径"):
        st.write("每个上海日期只保留 15:15 后最后一次总资产查询；首次观察可作为起点。"
                 "未收集的日期保持缺失。未知交易日历时不显示 Sharpe。")
        st.dataframe(series[selected], hide_index=True, width="stretch")


def render(root: Path) -> None:
    root = Path(root)
    st.subheader("自动模拟交易 · 前瞻观察")
    st.caption("真实模拟服务回执与本地持久账本。收益从接入后的观察累积，不能提前证明策略有效。")
    accounts, invalid_accounts = read_accounts(root)
    _service_panel(root, accounts)
    st.info("手机账户互通未验证。本页连接的是同花顺官方 SkillHub 专用模拟服务，"
            "不表示已连接你手机 App 里原有的模拟账户。")
    st.link_button("查看同花顺官方模拟炒股入口", "https://moni.10jqka.com.cn/index.shtml")
    if invalid_accounts:
        st.warning(f"{invalid_accounts} 份本地账户记录尚未完成验证，未作为已连接账户使用。")
    if not accounts:
        st.info("专用模拟账户未连接。先完成专用账户准备、最新数据和信号核验，再开启自动执行；本页不会自动开户或下单。")
        return
    selected = st.selectbox("观察策略", list(accounts), format_func=STRATEGY_LABELS.get,
                            key="auto_sim_strategy")
    st.caption("独立模拟账户尾号：" + _mask(accounts[selected]["account_id"]))
    a, b = st.columns(2)
    refresh_selected = a.button("刷新当前账户", key="auto_sim_refresh_selected")
    refresh_all = b.button("刷新全部账户", key="auto_sim_refresh_all")
    if refresh_selected or refresh_all:
        targets = list(accounts) if refresh_all else [selected]
        completed = 0
        with st.spinner("正在查询模拟账户；本操作不发送委托……"):
            for strategy_id in targets:
                try:
                    THSSimulationClient(root, strategy_id).save_observation()
                    completed += 1
                except (OSError, ValueError, RuntimeError, TypeError, KeyError):
                    pass  # Raw exceptions may contain provider/account details.
        if completed == len(targets):
            st.success(f"已刷新 {completed} 个模拟账户。")
        else:
            st.warning(f"已刷新 {completed}/{len(targets)} 个账户；未成功的查询保留原观察，不表示账户为空。")
    all_observations, invalid_observations = {}, 0
    for strategy_id, account in accounts.items():
        values, bad = read_observations(root, strategy_id, _account_binding(account))
        all_observations[strategy_id] = values
        invalid_observations += bad
    if invalid_observations:
        st.warning(f"{invalid_observations} 条观察记录未通过日期、账户或资产数值核验，未纳入展示和统计。")
    observations = all_observations[selected]
    if any(value.get("fixture_source") for values in all_observations.values() for value in values):
        st.warning("合成测试数据：仅用于界面与计算校验，不是实际账户收益。")
    _signal_panel(root, selected)
    if observations:
        latest = observations[-1]
        age = datetime.now(timezone.utc) - latest["_instant"]
        st.caption("最新账户查询：" + latest["_instant"].astimezone(SHANGHAI).strftime("%Y-%m-%d %H:%M:%S"))
        if age.total_seconds() > 300:
            st.warning("当前显示的是历史查询快照；请刷新或检查服务，不能据此确认当前账户状态。")
        a, b, c, d = st.columns(4)
        for cell, label, field in ((a, "账户总资产（元）", "nav"), (b, "可用资金（元）", "available_cash"),
                                   (c, "冻结资金（元）", "frozen_cash"), (d, "持仓市值（元）", "market_value")):
            value = _number(latest.get(field))
            cell.metric(label, "待核对" if value is None else f"{value:,.2f}")
    else:
        latest = {}
        st.warning("账户已配置，但尚无通过核验的账户查询；不能判定已经成功连接或产生收益。")
    intents, fills, problem = read_journal(root, _account_binding(accounts[selected]))
    positions_tab, trades_tab, journal_tab, comparison_tab = st.tabs(["持仓", "成交", "执行日志", "收益对比"])
    with positions_tab:
        if observations:
            _positions(latest)
        else:
            st.info("等待首次有效账户查询。")
    with trades_tab:
        _trades(latest, fills)
    with journal_tab:
        _journal(intents, problem)
    with comparison_tab:
        _comparison(all_observations, selected, _calendar(root))
