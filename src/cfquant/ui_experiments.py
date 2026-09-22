"""Interactive experiment archive and auditable trade/holding drill-down."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from .experiment_catalog import (NOTE_STATES, Experiment, discover_experiments,
                                filter_experiments, load_note, order_fill_state, read_ledger,
                                research_summary, save_note)


@st.cache_data(ttl=30, show_spinner=False)
def _catalogue(root: str, archives: bool) -> list[Experiment]:
    return discover_experiments(Path(root), archives)


@st.cache_data(ttl=60, show_spinner=False, max_entries=40)
def _ledger(root: str, record: Experiment, table: str, **filters):
    return read_ledger(Path(root), record, table, **filters)


def _csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, date_format="%Y-%m-%d").encode("utf-8-sig")


def _json(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def _pct(value) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.2%}"


def _label(record: Experiment) -> str:
    names = {"rolling_lightgbm": "年度滚动 LightGBM", "rolling_ridge": "年度滚动 Ridge",
             "ensemble": "模型集成", "economic": "经济逻辑多因子", "multifactor_raw": "标准化多因子",
             "multifactor_neutral": "中性化多因子", "momentum": "动量", "reversal": "反转",
             "low_volatility": "低波动"}
    signal = record.strategy.split("/")[-1]
    for key in sorted(names, key=len, reverse=True):
        if signal.startswith(key):
            signal = signal.replace(key, names[key], 1)
            break
    signal = signal.replace("__managed", " · 波动预算").replace("__steady", " · 稳定仓位")
    return f"{record.version} · {signal}｜{record.stage}｜{record.start} → {record.end}｜{record.id[:6]}"


def _table_export(frame: pd.DataFrame, matched: int, label: str, filename: str, key: str):
    if frame.empty:
        st.info("当前筛选条件没有记录。可扩大日期范围或清空股票代码。")
        return
    if matched > len(frame):
        st.warning(f"匹配 {matched:,} 行；为控制内存，仅载入前 {len(frame):,} 行。请缩小筛选范围后导出完整结果。")
    st.caption(f"匹配 {matched:,} 行 · 表格预览前 {min(1000, len(frame)):,} 行 · 下载包含已载入的 {len(frame):,} 行")
    st.dataframe(frame.head(1000), hide_index=True, width="stretch")
    st.download_button(f"下载{label} CSV", _csv(frame), filename, "text/csv", key=key)


def _overview(root: Path, record: Experiment, daily: pd.DataFrame):
    if record.checks.get("passed") is True:
        count = record.checks.get("checked_sessions", "全部")
        st.info(f"保存时账本核对通过：{count} 个交易日。此检查仅验证资金与成交一致，不代表策略收益达标。")
    elif record.checks.get("passed") is False:
        st.error("保存的账本检查未通过，请先检查错误记录。")
    else:
        st.caption("此档案未附单独的账本检查记录；不据此推断检查通过。")
    if not daily.empty and "nav" in daily:
        config = record.config.get("backtest", record.config)
        capital = config.get("initial_cash") or daily.iloc[0].get("opening_nav")
        if capital and float(capital) > 0:
            chart = daily[["date", "nav"]].copy()
            chart["单位净值"] = chart.nav / float(capital)
            fig = px.line(chart, x="date", y="单位净值", labels={"date": "日期"},
                          color_discrete_sequence=["#168579"])
            fig.update_layout(height=300, margin=dict(t=15, b=0), hovermode="x unified")
            st.plotly_chart(fig, width="stretch")
        _table_export(daily, len(daily), "逐日账本", f"{record.id}_daily.csv", f"exp_daily_{record.id}")
    with st.expander("查看实际运行参数与账本检查"):
        a, b = st.columns(2)
        with a:
            st.markdown("**运行参数**")
            st.json(record.config)
        with b:
            st.markdown("**保存的检查结果**")
            st.json(record.checks)
    st.caption(f"记录来源：{record.relative_path} · 更新于 {record.updated_at[:19]} UTC")


def _positions(root: Path, record: Experiment, daily: pd.DataFrame):
    if not record.has_details:
        st.info("公开快照只有组合逐日汇总，不含逐股持仓。使用本地数据重跑后，这里会自动显示完整明细。")
        return
    if daily.empty:
        st.info("此实验没有可读取的逐日账本。")
        return
    dates = list(daily.date.dt.strftime("%Y-%m-%d").drop_duplicates())
    left, right = st.columns([1, 2])
    chosen = left.selectbox("持仓截面日期", dates, index=len(dates)-1, key=f"exp_position_date_{record.id}")
    asset = right.text_input("筛选持仓股票代码", placeholder="例如 000001 或 .SZ；留空查看全部", key=f"exp_position_asset_{record.id}")
    snapshot, count = _ledger(str(root), record, "positions", start=chosen, end=chosen, asset=asset)
    day = daily[daily.date == pd.Timestamp(chosen)].iloc[0]
    a, b, c = st.columns(3)
    a.metric("当日组合净值（元）", f"{day['nav']:,.2f}")
    b.metric("当日现金占比", _pct(day.get("cash", 0) / day["nav"] if day["nav"] else None))
    c.metric("筛选后股票数", f"{len(snapshot):,}")
    if not snapshot.empty and {"weight", "asset"} <= set(snapshot.columns):
        top = snapshot.nlargest(12, "weight").sort_values("weight")
        fig = px.bar(top, x="weight", y="asset", orientation="h",
                     labels={"weight": "占组合总净值的权重", "asset": "股票"},
                     color_discrete_sequence=["#4265a6"])
        fig.update_layout(height=max(220, 24*len(top)+45), margin=dict(t=10, b=0), xaxis_tickformat=".1%")
        st.plotly_chart(fig, width="stretch")
        st.caption("显示筛选后权重最高的 12 只股票；权重分母仍为完整组合总净值。")
        snapshot = snapshot.sort_values("weight", ascending=False)
    _table_export(snapshot, count, "持仓截面", f"{record.id}_positions_{chosen}.csv", f"exp_positions_download_{record.id}")


def _transactions(root: Path, record: Experiment, daily: pd.DataFrame):
    if not record.has_details:
        st.info("公开快照不含逐笔成交或订单。可在概览下载逐日成本、换手与未成交数量。")
        return
    if daily.empty:
        st.info("此实验没有可读取的交易日期。")
        return
    kind = st.radio("明细类型", ["实际成交", "订单与未完全成交"], horizontal=True, key=f"exp_trade_kind_{record.id}")
    table = "trades" if kind == "实际成交" else "orders"
    first, last = daily.date.min().date(), daily.date.max().date()
    a, b, c, d = st.columns([1, 1, 1.4, 1])
    start = a.date_input("成交开始日期", first, min_value=first, max_value=last, key=f"exp_trade_start_{record.id}")
    end = b.date_input("成交结束日期", last, min_value=first, max_value=last, key=f"exp_trade_end_{record.id}")
    asset = c.text_input("成交股票代码", placeholder="输入代码片段", key=f"exp_trade_asset_{record.id}")
    side_name = d.selectbox("交易方向", ["全部", "买入", "卖出"], key=f"exp_trade_side_{record.id}")
    status = "all"
    if table == "orders":
        chosen = st.selectbox("订单状态", ["全部", "未完全成交（含部分成交）", "全部成交", "部分成交", "完全未成交", "流动性限制", "资金不足缩量"],
                              key=f"exp_trade_status_{record.id}")
        status = {"全部": "all", "未完全成交（含部分成交）": "unfilled", "全部成交": "filled",
                  "部分成交": "partial", "完全未成交": "blocked", "流动性限制": "liquidity_limited", "资金不足缩量": "cash_scaled"}[chosen]
        st.caption("未完全成交包括停牌、缺失开盘价、涨跌停及流动性限制等原因；以保存的 reason 字段为准。")
    if start > end:
        st.error("成交开始日期不能晚于结束日期。")
        return
    frame, count = _ledger(str(root), record, table, start=str(start), end=str(end), asset=asset,
                           side={"全部": "all", "买入": "buy", "卖出": "sell"}[side_name], status=status)
    a, b, c = st.columns(3)
    a.metric("匹配记录", f"{count:,}")
    if table == "trades" and not frame.empty:
        b.metric("筛选成交额（元）", f"{frame.notional.sum():,.2f}")
        c.metric("筛选交易成本（元）", f"{frame.cost.sum():,.2f}")
    elif table == "orders" and not frame.empty:
        execution_state = order_fill_state(frame)
        b.metric("完全未成交", int((execution_state == "blocked").sum()))
        c.metric("部分成交", int((execution_state == "partial").sum()))
        failed = frame.loc[execution_state != "filled"]
        reasons = failed.reason.fillna(failed.status).replace("", pd.NA).fillna(failed.status).value_counts()
        if not reasons.empty:
            st.dataframe(reasons.rename_axis("原因代码").reset_index(name="订单数"), hide_index=True, width="stretch")
    _table_export(frame, count, "筛选明细", f"{record.id}_{table}_filtered.csv", f"exp_trades_download_{record.id}_{table}")


def _notes(root: Path, record: Experiment):
    saved = load_note(root, record.id)
    st.caption("将本次观察、疑问和下一步计划保存在本地。笔记不修改收益、参数或任何账本。")
    with st.form(f"exp_note_form_{record.id}"):
        title = st.text_input("笔记标题", saved.get("title", ""), max_chars=160)
        state = st.selectbox("研究状态", NOTE_STATES,
                             index=NOTE_STATES.index(saved.get("state")) if saved.get("state") in NOTE_STATES else 0)
        body = st.text_area("观察与下一步", saved.get("body", ""), height=220, max_chars=50_000,
                            placeholder="例如：这次回撤集中在哪个月？交易成本来自哪些股票？下一步先验证什么？")
        submitted = st.form_submit_button("保存实验笔记", type="primary")
    if submitted:
        saved = save_note(root, record.id, title, body, state)
        st.success("实验笔记已保存。")
    if saved:
        st.caption(f"最近保存：{saved.get('updated_at_utc', '')[:19]} UTC · runs/notes/{record.id}.json")
    st.download_button("导出实验研究摘要 JSON", _json(research_summary(record, saved)),
                       f"{record.id}_research_summary.json", "application/json", key=f"exp_summary_{record.id}")


def render(root: Path):
    """Entry point for the application's 实验档案 navigation item."""
    root = Path(root).resolve()
    from .ui_design import heading
    heading("实验档案", "从净值回到每一笔成交：检索实验、核查持仓与未成交原因，并保存研究笔记。")
    a, b = st.columns([5, 1])
    archives = a.checkbox("包含旧实验归档", value=False, key="exp_include_archives")
    if b.button("刷新档案", key="exp_refresh"):
        _catalogue.clear()
        _ledger.clear()
    try:
        records = _catalogue(str(root), archives)
        if not records:
            st.info("还没有实验档案。先运行一个回测，或按复现指南放入公开证据文件。")
            return
        a, b, c = st.columns(3)
        a.metric("可查实验", len(records))
        b.metric("含逐股明细", sum(record.has_details for record in records))
        c.metric("公开聚合快照", sum(not record.has_details for record in records))
        with st.expander("筛选实验", expanded=True):
            query = st.text_input("搜索策略或实验名称", placeholder="例如 rolling_lightgbm、momentum 或 interactive", key="exp_search")
            a, b = st.columns(2)
            versions = a.multiselect("版本", sorted({x.version for x in records}, reverse=True), key="exp_versions")
            stages = b.multiselect("实验类型", sorted({x.stage for x in records}), key="exp_stages")
            lower = upper = None
            if st.checkbox("按回测覆盖日期筛选", key="exp_filter_period"):
                dates = [pd.Timestamp(v).date() for r in records for v in (r.start, r.end) if v]
                if dates:
                    a, b = st.columns(2)
                    lower = str(a.date_input("覆盖开始日期", min(dates), key="exp_filter_start"))
                    upper = str(b.date_input("覆盖结束日期", max(dates), key="exp_filter_end"))
                    st.caption("保留与该日期范围有交集的实验；不重新计算收益。")
        selected_records = filter_experiments(records, query=query, versions=versions, stages=stages, start=lower, end=upper)
        if not selected_records:
            st.info("没有匹配的实验，请调整筛选条件。")
            return
        by_id = {record.id: record for record in selected_records}
        selected = st.selectbox("选择实验档案", list(by_id), format_func=lambda key: _label(by_id[key]), key="experiment_archive_selected")
        record = by_id[selected]
        if record.warning:
            st.warning(record.warning)
        if not record.has_details:
            st.info("当前为公开聚合快照：可以查看曲线、逐日汇总和记录研究笔记；逐股持仓及成交需要本地完整实验。")
        a, b, c, d = st.columns(4)
        a.metric("累计净收益", _pct(record.metrics.get("total_return")))
        b.metric("最大回撤", _pct(record.metrics.get("max_drawdown")))
        c.metric("交易成本（元）", f"{record.metrics['total_cost']:,.0f}" if record.metrics.get("total_cost") is not None else "—")
        d.metric("交易日", record.metrics.get("sessions", "—"))
        st.caption(f"{record.start} 至 {record.end} · {record.stage} · {record.source}。不同区间的收益不能直接排名。")
        daily, _ = _ledger(str(root), record, "daily")
        overview, positions, trades, notes = st.tabs(["净值与参数", "持仓穿透", "成交与未成交", "研究笔记"])
        with overview:
            _overview(root, record, daily)
        with positions:
            _positions(root, record, daily)
        with trades:
            _transactions(root, record, daily)
        with notes:
            _notes(root, record)
    except (ValueError, OSError, KeyError, pd.errors.ParserError) as exc:
        st.error(f"这份档案暂时无法完整读取：{exc}")
        st.caption("可刷新目录或切换其他实验；原始数据和回测结果不会被更改。")
