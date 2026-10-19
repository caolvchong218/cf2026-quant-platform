"""RSRS research presentation and isolated replay; calculations live in rsrs.py."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
from uuid import uuid4

import pandas as pd
import plotly.express as px
import streamlit as st

from .ui_design import chart, heading, note


INDEX_NAMES = {"sh000300": "沪深300", "sh000016": "上证50", "sh000905": "中证500"}
STRATEGY_NAMES = {
    "slope": "原始斜率", "standard": "标准分", "corrected": "修正标准分",
    "right": "右偏标准分", "standard_ma": "标准分 + 均线过滤",
    "corrected_ma": "修正标准分 + 均线过滤", "right_ma": "右偏 + 均线过滤（预设主策略）",
    "right_ma_vol15": "自主改进 · 15% 波动目标", "right_ma_cash75": "固定75%仓位对照",
    "buy_hold": "买入持有", "ma20": "20日均线对照",
}
PERIOD_NAMES = {
    "historical_overlap": "原文重叠期", "post_publication": "发表后检验",
    "recent": "近期子样本",
}
MATERIALS = {
    "CF2026_Project2_Report.pdf": "最终研究报告 PDF",
    "CF2026_Project2_Beamer.pdf": "15 分钟 Beamer PDF",
    "CF2026_Project2_Presentation.pptx": "可编辑演示 PPTX",
    "CF2026_Project2_Speaker_Script.pdf": "逐页演讲稿 PDF",
}


def _json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}


def _csv(path):
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def load_evidence(root):
    folder = Path(root) / "evidence/project2_rsrs"
    performance = _csv(folder / "performance.csv")
    aliases = {"symbol": "index_id", "period": "period_id", "strategy": "strategy_id",
               "annualized_return": "cagr", "annualized_volatility": "volatility", "cost_bps_each_side": "cost_bps"}
    for old, new in aliases.items():
        if new not in performance and old in performance:
            performance[new] = performance[old]
    return folder, performance


def _safe_child(folder, relative):
    """Published artifact paths can only address files inside their evidence folder."""
    if not isinstance(relative, str) or not relative:
        return None
    folder = Path(folder).resolve()
    candidate = (folder / relative).resolve()
    return candidate if candidate.is_relative_to(folder) and candidate.is_file() else None


def _pct(value):
    try:
        return f"{float(value):.2%}" if math.isfinite(float(value)) else "—"
    except (TypeError, ValueError):
        return "—"


def _ratio(value):
    try:
        return f"{float(value):.3f}" if math.isfinite(float(value)) else "—"
    except (TypeError, ValueError):
        return "—"


def _metrics(row, prefix=""):
    columns = st.columns(4)
    for column, label, key, formatter in zip(
        columns, ["累计净收益", "复合年化收益", "年化夏普比率", "最大回撤"],
        ["total_return", "cagr", "sharpe", "max_drawdown"], [_pct, _pct, _ratio, _pct],
    ):
        column.metric(prefix + label, formatter(row.get(key, row.get("annualized_return") if key == "cagr" else None)))


def _download_frame(label, frame, filename, key):
    if not frame.empty:
        st.download_button(label, frame.to_csv(index=False).encode("utf-8-sig"), filename,
                           "text/csv", key=key)


def _filter(frame, index_id, period_id, cost=None):
    result = frame.copy()
    for names, value in [(('index_id', 'symbol'), index_id), (('period_id', 'period'), period_id)]:
        col = next((name for name in names if name in result), None)
        if col is not None:
            result = result[result[col].astype(str) == str(value)]
    if cost is not None and "cost_bps" in result:
        result = result[pd.to_numeric(result.cost_bps, errors="coerce") == float(cost)]
    return result


def _curves(folder, rows):
    parts = []
    for _, row in rows.iterrows():
        path = _safe_child(folder, row.get("daily_file"))
        if path is None:
            continue
        daily = _csv(path)
        if not {"date", "nav"}.issubset(daily):
            continue
        daily["date"] = pd.to_datetime(daily.date, errors="coerce")
        daily = daily.dropna(subset=["date", "nav"]).sort_values("date")
        daily["策略"] = STRATEGY_NAMES.get(row.strategy_id, row.strategy_id)
        parts.append(daily)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _protocol_tab(folder, protocol):
    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("研究问题与固定方案")
        st.write("2017 年 RSRS 研报的择时优势能否复现？差异由什么造成？在相同信号上加入波动目标，能否改善风险与收益的权衡？")
        st.markdown("**原文复现 → 差异归因 → 自主改进 → 发表后检验**")
        st.write("主策略预设为右偏 RSRS + 均线入场过滤；自主改进保留其进出场信号，用过去 20 个交易日波动率控制仓位，目标年化波动率 15%，最高仓位 100%。")
        st.caption("均线过滤来自原研报；自主改进是新增的波动率仓位管理。15% / 20 日为运行前固定设定，参数探索不会改写发布结论。")
        note("指数点位是研究代理，无法直接买卖；现金利息为 0、无杠杆、连续份额。发表后的历史数据也已被观察，不能称为前瞻盲测。")
    with right:
        st.subheader("从价格到下一交易日仓位")
        st.latex(r"H_i=\alpha_t+\beta_t L_i+\varepsilon_i,\quad i=t-N+1,\ldots,t")
        st.latex(r"z_t=\frac{\beta_t-\mu_{t-1,M}}{\sigma_{t-1,M}},\qquad s_t=z_tR_t^2\beta_t")
        st.latex(r"w_t=d_t\min\left(1,\frac{0.15}{\widehat\sigma_{t,20}}\right)")
        st.caption("标准化只使用当前斜率之前 M 个斜率。右偏版 N = 16、M = 300；阈值为 ±0.7。d 为带滞回的多头/现金信号。")
        st.write("收盘获得完整高低价 → 计算信号与波动 → 下一交易日开盘成交 → 收盘估值。原文未明确的执行细节逐项披露。")
    with st.expander("查看完整固定协议与输入身份"):
        if protocol:
            st.json(protocol)
        else:
            st.info("固定研究协议尚未生成。")
        manifest = _json(folder / "manifest.json")
        if manifest:
            st.json(manifest, expanded=False)


def _reference_tab(folder, index_id):
    st.subheader("目标图表：原文数值与本次复现")
    st.caption("原文图25、表5和图26针对沪深300；此处原文对照固定为沪深300，其他指数结果见实验页。")
    st.caption("只有数据、样本期、执行时点、成本和指标定义一致的结果，才具备数值复现可比性。无法完全对齐的项目保留标记。")
    reference = _json(folder / "source_reference.json")
    source_image = folder / "reference/source_p19.png"
    if source_image.exists():
        with st.expander("查看研报目标页原图"):
            st.image(str(source_image), caption="原研报第 19 页 · 来源及哈希见 source_reference.json", width="stretch")
    table = _csv(folder / "reproduction_table.csv")
    if not table.empty:
        if "index_id" in table or "symbol" in table:
            col = "index_id" if "index_id" in table else "symbol"
            table = table[table[col] == index_id]
        st.dataframe(table, width="stretch", hide_index=True)
        _download_frame("下载原文逐项对照", table, "rsrs_reproduction_table.csv", "p2_reference_download")
    else:
        st.info("原文对照表尚未生成。")
    if reference:
        with st.expander("原文出处、页码与抄录依据"):
            st.json(reference)


def _published_experiments(folder, filtered):
    st.subheader("同一期限、同一成本，保留所有对照")
    ids = list(dict.fromkeys(filtered.strategy_id.astype(str)))
    defaults = [x for x in ["right_ma", "right_ma_vol15", "right_ma_cash75", "buy_hold"] if x in ids] or ids[:3]
    choices = st.multiselect("净值图比较策略", ids, default=defaults,
                            format_func=lambda x: STRATEGY_NAMES.get(x, x), key="p2_compare")
    picked = st.selectbox("查看指标的策略", ids, index=ids.index("right_ma") if "right_ma" in ids else 0,
                          format_func=lambda x: STRATEGY_NAMES.get(x, x), key="p2_focus")
    selected = filtered[filtered.strategy_id == picked].iloc[0]
    _metrics(selected)
    st.caption(f"{selected.get('start', '—')} — {selected.get('end', '—')} · 年化 252 交易日 · 无风险利率 0% · 扣费日收益样本标准差。")
    if float(selected.get("total_return", 0)) < 0:
        st.warning("所选策略在这段历史区间亏损。该结果保留在研究对照中，不自动切换到更好看的策略。")
    data = _curves(folder, filtered[filtered.strategy_id.isin(choices)])
    if not data.empty:
        chart(px.line(data, x="date", y="nav", color="策略", labels={"date": "日期", "nav": "初始资金 = 1"}), 400, key="p2_published_nav")
        measure = st.radio("风险图", ["回撤", "仓位"], horizontal=True, key="p2_risk_measure")
        field = "drawdown" if measure == "回撤" else "exposure"
        if field in data:
            chart(px.line(data, x="date", y=field, color="策略", labels={"date": "日期", field: measure}).update_yaxes(tickformat=".0%"), 260, key="p2_published_risk")
        st.caption("净值从期初资金 1 开始，保留首日成交成本及损益。图表展示按该区间重新投入资金的结果。")
    elif choices:
        st.info("所选结果的逐日曲线尚未随证据发布；下方汇总表仍可查看。")
    visible = [c for c in ["strategy_id", "total_return", "cagr", "sharpe", "max_drawdown", "volatility", "annual_turnover", "exposure"] if c in filtered]
    labels = {"strategy_id": "策略", "total_return": "累计收益", "cagr": "年化收益", "sharpe": "夏普", "max_drawdown": "最大回撤", "volatility": "年化波动", "annual_turnover": "年化换手", "exposure": "平均仓位"}
    display = filtered[visible].copy()
    display["strategy_id"] = display.strategy_id.map(lambda x: STRATEGY_NAMES.get(x, x))
    display = display.rename(columns=labels)
    formats = {labels[c]: "{:.2%}" for c in visible if c not in {"strategy_id", "sharpe", "annual_turnover"}}
    formats.update({labels[c]: "{:.3f}" for c in visible if c in {"sharpe", "annual_turnover"}})
    st.dataframe(display.style.format(formats, na_rep="—"), width="stretch", hide_index=True)
    _download_frame("下载当前全部策略指标", filtered, "rsrs_selected_performance.csv", "p2_performance_download")


def run_exploration(root, parameters):
    """Run the public numerical API, saving a unique experiment without editing evidence."""
    from .rsrs import run_rsrs
    root = Path(root)
    if parameters.get("index_id") not in INDEX_NAMES or parameters.get("strategy_id") not in STRATEGY_NAMES:
        raise ValueError("请选择已登记的指数与策略")
    price_path = root / "data/private/project2_rsrs" / (parameters["index_id"] + ".csv")
    data = pd.read_csv(price_path, parse_dates=["date"]).set_index("date").sort_index()
    base_strategy = "right_ma" if parameters["strategy_id"] in {"right_ma_vol15", "right_ma_cash75"} else parameters["strategy_id"]
    asset = parameters["index_id"][2:] + ".SH"
    run = run_rsrs(data, base_strategy, parameters["start"], parameters["end"],
                   cost_bps=parameters["cost_bps"],
                   vol_target=parameters["vol_target"] if parameters["strategy_id"] == "right_ma_vol15" else None,
                   vol_window=parameters["vol_window"], asset=asset,
                   exposure_scale=.75 if parameters["strategy_id"] == "right_ma_cash75" else 1.0)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:10]
    folder = root / "runs/project2_rsrs_explorations" / stamp
    folder.mkdir(parents=True, exist_ok=False)
    identity = {"status": "user_parameter_exploration", "parameters": parameters,
                "source_price_sha256": hashlib.sha256(price_path.read_bytes()).hexdigest(),
                "published_evidence_unchanged": True, "metadata": run.metadata, "checks": run.checks,
                "metrics": run.metrics}
    (folder / "experiment.json").write_text(json.dumps(identity, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    for name in ["daily", "trades", "positions", "orders"]:
        frame = getattr(run.result, name)
        frame.to_csv(folder / (name + ".csv"), index=False)
    return {"folder": str(folder), "parameters": parameters, "metrics": run.metrics,
            "checks": run.checks, "daily": run.result.daily, "initial_cash": run.metadata["initial_cash"]}


def _replay(root, filtered, index_id, period_id, cost):
    st.subheader("本地独立回放")
    st.caption("调整参数会创建独立的探索实验。发布证据及预设 15% / 20 日改进不随交互修改。")
    sample = filtered.iloc[0] if not filtered.empty else {}
    start_default = pd.Timestamp(sample.get("start", "2020-01-02")).date()
    end_default = pd.Timestamp(sample.get("end", "2026-10-15")).date()
    period_token = (index_id, period_id)
    if st.session_state.get("p2_replay_period") != period_token:
        st.session_state["p2_replay_start"] = start_default
        st.session_state["p2_replay_end"] = end_default
        st.session_state["p2_replay_period"] = period_token
    a, b, c = st.columns(3)
    strategy = a.selectbox("回放策略", list(STRATEGY_NAMES), index=list(STRATEGY_NAMES).index("right_ma"),
                           format_func=STRATEGY_NAMES.get, key="p2_replay_strategy")
    start = b.date_input("回放开始日期", start_default, key="p2_replay_start")
    end = c.date_input("回放结束日期", end_default, key="p2_replay_end")
    a, b = st.columns(2)
    vol_target = a.number_input("探索目标年化波动率（%）", min_value=1., max_value=100., value=15., step=1., disabled=strategy != "right_ma_vol15", key="p2_vol_target") / 100
    vol_window = int(b.number_input("探索波动窗口（交易日）", min_value=5, max_value=252, value=20, disabled=strategy != "right_ma_vol15", key="p2_vol_window"))
    params = {"index_id": index_id, "period_id": period_id, "strategy_id": strategy, "start": str(start), "end": str(end),
              "cost_bps": float(cost), "vol_target": vol_target, "vol_window": vol_window}
    signature = json.dumps(params, sort_keys=True)
    previous = st.session_state.get("p2_exploration_signature")
    if previous is not None and previous != signature:
        st.session_state.pop("p2_exploration", None)
        st.info("参数已改变，旧探索结果已清除；重新运行后才会显示新结果。")
    st.session_state["p2_exploration_signature"] = signature
    available = (Path(root) / "data/private/project2_rsrs" / (index_id + ".csv")).is_file()
    if not available:
        st.info("本机缺少所选指数的私有行情快照，可继续浏览发布证据；放入 data/private/project2_rsrs 后可独立回放。")
    if start > end:
        st.error("回放开始日期不能晚于结束日期。")
    if st.button("运行独立探索实验", key="p2_run", type="primary", disabled=not available or start > end):
        st.session_state.pop("p2_exploration", None)
        try:
            with st.spinner("调用统一回测与账本核验，保存独立实验…"):
                st.session_state["p2_exploration"] = run_exploration(root, params)
        except (ValueError, ArithmeticError, OSError, KeyError) as exc:
            st.error(f"探索实验未完成：{exc}")
    outcome = st.session_state.get("p2_exploration")
    if outcome:
        st.warning("以下为当前参数的探索结果，不是重新挑选后的正式研究结论。")
        _metrics(outcome["metrics"], "探索 · ")
        st.json(outcome["checks"], expanded=False)
        daily = outcome["daily"]
        if outcome.get("initial_cash") and {"date", "nav"}.issubset(daily):
            curve = daily[["date", "nav"]].copy()
            curve["期初资金 = 1"] = curve.nav / outcome["initial_cash"]
            chart(px.line(curve, x="date", y="期初资金 = 1", labels={"date": "日期"}), 310, key="p2_exploration_nav")
        st.dataframe(daily.tail(20), hide_index=True, width="stretch")
        _download_frame("下载探索逐日账本", daily, "rsrs_exploration_daily.csv", "p2_exploration_download")
        st.caption("独立实验已保存：" + outcome["folder"])


def _diagnostics_tab(folder, index_id, period_id, cost):
    st.subheader("逐项改变假设，解释复现差异")
    st.write("一次只改变一个条件，观察交易、收益和风险如何变化。差异存在并不自动等于实现错误；只有通过对照才能确定其来源。")
    for filename, title in [("paired_comparison.csv", "同信号改进的成对比较"), ("diagnostics.csv", "执行时点、标准化、预热与成本诊断")]:
        frame = _filter(_csv(folder / filename), index_id, period_id,
                        cost if filename == "paired_comparison.csv" else None)
        st.markdown("**" + title + "**")
        if frame.empty:
            st.info("当前条件下暂无该诊断证据。")
        else:
            st.dataframe(frame, hide_index=True, width="stretch")
            _download_frame("下载" + title, frame, filename, "p2_" + filename)
    validation = _json(folder / "validation.json")
    with st.expander("实现与账本验证记录"):
        if validation:
            st.json(validation)
        else:
            st.info("验收记录尚未生成。")
    note("成交时点对照使用下一交易日收盘，主策略使用下一交易日开盘，两者均在信号形成之后。账本核对证明资金一致，不证明策略必然盈利。")


def _materials_tab(root, folder):
    st.subheader("研究材料与一键复现")
    st.write("团队：12412408 胡博文 · 12410819 谢明飞 · 12410436 王天希")
    st.caption("Project 2 · 2026 年 11 月 1 日截止 · 演示约 15 分钟")
    columns = st.columns(2)
    for i, (filename, label) in enumerate(MATERIALS.items()):
        path = Path(root) / "reports/project2_rsrs" / filename
        with columns[i % 2]:
            if path.is_file():
                st.download_button("下载" + label, path.read_bytes(), filename, key="p2_material_" + str(i), width="stretch")
            else:
                st.caption(label + "：准备中")
    st.markdown("**重新生成正式研究证据**")
    st.code("python scripts/run_project2_rsrs.py --help", language="shell")
    st.caption("运行参数与数据准备以项目 README 和固定协议为准。行情快照需要本地数据；公开仓库提供聚合指标、逐日研究净值和验证记录。")
    for filename, label in [("protocol.json", "固定研究协议"), ("manifest.json", "输入与输出身份"), ("validation.json", "验收证据")]:
        path = folder / filename
        if path.is_file():
            st.download_button("下载" + label, path.read_bytes(), filename, "application/json", key="p2_json_" + filename)


def render(root):
    root = Path(root)
    folder, performance = load_evidence(root)
    heading("RSRS · 从研报到可核验结论", "原文复现、差异解释与一项固定改进，在青序里完成同口径研究。", "COMPUTATIONAL FINANCE 2026 / PROJECT 02")
    protocol = _json(folder / "protocol.json")
    required = {"index_id", "period_id", "strategy_id", "cost_bps"}
    ready = not performance.empty and required.issubset(performance)
    if ready:
        a, b, c = st.columns([1, 1.4, 1])
        indices = list(dict.fromkeys(performance.index_id.astype(str)))
        index_id = a.selectbox("研究指数", indices, index=indices.index("sh000300") if "sh000300" in indices else 0, format_func=lambda x: INDEX_NAMES.get(x, x), key="p2_index")
        periods = list(dict.fromkeys(performance[performance.index_id == index_id].period_id.astype(str)))
        period_id = b.selectbox("研究时期", periods, index=periods.index("post_publication") if "post_publication" in periods else 0, format_func=lambda x: PERIOD_NAMES.get(x, x), key="p2_period")
        costs = sorted(pd.to_numeric(_filter(performance, index_id, period_id).cost_bps).unique().tolist())
        cost = c.selectbox("单边成本（bp）", costs, index=costs.index(10.) if 10. in costs else 0, format_func=lambda x: f"{x:g} bp", key="p2_cost")
        filtered = _filter(performance, index_id, period_id, cost)
    else:
        index_id, period_id, cost, filtered = "sh000300", "recent", 10., pd.DataFrame()
        st.info("正式研究证据尚未生成或尚不完整；可先查看研究协议与本地回放入口。")
    tabs = st.tabs(["研究协议", "原文对照", "实验与改进", "差异诊断", "材料与复现"])
    with tabs[0]:
        _protocol_tab(folder, protocol)
    with tabs[1]:
        _reference_tab(folder, index_id)
    with tabs[2]:
        if ready and not filtered.empty:
            _published_experiments(folder, filtered)
        st.divider()
        _replay(root, filtered, index_id, period_id, cost)
    with tabs[3]:
        _diagnostics_tab(folder, index_id, period_id, cost)
    with tabs[4]:
        _materials_tab(root, folder)
