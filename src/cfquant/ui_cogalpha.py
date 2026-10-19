"""Read-only, aggregate CogAlpha research UI; never constructs an API client."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from .ui_design import chart, heading, note


ARMS = {
    "fixed16": "固定 16 个公式", "llm_initial21": "大模型初代 · 21 槽位",
    "llm_fresh49": "同预算新提案 · 49 槽位", "llm_evolved49": "进化候选 · 49 槽位",
    "llm_evolved_diverse": "进化候选 + 去冗余改进",
    "literal_1": "原文 Listing 1", "literal_2": "原文 Listing 2", "literal_3": "原文 Listing 3",
}
MODELS = {"rank": "正方向等权排名", "ridge": "冻结 Ridge", "lightgbm": "冻结 LightGBM"}
PHASES = {"validation": "2023–2024 验证期", "test": "2025 起历史留出期"}
COSTS = {"paper": "规定费用（含最低佣金）", "zero": "零成本对照"}
MATERIALS = {
    "CF2026_Project2_CogAlpha_Report.pdf": "独立研究报告 PDF",
    "CF2026_Project2_CogAlpha_Beamer.pdf": "Beamer 演示 PDF",
    "CF2026_Project2_CogAlpha_Presentation.pptx": "可编辑演示 PPTX",
    "CF2026_Project2_CogAlpha_Speaker_Script.pdf": "对应演讲稿 PDF",
}
BUDGET_FIELDS = ("budget_cny", "calls", "max_calls", "successful_calls", "successful_cost_cny",
                 "uncertain_reserved_cny", "reserved_or_spent_cny", "remaining_cny", "remaining_calls",
                 "halted", "pricing_basis")
COLUMN_LABELS = {"id": "因子身份", "replicate": "重复", "arm": "研究路线", "model": "组合模型", "phase": "时期",
                 "proposal_slots": "提案槽位", "valid_candidates": "有效候选", "qualified_candidates": "合格候选",
                 "elite_candidates": "精英候选", "selected_count": "最终入选", "ic": "IC", "rank_ic": "Rank IC",
                 "ic_ir": "ICIR", "rank_ic_ir": "Rank ICIR", "mi": "互信息（nats）", "coverage": "有效覆盖率",
                 "valid_days": "有效日期", "quality_passed": "质量检查通过", "total_return": "累计净收益",
                 "cagr": "年化收益", "sharpe": "年化夏普", "max_drawdown": "最大回撤", "annual_turnover": "年化换手",
                 "aer": "年化超额收益", "information_ratio": "信息比率"}


def _json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {} if default is None else default


def _csv(path):
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _file(folder, relative):
    if not isinstance(relative, str) or not relative:
        return None
    base = Path(folder).resolve()
    candidate = (base / relative).resolve()
    return candidate if candidate.is_relative_to(base) and candidate.is_file() else None


def material_path(root, filename):
    """Only four declared outputs inside this checkout may become downloads."""
    if filename not in MATERIALS:
        return None
    root = Path(root).resolve()
    folder = (root / "reports/project2_cogalpha").resolve()
    return _file(folder, filename) if folder.is_relative_to(root) else None


def _number(value, pattern=".4f"):
    try:
        number = float(value)
        return format(number, pattern) if math.isfinite(number) else "—"
    except (ValueError, TypeError):
        return "—"


def _filter(frame, replicate=None, phase=None, model=None, cost=None):
    result = frame.copy()
    if replicate is not None and "replicate" in result:
        numbers = pd.to_numeric(result.replicate, errors="coerce")
        # Fixed/literal runs are a shared baseline; repeat 0 is not an LLM repeat.
        result = result[(numbers == float(replicate)) | (numbers == 0)]
    for column, value in [("phase", phase), ("model", model), ("cost_case", cost)]:
        if value is not None and column in result:
            result = result[result[column] == value]
    return result


def _download(frame, label, filename, key):
    if not frame.empty:
        st.download_button(label, frame.to_csv(index=False).encode("utf-8-sig"), filename, "text/csv", key=key)


def _readable(frame, columns=None):
    result = frame.copy() if columns is None else frame[[key for key in columns if key in frame]].copy()
    for key, labels in [("arm", ARMS), ("model", MODELS), ("phase", PHASES)]:
        if key in result:
            result[key] = result[key].map(lambda value: labels.get(value, value))
    result = result.rename(columns=COLUMN_LABELS)
    formats = {COLUMN_LABELS[key]: "{:.2%}" for key in ["coverage", "total_return", "cagr", "max_drawdown", "aer"] if COLUMN_LABELS[key] in result}
    formats.update({COLUMN_LABELS[key]: "{:.4f}" for key in ["ic", "rank_ic", "ic_ir", "rank_ic_ir", "mi", "sharpe", "annual_turnover", "information_ratio"] if COLUMN_LABELS[key] in result})
    return result.style.format(formats, na_rep="—")


def _protocol(root, folder, protocol):
    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("这次检验什么")
        st.write("大模型提出的公式能否提供预测信息？在相同候选槽位预算下，带反馈的进化是否优于继续提出新公式？去掉高度相似的因子是否改善稳定性？")
        st.markdown("**提出假设 → 因果公式检查 → 训练期评价 → 候选进化 → 冻结模型 → 历史对照**")
        st.caption("主要比较：LightGBM 下，进化 49 槽位与新提案 49 槽位的留出期 Rank IC；报告每次独立重复。投资收益和去冗余属于补充检验。")
        note("候选由模型提出不代表因子有效。质量检查、预测指标和扣费账本提供不同层次的证据；失败公式与负结果同样保留。")
    with right:
        st.subheader("固定实验边界")
        st.write("训练：2020–2022；验证：2023–2024；历史留出：2025 至快照末日。模型在 2022 年底训练一次，后续不重估。")
        st.write("每次重复：21 个初代槽位、两代各 14 个子代；同时生成同预算的新提案对照。最终最多选 20 个正方向训练因子。")
        st.latex(r"\mathrm{RankIC}_t=\mathrm{Corr}\big(\mathrm{rank}(f_t),\mathrm{rank}(r_{t+1:t+11})\big)")
        st.caption("标签采用下一开盘起 10 个交易日的收益，这是本地研究设定。ICIR 用日 IC 的均值除以样本标准差，不年化。")
    differences = pd.DataFrame([
        {"项目": "数据与股票池", "本地处理": "固定 1000 股 Tushare 快照，非历史沪深300成分；日期也不同。"},
        {"项目": "模型与规模", "本地处理": "实际服务、模型版本、随机重复和有限候选预算均单独记录。"},
        {"项目": "互信息 MI", "本地处理": "原估计器未公开；采用日截面五分位离散插件估计，单位 nats，存在有限样本偏差。"},
        {"项目": "最终因子与组合", "本地处理": "原最终表达式与完整选择实现不可得；本地规则提前固定，不能声称原表精确数值复现。"},
        {"项目": "已观察历史", "本地处理": "留出仅指实验流程隔离；不是前瞻盲测，模型预训练知识污染另列。"},
    ])
    st.subheader("与原论文的可比范围")
    st.dataframe(differences, hide_index=True, width="stretch")
    with st.expander("完整固定协议"):
        st.json(protocol) if protocol else st.info("研究协议尚未发布。")
    reference = _json(folder / "source_reference.json")
    if reference:
        with st.expander("论文来源、版本与复现目标"):
            st.json(reference)


def _factors(folder, replicate, phase):
    st.subheader("候选数量、质量与谱系")
    pool = _filter(_csv(folder / "pool_summary.csv"), replicate)
    if not pool.empty:
        st.dataframe(_readable(pool), hide_index=True, width="stretch")
        _download(pool, "下载候选池统计", "cogalpha_pool_summary.csv", "ca_pool_download")
    else:
        st.info("主体候选研究尚未发布；不会用计划槽位数冒充已生成或已通过的因子数。")
    raw = _json(folder / "proposals.json", [])
    cards = raw if isinstance(raw, list) else raw.get("proposals", []) if isinstance(raw, dict) else []
    cards = [card for card in cards if isinstance(card, dict) and isinstance(card.get("id"), str)]
    if replicate is not None:
        cards = [card for card in cards if card.get("replicate") in (None, 0, replicate, str(replicate), "0")]
    if cards:
        roles = list(dict.fromkeys(str(card.get("role", "未标注")) for card in cards))
        role = st.selectbox("研究角色", ["全部"] + roles, key="ca_role")
        if role != "全部":
            cards = [card for card in cards if str(card.get("role", "未标注")) == role]
        ids = list(dict.fromkeys(card["id"] for card in cards))
        lookup = {card["id"]: card for card in cards}
        selected = st.selectbox("检查候选公式", ids,
                                format_func=lambda key: str(lookup[key].get("label", key)) + " · " + key, key="ca_factor")
        card = lookup[selected]
        st.code(card.get("expression", "尚无可执行公式"), language="text")
        st.write(card.get("hypothesis", "没有记录机制假设。"))
        columns = st.columns(4)
        for col, label, key in zip(columns, ["层级", "代数", "角色", "质量状态"], ["layer", "generation", "role", "quality_status"]):
            col.caption(label + "：" + str(card.get(key, "未记录")))
        parents = card.get("parents", [])
        st.write("父候选：" + ("、".join(map(str, parents)) if isinstance(parents, list) and parents else str(parents) if parents else "初代 / 无父候选"))
        st.caption("来源：" + str(card.get("source", "未记录")) + "。公式来源、质量合格与收益有效分别判断。")
        with st.expander("完整候选记录"):
            st.json(card)
    factor_metrics = _filter(_csv(folder / "factor_metrics.csv"), replicate, phase)
    if not factor_metrics.empty:
        st.markdown("**所选时期的因子预测指标**")
        st.dataframe(_readable(factor_metrics), hide_index=True, width="stretch", height=330)
        _download(factor_metrics, "下载因子评价", "cogalpha_factor_metrics.csv", "ca_factor_download")
    for filename, label in [("lineage.json", "父链与代际记录"), ("selection.json", "按训练期冻结的选择")]:
        content = _json(folder / filename)
        if content:
            with st.expander(label):
                st.json(content, expanded=False)


def _experiments(folder, performance, filtered):
    st.subheader("预测与投资，分开评估")
    if filtered.empty:
        st.info("所选条件还没有完成的投资实验。尚未生成的主体结果不会显示为成功或零收益。")
        return
    ids = list(dict.fromkeys(filtered.arm.astype(str)))
    default = ids.index("llm_evolved49") if "llm_evolved49" in ids else 0
    focus = st.selectbox("当前研究路线", ids, index=default, format_func=lambda key: ARMS.get(key, key), key="ca_focus")
    row = filtered[filtered.arm == focus].iloc[0]
    for col, label, key, pattern in zip(st.columns(5), ["Rank IC", "Rank ICIR", "累计净收益", "年化夏普比率", "最大回撤"],
                                       ["rank_ic", "rank_ic_ir", "total_return", "sharpe", "max_drawdown"],
                                       [".4f", ".3f", ".2%", ".3f", ".2%"]):
        col.metric(label, _number(row.get(key), pattern))
    st.caption(f"{row.get('start', '—')} — {row.get('end', '—')} · 扣费日收益 · 252 日年化 · 年化无风险利率 0% · 首日损益保留。")
    if pd.notna(row.get("total_return")) and float(row.total_return) < 0:
        st.warning("这条路线在所选区间亏损。负结果作为研究证据保留，不自动改选收益较高的路线。")
    if pd.notna(row.get("rank_ic")) and float(row.rank_ic) <= 0:
        st.warning("所选路线的 Rank IC 未显示正向预测关系；不能把候选数量或生成成功当作预测有效。")
    if {"llm_evolved49", "llm_fresh49"}.issubset(ids):
        pair = filtered[filtered.arm.isin(["llm_evolved49", "llm_fresh49"])].set_index("arm")
        delta = pair.loc["llm_evolved49", "rank_ic"] - pair.loc["llm_fresh49", "rank_ic"]
        st.info("本次重复中，进化相对同预算新提案的 Rank IC 差值：" + _number(delta, "+.4f") + "。应结合全部重复判断。")
    defaults = [arm for arm in ["llm_evolved49", "llm_fresh49"] if arm in ids] or ids[:2]
    chosen = st.multiselect("曲线比较路线", ids, default=defaults, format_func=lambda key: ARMS.get(key, key), key="ca_curves")
    parts = []
    for _, candidate in filtered[filtered.arm.isin(chosen)].iterrows():
        path = _file(folder, candidate.get("daily_file"))
        daily = _csv(path) if path else pd.DataFrame()
        if {"date", "nav"}.issubset(daily):
            daily["date"] = pd.to_datetime(daily.date, errors="coerce")
            daily["路线"] = ARMS.get(candidate.arm, candidate.arm)
            parts.append(daily.dropna(subset=["date", "nav"]).sort_values("date"))
    if parts:
        curves = pd.concat(parts, ignore_index=True)
        plotted = curves
        if "benchmark_nav" in parts[0]:
            benchmark = parts[0][["date", "benchmark_nav"]].rename(columns={"benchmark_nav": "nav"})
            benchmark["路线"] = "沪深300价格指数"
            plotted = pd.concat([curves, benchmark], ignore_index=True)
        chart(px.line(plotted, x="date", y="nav", color="路线", labels={"date": "日期", "nav": "期初资金 = 1"}), 390, key="ca_nav")
        st.caption("基准采用研究记录中的同期沪深300价格指数；不是含分红全收益指数，也不承担策略模拟费用。")
        choice = st.radio("补充风险视图", ["回撤", "仓位"], horizontal=True, key="ca_risk")
        field = "drawdown" if choice == "回撤" else "exposure"
        if field in curves:
            chart(px.line(curves, x="date", y=field, color="路线", labels={"date": "日期", field: choice}).update_yaxes(tickformat=".0%"), 260, key="ca_risk_chart")
    elif chosen:
        st.caption("所选日线曲线尚未发布，汇总指标仍可查看。")
    st.markdown("**同模型、时期和成本下的全部路线**")
    st.dataframe(_readable(filtered, ["arm", "replicate", "rank_ic", "rank_ic_ir", "ic", "mi", "total_return", "cagr", "sharpe", "max_drawdown", "annual_turnover", "aer", "information_ratio"]), hide_index=True, width="stretch", height=330)
    _download(filtered, "下载当前投资与预测对照", "cogalpha_selected_performance.csv", "ca_performance_download")
    comparison = _csv(folder / "comparison.csv")
    if not comparison.empty:
        st.markdown("**全部重复的预设比较**")
        st.dataframe(comparison, hide_index=True, width="stretch")
        _download(comparison, "下载重复比较结果", "cogalpha_comparison.csv", "ca_comparison_download")
    with st.expander("查看全部实验，包含失败与对照"):
        st.dataframe(performance, hide_index=True, width="stretch")


def _budget(folder, protocol):
    st.subheader("模型调用与预算回执")
    model = protocol.get("model", {})
    st.caption("配置的模型：" + str(model.get("name", "尚未配置")) + "；实际调用情况以下方发布回执为准。")
    raw = _json(folder / "budget.json")
    budget = {key: raw[key] for key in BUDGET_FIELDS if key in raw} if isinstance(raw, dict) else {}
    if not budget:
        st.info("尚无发布的预算回执。配置存在不代表主体研究已调用模型或已完成；此页面只读，不会自动发起付费请求。")
        return
    for col, label, key, pattern in zip(st.columns(4), ["发起请求数", "成功请求数", "成功调用保守估算（元）", "剩余可用预算（元）"],
                                       ["calls", "successful_calls", "successful_cost_cny", "remaining_cny"], [".0f", ".0f", ".6f", ".4f"]):
        col.metric(label, _number(budget.get(key), pattern))
    ceiling = pd.to_numeric(budget.get("budget_cny"), errors="coerce")
    used = pd.to_numeric(budget.get("reserved_or_spent_cny"), errors="coerce")
    if pd.notna(ceiling) and pd.notna(used) and ceiling > 0:
        st.progress(min(1., max(0., float(used / ceiling))), text=f"已估算或预留 {_number(used, '.6f')} / {_number(ceiling, '.2f')} 元")
    note("金额是按回执用量和保守价格计算的预算记录，不是服务商正式账单。失败或未确认请求的预留仍占用预算，不把不确定费用当作零。")
    st.write("未确认请求预留：" + _number(budget.get("uncertain_reserved_cny"), ".6f") + " 元。")
    if budget.get("halted"):
        st.warning("预算回执标记已停止继续请求；请先核查研究记录中的停止原因。")
    with st.expander("完整预算摘要"):
        st.json(budget)
    st.download_button("下载预算摘要", json.dumps(budget, ensure_ascii=False, indent=2), "cogalpha_budget.json", "application/json", key="ca_budget_download")


def _materials(root, folder):
    st.subheader("独立交付与复现")
    st.write("12412408 胡博文 · 12410819 谢明飞 · 12410436 王天希")
    st.caption("CogAlpha 与 RSRS 分别保留协议、实验和材料。")
    columns = st.columns(2)
    for i, (filename, label) in enumerate(MATERIALS.items()):
        with columns[i % 2]:
            path = material_path(root, filename)
            if path:
                st.download_button("下载" + label, path.read_bytes(), filename, key="ca_material_" + str(i), width="stretch")
            else:
                st.caption(label + "：准备中")
    st.markdown("**查看命令行复现入口**")
    st.code("python scripts/run_project2_cogalpha.py --help", language="shell")
    st.caption("真实生成阶段会使用所选模型服务与本地行情，按固定预算运行。浏览此页面或下载材料不会发起模型请求。")
    for filename, label in [("manifest.json", "研究清单与输入身份"), ("validation.json", "验证记录")]:
        record = _json(folder / filename)
        if record:
            with st.expander(label):
                st.json(record, expanded=False)


def render(root):
    root = Path(root)
    folder = root / "evidence/project2_cogalpha"
    protocol = _json(folder / "protocol.json") or _json(root / "configs/project2_cogalpha.json")
    heading("CogAlpha · 因子如何被提出与检验", "跟踪公式从初代到进化的全过程，用同预算对照检验大模型是否带来额外价值。", "COMPUTATIONAL FINANCE 2026 / INDEPENDENT PAPER STUDY")
    note("受限机制复现 / 与原表非直接可比 / 已观察历史。原年度 LightGBM 主线与 RSRS 研究继续独立保留。")
    performance = _csv(folder / "performance.csv")
    required = {"replicate", "arm", "model", "phase", "cost_case", "rank_ic"}
    ready = not performance.empty and required.issubset(performance)
    if ready:
        a, b, c, d = st.columns(4)
        repeats = sorted(pd.to_numeric(performance.replicate, errors="coerce").dropna().astype(int).unique().tolist())
        repeats = [value for value in repeats if value > 0] or repeats
        replicate = a.selectbox("独立重复", repeats, format_func=lambda value: f"重复 {value}" if value else "共享固定对照", key="ca_replicate")
        phases = list(dict.fromkeys(performance.phase.astype(str)))
        phase = b.selectbox("评价时期", phases, index=phases.index("test") if "test" in phases else 0, format_func=lambda key: PHASES.get(key, key), key="ca_phase")
        models = list(dict.fromkeys(performance.model.astype(str)))
        model = c.selectbox("组合模型", models, index=models.index("lightgbm") if "lightgbm" in models else 0, format_func=lambda key: MODELS.get(key, key), key="ca_model")
        costs = list(dict.fromkeys(performance.cost_case.astype(str)))
        cost = d.selectbox("成本口径", costs, index=costs.index("paper") if "paper" in costs else 0, format_func=lambda key: COSTS.get(key, key), key="ca_cost")
        filtered = _filter(performance, replicate, phase, model, cost)
        st.caption("重复 0 的固定公式和原文例子作为共享对照显示；它们不算一次独立的大模型生成重复。")
    else:
        replicate, phase, filtered = None, None, pd.DataFrame()
        st.info("主体实验结果尚未发布。当前可查看固定协议；真实模型连通、候选生成和完整实验验收分别记录。")
    tabs = st.tabs(["协议与差异", "因子谱系与质量", "预测与投资实验", "模型调用与预算", "材料复现"])
    with tabs[0]:
        _protocol(root, folder, protocol)
    with tabs[1]:
        _factors(folder, replicate, phase)
    with tabs[2]:
        _experiments(folder, performance, filtered)
    with tabs[3]:
        _budget(folder, protocol)
    with tabs[4]:
        _materials(root, folder)
