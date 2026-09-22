"""File-based simulation handoff without claiming a broker connection."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from .experiment_catalog import discover_experiments, read_ledger
from .simulation_bridge import (OFFICIAL_SOURCES, connection_status, historical_watchlist,
                                parse_simulated_trades, reconcile_by_security, safe_csv,
                                save_simulated_receipt, synthetic_example_csv, validation_summary)


def _json(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


@st.cache_data(ttl=60, show_spinner=False)
def _historical_records(root: str):
    return [record for record in discover_experiments(Path(root)) if record.version == "V3" and record.has_details]


def _watchlist_downloads(watch: pd.DataFrame, metadata: dict):
    as_of = metadata["as_of"]
    st.info(f"历史观察组合 · {as_of} 收盘截面。权重是当时实际持仓，不是今天的买卖信号或委托文件。")
    a, b, c = st.columns(3)
    a.metric("历史证券数", len(watch))
    b.metric("历史股票权重", f"{metadata['stock_weight']:.2%}")
    c.metric("历史现金权重", f"{metadata['cash_weight']:.2%}")
    st.dataframe(watch, hide_index=True, width="stretch",
                 column_config={"reference_weight": st.column_config.NumberColumn("历史参考权重", format="%.4f"),
                                "code": st.column_config.TextColumn("证券代码"), "as_of": "历史日期"})
    a, b, c = st.columns(3)
    a.download_button("下载历史权重 CSV", safe_csv(watch), f"historical_watchlist_{as_of}.csv", "text/csv", key="bridge_watchlist_csv")
    codes = "\n".join(watch.code.astype(str)) + "\n"
    b.download_button("下载六位代码 TXT", codes.encode("utf-8"), f"historical_codes_{as_of}.txt", "text/plain", key="bridge_codes_txt")
    c.download_button("下载组合说明 JSON", _json(metadata), f"historical_watchlist_{as_of}.json", "application/json", key="bridge_watchlist_json")
    st.caption("代码 TXT 可用于人工维护模拟账户的自选观察列表；未声称同花顺支持本文件一键导入。CSV 中代码应按文本列打开。")
    st.caption("回测 units 为复权单位，不能当作交易股数。本文件只含历史权重、证券代码与日期，不包含下单数量。")


def _export_history(root: Path):
    records = _historical_records(str(root))
    if records:
        by_id = {record.id: record for record in records}
        chosen = st.selectbox("选择历史 V3 实验", list(by_id),
                              format_func=lambda key: f"{by_id[key].label} · {by_id[key].start} 至 {by_id[key].end}",
                              key="bridge_history_experiment")
        record = by_id[chosen]
        dates, _ = read_ledger(root, record, "daily")
        if dates.empty:
            st.info("该实验没有可用的日期账本。")
            return
        options = dates.date.dt.strftime("%Y-%m-%d").drop_duplicates().tolist()
        selected_date = st.selectbox("历史持仓日期", options, index=len(options)-1, key="bridge_history_date")
        watch, metadata = historical_watchlist(root, record, selected_date)
        _watchlist_downloads(watch, metadata)
        return
    csv = root / "evidence/connections/historical_watchlist_20260918.csv"
    meta = root / "evidence/connections/historical_watchlist_20260918.json"
    if csv.exists() and meta.exists():
        st.caption("本机缺少逐股账本，下面显示发布包内固定的历史观察截面；不能选择其他日期。")
        _watchlist_downloads(pd.read_csv(csv, dtype={"code": str}), json.loads(meta.read_text(encoding="utf-8")))
    else:
        st.info("本机未找到 V3 逐股持仓。完成本地回测后，可在这里导出历史观察组合；成交导入仍然可用。")


def _import_receipt(root: Path):
    mode = st.radio("成交文件来源", ["合成示例 · 离线演示", "上传模拟成交 CSV"], horizontal=True, key="bridge_source")
    synthetic = mode.startswith("合成")
    if synthetic:
        st.info("当前是 3 笔人工构造的教学成交，证券名称与价格为示例，不来自同花顺账号，也不是策略业绩。")
        raw = synthetic_example_csv()
        st.download_button("下载合成示例 CSV", raw, "synthetic_simulation_trades.csv", "text/csv", key="bridge_example_csv")
    else:
        st.caption("从同花顺/SuperMind 的模拟账户导出成交明细，上传 CSV。文件在本地验证，不向第三方上传。")
        upload = st.file_uploader("模拟账户成交 CSV", type=["csv", "txt"], key="bridge_upload")
        if upload is None:
            st.download_button("下载字段模板（含合成示例）", synthetic_example_csv(), "synthetic_simulation_template.csv", "text/csv", key="bridge_template_csv")
            return
        raw = upload.getvalue()
    tolerance = st.number_input("金额核对允差（元）", min_value=0.0, max_value=1.0, value=.05, step=.01,
                                 help="逐笔比较成交金额与成交价格 × 整数股数；默认允许 0.05 元舍入差。", key="bridge_amount_tolerance")
    result = parse_simulated_trades(raw, source_kind="synthetic_demo" if synthetic else "uploaded_simulation", amount_tolerance=tolerance)
    summary = validation_summary(result)
    a, b, c, d = st.columns(4)
    a.metric("读入成交", summary["rows"])
    b.metric("需要修正", summary["error_count"])
    c.metric("提示", summary["warning_count"])
    d.metric("疑似重复行", summary["duplicate_rows"])
    st.caption(f"识别编码：{result.encoding} · 成交数量按整数股校验 · 逐笔金额允差：{tolerance:.2f} 元")
    if not result.issues.empty:
        display = result.issues.copy()
        display.severity = display.severity.map({"error": "错误", "warning": "提示"})
        st.dataframe(display.rename(columns={"source_row": "数据行", "severity": "级别", "field": "字段", "message": "说明"}),
                     hide_index=True, width="stretch")
    if result.duplicate_rows:
        st.warning("疑似重复成交全部保留，并计入下面的金额。请先与原模拟平台核对，不能把重复行自动当作错误删除。")
    if result.valid:
        st.success("本地文件校验通过。该状态说明格式与逐笔金额一致，不代表账号已连接或来源已通过平台认证。")
    else:
        st.error("文件存在错误，请修正原 CSV 后重新上传。当前不能生成已验证回执或成交金额对账。")
    display_columns = ["source_row", "timestamp", "code", "exchange", "name", "side", "price", "quantity", "amount", "amount_difference", "duplicate_flag"]
    st.dataframe(result.normalized[display_columns], hide_index=True, width="stretch")
    with st.expander("识别到的字段对应关系"):
        st.json(result.column_mapping)
    if not result.valid:
        st.download_button("下载错误清单 CSV", safe_csv(result.issues), "simulation_import_issues.csv", "text/csv", key="bridge_issues_csv")
        return
    aggregate = reconcile_by_security(result)
    st.markdown("**按证券核对买卖金额**")
    st.caption("净现金流 = 卖出金额 − 买入金额，未扣手续费；数量为本文件中的净成交变化。两者都不是投资收益或期末持仓。")
    plot = aggregate.melt(id_vars=["asset"], value_vars=["buy_amount", "sell_amount"], var_name="direction", value_name="amount")
    plot.direction = plot.direction.map({"buy_amount": "买入金额", "sell_amount": "卖出金额"})
    fig = px.bar(plot, x="asset", y="amount", color="direction", barmode="group",
                 labels={"asset": "证券", "amount": "成交金额（元）", "direction": "方向"},
                 color_discrete_map={"买入金额": "#168579", "卖出金额": "#4265a6"})
    fig.update_layout(height=290, margin=dict(t=20, b=0), legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(aggregate.rename(columns={"buy_amount": "买入金额", "sell_amount": "卖出金额",
                                          "gross_cash_flow": "未扣费净现金流", "net_quantity_change": "净成交数量变化"}),
                 hide_index=True, width="stretch")
    prefix = "synthetic" if synthetic else "simulation"
    a, b, c = st.columns(3)
    a.download_button("下载规范化成交 CSV", safe_csv(result.normalized), f"{prefix}_normalized.csv", "text/csv", key="bridge_normalized_csv")
    b.download_button("下载证券金额对账 CSV", safe_csv(aggregate), f"{prefix}_reconciliation.csv", "text/csv", key="bridge_reconciled_csv")
    c.download_button("下载校验摘要 JSON", _json(summary), f"{prefix}_validation.json", "application/json", key="bridge_validation_json")
    with st.form("bridge_save_receipt"):
        label = st.text_input("本地回执名称", "合成演示回执" if synthetic else "模拟账户成交回执", max_chars=160)
        save = st.form_submit_button("保存已验证模拟回执", type="primary")
    if save:
        path, created = save_simulated_receipt(root, result, label=label)
        st.success(("已保存：" if created else "相同文件已保存，未重复写入：") + path.relative_to(root).as_posix())
    st.caption("回执仅保存在本机 runs/paper_imports；不提交买卖、不改变模拟账户、不计算缺少账户依据的收益。")


def _guide(root: Path):
    st.markdown("""1. **准备观察组合。** 在“历史组合导出”选择回测日期，下载权重表或代码清单，用于研究与人工维护观察列表。
2. **在官方模拟环境操作。** 打开同花顺模拟炒股或 SuperMind。账号登录、权限开通和模拟交易在官方环境中完成。
3. **导出成交明细。** SuperMind 官方帮助说明：在模拟交易的交易明细中选择日期范围并导出文件。若导出 Excel，可在表格软件中另存为 CSV。
4. **带回平台核对。** 上传模拟成交 CSV，检查金额与重复提示，导出规范化文件并保存回执。""")
    for title, url in OFFICIAL_SOURCES.items():
        st.markdown(f"[{title}]({url})")
    st.markdown("**当前边界**")
    st.write("普通同花顺模拟炒股的公开自动交易 API 尚未核实。SuperMind 官方文档介绍 Python 模拟策略，但本项目未登录或开通该环境，未完成接口握手。当前交付的是可使用的文件桥。")
    st.write("后续账户级验证需要实际模拟回执、初始资金、初始持仓、费用和每日资产记录；本模块不会仅凭买卖差额宣称收益。")
    guide = root / "docs/THS_CONNECTION.md"
    if guide.exists():
        st.download_button("下载同花顺衔接说明", guide.read_bytes(), "THS_CONNECTION.md", "text/markdown", key="bridge_guide")


def render(root: Path):
    root = Path(root).resolve()
    from .ui_design import heading
    heading("模拟交易衔接", "把研究组合带到模拟环境，再把成交回执带回平台核对。当前采用本地文件方式。")
    a, b, c = st.columns(3)
    a.metric("本地文件桥", "可用")
    b.metric("同花顺账号", "未连接")
    c.metric("本模块提交订单", "0")
    with st.expander("查看各项连接状态", expanded=False):
        modes = pd.DataFrame(connection_status()["modes"]).rename(columns={"capability": "能力", "status": "当前状态", "detail": "说明"})
        st.dataframe(modes, hide_index=True, width="stretch")
    exports, imports, guide = st.tabs(["历史组合导出", "模拟回执核对", "官方入口与步骤"])
    with exports:
        try:
            _export_history(root)
        except (ValueError, OSError, KeyError) as exc:
            st.error(f"历史观察组合暂时无法读取：{exc}")
    with imports:
        try:
            _import_receipt(root)
        except (ValueError, OSError, KeyError) as exc:
            st.error(f"成交文件暂时无法处理：{exc}")
    with guide:
        _guide(root)
