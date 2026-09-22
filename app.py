"""Thin Streamlit adapter: all numerical work lives in cfquant."""
from dataclasses import replace
from pathlib import Path
import json
import math
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from cfquant.config import Config
from cfquant.data import load_market, load_calendar, digest
from cfquant.factors import REGISTRY, compute
from cfquant.analytics import diagnostics, forward_returns
from cfquant.experiment import execute

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="青序 · 量化研究工作台", page_icon="📈", layout="wide")
from cfquant.ui_design import install
install()
with st.sidebar:
    st.markdown('<div class="qx-brand">青序 QUANT</div><div class="qx-subbrand">RESEARCH WORKBENCH · 2026</div>',unsafe_allow_html=True)

real = (ROOT/"data/processed/market.csv").exists()
datasets = {"课程原始快照 · 60只 · 2023–2025": "configs/baseline.yaml"} if real else {
    "合成演示样本": "configs/demo.yaml"}
expanded_config = ROOT/"configs/expanded.yaml"
if expanded_config.exists():
    expanded = Config.load(expanded_config)
    status_path = (ROOT/expanded.data_path).parent/"progress.json"
    if status_path.exists() and json.loads(status_path.read_text(encoding="utf-8")).get("status") == "complete":
        datasets["扩展快照 · 1000只 · 2020–2026"] = "configs/expanded.yaml"
page = st.sidebar.radio("研究空间", ["研究总览", "策略对比", "风险透镜", "回测实验", "因子诊断", "实验档案", "行情探索", "Qlib 接入", "模拟交易连接", "策略研究", "材料与答辩", "数据与复现", "扩展指南"], key="workspace_page")
# Aggregate research pages do not load 1.65 million price rows on every cold start.
fast_pages = {"研究总览", "策略对比", "风险透镜", "实验档案", "Qlib 接入", "模拟交易连接", "材料与答辩"}
if page in fast_pages:
    st.sidebar.caption("平台 v2.2.2 · 策略研究 V3")
    st.sidebar.caption("历史数据截至 2026-09-18")
    st.sidebar.caption("聚合研究可离线查看；逐股明细需本地快照。")
    if page in {"研究总览", "策略对比", "风险透镜"}:
        from cfquant.ui_workspace import overview, comparison, risk
        {"研究总览":overview,"策略对比":comparison,"风险透镜":risk}[page](ROOT)
    elif page == "实验档案":
        from cfquant.ui_experiments import render
        render(ROOT)
    elif page == "Qlib 接入":
        from cfquant.ui_qlib import render
        render(ROOT)
    elif page == "模拟交易连接":
        from cfquant.ui_connections import render
        render(ROOT)
    else:
        from cfquant.ui_materials import render
        render(ROOT)
    st.stop()
if page == "因子诊断":
    from cfquant.ui_explorers import factors
    mode=st.radio("诊断模式",["12因子研究","基础因子自由诊断"],horizontal=True,key="factor_mode")
    if mode=="12因子研究":
        factors(ROOT)
        st.stop()
if page == "策略研究":
    selected = list(datasets)[-1]
    st.sidebar.caption("策略研究使用固定的1000只历史股票池；其他页面可切换数据集。")
else:
    selected = st.sidebar.selectbox("数据集（模型策略使用其固定快照）", list(datasets), index=len(datasets)-1, key="dataset")
base = Config.load(ROOT/datasets[selected])
real = base.data_path != "data/sample/market.csv"
original_study = datasets[selected] == "configs/baseline.yaml"
if st.session_state.get("active_dataset") != selected:
    st.session_state.pop("run", None)
    st.session_state["active_dataset"] = selected


@st.cache_data
def read_data(path, calendar_path, file_hash, calendar_hash):
    return load_market(path), load_calendar(calendar_path)


@st.cache_data
def calculate(config_dict, file_hash, calendar_hash):
    return execute(ROOT, Config(**config_dict))


market, calendar = read_data(ROOT/base.data_path, ROOT/base.calendar_path,
                             digest(ROOT/base.data_path), digest(ROOT/base.calendar_path))
with st.sidebar:
    st.caption("真实 Tushare 快照" if real else "合成演示数据 · 非实证结果")
    st.caption(f"{market.asset.nunique()} 个资产 · {len(market):,} 条日频记录")
    st.caption("本地运行 · 不连接交易账户")




def line_chart(frame, x, y, **kwargs):
    fig = px.line(frame, x=x, y=y, color_discrete_sequence=["#158779", "#4265a6", "#d68a40", "#8f629c"], **kwargs)
    fig.update_layout(template="plotly_white", margin=dict(l=10,r=10,t=35,b=10), height=390,
                      paper_bgcolor="rgba(0,0,0,0)", legend_title_text="")
    st.plotly_chart(fig, width="stretch")


if page == "行情探索":
    from cfquant.ui_explorers import market_explorer
    market_explorer(market,calendar,real)
elif page == "策略研究":
    from cfquant.ui_research import render
    render(ROOT)
elif page == "回测实验":
    st.title("策略版本与回测")
    from cfquant.strategies import catalogue, outcome_status
    versions=catalogue(ROOT)
    ids=[v.id for v in versions]+['legacy_single_factor']
    labels={v.id:v.label+' · '+v.status for v in versions}
    labels['legacy_single_factor']='V1 · 单因子自由实验 · 历史失败对照'
    version=st.selectbox('选择策略版本',ids,format_func=labels.get,key='strategy_version')
    if st.session_state.get('active_strategy_version') != version:
        st.session_state.pop('run',None)
        st.session_state.pop('version_run',None)
        st.session_state['active_strategy_version']=version
    if version!='legacy_single_factor':
        from cfquant.ui_strategy_backtest import render as render_version
        render_version(ROOT,next(v for v in versions if v.id==version))
        st.stop()
    st.warning('历史失败实验：原始动量在1000股长区间亏损83.57%，不满足当前策略目标。此入口供课程对照与研究，不是推荐默认策略。')
    st.caption("参数提交后统一计算。研究结果与成交明细会保存到本地 runs 目录。")
    with st.form("experiment"):
        a,b,c=st.columns(3)
        factor=a.selectbox("因子",list(REGISTRY),key="backtest_factor",format_func=lambda x:{"momentum":"动量","reversal":"反转","low_volatility":"低波动"}.get(x,REGISTRY[x].label))
        window=b.number_input("因子窗口（0 使用各因子默认值）",0,120,0,
                              help="默认：动量20日、反转5日、低波动20日；自定义窗口至少2日。")
        freq=c.selectbox("调仓间隔（交易日）",[5,20,1,10])
        a,b,c=st.columns(3)
        start=a.date_input("开始日期",pd.Timestamp(base.start).date(),min_value=pd.Timestamp(base.start).date(),max_value=calendar[-1].date())
        end=b.date_input("结束日期",pd.Timestamp(base.end).date(),min_value=calendar[1].date(),max_value=calendar[-1].date())
        holdings=c.number_input("目标持仓数",1,market.asset.nunique(),min(base.holdings,market.asset.nunique()))
        a,b,c=st.columns(3)
        buy=a.number_input("买入成本（基点，1 bp = 0.01%）",0.,200.,base.buy_cost*10000,1.)
        sell=b.number_input("卖出成本（基点）",0.,200.,base.sell_cost*10000,1.)
        initial=c.number_input("初始资金（元）",1000.,100000000.,base.initial_cash,10000.)
        submitted=st.form_submit_button("运行并核对账本",type="primary",width="stretch")
    if submitted:
        try:
            cfg=replace(base,name="interactive",factor=factor,factor_params={"window":int(window) if window else REGISTRY[factor].default_window},
                        start=str(start),end=str(end),rebalance_every=freq,holdings=int(holdings),
                        buy_cost=buy/10000,sell_cost=sell/10000,initial_cash=initial)
            with st.spinner("计算交易、费用、净值并独立核对账本…"):
                result,metrics,folder=calculate(cfg.to_dict(),digest(ROOT/base.data_path),digest(ROOT/base.calendar_path))
            st.session_state["run"]=(result,metrics,str(folder),cfg.to_dict())
        except (ValueError,AssertionError) as exc:st.error(str(exc))
    if "run" in st.session_state:
        result,metrics,folder,used=st.session_state["run"]
        st.info(f"账本检查通过（仅核对资金与交易） · {used['factor']} · {used['start']} 至 {used['end']} · 每 {used['rebalance_every']} 个交易日调仓")
        passed,reasons=outcome_status(metrics)
        if not passed:st.error('策略验收未通过：'+'；'.join(reasons)+'。')
        else:st.warning('单因子实验完成；尚未核验同期指数超额，不作为合格推荐。')
        cols=st.columns(4)
        cols[0].metric("累计净收益",f"{metrics['total_return']:.2%}")
        cols[1].metric("最大回撤",f"{metrics['max_drawdown']:.2%}")
        sharpe=metrics.get('sharpe')
        cols[2].metric("年化夏普比率",f'{sharpe:.3f}' if sharpe is not None and math.isfinite(sharpe) else '—')
        cols[3].metric("累计成本（元）",f"{metrics['total_cost']:,.2f}")
        st.caption(f"本次结果的夏普口径：扣费日收益 · 年化无风险利率 {used['risk_free_annual']:.2%} · 每年 {used['annualization']} 个交易日 · 样本标准差。样本不足或波动接近零时显示“—”。")
        d=result.daily.copy();d["扣费净值"]=d.nav/used["initial_cash"];d["同成交路径毛归因"]=d.gross_attribution_nav/used["initial_cash"]
        line_chart(d,"date",["扣费净值","同成交路径毛归因"])
        st.caption("毛归因仅把已付费用放回不计息现金，不重新投资，也不是无成本策略的独立回测。")
        tab1,tab2,tab3=st.tabs(["逐日账本","成交与未成交","持仓"])
        with tab1:st.dataframe(result.daily,hide_index=True,width="stretch")
        with tab2:
            st.dataframe(result.trades,hide_index=True,width="stretch")
            st.markdown("**未成交 / 现金约束**")
            st.dataframe(result.orders[result.orders.status!="filled"],hide_index=True,width="stretch")
        with tab3:st.dataframe(result.positions,hide_index=True,width="stretch")
        st.download_button("下载逐日账本 CSV",result.daily.to_csv(index=False).encode("utf-8-sig"),"daily.csv","text/csv")
        st.caption(f"本地实验目录：{folder}")
    with st.expander("交易与估值口径"):
        st.write("仅做多、连续可分的复权总收益单位。先卖后买，费用按实际成交额扣除；买入资金不足时等比例缩减。开盘涨停不买入，开盘跌停不卖出，缺失报价或限制价格时不成交。")
        st.write("短期缺失收盘价按最后有效收盘价估值；连续超过20个交易日缺失后保守计零，保留单位及记录，价格恢复时重新估值。不是确认退市清偿价值。未建模整手、最低佣金、冲击成本及订单排队。")

elif page == "因子诊断":
    st.title("因子是否包含排序信息")
    st.caption("逐日横截面相关性与分组收益，独立于可执行组合净值。")
    a,b=st.columns([2,1])
    factor=a.selectbox("诊断因子",list(REGISTRY),format_func=lambda x:REGISTRY[x].label)
    horizon=b.selectbox("未来持有期（交易日）",[5,1,10,20])
    spec=REGISTRY[factor]
    st.info(f"{spec.hypothesis}  可能失效：{spec.failure_mode}")
    analysis_calendar=calendar[calendar<=base.end]
    analysis_market=market[market.date<=base.end]
    scores=compute(analysis_market,analysis_calendar,factor).loc[base.start:base.end]
    labels=forward_returns(analysis_market,analysis_calendar,horizon)
    diag=diagnostics(scores,labels,base.groups,base.min_assets)
    summary=diag["summary"]; cols=st.columns(4)
    cols[0].metric("IC 均值",f"{summary['ic_mean']:.4f}")
    cols[1].metric("Rank IC 均值",f"{summary['rank_ic_mean']:.4f}")
    cols[2].metric("因子覆盖率",f"{summary['mean_factor_coverage']:.2%}")
    cols[3].metric("有效 IC 日数",str(summary["ic_valid_days"]))
    d=diag["daily"].copy()
    d["Rank IC（20日均值）"]=d.rank_ic.rolling(20,min_periods=10).mean()
    line_chart(d,"date","Rank IC（20日均值）")
    means=diag["groups"].groupby("group").mean_forward_return.mean().reset_index()
    fig=px.bar(means,x="group",y="mean_forward_return",color_discrete_sequence=["#168579"],
               labels={"group":"从低分到高分","mean_forward_return":f"未来{horizon}日平均收益"})
    fig.update_layout(template="plotly_white",paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,width="stretch")
    st.caption("形成日先分组，再匹配标签；并列按资产代码稳定排序，缺失标签不重新分组。组间价差未扣费，重叠标签不能直接连乘为净值。")
    st.dataframe(diag["daily"],hide_index=True,width="stretch")
    st.download_button("下载因子诊断 CSV",diag["daily"].to_csv(index=False).encode("utf-8-sig"),f"{factor}_diagnostics.csv","text/csv")

elif page == "数据与复现":
    st.title("每个结果都有来源")
    st.write("当前数据：**真实 Tushare 本地快照**" if real else "当前数据：**固定种子合成样本，只用于演示和工程验证**")
    st.code(digest(ROOT/base.data_path),language=None)
    st.caption("当前 market.csv 的 SHA-256 校验值")
    if real:
        dataset_dir=(ROOT/base.data_path).parent
        quality=json.loads((dataset_dir/"quality.json").read_text(encoding="utf-8"))
        manifest=json.loads((dataset_dir/"manifest.json").read_text(encoding="utf-8"))
        st.write(f"行情覆盖：{market.date.min().date()} 至 {market.date.max().date()}；{len(calendar):,} 个交易日。")
        st.caption(f"股票池选取日：{manifest['selection_date']}。固定历史股票池，不是全市场或动态指数成分。")
        st.dataframe(pd.DataFrame(quality["assets"]),hide_index=True,width="stretch")
        st.json({k:v for k,v in quality.items() if k!="assets"},expanded=False)
    else:
        st.info("要复现实证报告，请使用自己的数据权限执行下载命令；样本演示的收益不能替代真实数据实验。")
    st.markdown("**命令行入口**")
    st.code(f'python -m cfquant.cli run --config {datasets[selected]}\npython -m pytest -q',language="bash")
    st.write("报告和使用指南位于 reports 与 docs。真实行情、凭据、缓存和运行目录默认不进入 Git；本机保留完整离线快照。")
    st.dataframe(market.head(100),hide_index=True,width="stretch")

else:
    st.title("为下一次 Project 留好接口")
    st.write("修改因子不需要修改成交核心。将新函数注册为 FactorSpec，即可复用相同的诊断与回测路径。")
    st.code('from cfquant.factors import FactorSpec, register\n\ndef my_factor(close, window):\n    return -(close.rolling(window).max() / close.rolling(window).min() - 1)\n\nregister(FactorSpec("range", "区间收敛", "range ratio",\n                    "小振幅可能反映稳定状态", 20,\n                    "趋势阶段可能失效", my_factor))',language="python")
    st.write("P2：在现有快照和实验记录上复现论文。P3：新增因子并验证因果性。P4：将模型预测转换成相同的日期×资产分数，训练/验证/测试严格按时间切分。P5：扩展组合权重与执行规则。")
    st.warning("前缀一致性与未来扰动测试能发现一类信息泄漏，但不能证明任意插件绝对安全。训练模型还需检查标签跨分割边界、预处理拟合范围及样本外选择。")
    st.markdown("**论文参考的取舍**")
    st.write("CogAlpha 提供因子代码生成与进化思路；本项目借鉴可解释因子卡、质量检查和时间泄漏测试，不声称复现其多智能体系统或论文收益。")
