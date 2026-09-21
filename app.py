"""Thin Streamlit adapter: all numerical work lives in cfquant."""
from dataclasses import replace
from pathlib import Path
import json
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
st.markdown("""<style>
.stApp {background:#f5f7fb;}
[data-testid="stSidebar"] {background:#12243a;}
[data-testid="stSidebar"] * {color:#e8eef6;}
[data-testid="stSidebar"] input {color:#172b45;}
[data-testid="stSidebar"] [data-baseweb="select"] * {color:#172b45;}
h1,h2,h3 {color:#12243a;letter-spacing:-.025em;}
[data-testid="stMetric"] {background:white;border:1px solid #e0e7ef;border-radius:10px;padding:16px;}
.hero-label {color:#168579;font-size:13px;letter-spacing:3px;font-weight:700;}
.hero-copy {color:#62748a;font-size:16px;margin-bottom:22px;}
</style>""", unsafe_allow_html=True)

real = (ROOT/"data/processed/market.csv").exists()
datasets = {"课程原始快照 · 60只 · 2023–2025": "configs/baseline.yaml"} if real else {
    "合成演示样本": "configs/demo.yaml"}
expanded_config = ROOT/"configs/expanded.yaml"
if expanded_config.exists():
    expanded = Config.load(expanded_config)
    status_path = (ROOT/expanded.data_path).parent/"progress.json"
    if status_path.exists() and json.loads(status_path.read_text(encoding="utf-8")).get("status") == "complete":
        datasets["扩展快照 · 1000只 · 2020–2026"] = "configs/expanded.yaml"
selected = st.sidebar.selectbox("数据集", list(datasets), key="dataset")
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
    st.markdown("## 青序 QUANT")
    st.caption("COMPUTATIONAL FINANCE · 2026")
    page = st.radio("研究空间", ["研究总览", "回测实验", "因子诊断", "数据与复现", "扩展指南"])
    st.divider()
    st.caption("真实 Tushare 快照" if real else "合成演示数据 · 非实证结果")
    st.caption(f"{market.asset.nunique()} 个资产 · {len(market):,} 条日频记录")
    st.caption("本地运行 · 不连接交易账户")

st.markdown('<div class="hero-label">QINGXU RESEARCH / PROJECT 01</div>', unsafe_allow_html=True)


def line_chart(frame, x, y, **kwargs):
    fig = px.line(frame, x=x, y=y, color_discrete_sequence=["#158779", "#4265a6", "#d68a40", "#8f629c"], **kwargs)
    fig.update_layout(template="plotly_white", margin=dict(l=10,r=10,t=35,b=10), height=390,
                      paper_bgcolor="rgba(0,0,0,0)", legend_title_text="")
    st.plotly_chart(fig, width="stretch")


if page == "研究总览":
    st.title("让每一次研究，都能被复现")
    st.markdown('<div class="hero-copy">从行情质量到成交账本，把研究结论建立在可检查的计算上。</div>', unsafe_allow_html=True)
    cols=st.columns(4)
    for col,label,value in zip(cols,["数据来源","研究资产","基础因子","执行时点"],
                               ["Tushare Pro" if real else "合成演示",str(market.asset.nunique()),"3","次日开盘"]):
        col.metric(label,value)
    st.markdown("### 一条完整的研究链路")
    st.info("数据快照  →  因子与诊断  →  目标组合  →  成交与费用  →  净值与解释")
    comparison=ROOT/"runs/study/comparison.csv"
    if original_study and comparison.exists():
        comp=pd.read_csv(comparison)
        curves=[]
        index_path=ROOT/"runs/study/run_index.json"
        run_index=json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
        display_names={"momentum_weekly":"动量 · 5日","reversal_weekly":"反转 · 5日",
                       "low_volatility_weekly":"低波动 · 5日","momentum_monthly":"动量 · 20日"}
        for name in comp.experiment:
            matches=[ROOT/run_index[name]/"daily.csv"] if name in run_index else sorted((ROOT/"runs").glob(f"{name}_*/daily.csv"))
            if matches:
                d=pd.read_csv(matches[-1],parse_dates=["date"])
                curves.append(pd.DataFrame({"日期":d.date,"净值":d.nav/base.initial_cash,"实验":display_names.get(name,name)}))
        if curves:line_chart(pd.concat(curves), "日期","净值",color="实验",title="固定研究方案 · 扣费后历史净值")
        st.caption("以上为固定股票池上的描述性历史实验。负收益结果原样保留，不代表未来表现。")
        table=comp[["experiment","total_return","sharpe","max_drawdown","total_cost"]].copy()
        table["experiment"]=table.experiment.map(display_names)
        table["total_return"]=table.total_return.map(lambda x:f"{x:.2%}")
        table["max_drawdown"]=table.max_drawdown.map(lambda x:f"{x:.2%}")
        table["sharpe"]=table.sharpe.map(lambda x:f"{x:.3f}")
        table["total_cost"]=table.total_cost.map(lambda x:f"{x:,.2f}")
        table.columns=["实验","累计净收益","Sharpe","最大回撤","费用（元）"]
        st.table(table.set_index("实验"))
    else:
        st.info("进入「回测实验」运行一次完整研究；无 Token 的新环境也可使用仓库内的合成样本。")
    st.markdown("### 先理解三个研究假设")
    for col,spec in zip(st.columns(3),REGISTRY.values()):
        with col:
            st.markdown(f"**{spec.label}**")
            st.write(spec.hypothesis)
            st.code(spec.formula,language=None)

elif page == "回测实验":
    st.title("配置一次可核验的回测")
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
        st.success(f"账本核对通过 · {used['factor']} · {used['start']} 至 {used['end']} · 每 {used['rebalance_every']} 个交易日调仓")
        cols=st.columns(4)
        cols[0].metric("累计净收益",f"{metrics['total_return']:.2%}")
        cols[1].metric("最大回撤",f"{metrics['max_drawdown']:.2%}")
        cols[2].metric("年化 Sharpe",f"{metrics['sharpe']:.3f}")
        cols[3].metric("累计成本（元）",f"{metrics['total_cost']:,.2f}")
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
