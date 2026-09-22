"""Current benchmark study; outcome qualification is separate from accounting."""
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from .benchmark_research import NAMES


def render(root:Path):
    folder=root/'evidence/research_v3'
    load=lambda n:json.loads((folder/n).read_text(encoding='utf-8'))
    decision=load('decision.json');m=load('metrics.json')
    st.subheader('V3 · 年度滚动 LightGBM + 波动预算')
    st.caption('2025-01-02 至 2026-09-18 · 417个交易日 · 买入10bp / 卖出15bp · 固定1000股历史池')
    cols=st.columns(5)
    for c,label,value in zip(cols,['累计净收益','年化净收益','超沪深300（百分点）','最大回撤','年化 Sharpe'],
         [f"{m['total_return']:.2%}",f"{m['annualized_return']:.2%}",f"{100*m['excess_return_pp']:+.2f}",f"{m['max_drawdown']:.2%}",f"{m['sharpe']:.2f}"]):c.metric(label,value)
    st.success(f"历史目标达标：沪深300价格指数 {m['benchmark_return']:.2%}；全收益指数 {m['total_return_benchmark']:.2%}；策略扣费后分别超过 {100*m['excess_return_pp']:.2f} / {100*m['excess_vs_total_return']:.2f} 个百分点。")
    st.warning('候选选择边界：V3验证期排名第一的价值质量组合在随后区间失败；当前LightGBM是查看本轮历史复核后选出的合格候选，不属于独立盲测成功。完整8个候选和旧版均保留。')
    tabs=st.tabs(['最新策略','候选与版本','压力与年度','训练时点','材料与复现'])
    with tabs[0]:
        curves=pd.read_csv(folder/'curves.csv',parse_dates=['date'])
        fig=px.line(curves,x='date',y='nav',color='series',labels={'date':'日期','nav':'净值','series':'曲线'},
                    color_discrete_sequence=['#168579','#4265a6','#965eb0','#be8842','#89949f'])
        fig.update_layout(height=440,legend=dict(orientation='h',y=1.2),margin=dict(t=70,b=0))
        st.plotly_chart(fig,width='stretch')
        st.write('12个经济因子同时保留标准化与行业/市值中性化表示，24个模型输入。每年使用此前3年已实现标签重训；50只目标持仓、20日调仓、20名缓冲、20%波动预算、最高95%股票仓位；单股/行业目标上限4%/25%。')
        st.caption(f"平均现金 {m['mean_cash_weight']:.2%}；风险约束为形成时目标，不能保证未来回撤。ETF复权曲线为未扣交易成本代理，全收益指数另列。")
        st.info('要切换并重跑任何版本，进入左侧“回测实验” → “选择策略版本”。该入口现在默认最新合格候选。')
    with tabs[1]:
        for name,label in [('validation.csv','2023–2024验证：先记录选择'),('test.csv','2025–2026历史复核：全候选保留')]:
            st.markdown('**'+label+'**')
            t=pd.read_csv(folder/name)[['candidate','total_return','excess_return_pp','max_drawdown','worst_year_excess']]
            t.columns=['候选','净收益','超额收益','最大回撤','最差年度超额']
            st.dataframe(t.style.format({c:'{:.2%}' for c in t.columns[1:]}),hide_index=True,width='stretch')
        st.caption('预选：economic__steady；本次开发候选：'+decision['selected']+'。稳定仓位LightGBM验证期回撤超过25%，未作为默认策略。')
    with tabs[2]:
        stress=pd.read_csv(folder/'stress.csv')[['experiment','total_return','benchmark_return','total_return_benchmark','max_drawdown','excess_vs_total_return']]
        stress.columns=['实验','策略收益','价格指数','全收益指数','最大回撤','超全收益指数']
        st.dataframe(stress.style.format({c:'{:.2%}' for c in stress.columns[1:]}),hide_index=True,width='stretch')
        st.caption('前三行同为2025–2026；连续运行行从2023开始，不能将不同区间收益直接相减。')
        st.markdown('**2023起连续运行，年度收益与同期指数**')
        years=pd.read_csv(folder/'continuous_years.csv')
        st.dataframe(years.style.format({c:'{:.2%}' for c in ['strategy_return','benchmark_return','excess_return_pp']}),hide_index=True,width='stretch')
        st.caption('不要求每年都战胜指数；某些阶段会落后，完整年份不隐藏。')
    with tabs[3]:
        folds=load('folds.json')
        st.dataframe(pd.DataFrame([{k:f[k] for k in ['year','cutoff','train_rows','max_training_label_exit','prediction_end_exclusive']} for f in folds]),hide_index=True,width='stretch')
        st.write('标签：次日开盘至20个交易日后开盘收益的当日横截面排名。训练标签退出日必须严格早于新模型形成日；未来标签扰动测试验证该边界。')
    with tabs[4]:
        st.code('python scripts/run_benchmark_research.py\npython scripts/qualify_benchmark_strategy.py\npython -m pytest -q',language='bash')
        st.json(load('acceptance.json'),expanded=False)
        enhanced=(root/'reports/platform_v4/CF2026_V4_Final_Report.pdf').exists()
        for suffix,label in [('Final_Report.pdf','下载最新报告'),('Beamer.pdf','下载最新演示')]:
            file=('CF2026_V4_' if enhanced else 'CF2026_V3_')+suffix
            p=root/('reports/platform_v4' if enhanced else 'reports/benchmark_v3')/file
            if p.exists():st.download_button(label,p.read_bytes(),file,'application/pdf')
        st.caption('材料与答辩页面另有可编辑 PowerPoint、逐页讲稿和现场演示步骤。策略研究仍为V3，平台增强版材料为V4。')
