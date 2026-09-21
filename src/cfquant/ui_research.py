"""Research dashboard uses persisted evidence; never silently retrains models."""
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

NAMES={'multifactor_raw':'标准化多因子','multifactor_neutral':'中性化多因子','ridge':'线性 Ridge','lightgbm':'LightGBM','mlp':'小型神经网络'}


def render(root:Path):
    folder=root/'evidence/research_v2'
    st.title('策略研究与验证')
    st.caption('固定历史股票池 · 先验证选择，再评估结果 · 所有收益均明确费用与区间')
    if not (folder/'selection.json').exists():
        st.info('升级研究正在运行，已有单因子实验可在其他页面查看。')
        return
    selection=json.loads((folder/'selection.json').read_text(encoding='utf-8'))
    selected=selection['selected']
    validation=pd.read_csv(folder/'validation.csv');test=pd.read_csv(folder/'test.csv')
    row=test[test.model==selected].iloc[0]
    tabs=st.tabs(['提交策略','模型对照','因子诊断','风险与失败分析','复现与验收'])
    with tabs[0]:
        st.subheader(NAMES.get(selected,selected)+' + 风险约束')
        st.write('训练：2020–2022　｜　验证选择：2023–2024　｜　最终评估：2025-01-02 至 2026-09-18')
        cols=st.columns(4)
        for c,label,value in zip(cols,['累计净收益','年化净收益','最大回撤','年化 Sharpe'],
                                 [f'{row.total_return:.2%}',f'{row.annualized_return:.2%}',f'{row.max_drawdown:.2%}',f'{row.sharpe:.2f}']):
            c.metric(label,value)
        curves=pd.read_csv(folder/'curves.csv',parse_dates=['date'])
        chart=px.line(curves,x='date',y='nav',color='series',labels={'date':'日期','nav':'起点净值 = 1','series':'曲线'})
        chart.update_layout(template='plotly_white',height=430,legend=dict(orientation='h',y=1.12),margin=dict(l=0,r=10,b=0,t=30))
        st.plotly_chart(chart,width='stretch')
        if pd.notna(row.get('cpi_real_return')):
            st.info(f"购买力比较：{int(row.cpi_start_month)}–{int(row.cpi_end_month)} 的共同完整月份内，策略 {row.cpi_strategy_return:.2%}，CPI 累计 {row.cpi_inflation:.2%}，实际收益 {row.cpi_real_return:.2%}。")
        st.write('组合规则：20 个交易日调仓、50 个目标持仓、持仓缓冲；单股与行业目标权重上限；12% 波动率目标与市场趋势仓位，最高股票敞口90%，其余现金按零利息计算。')
        st.caption('权重与波动率为构建目标，市场变化后实际权重会漂移。原动量全期结果已被观察过，本次最终区间不属于完全未知的盲测。')
        st.caption('沪深300曲线为未扣交易成本的价格指数：用于说明市场环境，不能当作同等可交易的含分红收益基准。')
        for name,label in [('CF2026_Final_Report.pdf','下载最终报告'),('CF2026_Beamer.pdf','下载演示 PDF')]:
            path=root/'reports/final'/name
            if path.exists():st.download_button(label,path.read_bytes(),name,'application/pdf',key=name)
    with tabs[1]:
        st.write('最终提交策略只依据验证期规则选择，不能用右表的最终结果重新选冠军。')
        for label,frame in [('验证期 · 用于选择',validation),('最终评估期 · 用于报告',test)]:
            st.markdown('**'+label+'**')
            data=frame[['model','annualized_return','max_drawdown','sharpe','total_cost','mean_cash_weight']].copy()
            data.model=data.model.map(NAMES)
            data.columns=['模型','年化净收益','最大回撤','Sharpe','费用（元）','平均现金比例']
            st.dataframe(data.style.format({'年化净收益':'{:.2%}','最大回撤':'{:.2%}','Sharpe':'{:.2f}','费用（元）':'{:,.0f}','平均现金比例':'{:.2%}'}),hide_index=True,width='stretch')
        st.download_button('下载完整模型对照',test.to_csv(index=False).encode('utf-8-sig'),'model_comparison.csv','text/csv')
    with tabs[2]:
        factors=pd.read_csv(folder/'factor_summary.csv')
        st.write('逐日横截面相关与分组诊断；因子越多不代表独立信息越多。')
        st.dataframe(factors,hide_index=True,width='stretch')
        st.plotly_chart(px.bar(factors,x='factor',y='rank_ic_mean',labels={'factor':'因子','rank_ic_mean':'平均 Rank IC'},color_discrete_sequence=['#168579']),width='stretch')
        cards=json.loads((folder/'factor_cards.json').read_text(encoding='utf-8'))
        choice=st.selectbox('查看因子定义',list(cards),format_func=lambda key:cards[key][0])
        st.code(cards[choice][1],language=None);st.write(cards[choice][3])
        d=pd.read_csv(folder/'factors'/choice/'daily.csv',parse_dates=['date'])
        g=pd.read_csv(folder/'factors'/choice/'groups.csv')
        left,right=st.columns(2)
        with left:
            d['20日平均Rank IC']=d.rank_ic.rolling(20).mean()
            st.plotly_chart(px.line(d,x='date',y='20日平均Rank IC'),width='stretch')
        with right:
            groups=g.groupby('group',as_index=False).mean_forward_return.mean()
            st.plotly_chart(px.bar(groups,x='group',y='mean_forward_return',labels={'group':'形成时分组','mean_forward_return':'平均未来20日收益'}),width='stretch')
        st.caption('分组收益是重叠的未来20日标签均值，未扣费，不能连乘成可交易净值；覆盖和有效人数见下表。')
        with st.expander('每日样本与缺失核验'):
            st.dataframe(d[['date','factor_count','valid_pairs','factor_coverage','ic','rank_ic']],hide_index=True)
            st.download_button('下载分组人数与标签均值',g.to_csv(index=False).encode('utf-8-sig'),choice+'_groups.csv')
        st.caption('先按日去极值和标准化，再对当日行业分类及对数市值回归取残差。缺失财报不会用未来值填充。')
    with tabs[3]:
        controls=pd.read_csv(folder/'controls.csv')
        st.subheader('同一模型的组合与成本对照')
        display=controls[['experiment','annualized_return','max_drawdown','total_cost','mean_cash_weight']].rename(columns={'experiment':'实验','annualized_return':'年化收益','max_drawdown':'最大回撤','total_cost':'费用（元）','mean_cash_weight':'平均现金比例'})
        st.dataframe(display.style.format({'年化收益':'{:.2%}','最大回撤':'{:.2%}','费用（元）':'{:,.0f}','平均现金比例':'{:.2%}'}),hide_index=True,width='stretch')
        risk=pd.read_csv(folder/'risk.csv',parse_dates=['date'])
        st.plotly_chart(px.line(risk,x='date',y='target_exposure',labels={'date':'形成日','target_exposure':'目标股票仓位'}),width='stretch')
        st.subheader('原始动量失败实验')
        st.write('2020–2026 年原始动量累计亏损 83.57%。以下按年呈现净收益与费用；同成交路径加回费用不等同于零费用策略重跑。')
        st.dataframe(pd.read_csv(folder/'legacy_years.csv'),hide_index=True,width='stretch')
    with tabs[4]:
        st.write('本页来自固定实验结果，不会在刷新页面时重新训练。真实行情与财务数据保留在本地，公开仓库提供代码、可分享样本和结果摘要。')
        st.code('python scripts/download_research_data.py --token-file YOUR_TOKEN_FILE\npython -m cfquant.cli research\npython -m pytest -q',language='bash')
        st.json(json.loads((folder/'acceptance.json').read_text(encoding='utf-8')),expanded=False)
        st.write('最终报告、Beamer PDF 与源文件、PowerPoint、讲稿和演示步骤位于交付目录。')
