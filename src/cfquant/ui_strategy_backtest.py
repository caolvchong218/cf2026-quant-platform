"""Every strategy version can be inspected and replayed from the backtest page."""
import json
from dataclasses import asdict
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from .strategies import replay, outcome_status


def _stored(root, spec):
    folder=root/f'runs/research_{spec.version}'/('test_'+spec.signal)
    public=root/spec.evidence_path
    metrics_file=folder/'metrics.json'
    if metrics_file.exists():
        metrics=json.loads(metrics_file.read_text(encoding='utf-8'))
        daily=pd.read_csv(folder/'daily.csv',parse_dates=['date'])
        config=json.loads((folder/'config.json').read_text(encoding='utf-8'))['backtest']
        return daily,metrics,folder,config
    table=pd.read_csv(public/'test.csv');key='candidate' if spec.version=='v3' else 'model'
    metrics=table[table[key]==spec.signal].iloc[0].to_dict()
    daily_path=public/'daily'/f'{spec.signal}.csv'
    daily=pd.read_csv(daily_path,parse_dates=['date']) if daily_path.exists() else pd.DataFrame()
    return daily,metrics,None,{'initial_cash':1e6,'start':'2025-01-02','end':'2026-09-18'}


def render(root:Path,spec):
    st.subheader(spec.label)
    st.caption(spec.status+' · 固定1000股历史池 · 下一开盘执行 · 所有费用按实际成交扣除')
    if spec.version=='v3':
        st.info('本轮候选在已观察的历史区间内研究和筛选，包含逐年滚动训练；不是未知盲测，也不保证未来收益。')
    else:
        st.warning('历史版本：用于对照。V2原预选针对CPI和回撤，不是当前战胜指数的默认策略。')
    daily,metrics,folder,used=_stored(root,spec)
    local=(root/spec.scores_path).exists() and (root/'data/private/research_v2/features.parquet').exists()
    with st.expander('修改参数并重新回测',expanded=False):
        st.caption('以下参数用于新的实验；下方结果始终标明实际使用的日期。更换策略会清除上一策略的结果。')
        with st.form('version_form_'+spec.id):
            a,b,c=st.columns(3)
            start=a.date_input('开始日期',pd.Timestamp('2025-01-02').date(),min_value=pd.Timestamp('2023-01-03').date(),max_value=pd.Timestamp('2026-09-18').date())
            end=b.date_input('结束日期',pd.Timestamp('2026-09-18').date(),min_value=pd.Timestamp('2023-01-03').date(),max_value=pd.Timestamp('2026-09-18').date())
            holdings=c.number_input('目标持仓数',10,100,50)
            a,b,c=st.columns(3)
            every=a.selectbox('调仓间隔（交易日）',[20,5,10,40])
            buy=b.number_input('买入费用（bp）',0.,200.,10.)
            sell=c.number_input('卖出费用（bp）',0.,200.,15.)
            initial=st.number_input('初始资金（元）',1000.,1e8,1e6,10000.)
            submit=st.form_submit_button('运行此版本并核对账本',type='primary',disabled=not local)
        if not local:
            st.caption('当前电脑缺少本地模型分数/真实快照，可查看公开固定结果；按复现指南生成数据后可重新运行。')
        if submit:
            try:
                with st.spinner('使用所选版本重新计算交易、费用和资金账本…'):
                    result,m,f,u=replay(root,spec,str(start),str(end),holdings,every,buy/10000,sell/10000,initial)
                st.session_state['version_run']=(spec.id,result,m,str(f),u)
            except (ValueError,AssertionError) as exc:
                st.error(str(exc))
    if st.session_state.get('version_run',(None,))[0]==spec.id:
        _,result,metrics,folder,used=st.session_state['version_run'];folder=Path(folder);daily=result.daily
    st.caption(f"显示结果：{spec.label} · {used['start']} 至 {used['end']} · 初始资金 {used['initial_cash']:,.0f} 元")
    st.info('账本检查：已通过。此状态仅说明资金与成交记录一致，不代表收益或风险达标。')
    if 'benchmark_return' not in metrics:
        bm=json.loads((root/'evidence/research_v2/benchmark_metrics.json').read_text())
        metrics={**metrics,'benchmark_return':bm['total_return'],
                 'excess_return_pp':metrics['total_return']-bm['total_return'],
                 'beats_benchmark':metrics['total_return']>bm['total_return']}
    passed,reasons=outcome_status(metrics)
    if passed:st.success('历史区间目标：扣费正收益、超过沪深300价格指数、最大回撤不高于25%。')
    else:st.error('历史区间未达标：'+'；'.join(reasons)+'。')
    cols=st.columns(5)
    for col,label,value in zip(cols,['累计净收益','沪深300同期收益','超额收益（百分点）','最大回撤','费用（元）'],
        [f"{metrics['total_return']:.2%}",f"{metrics['benchmark_return']:.2%}",
         f"{metrics['excess_return_pp']*100:+.2f}",f"{metrics['max_drawdown']:.2%}",f"{metrics['total_cost']:,.0f}"]):
        col.metric(label,value)
    if not daily.empty:
        plots=daily[['date']].assign(nav=daily.nav/used['initial_cash'],series=spec.label+' · 扣费')
        bp=root/'data/private/research_v2/benchmark.csv'
        if bp.exists():
            b=pd.read_csv(bp);b['date']=pd.to_datetime(b.trade_date.astype(str));b=b.set_index('date').close.sort_index()
            bn=b.reindex(daily.date).to_numpy()/b.loc[b.index<daily.date.iloc[0]].iloc[-1]
            plots=pd.concat([plots,daily[['date']].assign(nav=bn,series='沪深300 · 价格指数')],ignore_index=True)
        fig=px.line(plots,x='date',y='nav',color='series',labels={'date':'日期','nav':'净值','series':'策略'},
                    color_discrete_sequence=['#168579','#4265a6'])
        fig.update_layout(height=420,legend=dict(orientation='h',y=1.13),margin=dict(t=35,b=0))
        st.plotly_chart(fig,width='stretch')
        st.download_button('下载当前版本逐日账本',daily.to_csv(index=False).encode('utf-8-sig'),spec.id.replace('/','_')+'_daily.csv')
    with st.expander('策略规则与运行明细'):
        st.json(asdict(spec.policy))
        if folder:
            st.caption(f'本地实验：{folder}')
            for name,label in [('trades','成交'),('orders','订单与未成交'),('positions','持仓')]:
                p=folder/f'{name}.csv'
                if p.exists():
                    st.markdown('**'+label+'**');st.dataframe(pd.read_csv(p),hide_index=True,width='stretch')
