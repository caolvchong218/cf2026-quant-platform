"""Interactive comparison and risk views backed by published daily ledgers."""
from pathlib import Path
import io
import json
import zipfile
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from .strategies import catalogue
from .ui_design import heading, chart, note, COLORS


def load_evidence(root):
    specs = catalogue(root)
    ledgers, labels = {}, {}
    for s in specs:
        p = root/s.evidence_path/'daily'/f'{s.signal}.csv'
        if p.exists():
            ledgers[s.id] = pd.read_csv(p, parse_dates=['date'])
            labels[s.id] = s.label
    curves = pd.read_csv(root/'evidence/research_v3/curves.csv', parse_dates=['date'])
    benchmarks = {n:g.set_index('date').nav.sort_index() for n,g in curves.groupby('series')
                  if n in ['沪深300 · 价格指数','沪深300 · 全收益指数']}
    return ledgers, labels, benchmarks


def navigate(page):
    st.session_state['workspace_page'] = page


def _export(result):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as z:
        for name,value in result.items():
            if isinstance(value, pd.DataFrame):
                z.writestr(name+'.csv', value.to_csv().encode('utf-8-sig'))
            elif name == 'assumptions':
                z.writestr('methodology.json',json.dumps(value,ensure_ascii=False,default=str,indent=2))
    return stream.getvalue()


def overview(root):
    heading('青序 · 量化研究工作台', '从一个信号，到一笔成交，再到一项可核验的研究结论。', 'COMPUTATIONAL FINANCE 2026 / PROJECT 01')
    m = json.loads((root/'evidence/research_v3/metrics.json').read_text(encoding='utf-8'))
    st.caption('当前默认：V3 年度滚动 LightGBM + 波动预算　｜　历史区间 2025-01-02 — 2026-09-18')
    for c,label,value,delta in zip(st.columns(4),
            ['扣费累计收益','超过全收益指数','最大回撤','平均现金比例'],
            [f"{m['total_return']:.2%}",f"{m['excess_vs_total_return']*100:.2f} 个百分点",f"{m['max_drawdown']:.2%}",f"{m['mean_cash_weight']:.2%}"],
            ['年化 '+f"{m['annualized_return']:.2%}",'含红利再投资的基准','研究门槛 25%','允许现金 · 无杠杆']):
        c.metric(label,value,help=delta)
    left,right=st.columns([2.35,1])
    with left:
        st.subheader('净值轨迹')
        curves=pd.read_csv(root/'evidence/research_v3/curves.csv',parse_dates=['date'])
        curves=curves[curves.series.isin(['V3最新候选 · 扣费','沪深300 · 全收益指数','沪深300 · 价格指数'])]
        fig=px.line(curves,x='date',y='nav',color='series',labels={'nav':'初始资金 = 1','date':'日期'})
        chart(fig,390)
    with right:
        st.subheader('继续研究')
        st.markdown('**比较一个假设**')
        st.caption('把多版策略放在相同日期，比较扣费收益、回撤和超额。')
        st.button('打开策略对比 →',on_click=navigate,args=('策略对比',),width='stretch')
        st.markdown('**解释一段回撤**')
        st.caption('定位最深回撤、恢复时间与月度表现，查看现金和换手。')
        st.button('打开风险透镜 →',on_click=navigate,args=('风险透镜',),width='stretch')
        st.markdown('**核查一笔成交**')
        st.caption('按实验、资产与日期检索订单，保存研究笔记。')
        st.button('打开实验档案 →',on_click=navigate,args=('实验档案',),width='stretch')
    note('研究边界：V3 是观察历史结果后选出的开发候选，尚未通过独立盲测。2024 年曾落后指数，风险控制也不能保证未来回撤上限。')
    st.subheader('研究路径')
    for col,step,title,body,page in zip(st.columns(4),['01 / DATA','02 / SIGNAL','03 / PORTFOLIO','04 / EVIDENCE'],
            ['观察行情','检验因子','回放策略','准备展示'],
            ['检索股票、K线与成交量，查看原始样本。','12个经济因子，切换日期并查看IC与分组。','选择策略版本、修改参数，重新计算账本。','报告、Beamer、PPTX和逐页讲稿集中下载。'],
            ['行情探索','因子诊断','回测实验','材料与答辩']):
        with col:
            st.markdown(f'<div class="qx-step">{step}</div>',unsafe_allow_html=True)
            st.markdown('**'+title+'**');st.caption(body)
            st.button(title,on_click=navigate,args=(page,),key='home_'+page)


def _choose(root, multiple=False):
    ledgers,labels,benchmarks=load_evidence(root)
    ids=list(ledgers)
    if multiple:
        defaults=[ids[0]]+[i for i in ['v2/multifactor_raw','v3/economic__steady'] if i in ids]
        chosen=st.multiselect('比较策略（最多5个）',ids,default=defaults,format_func=labels.get,max_selections=5,key='compare_ids')
    else:
        chosen=[st.selectbox('分析策略',ids,format_func=labels.get,key='risk_id')]
    if not chosen:
        st.info('选择至少一个策略以开始比较。');return None
    lo=max(ledgers[i].date.min() for i in chosen).date()
    hi=min(ledgers[i].date.max() for i in chosen).date()
    a,b,c=st.columns([1,1,1.2])
    start=a.date_input('分析开始日期',lo,min_value=lo,max_value=hi,key=('compare' if multiple else 'risk')+'_start')
    end=b.date_input('分析结束日期',hi,min_value=lo,max_value=hi,key=('compare' if multiple else 'risk')+'_end')
    benchmark=c.selectbox('比较基准',list(benchmarks),index=1,key=('compare' if multiple else 'risk')+'_benchmark')
    if start>end:
        st.error('开始日期不能晚于结束日期。');return None
    from .analytics import compare_strategies
    try:
        result=compare_strategies({labels[i]:ledgers[i] for i in chosen},
            {benchmark:benchmarks[benchmark]},start=str(start),end=str(end),window=60)
    except ValueError as exc:
        st.error(str(exc));return None
    return result,benchmark,chosen,labels,ledgers


def comparison(root):
    heading('策略对比', '在相同历史区间比较不同策略，保留首日损益与实际费用。')
    data=_choose(root,True)
    if data is None:return
    result,bm,chosen,labels,_=data
    wealth=result['wealth']
    st.caption(f'共同区间：{wealth.index.min().date()} — {wealth.index.max().date()} · {len(wealth)} 个交易日 · 区间开始前归一为1')
    tab1,tab2,tab3=st.tabs(['净值与收益','风险与相关性','数据与口径'])
    with tab1:
        relative=st.toggle('显示相对基准净值',key='relative_nav')
        plot=wealth.div(wealth[bm],axis=0).drop(columns=bm) if relative else wealth
        fig=px.line(plot,labels={'value':'相对基准净值' if relative else '区间净值','index':'日期','date':'日期','variable':'策略'})
        if relative:fig.add_hline(y=1,line_dash='dot',line_color='#9daaba')
        chart(fig,440)
        summary=result['summary'].copy()
        cols=[c for c in ['total_return','annualized_return','max_drawdown','annualized_volatility','sharpe','total_cost','mean_cash_weight'] if c in summary]
        names={'total_return':'累计净收益','annualized_return':'年化收益','max_drawdown':'最大回撤','annualized_volatility':'年化波动','sharpe':'Sharpe','total_cost':'费用（元）','mean_cash_weight':'平均现金'}
        table=summary[cols].rename(columns=names)
        table.index.name='策略'
        formats={names[c]:('{:,.2f}' if c=='total_cost' else '{:.2f}' if c=='sharpe' else '{:.2%}') for c in cols}
        st.dataframe(table.style.format(formats,na_rep='—'),width='stretch')
    with tab2:
        chart(px.line(-result['drawdowns'],labels={'value':'回撤','index':'日期','date':'日期','variable':'策略'}).update_yaxes(tickformat='.0%'),340)
        corr=result['correlation']
        chart(px.imshow(corr,text_auto='.2f',zmin=-1,zmax=1,color_continuous_scale='RdBu',aspect='auto',labels={'color':'日收益相关性'}),380)
        st.caption('相关性基于同区间日收益，仅描述历史共同波动；高相关并不代表同等风险。')
    with tab3:
        st.write('区间收益从所选首日之前的净值开始计算，首日亏损不会因图表归一被抹掉。基准采用同一天收益；内部缺日不插值。日期筛选是在原有持仓路径上截取，不等同于在该日以现金重新启动回测。')
        st.json(result['assumptions'],expanded=False)
        st.dataframe(result['relative'],width='stretch',hide_index=True)
    st.download_button('下载本次比较包（CSV + 计算口径）',_export(result),'strategy_comparison.zip','application/zip')


def risk(root):
    heading('风险透镜', '看清收益来自哪些月份，回撤持续多久，以及资金如何暴露于市场。')
    data=_choose(root)
    if data is None:return
    result,bm,chosen,labels,ledgers=data
    name=labels[chosen[0]]
    r=result['returns'][name];nav=result['wealth'][name]
    loss=r[r<0]
    cols=st.columns(4)
    cols[0].metric('区间最大回撤',f"{result['drawdowns'][name].max():.2%}")
    cols[1].metric('最差单日',f'{r.min():.2%}')
    cols[2].metric('亏损交易日占比',f'{(r<0).mean():.1%}')
    cutoff=r.quantile(.05);tail=r[r<=cutoff]
    cols[3].metric('最差5%日平均收益',f'{tail.mean():.2%}',help='经验尾部均值，不是对未来损失的保证。')
    tabs=st.tabs(['月度表现','回撤与恢复','滚动风险','仓位与费用','现金配置实验'])
    with tabs[0]:
        monthly=result['monthly'][name].reset_index().pivot(index='year',columns='month',values=name).reindex(columns=range(1,13))
        monthly.columns=[f'{i}月' for i in monthly.columns]
        monthly.index=monthly.index.astype(str)
        chart(px.imshow(monthly*100,text_auto='.1f',color_continuous_scale=['#bd645e','#fbfcff','#168b94'],color_continuous_midpoint=0,
            aspect='auto',labels={'color':'月收益（%）','x':'','y':''}),290)
        st.caption('月收益按日收益复利计算；首尾月份可能不完整，空白月份没有样本。')
        chart(px.histogram(r.to_frame('日收益'),x='日收益',nbins=45,color_discrete_sequence=[COLORS[0]]).update_xaxes(tickformat='.1%'),300)
    with tabs[1]:
        fig=go.Figure(go.Scatter(x=nav.index,y=-result['drawdowns'][name],fill='tozeroy',line_color='#bd645e',name='回撤'))
        fig.update_yaxes(tickformat='.0%');chart(fig,330)
        episodes=result['episodes'];episodes=episodes[episodes.series==name].copy()
        if episodes.empty:st.info('当前区间没有深于1%的回撤。')
        else:
            display=episodes[['peak_date','trough_date','recovery_date','depth','duration_sessions','recovered']].copy()
            for col in ['peak_date','trough_date','recovery_date']:
                display[col]=pd.to_datetime(display[col]).dt.strftime('%Y-%m-%d').fillna('尚未恢复' if col=='recovery_date' else '区间起点前')
            display['recovered']=display.recovered.map({True:'已恢复',False:'尚未恢复'})
            display=display.rename(columns={'depth':'回撤深度','peak_date':'峰值日期','trough_date':'谷底日期','recovery_date':'恢复日期',
                'duration_sessions':'持续交易日','recovered':'状态'})
            st.dataframe(display.style.format({'回撤深度':'{:.2%}'}),hide_index=True,width='stretch')
        st.caption('以区间开始前净值作为初始高水位；未恢复回撤计至区间末尾，不能将其记为已恢复。')
    with tabs[2]:
        rolling=result['rolling'];view=rolling[rolling.series==name]
        chart(px.line(view,x='date',y=['annualized_volatility','downside_deviation'],labels={'value':'年化波动','date':'日期'}).update_yaxes(tickformat='.0%'),350)
        chart(px.line(view,x='date',y='rolling_return',labels={'rolling_return':'60日累计收益','date':'日期'}).update_yaxes(tickformat='.0%'),300)
        st.caption('滚动窗口为60个交易日；不足窗口不显示。波动使用252日年化，下行偏差目标为日收益0。')
    with tabs[3]:
        d=ledgers[chosen[0]];d=d[d.date.isin(nav.index)].copy()
        d['现金比例']=d.cash/d.nav
        d['区间累计费用']=d.cost.cumsum()
        chart(px.line(d,x='date',y='现金比例').update_yaxes(tickformat='.0%'),300)
        a,b=st.columns(2)
        with a:chart(px.line(d,x='date',y='区间累计费用'),280)
        with b:chart(px.bar(d,x='date',y='turnover',labels={'turnover':'双边成交额 / 开盘净值'}),280)
        st.caption('费用按实际成交扣除；展示累计费用并不是将其重新投资后的无费用回测。')
    with tabs[4]:
        weight=st.slider('投入策略的比例，其余持有现金',0,100,80,5,key='cash_weight')/100
        inflation=st.number_input('年化通胀假设（%）',0.,15.,2.,.5,key='inflation_assumption')/100
        # Static funding split: strategy sleeve is never rebalanced against cash.
        mixed=weight*nav+(1-weight)
        real=mixed/(1+inflation)**(np.arange(1,len(nav)+1)/252)
        dd=mixed/np.maximum.accumulate(np.r_[1.,mixed.to_numpy()])[1:]-1
        a,b,c=st.columns(3)
        a.metric('组合累计收益',f'{mixed.iloc[-1]-1:.2%}')
        b.metric('购买力变化（假设）',f'{real.iloc[-1]-1:.2%}')
        c.metric('组合最大回撤',f'{max(0.,-dd.min()):.2%}')
        chart(px.line(pd.DataFrame({'原策略':nav,'策略 + 现金':mixed,'通胀调整后（假设）':real}),labels={'value':'区间净值','index':'日期','date':'日期'}),330)
        note('这是静态资金分配的算术情景：现金零利息，策略仓位按历史净值同比例缩放，不在策略与现金之间再平衡。未重新模拟资金规模对最低费用、容量和成交的影响；通胀数值是输入假设，不是已下载的CPI。')
    st.download_button('下载风险分析包',_export(result),'risk_analysis.zip','application/zip')
