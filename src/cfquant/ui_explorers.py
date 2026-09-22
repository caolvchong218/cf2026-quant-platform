"""Price and factor explorers use existing observations, never future-filled data."""
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from .ui_design import heading, chart, note


def factors(root):
    heading('因子实验台', '统一查看12个经济因子，按日期重新汇总诊断，解释有效性与缺失。')
    folder=root/'evidence/research_v2'
    cards=json.loads((folder/'factor_cards.json').read_text(encoding='utf-8'))
    choice=st.selectbox('研究因子',list(cards),format_func=lambda x:cards[x][0]+' · '+x,key='factor_lab_choice')
    card=cards[choice]
    note(card[3]+'。诊断使用当日去极值、标准化及行业/市值中性化后的分数；20日未来标签只用于事后检验。')
    st.code(card[1],language=None)
    d=pd.read_csv(folder/'factors'/choice/'daily.csv',parse_dates=['date'])
    g=pd.read_csv(folder/'factors'/choice/'groups.csv',parse_dates=['date'])
    lo,hi=d.date.min().date(),d.date.max().date()
    a,b=st.columns(2)
    start=a.date_input('因子诊断开始日期',lo,min_value=lo,max_value=hi,key='factor_lab_start')
    end=b.date_input('因子诊断结束日期',hi,min_value=lo,max_value=hi,key='factor_lab_end')
    if start>end:st.error('开始日期不能晚于结束日期。');return
    d=d[d.date.between(pd.Timestamp(start),pd.Timestamp(end))].copy()
    g=g[g.date.between(pd.Timestamp(start),pd.Timestamp(end))].copy()
    if d.empty:st.info('所选范围内没有交易日。');return
    a,b,c,e=st.columns(4)
    a.metric('平均 Rank IC',f'{d.rank_ic.mean():.4f}')
    b.metric('平均 IC',f'{d.ic.mean():.4f}')
    c.metric('因子覆盖率',f'{d.factor_coverage.mean():.1%}')
    e.metric('有效标签配对日',str(d.rank_ic.notna().sum()))
    tabs=st.tabs(['时间稳定性','分组与覆盖','12因子比较','原始诊断'])
    with tabs[0]:
        d['20日平均 Rank IC']=d.rank_ic.rolling(20,min_periods=20).mean()
        fig=px.line(d,x='date',y='20日平均 Rank IC');fig.add_hline(y=0,line_dash='dot')
        chart(fig,340)
        annual=d.assign(年份=d.date.dt.year).groupby('年份').agg(平均Rank_IC=('rank_ic','mean'),有效日数=('rank_ic','count'),覆盖率=('factor_coverage','mean'))
        st.dataframe(annual.style.format({'平均Rank_IC':'{:.4f}','覆盖率':'{:.1%}'}),width='stretch')
    with tabs[1]:
        means=g.groupby('group').agg(平均未来收益=('mean_forward_return','mean'),平均形成样本=('formed_count','mean'),平均有效标签=('valid_count','mean')).reset_index()
        chart(px.bar(means,x='group',y='平均未来收益',labels={'group':'形成日从低分到高分分组'}).update_yaxes(tickformat='.1%'),320)
        st.dataframe(means,width='stretch',hide_index=True)
        chart(px.line(d,x='date',y='factor_coverage',labels={'factor_coverage':'覆盖率'}).update_yaxes(tickformat='.0%'),260)
        st.caption('组别在形成日先确定，再匹配未来标签。20日标签相互重叠、未扣费，组均值不是可交易净值，不能直接连乘。')
    with tabs[2]:
        rows=[]
        for factor,meta in cards.items():
            f=pd.read_csv(folder/'factors'/factor/'daily.csv',parse_dates=['date'])
            f=f[f.date.between(pd.Timestamp(start),pd.Timestamp(end))]
            rows.append({'因子':meta[0],'Rank IC':f.rank_ic.mean(),'覆盖率':f.factor_coverage.mean(),'有效日数':f.rank_ic.count()})
        comparison=pd.DataFrame(rows)
        fig=px.bar(comparison,x='Rank IC',y='因子',orientation='h',color='Rank IC',color_continuous_scale=['#bb655f','#e4ecef','#128b91'],color_continuous_midpoint=0)
        chart(fig,470)
        st.dataframe(comparison.style.format({'Rank IC':'{:.4f}','覆盖率':'{:.1%}'}),hide_index=True,width='stretch')
        st.caption('全部因子保持固定顺序，负结果一并显示。区间选择属于描述性分析，不构成新的样本外验证。')
    with tabs[3]:
        st.dataframe(d,hide_index=True,width='stretch')
    a,b=st.columns(2)
    a.download_button('下载所选区间 IC',d.to_csv(index=False).encode('utf-8-sig'),choice+'_diagnostics.csv','text/csv')
    b.download_button('下载所选区间分组',g.to_csv(index=False).encode('utf-8-sig'),choice+'_groups.csv','text/csv')


def market_explorer(market,calendar,real):
    heading('行情探索', '按股票和日期查看价格、成交量与缺失样本，为异常结果找到数据依据。')
    if not real:st.info('当前为固定种子的合成样本，用于演示功能，不能视作真实股票表现。')
    a,b=st.columns([2,1])
    asset=a.selectbox('股票代码（可输入搜索）',sorted(market.asset.unique()),key='market_asset')
    basis=b.radio('价格口径',['复权研究单位','原始价格'],horizontal=True,key='price_basis')
    full=market[market.asset==asset].sort_values('date').copy()
    first,last=full.date.min().date(),full.date.max().date()
    start_default=max(first,(pd.Timestamp(last)-pd.Timedelta(days=365)).date())
    a,b=st.columns(2)
    start=a.date_input('行情开始日期',start_default,min_value=first,max_value=last,key='market_start_'+asset)
    end=b.date_input('行情结束日期',last,min_value=first,max_value=last,key='market_end_'+asset)
    if start>end:st.error('开始日期不能晚于结束日期。');return
    # Raw OHLC is derived by the same-day adjusted/raw close ratio; no future normalization.
    ratio=full.close/full.raw_close if 'raw_close' in full else pd.Series(1.,index=full.index)
    if basis=='原始价格':
        for col in ['open','high','low','close']:full[col]=full[col]/ratio
    full['MA20']=full.close.rolling(20,min_periods=20).mean()
    full['MA60']=full.close.rolling(60,min_periods=60).mean()
    d=full[full.date.between(pd.Timestamp(start),pd.Timestamp(end))]
    if d.empty:st.info('所选日期内没有行情记录。');return
    expected=calendar[(calendar>=pd.Timestamp(start))&(calendar<=pd.Timestamp(end))]
    missing=expected.difference(pd.DatetimeIndex(d.date))
    a,b,c=st.columns(3)
    a.metric('区间记录',str(len(d)));b.metric('缺失交易日',str(len(missing)))
    c.metric('区间末价格',f'{d.close.iloc[-1]:.3f}',help='复权研究单位价格不能直接填入交易软件委托。')
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.06,row_heights=[.74,.26])
    fig.add_trace(go.Candlestick(x=d.date,open=d.open,high=d.high,low=d.low,close=d.close,name='OHLC',increasing_line_color='#bc665f',decreasing_line_color='#168b94'),row=1,col=1)
    for col,color in [('MA20','#b68a36'),('MA60','#5473b7')]:fig.add_trace(go.Scatter(x=d.date,y=d[col],name=col,line=dict(width=1.4,color=color)),row=1,col=1)
    fig.add_trace(go.Bar(x=d.date,y=d.volume/10000,name='成交量（万股）',marker_color='#96adbf'),row=2,col=1)
    fig.update_layout(xaxis_rangeslider_visible=False,legend=dict(y=1.08))
    fig.update_yaxes(title_text='万股',row=2,col=1)
    chart(fig,560)
    st.caption('均线使用所选区间之前已有的历史数据预热。红色K线表示上涨，绿色表示下跌；价格缺失不补造行情。')
    if len(missing):
        with st.expander('查看缺失交易日'):st.write(', '.join(x.strftime('%Y-%m-%d') for x in missing))
    with st.expander('查看所选行情'):st.dataframe(d,hide_index=True,width='stretch')
    st.download_button('下载所选行情 CSV',d.to_csv(index=False).encode('utf-8-sig'),asset+'_market.csv','text/csv')
