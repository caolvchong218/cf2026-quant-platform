"""Causal cross-sectional features and dated fundamental joins."""
from pathlib import Path
import numpy as np
import pandas as pd
from .data import load_market, load_calendar, write_json, digest

FEATURE_CARDS = {
    'momentum_60_20': ('中期动量', 'C(t-20)/C(t-60)-1', 1, '排除最近一个月的中期趋势'),
    'reversal_5': ('短期反转', '-(C(t)/C(t-5)-1)', 1, '短期超调后可能反弹'),
    'low_volatility_20': ('低波动', '-std(r,20)', 1, '低风险股票的风险收益特征'),
    'range_20': ('低振幅', '-mean((H-L)/C,20)', 1, '日内振幅及交易拥挤程度'),
    'trend_20': ('均线趋势', 'C/mean(C,20)-1', 1, '偏离均线的趋势延续'),
    'volume_trend': ('成交量变化', 'log(mean(V,5)/mean(V,20))', 1, '量能变化与价格发现'),
    'liquidity_20': ('流动性', '-mean(abs(r)/(rawC*V/1e6),20)', 1, '单位成交额价格冲击'),
    'turnover': ('低换手', '-turnover_rate', 1, '换手拥挤与交易成本'),
    'earnings_yield': ('盈利收益率', '1/PE_TTM (PE>0)', 1, '价格相对盈利的便宜程度'),
    'book_yield': ('账面市值比', '1/PB (PB>0)', 1, '价格相对净资产的便宜程度'),
    'dividend_yield': ('股息率', 'dv_ttm/100', 1, '现金分红相对价格'),
    'quality_roe': ('盈利质量', 'latest announced annual ROE/100', 1, '历史年度盈利能力'),
}
FEATURES = list(FEATURE_CARDS)


def standardize(frame):
    """Winsorize within the same formation date, then sample z-score."""
    finite = frame.replace([np.inf, -np.inf], np.nan)
    clipped = finite.clip(lower=finite.quantile(.01), upper=finite.quantile(.99), axis=1)
    return (clipped-clipped.mean()) / clipped.std(ddof=1).replace(0,np.nan)


def neutralize(frame, log_cap, industry):
    """OLS residual against industry dummies and log capitalization (FWL)."""
    output = pd.DataFrame(np.nan,index=frame.index,columns=frame.columns)
    groups = industry.fillna('UNKNOWN').astype(str)
    for col in frame:
        valid = frame[col].notna() & log_cap.notna()
        if valid.sum()<10:continue
        y=frame.loc[valid,col]; x=log_cap.loc[valid]; g=groups.loc[valid]
        y=y-y.groupby(g).transform('mean')
        x=x-x.groupby(g).transform('mean')
        denom=float(x@x)
        residual=y-x*float(x@y)/denom if denom>1e-12 else y
        sd=residual.std(ddof=1)
        if sd>1e-12:output.loc[valid,col]=residual/sd
    return output


def financial_asof(financials, dates):
    """Annual ROE becomes available strictly after announcement, never at period end."""
    f=financials.copy()
    for col in ['ann_date','end_date']:
        f[col]=pd.to_datetime(f[col].astype(str),format='%Y%m%d',errors='coerce')
    f=f.dropna(subset=['ann_date','end_date'])
    f=f[(f.end_date.dt.month==12)&(f.end_date<=f.ann_date)]
    # Conflicting copies with identical public identity are ambiguous; don't pick
    # the most favorable revised number. Exact duplicate rows are harmless.
    f=f.drop_duplicates(['ann_date','end_date','roe'])
    f=f[~f.duplicated(['ann_date','end_date'],keep=False)]
    f=f.sort_values(['ann_date','end_date']).drop_duplicates('ann_date',keep='last')
    if f.empty:return pd.Series(np.nan,index=dates)
    joined=pd.merge_asof(pd.DataFrame({'date':dates}),f[['ann_date','end_date','roe']],
                         left_on='date',right_on='ann_date',direction='backward',allow_exact_matches=False)
    # Ignore stale annual statements and missing vendor histories.
    roe=pd.to_numeric(joined.roe,errors='coerce').where((joined.date-joined.end_date).dt.days<=550)/100
    return pd.Series(roe.to_numpy(),index=dates)


def industry_history(records, dates):
    """Reconstruct valid intervals; ambiguous concurrent memberships stay unknown."""
    result=pd.Series('UNKNOWN',index=dates,dtype=object)
    count=pd.Series(0,index=dates)
    for row in records.drop_duplicates(['l1_code','in_date','out_date']).itertuples():
        start=pd.to_datetime(str(row.in_date),format='%Y%m%d',errors='coerce')
        end=pd.to_datetime(str(row.out_date),format='%Y%m%d',errors='coerce')
        if pd.isna(start):continue
        mask=(dates>=start)&((dates<end) if pd.notna(end) else True)
        # Overlap in the same L1 category is consistent, not a second industry.
        conflicting=mask & (result.to_numpy()!='UNKNOWN') & (result.to_numpy()!=row.l1_code)
        count.loc[conflicting]+=1
        result.loc[mask]=row.l1_code
    result.loc[count>0]='UNKNOWN'
    return result


def build_features(root: Path):
    source=root/'data/private/mainboard1000_20260918/data/processed'
    extra=root/'data/private/research_v2'
    market=load_market(source/'market.csv'); dates=load_calendar(source/'calendar.csv')
    def panel(col):return market.pivot(index='date',columns='asset',values=col).reindex(dates)
    close,volume=panel('close'),panel('volume'); rawclose=panel('raw_close')
    high,low=panel('high'),panel('low'); returns=close.pct_change(fill_method=None)
    raw={
        'momentum_60_20':(close.shift(20)/close.shift(60)-1).where(close.rolling(61).count()==61),
        'reversal_5':-(close/close.shift(5)-1).where(close.rolling(6).count()==6),
        'low_volatility_20':-returns.rolling(20,min_periods=20).std(),
        'range_20':-((high-low)/close).rolling(20,min_periods=20).mean(),
        'trend_20':close/close.rolling(20,min_periods=20).mean()-1,
        'volume_trend':np.log(volume.rolling(5,min_periods=5).mean()/volume.rolling(20,min_periods=20).mean()),
        'liquidity_20':-(returns.abs()/(rawclose*volume/1e6)).rolling(20,min_periods=20).mean(),
    }
    basic=pd.read_parquet(extra/'daily_basic.parquet')
    basic['date']=pd.to_datetime(basic.trade_date.astype(str),format='%Y%m%d')
    if basic.duplicated(['date','ts_code']).any():raise ValueError('Duplicate daily basic keys')
    def basic_panel(col):return basic.pivot(index='date',columns='ts_code',values=col).reindex(index=dates,columns=close.columns).astype(float)
    cap=basic_panel('total_mv'); pe=basic_panel('pe_ttm'); pb=basic_panel('pb')
    raw.update(turnover=-basic_panel('turnover_rate'),earnings_yield=1/pe.where(pe>0),
               book_yield=1/pb.where(pb>0),dividend_yield=basic_panel('dv_ttm')/100)
    financials=pd.read_parquet(extra/'financials.parquet')
    industry=pd.read_parquet(extra/'industries.parquet')
    roe=pd.DataFrame(np.nan,index=dates,columns=close.columns)
    sectors=pd.DataFrame('UNKNOWN',index=dates,columns=close.columns)
    for code in close.columns:
        roe[code]=financial_asof(financials[financials.ts_code==code],dates)
        sectors[code]=industry_history(industry[industry.ts_code==code],dates)
    raw['quality_roe']=roe
    opens=panel('open'); labels=opens.shift(-21)/opens.shift(-1)-1
    exit_dates=pd.Series(dates,index=dates).shift(-21)
    amount20=(rawclose*volume).rolling(20,min_periods=20).mean()
    # Main-board 5% limits are used only as a conservative ST-like eligibility
    # proxy, not claimed to be a complete historical ST classification.
    limit_ratio=panel('up_limit')/panel('down_limit')
    eligible=(rawclose>=3)&(amount20>=20_000_000)&(limit_ratio>1.15)&(cap>0)
    outputs=[]; audit=[]
    for date in dates:
        frame=pd.DataFrame({name:raw[name].loc[date] for name in FEATURES})
        active=eligible.loc[date] & (frame.notna().sum(axis=1)>=10)
        z=standardize(frame.loc[active])
        neutral=neutralize(z,np.log(cap.loc[date,active]),sectors.loc[date,active])
        piece=frame.copy()
        for name in FEATURES:
            piece['z_'+name]=z[name].reindex(piece.index)
            piece['n_'+name]=neutral[name].reindex(piece.index)
        piece['date']=date;piece['asset']=piece.index;piece['eligible']=active
        piece['industry']=sectors.loc[date];piece['log_cap']=np.log(cap.loc[date])
        piece['volatility']=-raw['low_volatility_20'].loc[date]
        piece['label']=labels.loc[date];piece['label_exit']=exit_dates.loc[date]
        outputs.append(piece.reset_index(drop=True))
        audit.append({'date':str(date.date()),'eligible':int(active.sum()),'industry_unknown':int((sectors.loc[date,active]=='UNKNOWN').sum()),
                      'quality_coverage':float(roe.loc[date,active].notna().mean()) if active.any() else None})
    frame=pd.concat(outputs,ignore_index=True)
    frame.to_parquet(extra/'features.parquet',index=False)
    write_json(extra/'feature_quality.json',{'dates':audit,'features':FEATURE_CARDS,
               'rows':len(frame),'data_sha256':digest(source/'market.csv'),
               'financial_caveat':'Vendor historical revisions are not a full vintage database; conflicting statement copies omitted.',
               'industry_caveat':'Vendor reconstructed historical intervals; ambiguous/missing membership UNKNOWN, no current-industry backfill.'})
    print(f'Feature table ready: {len(frame):,} rows',flush=True)
    return frame
