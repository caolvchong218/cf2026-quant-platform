"""Predeclared validation selection, held-period evaluation, and persisted evidence."""
from dataclasses import replace
from pathlib import Path
import json
import subprocess
import time
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from .features import FEATURES,build_features
from .models import fit_predict
from .risk import RiskPolicy,build_targets
from .data import load_market,load_calendar,write_json,digest
from .config import Config
from .engine import run_backtest
from .analytics import performance,reconcile,diagnostics,forward_returns
from .experiment import json_safe


def evaluate(root, scores, features, market, calendar, benchmark, name, start, end, risk=True, cost_scale=1.):
    from .factors import panel
    folder=root/'runs/research_v2'/name
    folder.mkdir(parents=True,exist_ok=True)
    cfg=Config(name=name,start=start,end=end,holdings=50,rebalance_every=20,
               buy_cost=.001*cost_scale,sell_cost=.0015*cost_scale,
               data_path='data/private/mainboard1000_20260918/data/processed/market.csv',
               calendar_path='data/private/mainboard1000_20260918/data/processed/calendar.csv')
    if risk:
        targets,exposures=build_targets(scores,features,panel(market,calendar,'close'),benchmark,start)
        exposures.to_csv(folder/'risk.csv',index=False)
    else:targets=None
    result=run_backtest(market,calendar,scores,cfg,target_weights=targets,participation_limit=.01)
    checks=reconcile(result,cfg.initial_cash,cfg.buy_cost,cfg.sell_cost)
    if not checks['passed']:raise AssertionError(checks)
    metrics=performance(result.daily,cfg.initial_cash,trades=result.trades)
    metrics['mean_cash_weight']=float((result.daily.cash/result.daily.nav).mean())
    metrics['partial_fill_orders']=int((result.orders.status=='liquidity_limited').sum())
    for what in ['daily','trades','positions','orders']:
        getattr(result,what).to_csv(folder/f'{what}.csv',index=False,float_format='%.12g')
    write_json(folder/'metrics.json',json_safe(metrics));write_json(folder/'checks.json',checks)
    write_json(folder/'config.json',{'backtest':cfg.to_dict(),'risk':risk,'policy':RiskPolicy().__dict__ if risk else None,'participation_limit':.01})
    return metrics,result


def cpi_comparison(daily,cpi):
    """Match complete monthly NAV returns to CPI month-on-month rates."""
    nav=daily.set_index('date').nav
    monthly=nav.resample('ME').last()
    # The first month lacks a previous month-end NAV; omit it explicitly.
    stock=monthly.pct_change(fill_method=None)
    monthly_index=stock.index.to_period('M').astype(str).str.replace('-','')
    stock.index=monthly_index
    c=cpi.copy();c['month']=c.month.astype(str)
    rates=pd.to_numeric(c.set_index('month').nt_mom,errors='coerce')/100
    merged=pd.concat([stock.rename('strategy'),rates.rename('cpi')],axis=1).dropna()
    # Exclude the currently incomplete final calendar month even if vendor CPI
    # unexpectedly contains it; no partial-month comparison.
    final=nav.index.max()
    # Require the final business day of the month; missing a holiday month-end
    # may conservatively omit a complete trading month, never include a partial one.
    if final < (final + pd.offsets.BMonthEnd(0)):
        merged=merged[merged.index<final.strftime('%Y%m')]
    if merged.empty:return {'months':0}
    a=float((1+merged.strategy).prod()-1);b=float((1+merged.cpi).prod()-1)
    return {'months':len(merged),'start_month':merged.index.min(),'end_month':merged.index.max(),
            'strategy_return':a,'inflation':b,'real_return':(1+a)/(1+b)-1,'beats_cpi':a>b}


def run_research(root):
    start_time=time.monotonic()
    extra=root/'data/private/research_v2'
    out=root/'runs/research_v2'
    # Preserve a completed or interrupted run before a repeat. No prior candidate
    # or failure is silently overwritten by a subsequent research command.
    if out.exists() and any(out.iterdir()):
        archive=root/'runs'/('research_v2_archive_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        out.rename(archive)
    out.mkdir(parents=True,exist_ok=True)
    sources={p.name:digest(p) for p in sorted((root/'src/cfquant').glob('*.py'))}
    cache_inputs=[extra/n for n in ['daily_basic.parquet','financials.parquet','industries.parquet']]
    cache_inputs += [root/'src/cfquant/features.py',root/'src/cfquant/data.py',root/'src/cfquant/__init__.py',
                     root/'data/private/mainboard1000_20260918/data/processed/market.csv',
                     root/'data/private/mainboard1000_20260918/data/processed/calendar.csv']
    identity={p.relative_to(root).as_posix():digest(p) for p in cache_inputs}
    cache_meta=extra/'features_cache.json'
    cached=json.loads(cache_meta.read_text()) if cache_meta.exists() else {}
    cache_valid=(cached.get('inputs')==identity and (extra/'features.parquet').exists()
                 and cached.get('output')==digest(extra/'features.parquet'))
    frame=pd.read_parquet(extra/'features.parquet') if cache_valid else build_features(root)
    if not cache_valid:write_json(cache_meta,{'inputs':identity,'output':digest(extra/'features.parquet')})
    market_path=root/'data/private/mainboard1000_20260918/data/processed/market.csv'
    market=load_market(market_path);calendar=load_calendar(market_path.parent/'calendar.csv')
    assets=pd.Index(sorted(market.asset.unique()),name='asset')
    bench=pd.read_csv(extra/'benchmark.csv');bench['date']=pd.to_datetime(bench.trade_date.astype(str))
    benchmark=bench.set_index('date').close.sort_index()
    cpi=pd.read_csv(extra/'cpi.csv')
    good=frame.eligible & (frame[['n_'+f for f in FEATURES]].notna().sum(axis=1)>=10)
    train=good & (frame.date>='2020-01-01') & (frame.date<='2022-12-31') & (frame.label_exit<'2023-01-01') & frame.label.notna()
    predict=good & (frame.date>='2022-12-01')
    columns=['n_'+f for f in FEATURES]
    x=frame.loc[train,columns].fillna(0).to_numpy(dtype=np.float32)
    y=(frame.loc[train].groupby('date').label.rank(pct=True)-.5).to_numpy(dtype=np.float32)
    xp=frame.loc[predict,columns].fillna(0).to_numpy(dtype=np.float32)
    predictions={}
    for name,prefix in [('multifactor_raw','z_'),('multifactor_neutral','n_')]:
        predictions[name]=frame[[prefix+f for f in FEATURES]].mean(axis=1).where(good)
    model_info={}
    for name in ['ridge','lightgbm','mlp']:
        print('Fitting '+name,flush=True)
        values,info=fit_predict(name,x,y,xp)
        series=pd.Series(np.nan,index=frame.index);series.loc[predict]=values
        predictions[name]=series;model_info[name]=info
    write_json(out/'models.json',json_safe(model_info))
    panels={}
    for name,series in predictions.items():
        panel=frame[['date','asset']].assign(score=series).pivot(index='date',columns='asset',values='score')
        panels[name]=panel.reindex(index=calendar,columns=assets)
    # Persist model scores locally before any final-period outcome is evaluated.
    score_frame=frame[['date','asset']].copy()
    for name,series in predictions.items():score_frame[name]=series
    score_frame.to_parquet(extra/'scores.parquet',index=False)
    validation=[]
    for name,scores in panels.items():
        print('Validation: '+name,flush=True)
        metrics,_=evaluate(root,scores,frame,market,calendar,benchmark,'validation_'+name,'2023-01-03','2024-12-31')
        validation.append({'model':name,**metrics})
    table=pd.DataFrame(validation)
    viable=table[(table.max_drawdown<=.20)&(table.annualized_return>0)]
    pool=viable if len(viable) else table
    chosen=pool.assign(calmar=pool.annualized_return/pool.max_drawdown.clip(lower=.01)).sort_values(['calmar','model'],ascending=[False,True]).iloc[0].model
    write_json(out/'selection.json',{'selected':chosen,'rule':'validation positive return and MDD<=20%, then highest CAGR/MDD; fallback same ratio',
                                  'validation':['2023-01-03','2024-12-31'],'training_rows':int(train.sum()),
                                  'purged_training_rows':int((good&(frame.date>='2020-01-01')&(frame.date<='2022-12-31')&~train).sum()),
                                  'test_touched_before_selection':False,
                                  'caveat':'Original single-factor full-period outcomes were already observed before this protocol.'})
    table.to_csv(out/'validation.csv',index=False)
    print('Validation selected '+chosen+'; now entering final-period evaluation.',flush=True)
    final=[]
    for name,scores in panels.items():
        metrics,result=evaluate(root,scores,frame,market,calendar,benchmark,'test_'+name,'2025-01-02','2026-09-18')
        cp=cpi_comparison(result.daily,cpi)
        write_json(out/('test_'+name)/'cpi.json',json_safe(cp))
        final.append({'model':name,'selected':name==chosen,**metrics,**{'cpi_'+k:v for k,v in cp.items()}})
    pd.DataFrame(final).to_csv(out/'test.csv',index=False)
    # Controls use the selected validation model, not the final-period winner.
    for suffix,risk,scale in [('no_risk',False,1.),('double_cost',True,2.),('zero_cost',True,0.)]:
        evaluate(root,panels[chosen],frame,market,calendar,benchmark,'control_'+suffix,'2025-01-02','2026-09-18',risk,scale)
    labels=forward_returns(market,calendar,20)
    factor_summary=[]
    evaluation_dates=calendar[(calendar>='2023-01-03')&(calendar<='2026-08-18')]
    for name in FEATURES:
        scores=frame.pivot(index='date',columns='asset',values='n_'+name).reindex(index=calendar,columns=assets)
        diag=diagnostics(scores.loc[evaluation_dates],labels.loc[evaluation_dates],groups=5,min_assets=30)
        factor_summary.append({'factor':name,**diag['summary']})
        folder=out/'factors'/name;folder.mkdir(parents=True,exist_ok=True)
        for key in ['daily','groups','spread']:diag[key].to_csv(folder/f'{key}.csv',index=False)
    pd.DataFrame(factor_summary).to_csv(out/'factor_summary.csv',index=False)
    write_json(out/'provenance.json',{'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
             'data_sha256':digest(market_path),'features_sha256':digest(extra/'features.parquet'),
             'protocol_sha256':digest(root/'docs/RESEARCH_PROTOCOL_V2.md'),
             'source_hashes':sources,
             'seconds':time.monotonic()-start_time})
    write_json(out/'complete.json',{'selected':chosen,'status':'complete'})
    print('Research completed: '+chosen,flush=True)
