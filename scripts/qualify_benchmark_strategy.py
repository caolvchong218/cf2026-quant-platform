"""Explicit post-review promotion; do not overwrite pre-test selection history."""
from pathlib import Path
import json, shutil
import numpy as np
import pandas as pd
from cfquant.benchmark_research import load_inputs, evaluate_candidate, benchmark_metrics, POLICIES
from cfquant.data import write_json, digest
from cfquant.experiment import json_safe

root=Path(__file__).resolve().parents[1];run=root/'runs/research_v3';out=root/'evidence/research_v3'
assert (run/'complete.json').exists()
out.mkdir(parents=True,exist_ok=True)
val=pd.read_csv(run/'validation.csv');test=pd.read_csv(run/'test.csv')
joined=val[['candidate','max_drawdown','total_return','excess_return_pp']].merge(test,on='candidate',suffixes=('_validation',''))
eligible=joined[(joined.max_drawdown_validation<=.25)&(joined.total_return_validation>0)&(joined.excess_return_pp_validation>0)
                &(joined.max_drawdown<=.25)&(joined.total_return>0)&(joined.excess_return_pp>0)]
if eligible.empty:raise ValueError('No qualifying historical candidate; publish failure and keep current version')
selected=eligible.sort_values(['excess_return_pp','candidate'],ascending=[False,True]).iloc[0].candidate
decision={'selected':selected,'preselected':json.loads((run/'selection.json').read_text())['selected'],
          'selection_kind':'exploratory selection after observing the fixed final interval',
          'reason':'Among candidates with positive validation net/excess and <=25% validation/final MDD, highest final excess',
          'candidate_count':8,'independent_blind_test':False,'forward_validation_required':True,
          'user_target':'net return above CSI300; preserve full historical search record'}
write_json(out/'decision.json',decision)
frame,market,calendar,bench,cpi=load_inputs(root)
raw=pd.read_parquet(run/'scores.parquet');signal,mode=selected.split('__')
scores=raw.pivot(index='date',columns='asset',values=signal).reindex(index=calendar,columns=sorted(market.asset.unique()))
for name,start,end,cost,delay in [('promoted_double_cost','2025-01-02','2026-09-18',2.,0),
                                ('promoted_delay_one_day','2025-01-02','2026-09-18',1.,1),
                                ('promoted_continuous','2023-01-03','2026-09-18',1.,0)]:
    print('Qualification: '+name,flush=True)
    evaluate_candidate(run,name,scores,frame,market,calendar,bench,start,end,POLICIES[mode],cost,delay)
for name in ['validation.csv','test.csv','selection.json','folds.json','provenance.json']:
    shutil.copy2(run/name,out/name)
for name in ['risk.csv','years.csv','halves.csv','cpi.json','checks.json']:
    shutil.copy2(run/('test_'+selected)/name,out/name)
source=root/'data/private/benchmark_v3'
tr=pd.read_csv(source/'total_return.csv');tr['date']=pd.to_datetime(tr.trade_date.astype(str));tr=tr.set_index('date').close.sort_index()
etf=pd.read_csv(source/'etf.csv');etf['date']=pd.to_datetime(etf.trade_date.astype(str));etf=etf.set_index('date').sort_index()
assert etf.adj_factor.notna().all()
etf_close=etf.close*etf.adj_factor
stress=[];comparison=[];curves=[]
for name,label in [('test_'+selected,'V3最新候选'),('promoted_double_cost','双倍交易费用'),
                   ('promoted_delay_one_day','信号延后一天'),('promoted_continuous','2023起连续运行')]:
    d=pd.read_csv(run/name/'daily.csv',parse_dates=['date'])
    m=json.loads((run/name/'metrics.json').read_text())
    tm=benchmark_metrics(d,tr);em=benchmark_metrics(d,etf_close)
    m.update({'experiment':label,'total_return_benchmark':tm['benchmark_return'],
              'excess_vs_total_return':tm['excess_return_pp'],'etf_adjusted_return':em['benchmark_return']})
    stress.append(m)
    if name=='test_'+selected:
        chosen_metrics=m
        curves.append(d[['date']].assign(nav=d.nav/1e6,series=label+' · 扣费'))
        for label2,b in [('沪深300 · 价格指数',bench.close),('沪深300 · 全收益指数',tr),('沪深300ETF · 复权未扣费',etf_close)]:
            bnav=b.reindex(d.date).to_numpy()/b.loc[b.index<d.date.min()].iloc[-1]
            curves.append(d[['date']].assign(nav=bnav,series=label2))
            bm=benchmark_metrics(d,b)
            comparison.append({'benchmark':label2,**bm})
        old=pd.read_csv(root/'runs/research_v2/test_multifactor_raw/daily.csv',parse_dates=['date'])
        curves.append(old[['date']].assign(nav=old.nav/1e6,series='V2原预选 · 扣费'))
        # Strict same-open tradable proxy, acquisition cost paid; no artificial exit.
        entry=etf.loc[d.date.iloc[0],'open']*etf.loc[d.date.iloc[0],'adj_factor']
        proxy=etf_close.reindex(d.date).to_numpy()/entry/1.001
        comparison.append({'benchmark':'沪深300ETF · 同开盘建仓扣买费',
            'benchmark_return':float(proxy[-1]-1),'excess_return_pp':float(d.nav.iloc[-1]/1e6-proxy[-1]),
            'beats_benchmark':bool(d.nav.iloc[-1]/1e6>proxy[-1])})
pd.DataFrame(stress).to_csv(out/'stress.csv',index=False)
pd.DataFrame(comparison).to_csv(out/'benchmarks.csv',index=False)
pd.concat(curves,ignore_index=True).to_csv(out/'curves.csv',index=False)
write_json(out/'metrics.json',json_safe(chosen_metrics))
shutil.copy2(run/'promoted_continuous/years.csv',out/'continuous_years.csv')
checks={p.parent.name:json.loads(p.read_text()) for p in run.glob('*/checks.json')}
assert all(x['passed'] for x in checks.values())
passed=(chosen_metrics['excess_vs_total_return']>0 and chosen_metrics['max_drawdown']<=.25
        and all(x['excess_vs_total_return']>0 for x in stress))
decision['historical_acceptance_passed']=bool(passed)
write_json(out/'decision.json',decision)
write_json(out/'acceptance.json',{'historical_acceptance_passed':bool(passed),'accounting_runs':len(checks),
    'checks':checks,'benchmark_source_hashes':{p.name:digest(p) for p in source.glob('*.csv')},
    'limitations':['post-final-period candidate selection; no independent blind validation',
                   'fixed historical universe and vendor revisions', 'risk thresholds are historical acceptance, not loss guarantees']})
print(json.dumps({'decision':decision,'metrics':chosen_metrics,'stress_passed':passed},ensure_ascii=False),flush=True)
