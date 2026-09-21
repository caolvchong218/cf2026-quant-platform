"""Publish small derived evidence only; never copy vendor rows or credentials."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from cfquant.data import digest,write_json
from cfquant.features import FEATURE_CARDS
from cfquant.analytics import performance

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/research_v2'; OUT=ROOT/'evidence/research_v2'
EXTRA=ROOT/'data/private/research_v2'
assert (RUN/'complete.json').exists(), 'Research is not complete'
OUT.mkdir(parents=True,exist_ok=True)
for name in ['selection.json','validation.csv','test.csv','factor_summary.csv','models.json','provenance.json']:
    shutil.copy2(RUN/name,OUT/name)
selection=json.loads((RUN/'selection.json').read_text(encoding='utf-8'))
chosen=selection['selected']
daily=pd.read_csv(RUN/('test_'+chosen)/'daily.csv',parse_dates=['date'])
curves=[daily[['date']].assign(nav=daily.nav/1e6,series='提交策略 · 扣费')]
benchmark=pd.read_csv(EXTRA/'benchmark.csv')
benchmark['date']=pd.to_datetime(benchmark.trade_date.astype(str))
benchmark=benchmark.set_index('date').sort_index()
base=benchmark.loc[benchmark.index<daily.date.min(),'close'].iloc[-1]
b=benchmark.reindex(daily.date).close
curves.append(pd.DataFrame({'date':daily.date.to_numpy(),'nav':b.to_numpy()/base,'series':'沪深300 · 价格指数'}))
curves.append(daily[['date']].assign(nav=1.,series='现金 · 零利息'))
pd.concat(curves,ignore_index=True).to_csv(OUT/'curves.csv',index=False)
benchmark_metrics=performance(pd.DataFrame({'nav':b.to_numpy()/base,'cost':0.,'turnover':0.,'max_weight':1.,
    'holdings_count':1,'blocked_orders':0,'stale_holdings':0,'gross_attribution_nav':b.to_numpy()/base}),1.)
write_json(OUT/'benchmark_metrics.json',benchmark_metrics)
control=[]
for name,label in [('test_'+chosen,'预选策略 / 基准费用'),('control_no_risk','相同打分 / 满仓等权组合'),
                   ('control_double_cost','预选策略 / 双倍费用'),('control_zero_cost','预选策略 / 零费用重跑')]:
    m=json.loads((RUN/name/'metrics.json').read_text())
    control.append({'experiment':label,**m})
pd.DataFrame(control).to_csv(OUT/'controls.csv',index=False)
shutil.copy2(RUN/('test_'+chosen)/'risk.csv',OUT/'risk.csv')
shutil.copy2(RUN/('test_'+chosen)/'cpi.json',OUT/'cpi.json')
write_json(OUT/'factor_cards.json',FEATURE_CARDS)
legacy=pd.read_csv(ROOT/'runs/expanded_momentum_5cd361c1844e/daily.csv',parse_dates=['date'])
legacy_years=[]
for year,g in legacy.groupby(legacy.date.dt.year):
    legacy_years.append({'year':int(year),'net_return':float((1+g['return']).prod()-1),'cost_yuan':float(g.cost.sum()),
                         'two_sided_turnover':float(g.turnover.sum()),'sessions':len(g)})
pd.DataFrame(legacy_years).to_csv(OUT/'legacy_years.csv',index=False)
q=json.loads((EXTRA/'feature_quality.json').read_text(encoding='utf-8'))
qa=pd.DataFrame(q['dates']);qa.to_csv(OUT/'feature_coverage.csv',index=False)
f=pd.read_parquet(EXTRA/'financials.parquet');d=pd.read_parquet(EXTRA/'daily_basic.parquet')
i=pd.read_parquet(EXTRA/'industries.parquet')
manifest=json.loads((EXTRA/'manifest.json').read_text(encoding='utf-8'))
mismatch=[name for name,expected in manifest['files'].items() if digest(EXTRA/name)!=expected]
assert not mismatch,mismatch
checks={p.parent.name:json.loads(p.read_text()) for p in RUN.glob('*/checks.json')}
assert len(checks)==13 and all(x['passed'] for x in checks.values())
stats={'price_assets':1000,'price_rows':1656599,'calendar_sessions':1690,'price_start':'2019-10-08','price_end':'2026-09-18',
       'daily_basic_rows':len(d),'daily_basic_duplicate_keys':int(d.duplicated(['ts_code','trade_date']).sum()),
       'financial_rows':len(f),'max_financial_rows_per_request':int(f.groupby('ts_code').size().max()),
       'financial_endpoint_cap':100,'industry_interval_rows':len(i),'feature_rows':q['rows'],
       'features':len(FEATURE_CARDS),'research_input_hashes_checked':len(manifest['files']),
       'eligible_mean':float(qa.eligible.mean()),'unknown_industry_fraction':float(qa.industry_unknown.sum()/qa.eligible.sum()),
       'annual_roe_coverage_mean':float(qa.quality_coverage.mean()),
       'research_storage_bytes':sum(p.stat().st_size for p in EXTRA.rglob('*') if p.is_file())}
write_json(OUT/'data_audit.json',stats)
# Small, complete diagnostic tables retain formed and valid counts for review.
for p in (RUN/'factors').glob('*/*.csv'):
    dest=OUT/'factors'/p.parent.name/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
acceptance={'status':'research accounting and source hashes passed','research_runs':checks,
 'data':stats,'selection_rule':selection['rule'],
 'implemented_extensions':['dated fundamentals and historical industry neutralization','risk-aware target weights and cash',
                           'prior-volume participation caps and partial fills','idempotent snapshot merge and revision audit'],
 'limitations':['fixed 2019 pool is not the whole market','vendor revisions are not a full vintage database',
                'fractional adjusted units, no exchange lots or detailed dividend cash ledger',
                'targets may drift after formation','MLP fixed 8 epochs did not claim convergence',
                'original full-period momentum outcomes were observed before the protocol']}
write_json(OUT/'acceptance.json',acceptance)
write_json(OUT/'manifest.json',{'files':{p.relative_to(OUT).as_posix():digest(p) for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'}})
print(json.dumps({'selected':chosen,'data':stats,'checks':len(checks)},ensure_ascii=False))
