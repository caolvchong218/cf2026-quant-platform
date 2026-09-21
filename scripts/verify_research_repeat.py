"""Re-run the selected strategy in an independently installed environment."""
from pathlib import Path
import json,sys,importlib.metadata
import pandas as pd
import numpy as np
from cfquant.data import load_market,load_calendar,write_json,digest
from cfquant.factors import panel
from cfquant.risk import build_targets
from cfquant.config import Config
from cfquant.engine import run_backtest
from cfquant.analytics import performance,reconcile
from cfquant.experiment import json_safe
ROOT=Path(__file__).resolve().parents[1];EXTRA=ROOT/'data/private/research_v2';RUN=ROOT/'runs/research_v2'
assert (RUN/'complete.json').exists()
selected=json.loads((RUN/'selection.json').read_text())['selected']
f=pd.read_parquet(EXTRA/'features.parquet');s=pd.read_parquet(EXTRA/'scores.parquet')
source=ROOT/'data/private/mainboard1000_20260918/data/processed'
market=load_market(source/'market.csv');calendar=load_calendar(source/'calendar.csv')
scores=s.pivot(index='date',columns='asset',values=selected)
b=pd.read_csv(EXTRA/'benchmark.csv');b['date']=pd.to_datetime(b.trade_date.astype(str));benchmark=b.set_index('date').close.sort_index()
weights,risk=build_targets(scores,f,panel(market,calendar,'close'),benchmark,'2025-01-02')
cfg=Config(start='2025-01-02',end='2026-09-18',holdings=50,rebalance_every=20,buy_cost=.001,sell_cost=.0015)
r=run_backtest(market,calendar,scores,cfg,target_weights=weights,participation_limit=.01)
checks=reconcile(r,cfg.initial_cash,cfg.buy_cost,cfg.sell_cost);assert checks['passed']
out=ROOT/'runs/research_v2_repro_clean';out.mkdir(parents=True,exist_ok=True)
comparisons={}
for field in ['daily','trades','positions','orders']:
    actual=getattr(r,field);path=out/(field+'.csv');actual.to_csv(path,index=False,float_format='%.12g')
    a=pd.read_csv(path);b=pd.read_csv(RUN/('test_'+selected)/(field+'.csv'))
    pd.testing.assert_frame_equal(a,b,rtol=1e-9,atol=5e-6)
    comparisons[field]={'rows':len(a),'matched':True}
metrics=performance(r.daily,cfg.initial_cash,trades=r.trades)
write_json(out/'checks.json',checks);write_json(out/'metrics.json',json_safe(metrics))
first=sorted((ROOT/'runs').glob('research_v2_archive_*'))[0]
for name in ['validation.csv','test.csv']:
    pd.testing.assert_frame_equal(pd.read_csv(first/name),pd.read_csv(RUN/name),atol=1e-12,rtol=1e-10)
write_json(ROOT/'evidence/research_v2/repeatability.json',{'passed':True,'selected':selected,'independent_environment':sys.version,
    'versions':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','lightgbm']},
    'matched_first_valid_run_all_5_models':True,'repeated_selected_tables':comparisons,
    'rtol':1e-9,'atol_currency_and_units':5e-6,'data_sha256':digest(source/'market.csv'),'checks':checks})
print('Independent selected-strategy repeat passed; all 5 model summaries match first valid run.',flush=True)
