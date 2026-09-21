"""Predeclared original zero-cost and same-period comparisons; not model selection."""
from pathlib import Path
from dataclasses import replace
import json
import pandas as pd
from cfquant.config import Config
from cfquant.data import load_market,load_calendar,write_json
from cfquant.factors import compute
from cfquant.engine import run_backtest
from cfquant.analytics import performance,reconcile
from cfquant.experiment import json_safe
ROOT=Path(__file__).resolve().parents[1]
cfg=Config.load(ROOT/'configs/expanded.yaml')
market=load_market(ROOT/cfg.data_path);dates=load_calendar(ROOT/cfg.calendar_path)
scores=compute(market,dates,cfg.factor,cfg.factor_params)
out=ROOT/'evidence/research_v2';rows=[]
for name,c in [('legacy_recomputed',cfg),('legacy_zero_cost',replace(cfg,buy_cost=0.,sell_cost=0.)),
               ('legacy_same_period',replace(cfg,start='2025-01-02'))]:
    print('Audit '+name,flush=True)
    r=run_backtest(market,dates,scores,c)
    checks=reconcile(r,c.initial_cash,c.buy_cost,c.sell_cost);assert checks['passed']
    metrics=performance(r.daily,c.initial_cash,trades=r.trades)
    rows.append({'experiment':name,'start':c.start,'end':c.end,**metrics})
    dest=ROOT/'runs/research_v2_audit'/name;dest.mkdir(parents=True,exist_ok=True)
    for field in ['daily','trades','positions','orders']:getattr(r,field).to_csv(dest/(field+'.csv'),index=False)
    write_json(dest/'checks.json',checks);write_json(dest/'metrics.json',json_safe(metrics))
pd.DataFrame(rows).to_csv(out/'legacy_controls.csv',index=False)
print(pd.DataFrame(rows)[['experiment','total_return','max_drawdown']].to_string(index=False),flush=True)
