"""Descriptive preprocessing diagnostics and uncertainty; never select a model."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from cfquant.features import FEATURES
from cfquant.data import write_json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'evidence/research_v2'
f=pd.read_parquet(ROOT/'data/private/research_v2/features.parquet')
f=f[(f.date>='2023-01-03')&(f.date<='2026-08-18')]
y=f.pivot(index='date',columns='asset',values='label')
eligible=f.pivot(index='date',columns='asset',values='eligible')
rows=[]
for name in FEATURES:
    for prefix,label in [('', 'raw'),('z_','winsor_z'),('n_','industry_size_residual')]:
        x=f.pivot(index='date',columns='asset',values=prefix+name).where(eligible)
        pair=x.notna()&y.notna();xs=x.where(pair);ys=y.where(pair)
        count=pair.sum(axis=1)
        ic=xs.corrwith(ys,axis=1).where(count>=30)
        ric=xs.rank(axis=1).corrwith(ys.rank(axis=1),axis=1).where(count>=30)
        rows.append({'factor':name,'processing':label,'ic_mean':ic.mean(),'rank_ic_mean':ric.mean(),
                     'valid_days':int(ric.notna().sum()),'mean_valid_pairs':count.mean()})
pd.DataFrame(rows).to_csv(OUT/'preprocessing.csv',index=False)
selected=json.loads((OUT/'selection.json').read_text())['selected']
# Read archived first run if a deterministic repeat is currently in progress.
candidates=[ROOT/'runs/research_v2']+sorted((ROOT/'runs').glob('research_v2_archive_*'),reverse=True)
source=next(p/('test_'+selected)/'daily.csv' for p in candidates if (p/('test_'+selected)/'daily.csv').exists())
r=pd.read_csv(source)['return'].to_numpy();rng=np.random.default_rng(20260921)
samples=[];n=len(r);block=20
for _ in range(2000):
    starts=rng.integers(0,n,size=int(np.ceil(n/block)))
    idx=((starts[:,None]+np.arange(block))%n).ravel()[:n]
    samples.append(np.prod(1+r[idx])**(252/n)-1)
low,high=np.quantile(samples,[.025,.975])
write_json(OUT/'uncertainty.json',{'method':'circular moving-block bootstrap of realized daily net returns',
 'block_sessions':block,'replicates':2000,'seed':20260921,'cagr_95_percentile_interval':[float(low),float(high)],
 'caveat':'Descriptive conditional interval only: fixed realized strategy, no model-selection or regime-change uncertainty.'})
print('Supplemental evidence complete',flush=True)
