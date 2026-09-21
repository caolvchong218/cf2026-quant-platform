"""Append research inputs without changing the accepted price snapshot."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import pandas as pd
from cfquant.data import TushareProvider, write_json, digest

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'data/private/research_v2'
DEST.mkdir(parents=True,exist_ok=True)
parser=argparse.ArgumentParser()
parser.add_argument('--token-file',help='Token file, otherwise use TUSHARE_TOKEN')
args=parser.parse_args()
p=TushareProvider(DEST/'raw',args.token_file,delay=.4)
codes=pd.read_csv(ROOT/'data/private/mainboard1000_20260918/data/processed/universe.csv').ts_code.tolist()

def fetch(code):
    base={'ts_code':code,'start_date':'20191001','end_date':'20260918'}
    daily=p.query('daily_basic',base,'ts_code,trade_date,total_mv,circ_mv,pe_ttm,pb,dv_ttm,turnover_rate')
    fina=p.query('fina_indicator',{**base,'start_date':'20180101'},'ts_code,ann_date,end_date,roe,roa,debt_to_assets,ocf_to_or')
    # Official fina_indicator limit is 100 rows, not the daily endpoint limit.
    # Fail closed: split the report-period range before accepting a capped reply.
    if len(daily)>=6000 or len(fina)>=100:raise ValueError('Possible truncation; split report-date range: '+code)
    return daily,fina

daily_frames,fina_frames=[],[]
with ThreadPoolExecutor(max_workers=4) as pool:
    try:
        for i,(daily,fina) in enumerate(pool.map(fetch,codes)):
            daily_frames.append(daily);fina_frames.append(fina)
            if (i+1)%10==0:
                print(f'Research fundamentals {i+1}/1000',flush=True)
                write_json(DEST/'progress.json',{'assets':i+1,'target':1000,'status':'downloading'})
    except Exception:
        pool.shutdown(wait=True,cancel_futures=True)
        raise
pd.concat(daily_frames,ignore_index=True).to_parquet(DEST/'daily_basic.parquet',index=False)
pd.concat(fina_frames,ignore_index=True).to_parquet(DEST/'financials.parquet',index=False)
classify=p.query('index_classify',{'level':'L1','src':'SW2021'},'index_code,industry_name')
sectors=[]
for code in classify.index_code:
    for flag in ['N','Y']:
        frame=p.query('index_member_all',{'l1_code':code,'is_new':flag},'ts_code,l1_code,l1_name,in_date,out_date,is_new')
        if len(frame)>=2000:raise ValueError('Industry truncated; query constituent symbols individually')
        sectors.append(frame)
industries=pd.concat(sectors,ignore_index=True).drop_duplicates()
industries[industries.ts_code.isin(codes)].to_parquet(DEST/'industries.parquet',index=False)
p.query('cn_cpi',{'start_m':'201912','end_m':'202608'},'month,nt_mom,nt_yoy').to_csv(DEST/'cpi.csv',index=False)
p.query('index_daily',{'ts_code':'000300.SH','start_date':'20191001','end_date':'20260918'},'ts_code,trade_date,open,close').to_csv(DEST/'benchmark.csv',index=False)
files=sorted((DEST/'raw').glob('*.json'))+list(DEST.glob('*.parquet'))+list(DEST.glob('*.csv'))
write_json(DEST/'manifest.json',{'files':{f.relative_to(DEST).as_posix():digest(f) for f in files},'provider':'Tushare Pro','period':['20191001','20260918']})
write_json(DEST/'progress.json',{'assets':1000,'target':1000,'status':'complete'})
print('Research inputs complete',flush=True)
