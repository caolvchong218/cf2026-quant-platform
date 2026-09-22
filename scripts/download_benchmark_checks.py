"""Discover the total-return index, retain source identity and optional ETF proxy."""
from pathlib import Path
import argparse
import pandas as pd
from cfquant.data import TushareProvider, write_json, digest

root = Path(__file__).resolve().parents[1]
out = root/'data/private/benchmark_v3'
parser=argparse.ArgumentParser()
parser.add_argument('--token-file',default='D:/Desktop/tushare_token.txt')
args=parser.parse_args()
p = TushareProvider(out/'raw', args.token_file, delay=.4)
status = {}
for market in ['CSI', 'SSE']:
    try:
        directory = p.query('index_basic', {'market': market}, 'ts_code,name,market,publisher')
        directory.to_csv(out/f'directory_{market}.csv', index=False)
        found = directory[directory.ts_code.str.startswith('H00300', na=False)]
        status[market] = {'directory_rows': len(directory), 'matches': found.to_dict('records')}
        if len(found) == 1:
            code = str(found.iloc[0].ts_code)
            b = p.query('index_daily', {'ts_code': code, 'start_date': '20191001', 'end_date': '20260918'},
                        'ts_code,trade_date,open,close')
            b.to_csv(out/'total_return.csv', index=False)
            status['total_return'] = {'code': code, 'rows': len(b)}
    except RuntimeError as exc:
        status[market] = {'unavailable': str(exc)}
    write_json(out/'status.json', status)
try:
    params = {'ts_code': '510330.SH', 'start_date': '20191001', 'end_date': '20260918'}
    daily = p.query('fund_daily', params, 'ts_code,trade_date,open,close')
    adj = p.query('fund_adj', params, 'ts_code,trade_date,adj_factor')
    assert not daily.duplicated(['ts_code','trade_date']).any()
    assert not adj.duplicated(['ts_code','trade_date']).any()
    daily.merge(adj, on=['ts_code','trade_date'], how='left', validate='one_to_one').to_csv(out/'etf.csv', index=False)
    status['etf'] = {'code': '510330.SH', 'rows': len(daily), 'adjustment_rows': len(adj)}
except RuntimeError as exc:
    status['etf'] = {'unavailable': str(exc)}
write_json(out/'status.json', status)
write_json(out/'manifest.json', {'files': {p.relative_to(out).as_posix(): digest(p)
    for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'}})
print(status, flush=True)
