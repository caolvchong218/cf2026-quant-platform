"""Idempotent snapshot merging with one adjustment anchor per asset."""
from pathlib import Path
import pandas as pd
import numpy as np
from .data import load_market,write_json,digest


def merge_snapshots(existing, incoming):
    for frame in (existing,incoming):
        if frame.duplicated(['date','asset']).any():raise ValueError('Duplicate keys within a source')
    old=existing.set_index(['date','asset'])
    new=incoming.set_index(['date','asset'])
    overlap=old.index.intersection(new.index)
    raw_fields=['raw_open','raw_high','raw_low','raw_close','adj_factor','volume']
    changed=0
    if len(overlap):
        changed=int((~np.isclose(old.loc[overlap,raw_fields].to_numpy(dtype=float),
                                new.loc[overlap,raw_fields].to_numpy(dtype=float),equal_nan=True)).any(axis=1).sum())
    merged=pd.concat([existing,incoming]).drop_duplicates(['date','asset'],keep='last').sort_values(['asset','date']).copy()
    reference=merged.groupby('asset').adj_factor.transform('first')
    for field in ['open','high','low','close']:
        merged[field]=merged['raw_'+field]*merged.adj_factor/reference
    merged['adjustment_reference']=reference
    merged=merged.sort_values(['date','asset']).reset_index(drop=True)
    return merged,{'previous_rows':len(existing),'incoming_rows':len(incoming),
                   'overlapping_rows':len(overlap),'revised_raw_rows':changed,'result_rows':len(merged)}


def write_merged_snapshot(existing_path,incoming_path,destination):
    destination=Path(destination)
    if destination.exists():raise ValueError('Choose a new destination to preserve accepted snapshots')
    combined,audit=merge_snapshots(load_market(existing_path),load_market(incoming_path))
    destination.mkdir(parents=True)
    combined.to_csv(destination/'market.csv',index=False,float_format='%.12g')
    write_json(destination/'merge_audit.json',{**audit,'old_hash':digest(Path(existing_path)),
                 'incoming_hash':digest(Path(incoming_path)),'output_hash':digest(destination/'market.csv')})
    return audit
