"""Signal-date portfolio construction: turnover buffers, exposure and risk limits."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskPolicy:
    holdings: int = 50
    rebalance_every: int = 20
    buffer: int = 20
    target_volatility: float = .12
    max_exposure: float = .90
    max_stock_weight: float = .04
    max_industry_weight: float = .25
    trend_window: int = 120
    defensive_scale: float = .25


def constrained_weights(scores, volatility, industry, exposure, policy, previous=()):
    valid=scores.dropna().sort_index().sort_values(ascending=False,kind='stable')
    top=valid.index[:policy.holdings+policy.buffer]
    retained=[x for x in previous if x in top]
    selected=retained+[x for x in valid.index if x not in retained]
    selected=selected[:policy.holdings]
    result=pd.Series(0.,index=scores.index)
    if not selected:return result
    vol=volatility.reindex(selected).clip(lower=.005).fillna(.02)
    w=1/vol; w=w/w.sum()*exposure
    # Capping leaves cash; no repeated redistribution can violate another cap.
    w=w.clip(upper=policy.max_stock_weight)
    groups=industry.reindex(selected).fillna('UNKNOWN')
    for group in groups.unique():
        idx=groups.index[groups==group];total=w.loc[idx].sum()
        if total>policy.max_industry_weight:w.loc[idx]*=policy.max_industry_weight/total
    result.loc[selected]=w
    return result


def build_targets(scores, features, close, benchmark_close, start, policy=RiskPolicy()):
    """All estimates end on formation date; engine consumes the previous row."""
    dates=scores.index
    pivot=lambda col:features.pivot(index='date',columns='asset',values=col).reindex(index=dates,columns=scores.columns)
    vol=pivot('volatility'); industry=pivot('industry')
    returns=close.pct_change(fill_method=None)
    bench=benchmark_close.reindex(dates)
    trend=bench.rolling(policy.trend_window,min_periods=policy.trend_window).mean()
    execution_dates=dates[dates>=pd.Timestamp(start)]
    formations={dates[dates.get_loc(d)-1] for d in execution_dates[::policy.rebalance_every] if dates.get_loc(d)>0}
    targets=pd.DataFrame(0.,index=dates,columns=scores.columns)
    current=pd.Series(0.,index=scores.columns); log=[]
    for date in dates:
        if date in formations:
            row=scores.loc[date]
            seed=constrained_weights(row,vol.loc[date],industry.loc[date],1.,policy,current[current>0].index)
            if seed.sum()>0:
                normalized=seed/seed.sum()
                past=returns.loc[:date].tail(60)
                selected=normalized[normalized>0]
                # Missing past observations contribute zero here solely to a risk
                # estimate; missing execution quotes remain blocked in the engine.
                hist=past[selected.index].fillna(0).dot(selected)
                sigma=float(hist.std(ddof=1)*np.sqrt(252))
                risk_scale=min(1.,policy.target_volatility/sigma) if sigma>1e-8 else 0.
            else:sigma=np.nan;risk_scale=0.
            defensive=not (pd.notna(trend.loc[date]) and bench.loc[date]>=trend.loc[date])
            exposure=policy.max_exposure*risk_scale*(policy.defensive_scale if defensive else 1.)
            current=constrained_weights(row,vol.loc[date],industry.loc[date],exposure,policy,current[current>0].index)
            log.append({'date':date,'target_exposure':float(current.sum()),'estimated_volatility':sigma,'defensive':defensive,
                        'max_stock_weight':float(current.max()),'holdings':int((current>0).sum())})
        targets.loc[date]=current
    return targets,pd.DataFrame(log)
