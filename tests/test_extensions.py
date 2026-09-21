import numpy as np
import pandas as pd
import pytest
from cfquant.features import neutralize,financial_asof,industry_history
from cfquant.risk import constrained_weights,RiskPolicy,build_targets
from cfquant.incremental import merge_snapshots
from cfquant.engine import run_backtest
from cfquant.analytics import reconcile
from cfquant.config import Config
from test_research import fixture_data


def test_neutralization_removes_industry_and_size_exposure():
    rng=np.random.default_rng(3)
    groups=pd.Series(np.repeat(['A','B','C'],20))
    cap=pd.Series(rng.normal(size=60))
    scores=pd.DataFrame({'factor':cap*3+groups.map({'A':5,'B':-3,'C':9})+rng.normal(size=60)})
    residual=neutralize(scores,cap,groups).factor
    assert residual.groupby(groups).mean().abs().max()<1e-12
    assert abs(residual@cap)<1e-10
    assert residual.std()==pytest.approx(1.)


def test_financials_use_announcement_lag_not_statement_end():
    f=pd.DataFrame({'ann_date':['20200430','20210420'],'end_date':['20191231','20201231'],'roe':[10.,99.]})
    dates=pd.to_datetime(['20200429','20200430','20200506','20210420','20210421'])
    actual=financial_asof(f,dates)
    assert actual.iloc[:2].isna().all()
    assert actual.iloc[2]==pytest.approx(.10)
    assert actual.iloc[3]==pytest.approx(.10)
    assert actual.iloc[4]==pytest.approx(.99)
    f.loc[1,'roe']=-999
    assert financial_asof(f,dates).iloc[:4].equals(actual.iloc[:4])


def test_industry_intervals_do_not_backfill_current_membership():
    records=pd.DataFrame({'l1_code':['A','B'],'in_date':['20200102','20220101'],'out_date':['20220101',None]})
    dates=pd.to_datetime(['20191231','20200102','20211231','20220101'])
    assert industry_history(records,dates).tolist()==['UNKNOWN','A','A','B']


def test_portfolio_caps_leave_cash_without_leverage():
    idx=pd.Index([f'A{x}' for x in range(60)])
    scores=pd.Series(np.arange(60.),index=idx)
    vol=pd.Series(.01,index=idx)
    sectors=pd.Series(['ONE']*60,index=idx)
    w=constrained_weights(scores,vol,sectors,.9,RiskPolicy())
    assert w.sum()<=.25+1e-12
    assert w.max()<=.04+1e-12
    assert (w>=0).all()


def test_external_weights_execute_on_next_open_with_cash():
    market,dates=fixture_data(40)
    scores=pd.DataFrame(1.,index=dates,columns=['A','B','C'])
    weights=pd.DataFrame(0.,index=dates,columns=scores.columns)
    weights.loc[dates[21]:,'A']=.30
    cfg=Config(start=str(dates[22].date()),end=str(dates[-1].date()),holdings=2,initial_cash=1000)
    result=run_backtest(market,dates,scores,cfg,target_weights=weights)
    assert result.daily.iloc[0].cash>690
    assert set(result.trades.asset)=={'A'}
    assert reconcile(result,1000,cfg.buy_cost,cfg.sell_cost)['passed']
    weights.loc[dates[22],'A']=1.2
    with pytest.raises(ValueError,match='sum'):
        run_backtest(market,dates,scores,cfg,target_weights=weights)


def test_liquidity_cap_uses_only_prior_volume_and_charges_only_fills():
    market,dates=fixture_data(40)
    scores=pd.DataFrame(1.,index=dates,columns=['A','B','C'])
    cfg=Config(start=str(dates[22].date()),end=str(dates[22].date()),holdings=2,initial_cash=1e8)
    actual=run_backtest(market,dates,scores,cfg,participation_limit=.01)
    assert (actual.orders.status=='liquidity_limited').any()
    altered=market.copy();altered.loc[altered.date>=dates[22],'volume']*=1000
    changed=run_backtest(altered,dates,scores,cfg,participation_limit=.01)
    pd.testing.assert_frame_equal(actual.trades,changed.trades)
    assert reconcile(actual,cfg.initial_cash,cfg.buy_cost,cfg.sell_cost)['passed']


def test_incremental_matches_full_adjustment_and_repeated_import_is_idempotent():
    market,dates=fixture_data(40)
    for field in ['high','low']:market['raw_'+field]=market[field]
    market['adjustment_reference']=1.
    market.loc[market.date>=dates[20],'adj_factor']=2.
    first=market[market.date<=dates[25]].copy()
    second=market[market.date>=dates[20]].copy()
    merged,audit=merge_snapshots(first,second)
    full,_=merge_snapshots(market.iloc[:0],market)
    pd.testing.assert_frame_equal(merged,full)
    again,repeat=merge_snapshots(merged,second)
    pd.testing.assert_frame_equal(again,merged)
    assert audit['overlapping_rows']==18
    assert repeat['revised_raw_rows']==0


def test_cpi_compares_identical_complete_months_only():
    from cfquant.research import cpi_comparison
    dates=pd.bdate_range('2025-01-02','2025-04-28')
    daily=pd.DataFrame({'date':dates,'nav':np.linspace(100,120,len(dates))})
    cpi=pd.DataFrame({'month':[202501,202502,202503,202504],'nt_mom':[1,1,1,99]})
    result=cpi_comparison(daily,cpi)
    assert result['start_month']=='202502'
    assert result['end_month']=='202503'
    assert result['months']==2
    assert result['inflation']==pytest.approx(1.01**2-1)


def test_risk_targets_are_invariant_to_future_price_changes():
    dates=pd.bdate_range('2020-01-01',periods=160)
    assets=['A','B']; rng=np.random.default_rng(7)
    close=pd.DataFrame(100*np.exp(np.cumsum(rng.normal(0,.01,(160,2)),axis=0)),index=dates,columns=assets)
    scores=pd.DataFrame([[2.,1.]]*160,index=dates,columns=assets)
    frame=pd.MultiIndex.from_product([dates,assets],names=['date','asset']).to_frame(index=False)
    frame['volatility']=.01;frame['industry']='A'
    benchmark=close.mean(axis=1)
    original,_=build_targets(scores,frame,close,benchmark,dates[125])
    altered=close.copy();altered.loc[dates[146]:]*=100
    changed,_=build_targets(scores,frame,altered,benchmark.where(benchmark.index<dates[146],benchmark*100),dates[125])
    pd.testing.assert_frame_equal(original.loc[:dates[145]],changed.loc[:dates[145]])
