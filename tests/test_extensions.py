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
