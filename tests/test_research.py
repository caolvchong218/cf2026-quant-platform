"""Independent fixtures for timing, accounting, data gaps, diagnostics and plugins."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from cfquant.config import Config
from cfquant.data import load_market
from cfquant.factors import compute, REGISTRY, FactorSpec, register, causal_check
from cfquant.engine import run_backtest
from cfquant.analytics import diagnostics, forward_returns, performance, reconcile


def fixture_data(n=35, assets=("A", "B", "C")):
    calendar = pd.bdate_range("2024-01-01", periods=n)
    rows = []
    for j, date in enumerate(calendar):
        for k, asset in enumerate(assets):
            price = 10 + k + j*(k+1)*.1
            rows.append(dict(date=date, asset=asset, open=price, high=price*1.03, low=price*.98,
                             close=price*1.01, raw_open=price, raw_close=price*1.01,
                             volume=10000, adj_factor=1, up_limit=price*1.1, down_limit=price*.9))
    return pd.DataFrame(rows), calendar


def config(calendar, **kwargs):
    return Config(start=str(calendar[22].date()), end=str(calendar[-1].date()), holdings=2,
                  initial_cash=1000, **kwargs)


def test_factors_manual():
    m,c = fixture_data()
    for name, window in [("momentum",20),("reversal",5),("low_volatility",20)]:
        actual = compute(m,c,name)
        prices=m[m.asset=="A"].close.to_numpy()
        expected = (prices[-1]/prices[-1-window]-1)
        if name=="reversal": expected=-expected
        if name=="low_volatility":
            r=prices[1:]/prices[:-1]-1
            expected=-np.std(r[-20:],ddof=1)
        assert actual.iloc[-1]["A"] == pytest.approx(expected)


def test_factor_calendar_gap_does_not_compress():
    m,c=fixture_data()
    m=m[~((m.asset=="A") & (m.date==c[15]))]
    result=compute(m,c,"momentum")
    assert pd.isna(result.loc[c[-1],"A"])
    assert pd.notna(result.loc[c[-1],"B"])


@pytest.mark.parametrize("name",["momentum","reversal","low_volatility"])
def test_no_future_leakage(name):
    m,c=fixture_data(70)
    assert causal_check(m,c,name)["passed"]


def test_deliberately_leaky_plugin_is_detected():
    m,c=fixture_data(70)
    spec=FactorSpec("test_leak","test","future","test",2,"test",lambda x,w:x.shift(-1))
    register(spec)
    try:
        assert not causal_check(m,c,spec.name)["passed"]
    finally:
        REGISTRY.pop(spec.name)


def test_new_plugin_runs_without_engine_changes():
    m,c=fixture_data()
    spec=FactorSpec("test_range","test","range","test",3,"test",lambda x,w: -(x.rolling(w).max()/x.rolling(w).min()-1))
    register(spec)
    try:
        scores=compute(m,c,spec.name)
        out=run_backtest(m,c,scores,config(c))
        assert len(out.trades)>0
        assert causal_check(m,c,spec.name)["passed"]
    finally:
        REGISTRY.pop(spec.name)


def test_exact_flat_price_cost_and_cash():
    m,c=fixture_data()
    for col in ["open","close","raw_open","raw_close"]:m[col]=10.
    m["up_limit"],m["down_limit"]=11.,9.
    scores=pd.DataFrame(1.,index=c,columns=["A","B","C"])
    cfg=config(c, buy_cost=.01,sell_cost=.02,rebalance_every=100)
    out=run_backtest(m,c,scores,cfg)
    buy=1000/1.03
    assert out.daily.iloc[0].buy_notional==pytest.approx(buy)
    assert out.daily.iloc[0].nav==pytest.approx(1000-buy*.01)
    assert out.daily.iloc[0].cash==pytest.approx(1000-buy*1.01)
    assert out.daily.iloc[0].turnover==pytest.approx(buy/1000)
    assert (out.trades.date>out.trades.signal_date).all()
    assert reconcile(out,1000,.01,.02)["passed"]


def test_zero_cost_buy_and_hold_matches_hand_calculation():
    m,c=fixture_data()
    scores=pd.DataFrame({"A":1.,"B":0.,"C":0.},index=c)
    cfg=replace(config(c,buy_cost=0,sell_cost=0,rebalance_every=100),holdings=1)
    out=run_backtest(m,c,scores,cfg)
    a=m[m.asset=="A"].set_index("date")
    expected=1000*a.loc[c[-1],"close"]/a.loc[c[22],"open"]
    assert out.daily.nav.iloc[-1]==pytest.approx(expected)


def test_signal_spike_not_traded_same_day():
    m,c=fixture_data()
    scores=pd.DataFrame(0.,index=c,columns=["A","B","C"])
    scores.loc[c[22]:,"C"]=10
    cfg=replace(config(c,rebalance_every=1),holdings=1)
    out=run_backtest(m,c,scores,cfg)
    buys=out.trades.query("side == 'buy'")
    assert buys.iloc[0].asset=="A"
    assert buys[buys.asset=="C"].date.min()==c[23]


def test_missing_open_no_fee_no_fill():
    m,c=fixture_data()
    m.loc[(m.date==c[22])&(m.asset=="C"),["open","raw_open"]]=np.nan
    scores=pd.DataFrame({"A":0.,"B":1.,"C":2.},index=c)
    out=run_backtest(m,c,scores,config(c))
    assert not ((out.trades.date==c[22])&(out.trades.asset=="C")).any()
    assert "missing_open" in out.orders.reason.values
    assert reconcile(out,1000,.001,.0015)["passed"]


@pytest.mark.parametrize("side",["buy","sell"])
def test_price_limit_blocks_only_requested_side(side):
    m,c=fixture_data()
    scores=pd.DataFrame({"A":2.,"B":0.,"C":0.},index=c)
    cfg=replace(config(c,rebalance_every=1),holdings=1)
    if side=="buy":
        mask=(m.date==c[22])&(m.asset=="A")
        m.loc[mask,"up_limit"]=m.loc[mask,"raw_open"]
        date=c[22]
    else:
        scores.loc[c[22]:,"A"]=0
        scores.loc[c[22]:,"B"]=3
        mask=(m.date==c[23])&(m.asset=="A")
        m.loc[mask,"down_limit"]=m.loc[mask,"raw_open"]
        date=c[23]
    out=run_backtest(m,c,scores,cfg)
    assert not ((out.trades.date==date)&(out.trades.asset=="A")&(out.trades.side==side)).any()
    assert (out.daily.cash>=0).all()


def test_stale_held_close_flagged_not_liquidated():
    m,c=fixture_data()
    m.loc[(m.date==c[23])&(m.asset=="A"),"close"]=np.nan
    scores=pd.DataFrame({"A":2.,"B":0.,"C":0.},index=c)
    cfg=replace(config(c,rebalance_every=100),holdings=1)
    out=run_backtest(m,c,scores,cfg)
    assert out.daily.set_index("date").loc[c[23],"stale_holdings"]==1
    assert out.positions[out.positions.date==c[23]].iloc[0].mark_price==pytest.approx(
        m[(m.date==c[22])&(m.asset=="A")].close.iloc[0])


def test_split_continuity_adjusted_units():
    m,c=fixture_data()
    m["raw_open"]=20.;m["raw_close"]=20.;m["adj_factor"]=1.
    after=m.date>=c[25]
    m.loc[after,["raw_open","raw_close"]]=10.
    m.loc[after,"adj_factor"]=2.
    m["open"]=m.raw_open*m.adj_factor;m["close"]=m.raw_close*m.adj_factor
    m["up_limit"]=m.raw_open*1.1;m["down_limit"]=m.raw_open*.9
    scores=pd.DataFrame(1.,index=c,columns=["A","B","C"])
    out=run_backtest(m,c,scores,config(c,buy_cost=0,sell_cost=0,rebalance_every=100))
    assert np.allclose(out.daily.nav,1000)


def test_rank_ic_average_ties_and_constant_missing():
    c=pd.bdate_range("2024-01-01",periods=2)
    scores=pd.DataFrame([[1,1,3,4],[2,2,2,2]],index=c,columns=list("ABCD"))
    labels=pd.DataFrame([[1,2,3,4],[1,2,3,4]],index=c,columns=list("ABCD"))
    out=diagnostics(scores,labels,2,4)
    expected=np.corrcoef([1.5,1.5,3,4],[1,2,3,4])[0,1]
    assert out["daily"].iloc[0].rank_ic==pytest.approx(expected)
    assert pd.isna(out["daily"].iloc[1].ic)
    assert pd.isna(out["daily"].iloc[1].rank_ic)


def test_missing_label_never_regroups():
    c=pd.bdate_range("2024-01-01",periods=1)
    scores=pd.DataFrame([[1,2,3,4]],index=c,columns=list("ABCD"))
    labels=pd.DataFrame([[np.nan,2,3,4]],index=c,columns=list("ABCD"))
    out=diagnostics(scores,labels,2,4)
    groups=out["groups"].set_index("group")
    assert groups.loc[1,"formed_count"]==2
    assert groups.loc[1,"valid_count"]==1
    assert groups.loc[1,"mean_forward_return"]==2
    assert pd.isna(out["daily"].iloc[0].ic)


def test_forward_label_uses_next_open_and_trailing_missing():
    m,c=fixture_data()
    y=forward_returns(m,c,5)
    a=m[m.asset=="A"].set_index("date").open
    assert y.loc[c[10],"A"]==pytest.approx(a.loc[c[16]]/a.loc[c[11]]-1)
    assert y.iloc[-6:].isna().all().all()


def test_drawdown_includes_initial_capital():
    m,c=fixture_data()
    scores=compute(m,c,"momentum")
    out=run_backtest(m,c,scores,config(c))
    d=out.daily.iloc[:3].copy()
    d["nav"]=[1200,900,1000]
    metrics=performance(d,1000,trades=out.trades)
    assert metrics["max_drawdown"]==pytest.approx(.25)
    d["nav"]=[900,900,900]
    assert performance(d,1000)["max_drawdown"]==pytest.approx(.1)


def test_loader_rejects_duplicate_keys():
    from io import StringIO
    m,c=fixture_data()
    m=pd.concat([m,m.iloc[:1]])
    path=StringIO();m.to_csv(path,index=False);path.seek(0)
    with pytest.raises(ValueError,match="Duplicate"):
        load_market(path)


def test_long_missing_holding_written_down_without_fake_sale():
    m,c=fixture_data(55)
    m=m[~((m.asset=="A")&(m.date>=c[24]))]
    scores=pd.DataFrame({"A":2.,"B":0.,"C":0.},index=c)
    cfg=replace(config(c,rebalance_every=100,buy_cost=0,sell_cost=0),holdings=1,max_stale_sessions=20)
    out=run_backtest(m,c,scores,cfg)
    assert out.daily.iloc[-1].written_down_holdings==1
    assert out.daily.iloc[-1].market_value==0
    assert len(out.trades)==1
    assert reconcile(out,1000,0,0)["passed"]


def test_repeatability():
    m,c=fixture_data()
    scores=compute(m,c,"momentum")
    first=run_backtest(m,c,scores,config(c))
    second=run_backtest(m,c,scores,config(c))
    pd.testing.assert_frame_equal(first.daily,second.daily)
    pd.testing.assert_frame_equal(first.trades,second.trades)


@pytest.mark.parametrize("values",[{"rebalance_every":0},{"buy_cost":-1},{"initial_cash":float("nan")},{"groups":1}])
def test_invalid_config(values):
    with pytest.raises(ValueError):Config(**values)
