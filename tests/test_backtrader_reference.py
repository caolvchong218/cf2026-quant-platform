"""Independent library cross-check of cash, commissions and close-marked NAV."""
import numpy as np
import pandas as pd
import pytest
from cfquant.config import Config
from cfquant.engine import run_backtest
bt = pytest.importorskip("backtrader")


def test_fixed_fill_ledger_against_backtrader():
    class BuyOnce(bt.Strategy):
        def __init__(self):
            self.record=[]
        def next(self):
            if len(self)==1:
                self.buy(size=40)
            self.record.append((self.data.datetime.date(0),self.broker.getvalue(),self.broker.getcash()))
    dates=pd.bdate_range("2024-01-01",periods=5)
    df=pd.DataFrame({"open":[10,11,12,13,14],"high":[11,12,13,14,15],
                     "low":[9,10,11,12,13],"close":[10.5,11.5,12.5,13.5,14.5],
                     "volume":10000},index=dates)
    cerebro=bt.Cerebro()
    cerebro.broker.setcash(1000)
    cerebro.broker.setcommission(commission=.001)
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    cerebro.addstrategy(BuyOnce)
    result=cerebro.run()[0]
    cash=1000-40*11*(1+.001)
    expected=np.r_[1000,cash+40*df.close.iloc[1:].to_numpy()]
    np.testing.assert_allclose([r[1] for r in result.record],expected,rtol=1e-12)
    assert result.record[-1][2]==pytest.approx(cash)


def test_platform_buy_hold_matches_independent_backtrader():
    dates=pd.bdate_range("2024-01-01",periods=8)
    df=pd.DataFrame({"open":np.arange(10.,18.),"close":np.arange(10.5,18.5)},index=dates)
    df["high"]=df.close+1;df["low"]=df.open-1;df["volume"]=10000
    quantity=1000/(1+.001+.0015)/11
    class Reference(bt.Strategy):
        def __init__(self):self.values=[]
        def next(self):
            if len(self)==1:self.buy(size=quantity)
            self.values.append(self.broker.getvalue())
    cerebro=bt.Cerebro()
    cerebro.broker.setcash(1000)
    cerebro.broker.setcommission(commission=.001)
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    cerebro.addstrategy(Reference)
    reference=cerebro.run()[0]
    market=df.rename_axis("date").reset_index()
    market["asset"]="A";market["raw_open"]=market.open;market["raw_close"]=market.close
    market["adj_factor"]=1.;market["up_limit"]=market.open*1.1;market["down_limit"]=market.open*.9
    cfg=Config(start=str(dates[1].date()),end=str(dates[-1].date()),initial_cash=1000,
               holdings=1,rebalance_every=100,buy_cost=.001,sell_cost=.0015)
    platform=run_backtest(market,dates,pd.DataFrame(1.,index=dates,columns=["A"]),cfg)
    np.testing.assert_allclose(platform.daily.nav,reference.values[1:],atol=1e-9,rtol=1e-12)
