import numpy as np
import pandas as pd
import pytest
from rsrs import factors, decisions, backtest, audit_ledger, metrics


def prices(n=900):
    rng = np.random.default_rng(189)
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    return pd.DataFrame({"low": p * .98, "high": p * (1.02+rng.uniform(0,.01,n)),
                         "open": p, "close": p * 1.001},
                        index=pd.bdate_range("2010-01-01", periods=n))


def test_ols_matches_lstsq_and_strict_warmup():
    d=prices(); f=factors(d,18,600)
    for t in [17,200,650,899]:
        x=d.low.iloc[t-17:t+1].to_numpy();y=d.high.iloc[t-17:t+1].to_numpy()
        coef=np.linalg.lstsq(np.column_stack([np.ones(18),x]),y,rcond=None)[0]
        residual=y-(coef[0]+coef[1]*x)
        assert f.beta.iloc[t]==pytest.approx(coef[1],abs=1e-11)
        assert f.r2.iloc[t]==pytest.approx(1-(residual**2).sum()/((y-y.mean())**2).sum())
    assert f.z.iloc[:617].isna().all()
    assert np.isfinite(f.z.iloc[617])
    past=f.beta.iloc[17:617]
    assert f.z.iloc[617]==pytest.approx((f.beta.iloc[617]-past.mean())/past.std(ddof=0))


def test_future_mutation_cannot_change_past():
    d=prices();f=factors(d)
    changed=d.copy();changed.iloc[700:]*=2
    pd.testing.assert_frame_equal(f.iloc[:700],factors(changed).iloc[:700])


def test_entry_filter_does_not_force_exit():
    score=pd.Series([.8,.9,0,-.8,.9,.9]);gate=pd.Series([True,False,False,False,False,True])
    assert decisions(score,.7,-.7,gate).tolist()==[1,1,1,0,0,1]


def test_next_open_execution_and_gap_not_earned_before_buy():
    d=pd.DataFrame({"open":[100.,200.,300.,150.],"close":[100.,200.,300.,150.]},
                   index=pd.date_range("2020-01-01",periods=4))
    s=pd.Series([1,0,0,0],index=d.index)
    ledger=backtest(d,s,"2020-01-01","2020-01-04",0)
    assert ledger.equity.tolist()==pytest.approx([1,1,1.5,1.5])
    assert ledger.side.tolist()==["","BUY","SELL",""]
    audit_ledger(ledger,0)


def test_fees_reconciliation_initial_drawdown_and_benchmark():
    d=pd.DataFrame({"open":[100.,100.,100.],"close":[90.,100.,100.]},
                   index=pd.date_range("2020-01-01",periods=3))
    s=pd.Series([1,1,0],index=d.index)
    ledger=backtest(d,s,"2020-01-02","2020-01-03",10)
    assert ledger.equity.iloc[-1]==pytest.approx(1/1.001)
    audit_ledger(ledger,10)
    assert metrics(ledger)["max_drawdown"]==pytest.approx(1-1/1.001)


def test_no_implicit_alignment_or_duplicate_dates():
    d=prices(30);s=pd.Series(1,index=d.index)
    with pytest.raises(ValueError):backtest(d,s.iloc[1:],"2010","2011")
    bad=d.copy();bad.index=[d.index[0]]*30
    with pytest.raises(ValueError):backtest(bad,pd.Series(1,index=bad.index),"2010","2011")


def test_degenerate_ols_stays_unavailable():
    d=prices(50);d["high"]=3.;d["low"]=2.
    assert factors(d,5,10).z.isna().all()
