import numpy as np
import pandas as pd
import pytest
from cfquant.research_lab import training_dates,blend_target_weights,neutralize_panel,select_factors,research_cards
from cfquant.features import neutralize
from cfquant.benchmark_research import benchmark_metrics,version_only_cache_match
import hashlib


def test_formula_names_never_replace_economic_factors_or_explanations():
    cards=research_cards()
    assert len(cards)==28 and len({c['id'] for c in cards})==28
    by_id={c['id']:c for c in cards}
    assert by_id['range_20']['expression'].startswith('-mean')
    assert by_id['dsl_range_20']['id']!='range_20'
    with pytest.raises(ValueError,match='Duplicate factor'):
        research_cards([{'id':'range_20','expression':'returns(close, 5)'}])


def test_release_number_only_preserves_legacy_cache_but_policy_edits_fail(tmp_path):
    old=b'__version__ = "2.0.0"\r\nimport pandas as _pd\r\n_pd.set_option("compute.use_numexpr", False)\r\n'
    expected=hashlib.sha256(old).hexdigest();path=tmp_path/'__init__.py'
    path.write_bytes(old.replace(b'2.0.0',b'2.3.0').replace(b'\r\n',b'\n'))
    assert version_only_cache_match(path,expected,'src/cfquant/__init__.py')
    assert not version_only_cache_match(path,expected,'src/cfquant/data.py')
    path.write_bytes(path.read_bytes().replace(b'False',b'True'))
    assert not version_only_cache_match(path,expected,'src/cfquant/__init__.py')


def test_training_requires_realized_exit_and_prior_date():
    dates=pd.date_range('2020-01-01',periods=5)
    exits=dates+pd.Timedelta(days=2)
    assert training_dates(dates,exits,'2020-01-05').tolist()==[True,True,False,False,False]


def test_descending_vendor_benchmark_uses_previous_session_not_earliest():
    dates=pd.date_range('2020-01-01',periods=4)
    benchmark=pd.Series([100.,200.,220.,240.],index=dates)
    daily=pd.DataFrame({'date':dates[2:],'nav':[1100.,1200.]})
    first=benchmark_metrics(daily,benchmark,1000.)
    reverse=benchmark_metrics(daily,benchmark.sort_index(ascending=False),1000.)
    assert first==reverse
    assert reverse['benchmark_return']==pytest.approx(.2)
    with pytest.raises(ValueError):benchmark_metrics(daily,pd.concat([benchmark,benchmark.iloc[:1]]),1000.)


def test_target_blending_nets_overlaps_and_leaves_cash():
    first=pd.DataFrame([[.4,.2]],columns=['a','b'])
    second=pd.DataFrame([[.2,.4]],columns=['a','b'])
    result=blend_target_weights(first,second)
    np.testing.assert_allclose(result,[[.3,.3]])
    assert result.sum(axis=1).iloc[0]==pytest.approx(.6)
    with pytest.raises(ValueError):blend_target_weights(first,second,-.1)
    with pytest.raises(ValueError):blend_target_weights(first,second[['b','a']])


def test_panel_neutralization_matches_cross_section_and_prefix():
    rng=np.random.default_rng(43);dates=pd.date_range('2020-01-01',periods=4)
    cols=[str(i) for i in range(20)]
    values=pd.DataFrame(rng.normal(size=(4,20)),index=dates,columns=cols)
    cap=pd.DataFrame(rng.normal(size=(4,20)),index=dates,columns=cols)
    industry=pd.DataFrame([['a']*10+['b']*10]*4,index=dates,columns=cols)
    out=neutralize_panel(values,cap,industry)
    expected=neutralize(values.iloc[0].to_frame('v'),cap.iloc[0],industry.iloc[0]).v
    np.testing.assert_allclose(out.iloc[0],expected)
    pd.testing.assert_frame_equal(out.iloc[:2],neutralize_panel(values.iloc[:2],cap.iloc[:2],industry.iloc[:2]))


def test_screening_does_not_read_holdout_and_excludes_redundant():
    rng=np.random.default_rng(41);dates=pd.date_range('2020-01-01',periods=80)
    y=pd.DataFrame(rng.normal(size=(80,12)),index=dates)
    factors={'a':y.copy(),'duplicate':y*2,'other':pd.DataFrame(rng.normal(size=y.shape),index=dates)}
    selected,_=select_factors(factors,y,dates[:65],limit=2)
    y.iloc[65:]*=-100
    selected2,_=select_factors(factors,y,dates[:65],limit=2)
    assert selected==selected2 and not {'a','duplicate'}.issubset(selected)
