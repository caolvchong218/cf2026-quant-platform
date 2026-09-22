"""Public UI flow works with the portable demo when real data are absent."""
from pathlib import Path
from streamlit.testing.v1 import AppTest

def test_navigation_and_default_factor_window():
    root=Path(__file__).resolve().parents[1]
    app=AppTest.from_file(str(root/"app.py"),default_timeout=60).run()
    assert not app.exception
    app.sidebar.radio[0].set_value("回测实验").run()
    from cfquant.strategies import catalogue
    assert app.selectbox(key='strategy_version').value == catalogue(root)[0].id
    assert {x.label:x.value for x in app.metric}['年化夏普比率']=='1.255'
    app.selectbox(key='strategy_version').set_value('v2/multifactor_raw').run()
    assert {x.label:x.value for x in app.metric}['年化夏普比率']=='0.573'
    assert any(x.value == '6.12%' for x in app.metric)
    assert any('未达标' in x.value for x in app.error)
    app.selectbox(key='strategy_version').set_value('legacy_single_factor').run()
    app.sidebar.selectbox[0].set_value(app.sidebar.selectbox[0].options[0]).run()
    app.selectbox(key="backtest_factor").set_value("reversal")
    app.button[0].click().run(timeout=60)
    assert not app.exception
    assert app.session_state["run"][3]["factor_params"]["window"]==5
    expected_sharpe=app.session_state['run'][1]['sharpe']
    assert {x.label:x.value for x in app.metric}['年化夏普比率']==f'{expected_sharpe:.3f}'
    assert any('本次结果的夏普口径' in x.value for x in app.caption)
    assert any('账本检查通过' in x.value for x in app.info)
    for page in ["因子诊断","数据与复现","扩展指南"]:
        app.sidebar.radio[0].set_value(page).run(timeout=60)
        assert not app.exception
