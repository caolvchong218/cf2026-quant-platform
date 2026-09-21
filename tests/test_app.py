"""Public UI flow works with the portable demo when real data are absent."""
from pathlib import Path
from streamlit.testing.v1 import AppTest

def test_navigation_and_default_factor_window():
    root=Path(__file__).resolve().parents[1]
    app=AppTest.from_file(str(root/"app.py"),default_timeout=60).run()
    assert not app.exception
    app.sidebar.radio[0].set_value("回测实验").run()
    app.selectbox(key="backtest_factor").set_value("reversal")
    app.button[0].click().run(timeout=60)
    assert not app.exception
    assert app.session_state["run"][3]["factor_params"]["window"]==5
    assert len(app.success)==1
    for page in ["因子诊断","数据与复现","扩展指南"]:
        app.sidebar.radio[0].set_value(page).run(timeout=60)
        assert not app.exception
