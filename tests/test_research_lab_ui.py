from pathlib import Path
from streamlit.testing.v1 import AppTest
import json
import pytest
from cfquant.ui_factor_lab import save_proposals


def test_proposals_are_immutable_validated_and_credentials_are_not_saved(tmp_path):
    proposal={'id':'arbitrary','label':'example','expression':'returns(close, 20)','hypothesis':'trend','api_key':'never-write-this'}
    answer={'provider':'offline','model':'deterministic','api_key':'never-write-this','proposals':[proposal]}
    first=save_proposals(tmp_path,answer)
    path=next((tmp_path/'runs/factor_plans').glob('*.json'));before=path.read_bytes()
    answer['model']='changed';assert save_proposals(tmp_path,answer)==first
    assert path.read_bytes()==before and b'never-write-this' not in before
    assert json.loads(before)['status']=='dsl_validated_not_evaluated'
    with pytest.raises(ValueError):save_proposals(tmp_path,{'proposals':[dict(proposal,expression='close.shift(-1)')]})


def test_new_pages_keep_mainline_and_load_without_model_calls():
    root=Path(__file__).resolve().parents[1]
    app=AppTest.from_file(str(root/'app.py'),default_timeout=60).run()
    assert not app.exception
    for page in ['因子研发','策略验证','研究总览']:
        app.sidebar.radio(key='workspace_page').set_value(page).run(timeout=60)
        assert not app.exception
    assert any(m.label=='年化夏普比率' and m.value=='1.255' for m in app.metric)


def test_formula_validation_and_offline_model_are_explicit():
    root=Path(__file__).resolve().parents[1]
    app=AppTest.from_file(str(root/'app.py'),default_timeout=60).run()
    app.sidebar.radio(key='workspace_page').set_value('因子研发').run()
    app.text_input(key='lab_expression').set_value('__import__("os")').run()
    assert app.button(key='lab_save').disabled
    app.button(key='lab_request').click().run()
    assert not app.exception
    assert app.session_state['lab_model_answer']['provider']=='offline'


def test_backtest_switches_to_auxiliary_result_and_keeps_mainline_default():
    root=Path(__file__).resolve().parents[1]
    app=AppTest.from_file(str(root/'app.py'),default_timeout=60).run()
    app.sidebar.radio(key='workspace_page').set_value('回测实验').run(timeout=60)
    assert not app.exception
    assert app.selectbox(key='strategy_version').value=='v3/rolling_lightgbm__managed'
    app.selectbox(key='strategy_version').set_value('v4/mined_ridge').run(timeout=60)
    assert not app.exception
    assert any(m.label=='累计净收益' and m.value=='18.37%' for m in app.metric)
