"""User journeys for the enhanced workbench, including public-data operation."""
from pathlib import Path
import json
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]


def app_at(page):
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception
    return app


def test_comparison_empty_selection_is_actionable_and_exports_real_results():
    app=app_at('策略对比')
    assert len(app.get('download_button'))==1
    app.multiselect(key='compare_ids').set_value([]).run()
    assert not app.exception
    assert any('至少一个策略' in x.value for x in app.info)


def test_risk_cash_allocation_and_original_maximum_drawdown():
    app=app_at('风险透镜')
    assert any(x.value=='9.88%' for x in app.metric)
    app.slider(key='cash_weight').set_value(0).run()
    assert not app.exception
    metrics={x.label:x.value for x in app.metric}
    assert metrics['组合累计收益']=='0.00%'
    assert metrics['购买力变化（假设）'].startswith('-')
    assert metrics['区间最大回撤']=='9.88%'


def test_twelve_factors_are_accessible_from_primary_navigation():
    app=app_at('因子诊断')
    assert len(app.selectbox(key='factor_lab_choice').options)==12
    app.selectbox(key='factor_lab_choice').set_value('quality_roe').run()
    assert not app.exception
    assert any('公告' in str(x.value) or 'annual ROE' in str(x.value) for x in app.code)


def test_experiment_archive_and_materials_pages_are_available():
    app=app_at('实验档案')
    assert any(x.value=='实验档案' for x in app.title)
    app.sidebar.radio[0].set_value('材料与答辩').run()
    assert not app.exception
    assert len(app.get('download_button'))>=3


def test_sharpe_is_visible_and_rate_changes_risk_metrics_without_changing_nav():
    app=app_at('研究总览')
    assert {x.label:x.value for x in app.metric}['年化夏普比率']=='1.255'
    app.sidebar.radio[0].set_value('风险透镜').run()
    before={x.label:x.value for x in app.metric}
    assert before['年化夏普比率']=='1.255'
    app.number_input(key='risk_risk_free').set_value(2.).run()
    assert not app.exception
    after={x.label:x.value for x in app.metric}
    assert after['年化夏普比率']=='1.080'
    assert after['区间最大回撤']==before['区间最大回撤']=='9.88%'
    assert after['组合累计收益']==before['组合累计收益']
    assert any('0.619' in x.value for x in app.caption)
    assert any(json.loads(x.proto.spec)['layout'].get('yaxis',{}).get('title',{}).get('text')=='60日滚动年化夏普'
               for x in app.get('plotly_chart'))
