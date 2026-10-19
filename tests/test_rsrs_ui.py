"""User-facing RSRS selection, offline behavior, and isolated experiment state."""
from pathlib import Path
import json

import pandas as pd
from streamlit.testing.v1 import AppTest

from cfquant import ui_rsrs


def _app(root):
    code = "from pathlib import Path\nfrom cfquant.ui_rsrs import render\nrender(Path(" + repr(str(root)) + "))"
    script = root / "rsrs_ui_test_app.py"
    script.write_text(code, encoding="utf-8")
    return AppTest.from_file(str(script), default_timeout=30).run()


def _evidence(root):
    evidence = root / "evidence/project2_rsrs"
    (evidence / "daily").mkdir(parents=True)
    rows = []
    for index in ["sh000300", "sh000016"]:
        for period in ["historical_overlap", "post_publication"]:
            for cost in [0., 10.]:
                for strategy in ["right_ma", "right_ma_vol15", "buy_hold"]:
                    total = -.2 if strategy == "right_ma" else .1
                    if index == "sh000016":
                        total += .02
                    if period == "post_publication":
                        total += .04
                    total -= cost / 10000
                    filename = f"daily/{index}_{period}_{cost}_{strategy}.csv"
                    pd.DataFrame({"date": ["2020-01-02", "2020-01-03"], "nav": [1., 1. + total],
                                  "exposure": [.7, .8], "drawdown": [0., min(0., total)]}).to_csv(evidence / filename, index=False)
                    rows.append({"index_id": index, "period_id": period, "strategy_id": strategy,
                                 "cost_bps": cost, "total_return": total, "cagr": total / 2,
                                 "sharpe": -.5 if total < 0 else .8, "max_drawdown": abs(min(total, 0)),
                                 "start": "2020-01-02", "end": "2020-01-03", "daily_file": filename})
    pd.DataFrame(rows).to_csv(evidence / "performance.csv", index=False)
    (evidence / "protocol.json").write_text(json.dumps({"fixed_before_run": True}), encoding="utf-8")
    return evidence


def test_rsrs_page_keeps_losses_visible_and_filters_all_four_dimensions(tmp_path):
    _evidence(tmp_path)
    app = _app(tmp_path)
    assert not app.exception
    assert [t.label for t in app.tabs] == ["研究协议", "原文对照", "实验与改进", "差异诊断", "材料与复现"]
    assert app.selectbox(key="p2_focus").value == "right_ma"
    assert app.selectbox(key="p2_period").value == "post_publication"
    assert {x.label: x.value for x in app.metric}["累计净收益"] == "-16.10%"
    assert any("亏损" in x.value for x in app.warning)
    app.selectbox(key="p2_index").set_value("sh000016").run()
    app.selectbox(key="p2_period").set_value("post_publication").run()
    app.selectbox(key="p2_cost").set_value(0.).run()
    assert {x.label: x.value for x in app.metric}["累计净收益"] == "-14.00%"
    app.selectbox(key="p2_focus").set_value("right_ma_vol15").run()
    assert {x.label: x.value for x in app.metric}["累计净收益"] == "16.00%"
    assert not app.exception


def test_rsrs_missing_private_data_disables_replay_but_not_published_view(tmp_path):
    _evidence(tmp_path)
    app = _app(tmp_path)
    assert not app.exception
    assert app.button(key="p2_run").disabled
    assert any("缺少" in x.value for x in app.info)
    assert len(app.metric) == 4


def test_rsrs_empty_checkout_is_readable_without_large_market_load(tmp_path):
    app = _app(tmp_path)
    assert not app.exception
    assert app.button(key="p2_run").disabled
    assert len(app.tabs) == 5
    assert any("尚未生成" in x.value for x in app.info)


def test_parameter_change_clears_old_replay_and_preserves_published_evidence(tmp_path, monkeypatch):
    evidence = _evidence(tmp_path)
    before = (evidence / "performance.csv").read_bytes()
    data = tmp_path / "data/private/project2_rsrs"
    data.mkdir(parents=True)
    (data / "sh000300.csv").write_text("date,open,high,low,close\n", encoding="utf-8")
    requests = []

    def replay(root, parameters):
        requests.append(parameters)
        return {"folder": str(root / "runs/test"), "parameters": parameters,
                "metrics": {"total_return": .09, "annualized_return": .045, "max_drawdown": .03, "sharpe": .7},
                "checks": {"capital_reconstruction": True}, "daily": pd.DataFrame({"date": ["2020-01-02"], "nav": [1.09]})}

    monkeypatch.setattr(ui_rsrs, "run_exploration", replay)
    app = _app(tmp_path)
    app.button(key="p2_run").click().run()
    assert not app.exception and len(requests) == 1
    assert any(x.label == "探索 · 累计净收益" and x.value == "9.00%" for x in app.metric)
    app.selectbox(key="p2_replay_strategy").set_value("right_ma_vol15").run()
    assert "p2_exploration" not in app.session_state
    assert not any(x.label.startswith("探索") for x in app.metric)
    assert any("旧探索结果已清除" in x.value for x in app.info)
    app.button(key="p2_run").click().run()
    app.number_input(key="p2_vol_target").set_value(20.).run()
    assert "p2_exploration" not in app.session_state
    assert (evidence / "performance.csv").read_bytes() == before


def test_rsrs_page_is_in_fast_navigation_and_keeps_default_mainline():
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "app.py"), default_timeout=60).run()
    assert app.sidebar.radio(key="workspace_page").value == "研究总览"
    app.sidebar.radio(key="workspace_page").set_value("研报复现 · P2").run(timeout=60)
    assert not app.exception
    assert len(app.tabs) == 5
    assert not any(x.key == "dataset" for x in app.sidebar.selectbox)
    app.sidebar.radio(key="workspace_page").set_value("研究总览").run(timeout=60)
    assert {x.label: x.value for x in app.metric}["年化夏普比率"] == "1.255"


def test_curve_path_cannot_leave_published_evidence(tmp_path):
    evidence = _evidence(tmp_path)
    outside = tmp_path / "not_evidence.csv"
    outside.write_text("date,nav\n2020-01-02,1\n", encoding="utf-8")
    assert ui_rsrs._safe_child(evidence, str(outside)) is None
    assert ui_rsrs._safe_child(evidence, "../../not_evidence.csv") is None
