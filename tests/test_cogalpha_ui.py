"""Read-only CogAlpha evidence display, immutable controls, and download scope."""
import json
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from cfquant.ui_cogalpha import MATERIALS, _file, material_path


def _app(root):
    script = root / "cogalpha_ui_test_app.py"
    script.write_text("from pathlib import Path\nfrom cfquant.ui_cogalpha import render\nrender(Path(" + repr(str(root)) + "))", encoding="utf-8")
    return AppTest.from_file(str(script), default_timeout=30).run()


def _evidence(root):
    folder = root / "evidence/project2_cogalpha"
    (folder / "daily").mkdir(parents=True)
    rows = []
    for replicate in [1, 2]:
        for model in ["rank", "ridge", "lightgbm"]:
            for phase in ["validation", "test"]:
                for cost in ["paper", "zero"]:
                    for arm in ["llm_fresh49", "llm_evolved49"]:
                        total = -.12 if arm == "llm_evolved49" else .08
                        total += (replicate - 1) * .01 + (.02 if phase == "validation" else 0) + (.03 if cost == "zero" else 0)
                        total += .01 if model == "ridge" else 0.
                        rank_ic = -.01 if arm == "llm_evolved49" else .015
                        filename = f"daily/{replicate}_{model}_{phase}_{cost}_{arm}.csv"
                        pd.DataFrame({"date": ["2025-01-02", "2025-01-03"], "nav": [1., 1.+total], "benchmark_nav": [1., 1.02],
                                      "drawdown": [0., -min(total, 0.)], "exposure": [.8, .9]}).to_csv(folder / filename, index=False)
                        rows.append({"replicate": replicate, "arm": arm, "model": model, "phase": phase, "cost_case": cost,
                                     "start": "2025-01-02", "end": "2025-01-03", "total_return": total, "cagr": total,
                                     "sharpe": -.5, "max_drawdown": .2, "rank_ic": rank_ic, "rank_ic_ir": -.1,
                                     "daily_file": filename})
    # Shared fixed baseline remains visible for either independent LLM repeat.
    rows.append({**rows[0], "replicate": 0, "arm": "fixed16", "model": "lightgbm", "phase": "test", "cost_case": "paper", "rank_ic": .02})
    pd.DataFrame(rows).to_csv(folder / "performance.csv", index=False)
    (folder / "proposals.json").write_text(json.dumps([{"id": "proposal1", "label": "示例候选", "expression": "returns(close, 20)",
                                                         "hypothesis": "趋势可能延续", "parents": ["seed1"], "role": "趋势", "generation": 1,
                                                         "source": "real_provider_receipt", "quality_status": "failed"}], ensure_ascii=False), encoding="utf-8")
    return folder


def test_empty_evidence_is_explicit_and_cannot_construct_a_paid_client(tmp_path, monkeypatch):
    from cfquant import research_api

    def forbidden(*args, **kwargs):
        raise AssertionError("Reading this page must never create an API client")

    monkeypatch.setattr(research_api, "ResearchAPI", forbidden)
    app = _app(tmp_path)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["协议与差异", "因子谱系与质量", "预测与投资实验", "模型调用与预算", "材料复现"]
    assert any("主体实验结果尚未发布" in item.value for item in app.info)
    assert any("尚无发布的预算回执" in item.value for item in app.info)
    assert len(app.text_input) == 0
    assert len(app.metric) == 0


def test_default_primary_comparison_keeps_negative_results_and_all_filters_work(tmp_path):
    _evidence(tmp_path)
    app = _app(tmp_path)
    assert not app.exception
    assert app.selectbox(key="ca_phase").value == "test"
    assert app.selectbox(key="ca_model").value == "lightgbm"
    assert app.selectbox(key="ca_cost").value == "paper"
    assert app.selectbox(key="ca_focus").value == "llm_evolved49"
    assert {item.label: item.value for item in app.metric}["累计净收益"] == "-12.00%"
    assert {item.label: item.value for item in app.metric}["Rank IC"] == "-0.0100"
    assert any("亏损" in item.value for item in app.warning)
    assert any("未显示正向预测关系" in item.value for item in app.warning)
    assert "固定 16 个公式" in app.selectbox(key="ca_focus").options
    app.selectbox(key="ca_replicate").set_value(2).run()
    app.selectbox(key="ca_phase").set_value("validation").run()
    app.selectbox(key="ca_cost").set_value("zero").run()
    app.selectbox(key="ca_model").set_value("ridge").run()
    assert not app.exception
    assert {item.label: item.value for item in app.metric}["累计净收益"] == "-5.00%"
    app.selectbox(key="ca_focus").set_value("llm_fresh49").run()
    assert {item.label: item.value for item in app.metric}["累计净收益"] == "15.00%"


def test_budget_reports_estimates_and_reservations_without_secrets_or_api_controls(tmp_path):
    folder = _evidence(tmp_path)
    budget = {"budget_cny": 50, "calls": 3, "max_calls": 400, "successful_calls": 2,
              "successful_cost_cny": .003, "uncertain_reserved_cny": .01,
              "reserved_or_spent_cny": .013, "remaining_cny": 49.987, "remaining_calls": 397,
              "halted": True, "pricing_basis": "conservative test fixture", "api_key": "never-display-this"}
    (folder / "budget.json").write_text(json.dumps(budget), encoding="utf-8")
    app = _app(tmp_path)
    assert not app.exception
    assert {item.label: item.value for item in app.metric}["成功请求数"] == "2"
    assert {item.label: item.value for item in app.metric}["成功调用保守估算（元）"] == "0.003000"
    assert any("停止继续请求" in item.value for item in app.warning)
    assert not any("never-display-this" in item.value for item in app.json)
    assert len(app.text_input) == 0


def test_materials_are_fixed_checkout_files_and_paths_cannot_escape(tmp_path):
    folder = tmp_path / "reports/project2_cogalpha"
    folder.mkdir(parents=True)
    filename = next(iter(MATERIALS))
    report = folder / filename
    report.write_bytes(b"test-only PDF fixture")
    outside = tmp_path / "unrelated.pdf"
    outside.write_bytes(b"unrelated")
    assert material_path(tmp_path, filename) == report.resolve()
    assert material_path(tmp_path, str(outside)) is None
    assert material_path(tmp_path, "../../unrelated.pdf") is None
    assert _file(folder, "../../unrelated.pdf") is None
    assert _file(folder, str(outside)) is None
    app = _app(tmp_path)
    assert not app.exception
    assert len(app.get("download_button")) == 1


def test_cogalpha_navigation_preserves_rsrs_and_mainline_default():
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "app.py"), default_timeout=60).run()
    assert app.sidebar.radio(key="workspace_page").value == "研究总览"
    app.sidebar.radio(key="workspace_page").set_value("论文复现 · CogAlpha").run(timeout=60)
    assert not app.exception and len(app.tabs) == 5
    assert not any(item.key == "dataset" for item in app.sidebar.selectbox)
    app.sidebar.radio(key="workspace_page").set_value("研报复现 · P2").run(timeout=60)
    assert not app.exception and len(app.tabs) == 5
    app.sidebar.radio(key="workspace_page").set_value("研究总览").run(timeout=60)
    assert {item.label: item.value for item in app.metric}["年化夏普比率"] == "1.255"
