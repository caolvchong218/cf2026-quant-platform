"""Evidence-oriented archive tests: lazy reads, exact filters and safe notes."""
from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest

from cfquant.experiment_catalog import (discover_experiments, filter_experiments, load_note,
                                       read_ledger, research_summary, save_note)


def _json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def archived_run(tmp_path):
    folder = tmp_path / "runs/research_v3/test_rolling_lightgbm__managed"
    folder.mkdir(parents=True)
    _json(folder / "config.json", {"backtest": {"name": folder.name, "start": "2025-01-02", "end": "2025-01-06", "initial_cash": 1000}})
    _json(folder / "metrics.json", {"total_return": .02, "max_drawdown": .01, "total_cost": 2, "sessions": 3})
    _json(folder / "checks.json", {"passed": True, "checked_sessions": 3})
    pd.DataFrame({"date": ["2025-01-02", "2025-01-03", "2025-01-06"], "nav": [1000, 990, 1020],
                  "cash": [500, 490, 520], "opening_nav": [1000, 1000, 990]}).to_csv(folder / "daily.csv", index=False)
    pd.DataFrame([
        ["2025-01-02", "000001.SZ", "buy", "filled", "", 10, 10],
        ["2025-01-03", "000001.SZ", "buy", "partial", "participation_limit", 10, 4],
        ["2025-01-03", "000002.SZ", "buy", "blocked", "missing_open", 5, 0],
        ["2025-01-06", "600001.SH", "sell", "filled", "", 6, 6],
        ["2025-01-06", "000001.SZ", "sell", "partial", "participation_limit", 10, 7],
    ], columns=["date", "asset", "side", "status", "reason", "requested_units", "filled_units"]).to_csv(folder / "orders.csv", index=False)
    pd.DataFrame({"date": ["2025-01-06"], "asset": ["000001.SZ"], "weight": [.4], "value": [408]}).to_csv(folder / "positions.csv", index=False)
    pd.DataFrame({"date": ["2025-01-02"], "asset": ["000001.SZ"], "side": ["buy"], "notional": [500], "cost": [2]}).to_csv(folder / "trades.csv", index=False)
    return tmp_path, folder


def test_discovery_does_not_read_private_csv(archived_run, monkeypatch):
    root, _ = archived_run
    def no_csv(*args, **kwargs):
        raise AssertionError("Discovery must not open private CSV ledgers")
    monkeypatch.setattr(pd, "read_csv", no_csv)
    records = discover_experiments(root)
    assert len(records) == 1
    assert records[0].version == "V3"
    assert records[0].stage == "最终历史区间"
    assert records[0].checks["passed"] is True


def test_archive_and_yaml_discovery(archived_run):
    root, _ = archived_run
    old = root / "runs/research_v2_archive_old/test_old"
    old.mkdir(parents=True)
    _json(old / "metrics.json", {})
    (old / "daily.csv").write_text("date,nav\n2024-01-02,1000\n")
    (old / "config.yaml").write_text("name: old\nstart: '2024-01-02'\nend: '2024-01-02'\nfactor: momentum\n")
    assert len(discover_experiments(root)) == 1
    all_records = discover_experiments(root, include_archives=True)
    assert len(all_records) == 2
    assert next(record for record in all_records if record.version == "V2").start == "2024-01-02"


def test_filters_overlap_dates_and_search(archived_run):
    records = discover_experiments(archived_run[0])
    assert filter_experiments(records, query="ROLLING managed", start="2025-01-03", end="2025-01-04") == records
    assert filter_experiments(records, start="2025-01-07") == []
    assert filter_experiments(records, versions=["V2"]) == []
    with pytest.raises(ValueError, match="开始日期"):
        filter_experiments(records, start="2025-02-01", end="2025-01-01")


def test_trade_filters_include_partial_fills_across_chunks(archived_run):
    root, _ = archived_run
    record = discover_experiments(root)[0]
    frame, matched = read_ledger(root, record, "orders", start="2025-01-03", end="2025-01-06",
                                 status="unfilled", side="buy", chunk_size=2)
    assert matched == 2
    assert frame.status.tolist() == ["partial", "blocked"]
    specific, count = read_ledger(root, record, "orders", asset="000001.SZ", side="sell", status="unfilled", chunk_size=2)
    assert count == 1 and specific.iloc[0].filled_units == 7
    assert read_ledger(root, record, "orders", asset=".*")[1] == 0


def test_reader_reports_full_match_count_when_capped(archived_run):
    root, _ = archived_run
    frame, matched = read_ledger(root, discover_experiments(root)[0], "orders", max_rows=2, chunk_size=1)
    assert len(frame) == 2 and matched == 5


def test_liquidity_limits_and_cash_scaling_classify_actual_fills(archived_run):
    root, folder = archived_run
    record = discover_experiments(root)[0]
    pd.DataFrame([
        ["2025-01-03", "000001.SZ", "sell", "liquidity_limited", 10, 0],
        ["2025-01-03", "000002.SZ", "buy", "cash_scaled", 10, 4],
    ], columns=["date", "asset", "side", "status", "requested_units", "filled_units"]).to_csv(folder / "orders.csv", index=False)
    assert read_ledger(root, record, "orders", status="unfilled")[1] == 2
    zero, _ = read_ledger(root, record, "orders", status="blocked")
    assert zero.iloc[0].status == "liquidity_limited"
    partial, _ = read_ledger(root, record, "orders", status="partial")
    assert partial.iloc[0].status == "cash_scaled"


def test_public_fallback_preserves_limits_and_capital(tmp_path):
    evidence = tmp_path / "evidence/research_v3"
    (evidence / "daily").mkdir(parents=True)
    pd.DataFrame([{"candidate": "rolling_lightgbm__managed", "total_return": .03}]).to_csv(evidence / "test.csv", index=False)
    pd.DataFrame({"date": ["2025-01-02", "2025-01-03"], "nav": [990, 1030], "opening_nav": [1000, 990]}).to_csv(evidence / "daily/rolling_lightgbm__managed.csv", index=False)
    _json(evidence / "decision.json", {"selected": "rolling_lightgbm__managed"})
    _json(evidence / "checks.json", {"passed": True})
    record = discover_experiments(tmp_path)[0]
    assert not record.has_details
    assert record.config["initial_cash"] == 1000
    assert record.start == "2025-01-02" and record.end == "2025-01-03"
    assert read_ledger(tmp_path, record, "daily")[1] == 2
    assert read_ledger(tmp_path, record, "positions")[1] == 0


def test_note_roundtrip_is_atomic_and_result_files_unchanged(archived_run):
    root, folder = archived_run
    record = discover_experiments(root)[0]
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    save_note(root, record.id, "  成本复核  ", "先检查换手贡献", "复核中")
    saved = save_note(root, record.id, "成本复核", "已核对全部成交", "已复核")
    assert load_note(root, record.id) == saved
    assert saved["body"] == "已核对全部成交"
    assert len(list((root / "runs/notes").iterdir())) == 1
    assert before == {p.name: p.read_bytes() for p in folder.iterdir()}
    bad = replace(record, config={"token": "never-export", "nested": {"api_key": "secret", "holdings": 50}})
    payload = research_summary(bad, saved)
    assert "never-export" not in json.dumps(payload)
    assert payload["config"]["nested"] == {"holdings": 50}


@pytest.mark.parametrize("unsafe_id", ["../config", "a/../../config", "D:/secret", "a" * 23, "x" * 24])
def test_note_rejects_unsafe_ids(tmp_path, unsafe_id):
    with pytest.raises(ValueError, match="ID"):
        save_note(tmp_path, unsafe_id, "x", "x")
    assert not (tmp_path / "runs").exists()


def test_reader_rejects_path_escape(archived_run):
    root, _ = archived_run
    record = replace(discover_experiments(root)[0], relative_path="../../outside")
    with pytest.raises(ValueError, match="路径超出"):
        read_ledger(root, record, "daily")


def test_archive_ui_can_filter_and_save_note(archived_run):
    from streamlit.testing.v1 import AppTest
    root, _ = archived_run
    script = f"from pathlib import Path\nfrom cfquant.ui_experiments import render\nrender(Path({str(root)!r}))"
    app = AppTest.from_string(script, default_timeout=30).run()
    assert not app.exception
    assert any(metric.value == "2.00%" for metric in app.metric)
    record = discover_experiments(root)[0]
    app.radio(key=f"exp_trade_kind_{record.id}").set_value("订单与未完全成交").run()
    app.selectbox(key=f"exp_trade_status_{record.id}").set_value("未完全成交（含部分成交）").run()
    assert not app.exception
    assert any(metric.label == "匹配记录" and metric.value == "3" for metric in app.metric)
    next(item for item in app.text_input if item.label == "笔记标题").set_value("课堂核查")
    next(item for item in app.text_area if item.label == "观察与下一步").set_value("核对部分成交原因")
    next(item for item in app.button if item.label == "保存实验笔记").click().run()
    assert not app.exception
    assert load_note(root, record.id)["title"] == "课堂核查"
