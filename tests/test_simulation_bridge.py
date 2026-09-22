"""Simulation files retain audit evidence and cannot create trade instructions."""
from dataclasses import replace
import io
import json
from pathlib import Path

import pandas as pd
import pytest

from cfquant.experiment_catalog import Experiment
from cfquant.simulation_bridge import (connection_status, historical_watchlist, normalize_code,
                                      parse_simulated_trades, reconcile_by_security, safe_csv,
                                      save_simulated_receipt, synthetic_example_csv, validation_summary)


NOW = "2026-09-22 12:00:00"


def _csv(rows, *, encoding="utf-8-sig"):
    columns = ["交易时间", "证券代码", "证券名称", "操作", "交易价格", "交易数量", "交易金额"]
    return pd.DataFrame(rows, columns=columns).to_csv(index=False).encode(encoding)


def _valid_rows():
    return [["2026-09-18 09:35:00", "000001", "示例甲", "买入", "10", "100", "1000"],
            ["2026-09-18 14:10:00", "600000.SH", "示例乙", "卖出", "8.25", "200", "1650"]]


@pytest.mark.parametrize("encoding", ["utf-8-sig", "gb18030", "utf-16"])
def test_official_chinese_columns_encodings_and_leading_zero(encoding):
    result = parse_simulated_trades(_csv(_valid_rows(), encoding=encoding), now=NOW)
    assert result.valid
    assert result.normalized.code.tolist() == ["000001", "600000"]
    assert result.normalized.side.tolist() == ["buy", "sell"]
    assert result.normalized.exchange.tolist() == ["SZ", "SH"]
    assert b"000001" in safe_csv(result.normalized)


def test_normalized_export_can_be_imported_without_losing_codes():
    first = parse_simulated_trades(_csv(_valid_rows()), now=NOW)
    second = parse_simulated_trades(safe_csv(first.normalized), now=NOW)
    assert second.valid
    assert first.normalized.fingerprint.tolist() == second.normalized.fingerprint.tolist()


def test_aliases_separate_date_time_and_thousands_separators():
    frame = pd.DataFrame({"成交日期": ["20260918"], "成交时间": ["09:35:00"], "股票代码": ["SZ000001"],
                          "买卖方向": ["买入"], "成交价格(元)": ["10.00"], "成交数量(股)": ["1,000"],
                          "成交金额(元)": ["10,000.00"]})
    result = parse_simulated_trades(frame.to_csv(index=False).encode("gb18030"), now=NOW)
    assert result.valid and result.normalized.iloc[0].quantity == 1000
    assert result.normalized.iloc[0].timestamp == "2026-09-18 09:35:00"


def test_date_only_is_marked_without_fabricated_execution_time():
    rows = _valid_rows()[:1]
    rows[0][0] = "2026-09-18"
    result = parse_simulated_trades(_csv(rows), now=NOW)
    assert result.valid
    assert result.normalized.iloc[0].timestamp == "2026-09-18"
    assert result.normalized.iloc[0].time_precision == "date"
    assert (result.issues.field == "timestamp").any()


def test_row_errors_are_not_silently_dropped_and_block_saving(tmp_path):
    rows = _valid_rows()
    rows.append(["2099-01-01 09:35:00", "not-code", "坏行", "撤单", "NaN", "2.5", "100"])
    result = parse_simulated_trades(_csv(rows), now=NOW)
    assert not result.valid and len(result.normalized) == 3
    assert set(result.issues[result.issues.severity == "error"].field) >= {"timestamp", "code", "side", "price", "quantity"}
    with pytest.raises(ValueError, match="错误"):
        save_simulated_receipt(tmp_path, result)
    assert not (tmp_path / "runs").exists()


def test_amount_tolerance_checks_gross_amount_with_decimal_arithmetic():
    rows = _valid_rows()[:1]
    rows[0][-1] = "1000.05"
    assert parse_simulated_trades(_csv(rows), now=NOW).valid
    rows[0][-1] = "1000.06"
    bad = parse_simulated_trades(_csv(rows), now=NOW)
    assert not bad.valid and "amount" in bad.issues.field.tolist()


def test_time_only_and_negative_share_counts_are_invalid():
    rows = _valid_rows()[:1]
    rows[0][0], rows[0][-2] = "09:35:00", "-100"
    result = parse_simulated_trades(_csv(rows), now=NOW)
    assert not result.valid
    assert set(result.issues.field) >= {"timestamp", "quantity"}


def test_duplicates_preserved_and_summed_with_explicit_flags():
    rows = _valid_rows()
    rows.append(rows[0].copy())
    result = parse_simulated_trades(_csv(rows), now=NOW)
    assert result.valid and result.duplicate_rows == 2
    summary = reconcile_by_security(result).set_index("code")
    assert summary.loc["000001", "buy_amount"] == 2000
    assert summary.loc["000001", "record_count"] == 2
    status = validation_summary(result)
    assert status["duplicates_removed"] is False and status["profit_or_return_computed"] is False


def test_conflicting_trade_ids_warn_without_deduplicating():
    frame = pd.read_csv(io.BytesIO(_csv(_valid_rows())), dtype=str)
    frame["成交编号"] = ["same-id", "same-id"]
    result = parse_simulated_trades(frame.to_csv(index=False).encode("utf-8"), now=NOW)
    assert result.valid and result.duplicate_rows == 2 and len(result.normalized) == 2


def test_csv_formula_name_is_escaped_but_never_executed():
    rows = _valid_rows()[:1]
    rows[0][2] = "=HYPERLINK(\"https://example.invalid\")"
    result = parse_simulated_trades(_csv(rows), now=NOW)
    assert result.valid and (result.issues.field == "name").any()
    assert b"'=HYPERLINK" in safe_csv(result.normalized)


def test_short_codes_warn_and_conflicting_market_is_rejected():
    code, exchange, warnings = normalize_code("1")
    assert (code, exchange) == ("000001", "SZ") and warnings
    with pytest.raises(ValueError, match="不一致"):
        normalize_code("000001.SZ", "SH")
    with pytest.raises(ValueError):
        normalize_code("=SUM(1,2)")


def test_receipt_atomic_idempotent_and_label_cannot_escape(tmp_path):
    result = parse_simulated_trades(synthetic_example_csv(), source_kind="synthetic_demo", now=NOW)
    path, created = save_simulated_receipt(tmp_path, result, label="../../anything")
    assert created and path.parent == tmp_path / "runs/paper_imports"
    again, created = save_simulated_receipt(tmp_path, result)
    assert again == path and not created
    assert len(list(path.parent.iterdir())) == 1
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["validation"]["source_kind"] == "synthetic_demo"
    assert stored["validation"]["orders_submitted"] == 0
    assert len(stored["trades"]) == 3
    with pytest.raises(ValueError, match="摘要"):
        save_simulated_receipt(tmp_path, replace(result, source_sha256="../../outside"))


def test_historical_watchlist_never_exports_adjusted_share_units(tmp_path):
    folder = tmp_path / "runs/test"
    folder.mkdir(parents=True)
    pd.DataFrame({"date": ["2026-09-18"], "asset": ["000001.SZ"], "units": [123.456],
                  "mark_price": [15.123], "value": [500], "weight": [.5]}).to_csv(folder / "positions.csv", index=False)
    pd.DataFrame({"date": ["2026-09-18"], "nav": [1000], "cash": [500]}).to_csv(folder / "daily.csv", index=False)
    record = Experiment("a"*24, "runs/test", "test", "rolling_lightgbm__managed", "V3", "最终历史区间",
                        "2026-09-18", "2026-09-18", "本地完整实验", "")
    watch, metadata = historical_watchlist(tmp_path, record, "2026-09-18")
    assert watch.iloc[0].code == "000001"
    assert not {"units", "quantity", "shares", "price", "mark_price"} & set(watch.columns)
    assert metadata["order_file"] is False and metadata["share_quantity_exported"] is False
    assert metadata["as_of"] == "2026-09-18" and metadata["cash_weight"] == .5
    with pytest.raises(ValueError, match="公开"):
        historical_watchlist(tmp_path, replace(record, source="公开聚合快照"), "2026-09-18")


def test_connection_status_does_not_claim_an_authenticated_account():
    status = connection_status()
    assert status["file_bridge_available"] is True
    assert status["account_connected"] is False
    assert status["automated_trade_api_connected"] is False
    assert status["orders_submitted"] == 0


def test_offline_ui_demonstrates_and_saves_synthetic_receipt(tmp_path):
    from streamlit.testing.v1 import AppTest
    script = f"from pathlib import Path\nfrom cfquant.ui_connections import render\nrender(Path({str(tmp_path)!r}))"
    app = AppTest.from_string(script, default_timeout=30).run()
    assert not app.exception
    assert any(metric.label == "同花顺账号" and metric.value == "未连接" for metric in app.metric)
    assert any(metric.label == "读入成交" and metric.value == "3" for metric in app.metric)
    next(button for button in app.button if button.label == "保存已验证模拟回执").click().run()
    assert not app.exception
    saved = list((tmp_path / "runs/paper_imports").glob("*.json"))
    assert len(saved) == 1
    assert json.loads(saved[0].read_text(encoding="utf-8"))["validation"]["source_kind"] == "synthetic_demo"
