"""Synthetic fixtures exercise monitoring; no vendor records or live network."""
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

import pandas as pd
import pytest

from cfquant import ui_simulation_auto as ui
from cfquant.simulation_execution import ExecutionConfig, SimulationExecutor
from cfquant.ths_simulation import DEPARTMENT, PROVIDER


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def account(root, strategy="v3_managed", account_id="9876543210001"):
    value = {
        "provider": PROVIDER, "department": DEPARTMENT, "simulation_only": True,
        "strategy_id": strategy, "status": "ready", "username": "skill_1234567890123",
        "account_id": account_id, "shareholders": {"1": "synthetic-only"},
        "mobile_account_linked": False,
    }
    write_json(root / "runs/simulation_auto/accounts" / f"{strategy}.json", value)
    return value


def observation(instant, nav=100_000, *, strategy="v3_managed", binding="test-binding"):
    return {
        "provider": PROVIDER, "simulation_only": True, "strategy_id": strategy,
        "account_binding": binding, "asof": instant.isoformat(), "_instant": instant,
        "nav": nav, "cash": nav, "available_cash": nav, "frozen_cash": 0,
        "market_value": 0, "positions": {}, "available_shares": {}, "trades_today": [],
        "fixture_source": "synthetic_test",
    }


def save_observation(root, value, name="observation"):
    clean = {key: item for key, item in value.items() if key != "_instant"}
    path = root / "runs/simulation_auto/observations" / value["strategy_id"] / f"{name}.json"
    write_json(path, clean)
    return path


def app(root):
    from streamlit.testing.v1 import AppTest
    script = f"from pathlib import Path\nfrom cfquant.ui_simulation_auto import render\nrender(Path({str(root)!r}))"
    return AppTest.from_string(script, default_timeout=30).run()


def visible_text(view):
    messages = []
    for kind in ("info", "warning", "error", "caption", "markdown", "success"):
        messages.extend(str(item.value) for item in getattr(view, kind))
    messages.extend(f"{item.label}:{item.value}" for item in view.metric)
    messages.extend(str(item.value) for item in view.dataframe)
    return "\n".join(messages)


def test_empty_dashboard_never_provisions_accounts_or_claims_connection(tmp_path, monkeypatch):
    monkeypatch.setattr(ui, "THSSimulationClient", lambda *args: pytest.fail("No network on render"))
    view = app(tmp_path)
    assert not view.exception
    text = visible_text(view)
    assert "专用模拟账户未连接" in text
    assert "手机账户互通未验证" in text
    assert "0 / 5" in text
    assert not (tmp_path / "runs").exists()
    assert next(button for button in view.button if button.key == "auto_sim_enable").disabled


def test_configured_account_without_observation_does_not_claim_connection(tmp_path, monkeypatch):
    identity = account(tmp_path)
    monkeypatch.setattr(ui, "THSSimulationClient", lambda *args: pytest.fail("No network on render"))
    view = app(tmp_path)
    assert not view.exception
    text = visible_text(view)
    assert "不能判定已经成功连接或产生收益" in text
    assert "观察期不足" in text
    assert identity["account_id"] not in text
    assert identity["username"] not in text
    assert "••••0001" in text


def test_synthetic_observations_are_labelled_and_provider_identifiers_hidden(tmp_path, monkeypatch):
    identity = account(tmp_path)
    value = observation(datetime(2020, 1, 2, 8, tzinfo=timezone.utc),
                        binding=ui._account_binding(identity))
    value["positions"] = {"600000.SH": 100}
    value["available_shares"] = {"600000.SH": 0}
    value["trades_today"] = [{
        "usrid": identity["account_id"], "secret_extra": "must-not-display",
        "zqdm": "600000", "mmlb": "B", "cjsl": 100, "cjg": 10,
        "cjsj": "2020-01-02 09:40:00", "entrust_no": "PRIVATEORDER1234",
        "business_no": "PRIVATEFILL4567",
    }]
    save_observation(tmp_path, value)
    monkeypatch.setattr(ui, "THSSimulationClient", lambda *args: pytest.fail("No network on render"))
    view = app(tmp_path)
    assert not view.exception
    text = visible_text(view)
    assert "合成测试数据" in text
    assert "当前显示的是历史查询快照" in text
    assert "100,000.00" in text
    assert "观察期不足" in text
    for private in (identity["account_id"], "must-not-display", "PRIVATEORDER1234", "PRIVATEFILL4567"):
        assert private not in text
    assert "600000.SH" in text


def test_refresh_button_only_queries_selected_account(tmp_path, monkeypatch):
    identity = account(tmp_path)
    account(tmp_path, "etf_buy_hold", "9876543210002")
    calls = []

    class ReadOnlyClient:
        def __init__(self, root, strategy):
            self.root, self.strategy = root, strategy

        def save_observation(self):
            calls.append(self.strategy)
            value = observation(datetime(2020, 1, 2, 8, tzinfo=timezone.utc),
                                binding=ui._account_binding(identity))
            path = save_observation(self.root, value)
            return value, path

        def submit_limit(self, **kwargs):
            pytest.fail("A refresh must never submit an order.")

    monkeypatch.setattr(ui, "THSSimulationClient", ReadOnlyClient)
    view = app(tmp_path)
    assert calls == []
    next(button for button in view.button if button.key == "auto_sim_refresh_selected").click().run()
    assert not view.exception and calls == ["v3_managed"]
    assert "已刷新 1 个模拟账户" in visible_text(view)


def test_unreachable_refresh_keeps_previous_snapshot_and_sanitizes_error(tmp_path, monkeypatch):
    identity = account(tmp_path)
    save_observation(tmp_path, observation(datetime(2020, 1, 2, 8, tzinfo=timezone.utc),
                                          binding=ui._account_binding(identity)))

    class FailingClient:
        def __init__(self, *_):
            pass

        def save_observation(self):
            raise RuntimeError("secret-account-and-url")

    monkeypatch.setattr(ui, "THSSimulationClient", FailingClient)
    view = app(tmp_path)
    next(button for button in view.button if button.key == "auto_sim_refresh_selected").click().run()
    assert not view.exception
    text = visible_text(view)
    assert "未成功的查询保留原观察" in text
    assert "secret-account-and-url" not in text
    assert "100,000.00" in text


def test_enabled_switch_without_heartbeat_is_not_reported_running(tmp_path):
    account(tmp_path)
    write_json(tmp_path / "runs/simulation_auto/settings.json", {"enabled": True})
    view = app(tmp_path)
    assert not view.exception
    text = visible_text(view)
    assert "不能据此判断自动下单正在运行" in text
    assert "未检测到近期心跳" in text


def test_pause_button_uses_service_switch_without_submitting(tmp_path, monkeypatch):
    account(tmp_path)
    path = tmp_path / "runs/simulation_auto/settings.json"
    write_json(path, {"enabled": True, "interval_seconds": 60, "token_file": "private-path"})
    calls = []

    def switch(root, enabled):
        calls.append(enabled)
        settings = ui._json(path)
        settings["enabled"] = enabled
        write_json(path, settings)
        return True

    monkeypatch.setattr(ui, "_set_enabled", switch)
    view = app(tmp_path)
    next(button for button in view.button if button.key == "auto_sim_pause").click().run()
    assert not view.exception and calls == [False]
    assert ui._json(path)["interval_seconds"] == 60
    assert "private-path" not in visible_text(view)
    assert next(metric for metric in view.metric if metric.label == "自动委托开关").value == "暂停"


def test_signal_dataclass_and_service_pending_status_are_visible(tmp_path, monkeypatch):
    from cfquant import simulation_service

    @dataclass(frozen=True)
    class SyntheticSignal:
        signal_asof: date = date(2026, 1, 5)
        execution_date: date = date(2026, 1, 6)
        target_weight: float = .95
        execution_asset: str = "510300.SH"

    account(tmp_path, "rsrs_standard_ma")
    monkeypatch.setattr(simulation_service, "latest_signals", lambda _: {"rsrs_standard_ma": SyntheticSignal()})
    write_json(tmp_path / "runs/simulation_auto/service_status.json", {
        "asof": datetime.now(timezone.utc).isoformat(),
        "strategies": [{"strategy_id": "rsrs_standard_ma", "status": "adapter_pending",
                        "detail": "private-provider-detail"}],
    })
    view = app(tmp_path)
    assert not view.exception
    text = visible_text(view)
    assert "2026-01-05" in text and "2026-01-06" in text
    assert "目标仓位：95%" in text and "策略执行适配待完成" in text
    assert "private-provider-detail" not in text


@pytest.mark.parametrize("strategy,column", [
    ("v3_managed", "v3_lightgbm_frozen_reference"),
    ("cogalpha_rank", "cogalpha_731_rank_frozen_reference"),
])
def test_equity_signal_panel_marks_frozen_reference_adaptation(tmp_path, monkeypatch, strategy, column):
    from cfquant import simulation_service
    account(tmp_path, strategy)
    monkeypatch.setattr(simulation_service, "latest_equity_status", lambda _: {
        "signal_asof": "2026-10-20", "execution_date": "2026-10-21",
        "reference_snapshot_date": "2026-09-18", "score_coverage": {column: 976},
        "variant_labels": {column: "Synthetic frozen-reference fixture"},
    })
    monkeypatch.setattr(simulation_service, "latest_signals",
                        lambda _: pytest.fail("Stock panel must use the stock readiness helper."))
    view = app(tmp_path)
    assert not view.exception
    text = visible_text(view)
    assert "股票分数截面：2026-10-20" in text and "计划执行日：2026-10-21" in text
    assert "976 只股票" in text and "冻结截至 2026-09-18" in text
    assert "不是原历史策略" in text and "不等于委托或收益已经验证" in text


def test_calendar_uses_service_verified_manifest(tmp_path):
    directory = tmp_path / "runs/simulation_auto/market/20260105"
    directory.mkdir(parents=True)
    path = directory / "trade_cal.csv"
    path.write_bytes(b"cal_date,is_open\n20260105,1\n20260106,0\n20260107,1\n")
    manifest = {"ready": True, "files": {"trade_cal.csv": hashlib.sha256(path.read_bytes()).hexdigest()}}
    write_json(directory / "manifest.json", manifest)
    assert ui._calendar(tmp_path) == {date(2026, 1, 5), date(2026, 1, 7)}
    path.write_bytes(b"cal_date,is_open\n20260105,0\n")
    assert ui._calendar(tmp_path) is None


def test_observations_reject_wrong_accounts_future_times_and_invalid_nav(tmp_path):
    identity = account(tmp_path)
    binding = ui._account_binding(identity)
    now = datetime(2026, 1, 2, 8, tzinfo=timezone.utc)
    valid = observation(now, binding=binding)
    save_observation(tmp_path, valid, "valid")
    save_observation(tmp_path, {**valid, "account_binding": "other"}, "wrong")
    save_observation(tmp_path, {**valid, "nav": -1}, "negative")
    save_observation(tmp_path, {**valid, "asof": (now + timedelta(seconds=1)).isoformat()}, "future")
    save_observation(tmp_path, {**valid, "asof": "2026-01-02T08:00:00"}, "naive")
    values, rejected = ui.read_observations(tmp_path, "v3_managed", binding, now=now)
    assert len(values) == 1 and rejected == 4


def test_daily_close_deduplicates_and_retains_initial_baseline_only_when_needed():
    baseline = datetime(2026, 1, 5, 2, tzinfo=timezone.utc)  # 10:00 Shanghai.
    values = [
        observation(baseline, 100),
        observation(baseline + timedelta(hours=4), 101),  # 14:00, ignored.
        observation(baseline + timedelta(days=1, hours=5, minutes=15), 102),
        observation(baseline + timedelta(days=1, hours=6), 103),
        observation(baseline + timedelta(days=2, hours=4), 104),  # Intraday only, ignored.
        observation(baseline + timedelta(days=5, hours=6), 999),  # Saturday, ignored.
    ]
    frame = ui.daily_observations(values)
    assert frame.date.tolist() == ["2026-01-05", "2026-01-06"]
    assert frame.nav.tolist() == [100, 103]
    values.append(observation(baseline + timedelta(hours=6), 100.5))
    assert ui.daily_observations(values).nav.tolist() == [100.5, 103]


def test_verified_calendar_excludes_weekday_holidays():
    initial = datetime(2026, 1, 5, 8, tzinfo=timezone.utc)
    observations = [observation(initial + timedelta(days=i), 100 + i) for i in range(3)]
    days = {date(2026, 1, 5), date(2026, 1, 7)}
    assert ui.daily_observations(observations, days).date.tolist() == ["2026-01-05", "2026-01-07"]


def test_sharpe_requires_enough_intervals_and_complete_verified_calendar():
    days = list(pd.bdate_range("2026-01-05", periods=22).date)
    values = [
        observation(datetime.combine(day, datetime.min.time(), timezone.utc) + timedelta(hours=8),
                    100 + i + (i % 2) / 2)
        for i, day in enumerate(days)
    ]
    assert ui.performance_summary(values[:20], set(days))["sharpe_reason"] == "观察期不足"
    assert ui.performance_summary(values)["sharpe_reason"] == "交易日覆盖未核验"
    complete = ui.performance_summary(values, set(days))
    assert complete["sharpe"] is not None
    assert complete["cumulative_change"] == pytest.approx(values[-1]["nav"] / values[0]["nav"] - 1)
    missing = ui.performance_summary(values[:5] + values[6:], set(days))
    assert missing["sharpe"] is None and missing["sharpe_reason"] == "缺少交易日收盘观察"
    flat = [{**value, "nav": 100} for value in values]
    assert ui.performance_summary(flat, set(days))["sharpe_reason"] == "波动不足，无法计算"


def test_common_period_uses_exact_intersection_and_rebases_without_filling():
    first = pd.DataFrame({"date": ["2026-01-01", "2026-01-02", "2026-01-05"], "nav": [100, 120, 150]})
    second = pd.DataFrame({"date": ["2026-01-02", "2026-01-03", "2026-01-05"], "nav": [200, 300, 220]})
    frame = ui.common_period_comparison({"v3_managed": first, "etf_buy_hold": second})
    assert set(frame["日期"]) == {"2026-01-02", "2026-01-05"}
    first_returns = frame[frame["策略"] == ui.STRATEGY_LABELS["v3_managed"]]["累计资产变化"].tolist()
    assert first_returns == pytest.approx([0, .25])
    assert ui.common_period_comparison({"v3_managed": first}).empty
    assert ui.common_period_comparison({"v3_managed": first, "etf_buy_hold": second.iloc[:1]}).empty


def test_zero_nav_is_preserved_as_a_total_loss_not_hidden_as_bad_data(tmp_path):
    identity = account(tmp_path)
    binding = ui._account_binding(identity)
    first = observation(datetime(2026, 1, 5, 8, tzinfo=timezone.utc), nav=100, binding=binding)
    last = observation(datetime(2026, 1, 6, 8, tzinfo=timezone.utc), nav=0, binding=binding)
    save_observation(tmp_path, first, "first")
    save_observation(tmp_path, last, "last")
    values, rejected = ui.read_observations(tmp_path, "v3_managed", binding,
                                           now=datetime(2026, 1, 7, tzinfo=timezone.utc))
    assert len(values) == 2 and rejected == 0
    summary = ui.performance_summary(values)
    assert summary["cumulative_change"] == -1
    assert summary["max_drawdown"] == 1
    assert ui.performance_summary([last])["cumulative_change"] is None


def test_read_only_journal_filters_binding_and_does_not_create_missing_db(tmp_path):
    assert ui.read_journal(tmp_path, "missing")[0].empty
    assert not (tmp_path / "runs").exists()
    path = tmp_path / "runs/simulation_auto/orders.sqlite"
    cfg = ExecutionConfig("account-a", frozenset({"600000.SH"}), (date(2026, 1, 5),))
    SimulationExecutor(path, object(), cfg)
    with sqlite3.connect(path) as conn:
        for binding in ("account-a", "account-b"):
            conn.execute("""INSERT INTO intents(account,intent_id,fingerprint,strategy_id,asset,side,
                quantity,limit_price,signal_asof,execution_date,status,created_at,updated_at)
                VALUES(?, 'decision', 'hash','v3_managed','600000.SH','buy',100,10,
                '2026-01-05','2026-01-06','unknown','2026-01-06','2026-01-06')""", (binding,))
    intents, fills, problem = ui.read_journal(tmp_path, "account-a")
    assert len(intents) == 1 and fills.empty and not problem
    assert ui.read_journal(tmp_path, "' OR 1=1 --")[0].empty
