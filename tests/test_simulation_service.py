from datetime import datetime
from types import SimpleNamespace

import pandas as pd

from cfquant import simulation_service as service


def test_service_has_no_creation_or_submission_on_empty_unconfigured_repo(tmp_path):
    result = service.run_cycle(tmp_path, now=datetime.fromisoformat("2026-10-20T19:00:00+08:00"))
    assert result["enabled"] is False
    assert not (tmp_path / "runs/simulation_auto/accounts").exists()
    assert (tmp_path / "runs/simulation_auto/service_status.json").is_file()
    assert all(row["status"] in {"adapter_pending", "blocked"} for row in result["strategies"])


def test_settings_pause_preserves_quote_source_and_limit(tmp_path):
    original = service.load_settings(tmp_path)
    assert original["enabled"] is False and original["quote_source"] == "tencent_public"
    service.set_enabled(tmp_path, True)
    service.set_enabled(tmp_path, False)
    after = service.load_settings(tmp_path)
    assert not after["enabled"] and after["max_order_notional"] == original["max_order_notional"]


def test_off_session_does_not_fetch_quote_or_touch_account_provider(tmp_path, monkeypatch):
    class Client:
        account_binding = "fixture"
        def __init__(self, *args):
            pass
        def inspect(self):
            raise AssertionError("No off-session provider access")
    class Executor:
        def journal(self):
            return []
    calendar = pd.DataFrame({"cal_date": ["2026-10-20", "2026-10-21"], "is_open": [1, 1]})
    monkeypatch.setattr(service, "market_calendar", lambda root: calendar)
    monkeypatch.setattr(service, "_executor", lambda *args: Executor())
    monkeypatch.setattr(service, "latest_signals", lambda root: {
        name: SimpleNamespace(signal_asof=datetime(2026, 10, 20).date(),
                              execution_date=datetime(2026, 10, 21).date(), target_weight=.95)
        for name in service.STRATEGY_IDS})
    service.set_enabled(tmp_path, True)
    result = service.run_cycle(
        tmp_path, now=datetime.fromisoformat("2026-10-20T19:00:00+08:00"),
        client_factory=Client, quote_fetcher=lambda **kw: (_ for _ in ()).throw(AssertionError("No quote")))
    assert all(row["status"] == "waiting_session" for row in result["strategies"]
               if row["strategy_id"] in service.STRATEGY_IDS)
