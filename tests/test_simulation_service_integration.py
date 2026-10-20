"""Bounded orchestration checks: fake providers/clocks, no network or real orders."""
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from cfquant import equity_simulation as equity
from cfquant import simulation_execution as execution
from cfquant import simulation_service as service
from cfquant.simulation_signals import EXECUTION_ASSET, RawQuote, SHANGHAI


class Clock:
    value = None

    @classmethod
    def set(cls, value):
        cls.value = FrozenDateTime.fromisoformat(value)


class FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return Clock.value.astimezone(tz) if tz else Clock.value.replace(tzinfo=None)


def calendar():
    return pd.DataFrame({"cal_date": ["2026-10-20", "2026-10-21", "2026-10-22"],
                         "is_open": [1, 1, 1]})


def enabled_settings(root, names):
    settings = service.load_settings(root)
    settings.update(enabled=True, strategies=names)
    service.write_json(service.directory(root) / "settings.json", settings)
    return settings


class FakeExecutor:
    def __init__(self, journal=()):
        self.rows = list(journal)
        self.calls = []

    def journal(self):
        return self.rows

    def reconcile(self):
        return SimpleNamespace(issues=())

    def execute(self, intent, **kwargs):
        self.calls.append((intent, Clock.value))
        return execution.ExecutionResult(intent.intent_id, "accepted")


@pytest.mark.parametrize("change", ["window_elapsed", "strategy_disabled"])
def test_etf_rechecks_live_window_and_strategy_permission_before_submit(tmp_path, monkeypatch, change):
    Clock.set("2026-10-21T09:44:55+08:00")
    monkeypatch.setattr(service, "datetime", FrozenDateTime)
    monkeypatch.setattr(service, "LABELS", {"etf_buy_hold": "ETF"})
    monkeypatch.setattr(service, "market_calendar", lambda root: calendar())
    monkeypatch.setattr(service, "latest_equity_status", lambda root: (_ for _ in ()).throw(ValueError()))
    monkeypatch.setattr(service, "latest_signals", lambda root: {
        "etf_buy_hold": SimpleNamespace(signal_asof=date(2026, 10, 20),
                                         execution_date=date(2026, 10, 21), target_weight=.95)})
    settings = enabled_settings(tmp_path, ["etf_buy_hold"])
    executor = FakeExecutor()
    monkeypatch.setattr(service, "_executor", lambda *args, **kwargs: executor)

    class Client:
        account_binding = "fixture-account"
        strategy_id = "etf_buy_hold"

        def __init__(self, *args):
            pass

        def save_observation(self):
            return None

        def inspect(self):
            if change == "window_elapsed":
                Clock.value += timedelta(seconds=65)
            else:
                changed = dict(settings, strategies=[])
                service.write_json(service.directory(tmp_path) / "settings.json", changed)
            return {"positions": {}, "available_shares": {}, "available_cash": 200_000.0}

    def quote_fetcher(**kwargs):
        return RawQuote(EXECUTION_ASSET, 4.0, Clock.value, "raw")

    def planner(bundle, **kwargs):
        return execution.OrderIntent(
            "fixture-intent", "etf_buy_hold", "fixture-account", EXECUTION_ASSET,
            "buy", 100, 4.0, date(2026, 10, 20), date(2026, 10, 21))

    monkeypatch.setattr(service, "plan_order", planner)
    service.run_cycle(tmp_path, client_factory=Client, quote_fetcher=quote_fetcher)
    assert not executor.calls, "ETF orders must not use an expired cycle-entry window or revoked strategy selection."


@pytest.mark.parametrize("pending", ["accepted", "unknown", "partial"])
def test_pending_equity_order_prevents_another_plan_or_submission(tmp_path, monkeypatch, pending):
    Clock.set("2026-10-21T09:36:00+08:00")
    monkeypatch.setattr(service, "datetime", FrozenDateTime)
    monkeypatch.setattr(service, "LABELS", {"cogalpha_rank": "Cog"})
    monkeypatch.setattr(service, "market_calendar", lambda root: calendar())
    monkeypatch.setattr(service, "latest_signals", lambda root: {})
    monkeypatch.setattr(service, "latest_equity_status", lambda root: {
        "signal_asof": "2026-10-20", "execution_date": "2026-10-21"})
    monkeypatch.setattr(service, "_equity_bundle", lambda root: (tmp_path, {}))
    monkeypatch.setattr(service.pd, "read_parquet", lambda *args, **kwargs: pd.DataFrame({"asset": ["600000.SH"]}))
    enabled_settings(tmp_path, ["cogalpha_rank"])
    executor = FakeExecutor([execution.ExecutionResult("existing", pending)])
    monkeypatch.setattr(service, "_executor", lambda *args, **kwargs: executor)
    monkeypatch.setattr(service, "_equity_cycle", lambda *args, **kwargs: pytest.fail("Must not plan again."))

    class Client:
        account_binding = "fixture-account"
        strategy_id = "cogalpha_rank"

        def __init__(self, *args):
            pass

        def save_observation(self):
            return None

    result = service.run_cycle(tmp_path, client_factory=Client)
    assert result["strategies"][0]["status"] == "reconciliation_pending"
    assert not executor.calls


def test_cog_daily_selection_survives_filled_exits_and_worker_restart(tmp_path):
    assets = [f"{600000 + i}.SH" for i in range(70)]
    scores = pd.DataFrame({"asset": assets, "date": pd.Timestamp("2026-10-20"),
                           "eligible": True, "cogalpha_731_rank_frozen_reference": range(70)})
    receipt = {"signal_date": "2026-10-20",
               "files_sha256": {"current_scores.parquet": "fixed-input-hash"}}
    client = SimpleNamespace(account_binding="fixture-account")
    before = {asset: 100 for asset in assets[:50]}
    initial = service._freeze_equity_decision(
        tmp_path, "cogalpha_rank", client, tmp_path, receipt, scores,
        {"positions": before}, calendar(), date(2026, 10, 21))
    after = {asset: shares for asset, shares in before.items() if asset not in initial.sell_candidates}
    restored = service._freeze_equity_decision(
        tmp_path, "cogalpha_rank", client, tmp_path, receipt, scores,
        {"positions": after}, calendar(), date(2026, 10, 21))
    assert restored == initial
    assert len(restored.sell_candidates) == 5


@pytest.mark.parametrize("snapshot_seconds", [0, 31])
def test_equity_quote_must_still_be_fresh_after_slow_broker_snapshot(tmp_path, monkeypatch, snapshot_seconds):
    """A fresh pre-plan quote is insufficient when broker reads take 31 seconds."""
    Clock.set("2026-10-21T09:36:00+08:00")
    monkeypatch.setattr(service, "datetime", FrozenDateTime)
    monkeypatch.setattr(execution, "datetime", FrozenDateTime)
    monkeypatch.setattr(execution, "_aware", lambda value: isinstance(value, datetime)
                        and value.tzinfo is not None and value.utcoffset() is not None)
    monkeypatch.setattr(equity, "_now", lambda now: Clock.value)
    settings = enabled_settings(tmp_path, ["v3_managed"])
    scores = pd.DataFrame({"asset": ["600000.SH"]})
    receipt = {"signal_date": "2026-10-20"}
    monkeypatch.setattr(service, "_equity_bundle", lambda root: (tmp_path, receipt))
    monkeypatch.setattr(service.pd, "read_parquet", lambda *args, **kwargs: scores)
    target = equity.EquityTargets("v3_managed", date(2026, 10, 20), date(2026, 10, 21),
                                  {"600000.SH": .04}, True, {})
    monkeypatch.setattr(service, "_freeze_equity_decision", lambda *args: target)
    last_quote = {"asof": Clock.value}
    submitted_ages = []

    class Client:
        account_binding = "fixture-account"
        strategy_id = "v3_managed"
        inspections = 0

        def inspect(self):
            self.inspections += 1
            if self.inspections > 1:
                Clock.value += timedelta(seconds=snapshot_seconds)
            return {"positions": {}, "available_shares": {}, "available_cash": 200_000.0,
                    "nav": 200_000.0, "account_binding": self.account_binding,
                    "asof": Clock.value.isoformat()}

        def submit_limit(self, **kwargs):
            submitted_ages.append((Clock.value - last_quote["asof"]).total_seconds())
            return "fixture-order"

    client = Client()
    executor = execution.SimulationExecutor(
        tmp_path / "fixture.sqlite", service._QuotedSimulationBroker(client, tmp_path),
        execution.ExecutionConfig("fixture-account", frozenset({"600000.SH"}),
                                   (date(2026, 10, 20), date(2026, 10, 21)),
                                   enabled=True))

    def quote_fetcher(assets, **kwargs):
        last_quote["asof"] = Clock.value
        return {asset: equity.EquityQuote(asset, 10.0, Clock.value, 11.0, 9.0) for asset in assets}

    item = {}
    service._equity_cycle(tmp_path, "v3_managed", client, executor, calendar(),
                          settings, item, now=Clock.value, quote_fetcher=quote_fetcher)
    assert client.inspections == 2
    expected = "rejected" if snapshot_seconds > 30 else "accepted"
    assert submitted_ages == ([] if snapshot_seconds > 30 else [0])
    assert item["status"] == expected
    assert executor.journal()[0].status == expected


def test_equity_status_without_market_calendar_is_a_readiness_error(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "_equity_bundle", lambda root: (tmp_path, {
        "signal_date": "2026-10-20", "reference_snapshot_date": "2026-09-18",
        "score_coverage": {}, "variant_labels": {},
    }))
    monkeypatch.setattr(service, "market_calendar", lambda root: None)
    with pytest.raises(ValueError):
        service.latest_equity_status(tmp_path)
