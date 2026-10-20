"""Failure-oriented tests; no network calls, provider login, or pretend profits."""
from dataclasses import replace
from datetime import date, datetime, timedelta
import multiprocessing
from pathlib import Path
import sqlite3

import pytest

from cfquant.simulation_execution import (
    BrokerFill, BrokerOrder, BrokerSnapshot, ExecutionConfig, OrderIntent,
    SHANGHAI, SimulationExecutor, SubmitReceipt,
)


NOW = datetime(2026, 10, 20, 10, 0, tzinfo=SHANGHAI)
ACCOUNT = "simulation-test"
ASSET = "600000.SH"


class FakeBroker:
    simulation_only = True

    def __init__(self, cash=100_000, shares=1000):
        self.state = BrokerSnapshot(ACCOUNT, NOW, cash, {ASSET: shares}, True)
        self.orders = []
        self.fills = []
        self.calls = []
        self.error = None
        self.receipt = None
        self.db_path = None

    def snapshot(self):
        return self.state

    def submit(self, intent):
        self.calls.append(intent)
        if self.db_path:
            with sqlite3.connect(self.db_path) as conn:
                status = conn.execute("SELECT status FROM intents WHERE intent_id=?",
                                      (intent.intent_id,)).fetchone()[0]
            assert status == "submitting", "Reservation must commit before the remote call."
        if self.error:
            raise self.error
        return self.receipt or SubmitReceipt(ACCOUNT, "order-" + intent.intent_id)

    def query_orders(self):
        return self.orders

    def query_fills(self):
        return self.fills


def config(**changes):
    return replace(ExecutionConfig(
        ACCOUNT, frozenset((ASSET, "159919.SZ")),
        (date(2026, 10, 16), date(2026, 10, 19), date(2026, 10, 20)),
        enabled=True, fee_buffer_bps=0,
    ), **changes)


def intent(**changes):
    return replace(OrderIntent("decision-1", "v3", ACCOUNT, ASSET, "buy", 100, 10.0,
                               date(2026, 10, 19), date(2026, 10, 20)), **changes)


def order(*, status="open", asof=NOW, **changes):
    return replace(BrokerOrder("order-decision-1", "decision-1", ACCOUNT, ASSET, "buy",
                               100, 10.0, status, asof), **changes)


def fill(*, quantity=100, **changes):
    return replace(BrokerFill("fill-1", "order-decision-1", ACCOUNT, quantity, 9.99, NOW),
                   **changes)


def executor(tmp_path, broker=None, cfg=None):
    broker = broker or FakeBroker()
    broker.db_path = tmp_path / "journal.sqlite"
    return SimulationExecutor(broker.db_path, broker, cfg or config()), broker


def test_disabled_by_default(tmp_path):
    cfg = ExecutionConfig(ACCOUNT, frozenset((ASSET,)), (date(2026, 10, 19), date(2026, 10, 20)))
    runner, broker = executor(tmp_path, cfg=cfg)
    assert runner.execute(intent(), now=NOW).reason_code == "execution_disabled"
    assert broker.calls == []


def test_reservation_durable_before_submit_and_duplicates_survive_restart(tmp_path):
    runner, broker = executor(tmp_path)
    result = runner.execute(intent(), now=NOW)
    assert (result.status, result.filled_quantity) == ("accepted", 0)
    again = SimulationExecutor(runner.db_path, broker, config()).execute(intent(), now=NOW)
    assert again.duplicate and again.status == "accepted"
    assert len(broker.calls) == 1
    changed = runner.execute(intent(limit_price=10.1), now=NOW)
    assert changed.reason_code == "intent_id_conflict"
    assert len(broker.calls) == 1


@pytest.mark.parametrize("change,reason", [
    ({"account_binding": "different"}, "account_mismatch"),
    ({"asset": "688001.SH"}, "asset_not_allowed"),
    ({"asset": "300001.SZ"}, "asset_not_allowed"),
    ({"asset": "920001.BJ"}, "asset_not_allowed"),
    ({"quantity": 1}, "invalid_lot_size"),
    ({"quantity": 150}, "invalid_lot_size"),
    ({"quantity": -100}, "invalid_lot_size"),
    ({"quantity": True}, "invalid_lot_size"),
    ({"quantity": 10**400}, "invalid_lot_size"),
    ({"limit_price": float("nan")}, "invalid_limit_price"),
    ({"limit_price": float("inf")}, "invalid_limit_price"),
    ({"limit_price": 0}, "invalid_limit_price"),
    ({"side": "short"}, "invalid_side"),
    ({"signal_asof": date(2026, 10, 16)}, "stale_or_future_signal"),
    ({"signal_asof": date(2026, 10, 20)}, "stale_or_future_signal"),
    ({"execution_date": date(2026, 10, 21)}, "wrong_execution_date"),
    ({"limit_price": 1000}, "order_notional_limit"),
])
def test_preflight_blocks_bad_or_stale_intents(tmp_path, change, reason):
    runner, broker = executor(tmp_path)
    assert runner.execute(intent(**change), now=NOW).reason_code == reason
    assert not broker.calls and not runner.journal()


@pytest.mark.parametrize("hour,minute", [(9, 29), (11, 30), (12, 59), (15, 0)])
def test_session_boundaries(tmp_path, hour, minute):
    runner, broker = executor(tmp_path)
    result = runner.execute(intent(), now=NOW.replace(hour=hour, minute=minute))
    assert result.reason_code == "outside_continuous_session"
    assert not broker.calls


def test_timezone_and_exchange_holiday_fail_closed(tmp_path):
    runner, _ = executor(tmp_path)
    assert runner.execute(intent(), now=NOW.replace(tzinfo=None)).reason_code == "timezone_required"
    holiday = NOW.replace(day=21)
    result = runner.execute(intent(execution_date=holiday.date()), now=holiday)
    assert result.reason_code == "not_trading_day"


@pytest.mark.parametrize("change,reason", [
    ({"asof": NOW - timedelta(seconds=31)}, "stale_snapshot"),
    ({"asof": NOW + timedelta(seconds=1)}, "stale_snapshot"),
    ({"account_binding": "wrong"}, "snapshot_account_mismatch"),
    ({"is_simulation": False}, "simulation_account_required"),
    ({"available_cash": float("nan")}, "invalid_available_cash"),
    ({"available_shares": {ASSET: -1}}, "invalid_available_shares"),
])
def test_invalid_snapshot_blocks_send(tmp_path, change, reason):
    runner, broker = executor(tmp_path)
    broker.state = replace(broker.state, **change)
    assert runner.execute(intent(), now=NOW).reason_code == reason
    assert not broker.calls


def test_non_simulation_adapter_cannot_touch_account(tmp_path):
    runner, broker = executor(tmp_path)
    broker.simulation_only = False
    broker.snapshot = lambda: pytest.fail("Must not access a real account.")
    assert runner.execute(intent(), now=NOW).reason_code == "simulation_adapter_required"
    assert runner.reconcile(now=NOW).issues == ("simulation_adapter_or_timezone_required",)


def test_pending_reservations_prevent_overspend_and_oversell(tmp_path):
    runner, broker = executor(tmp_path, FakeBroker(cash=1500, shares=100))
    assert runner.execute(intent(), now=NOW).status == "accepted"
    assert runner.execute(intent(intent_id="decision-2"), now=NOW).reason_code == "insufficient_available_cash"
    assert runner.execute(intent(intent_id="sell-1", side="sell"), now=NOW).status == "accepted"
    assert runner.execute(intent(intent_id="sell-2", side="sell"), now=NOW).reason_code == "insufficient_sellable_shares"
    assert len(broker.calls) == 2


def test_t_plus_one_uses_sellable_not_total_holdings(tmp_path):
    runner, broker = executor(tmp_path, FakeBroker(shares=0))
    assert runner.execute(intent(side="sell"), now=NOW).reason_code == "insufficient_sellable_shares"
    assert not broker.calls


def test_fee_reserve_and_daily_cap(tmp_path):
    runner, broker = executor(tmp_path, FakeBroker(cash=1000), config(fee_buffer_bps=25))
    assert runner.execute(intent(), now=NOW).reason_code == "insufficient_available_cash"
    runner, broker = executor(tmp_path, cfg=config(max_daily_notional=1500))
    assert runner.execute(intent(), now=NOW).status == "accepted"
    assert runner.execute(intent(intent_id="second"), now=NOW).reason_code == "daily_notional_limit"


def test_timeout_is_sticky_unknown_and_does_not_leak_exception(tmp_path):
    runner, broker = executor(tmp_path, FakeBroker(cash=1500))
    broker.error = TimeoutError("Authorization Bearer super-secret-token raw=response")
    result = runner.execute(intent(), now=NOW)
    assert result.status == "unknown" and result.reason_code == "submission_outcome_unknown"
    broker.error = None
    assert runner.execute(intent(), now=NOW).status == "unknown"
    assert runner.execute(intent(intent_id="second"), now=NOW).reason_code == "insufficient_available_cash"
    assert runner.reconcile(now=NOW).updated == 0
    assert runner.journal()[0].status == "unknown"
    assert len(broker.calls) == 1
    assert b"super-secret-token" not in runner.db_path.read_bytes()


def test_process_interruption_does_not_retry(tmp_path):
    runner, broker = executor(tmp_path)
    broker.error = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        runner.execute(intent(), now=NOW)
    broker.error = None
    reopened = SimulationExecutor(runner.db_path, broker, config())
    assert reopened.journal()[0].status == "unknown"
    assert reopened.execute(intent(), now=NOW).duplicate
    assert len(broker.calls) == 1


@pytest.mark.parametrize("receipt", [
    SubmitReceipt(ACCOUNT, None),
    SubmitReceipt("wrong-account", "broker-1"),
    SubmitReceipt(ACCOUNT, "bad\nid"),
    SubmitReceipt(ACCOUNT, "broker-1", "filled"),
])
def test_invalid_receipt_is_not_assumed_accepted(tmp_path, receipt):
    runner, broker = executor(tmp_path)
    broker.receipt = receipt
    assert runner.execute(intent(), now=NOW).status == "unknown"
    assert runner.execute(intent(), now=NOW).duplicate
    assert len(broker.calls) == 1


def test_definitive_rejection_releases_reserve_but_keeps_intent_id(tmp_path):
    runner, broker = executor(tmp_path, FakeBroker(cash=1000))
    broker.receipt = SubmitReceipt(ACCOUNT, None, "rejected")
    assert runner.execute(intent(), now=NOW).status == "rejected"
    broker.receipt = None
    assert runner.execute(intent(), now=NOW).status == "rejected"
    assert runner.execute(intent(intent_id="new-decision"), now=NOW).status == "accepted"


def test_reconcile_unknown_only_from_exact_order_evidence(tmp_path):
    runner, broker = executor(tmp_path)
    broker.error = TimeoutError()
    runner.execute(intent(), now=NOW)
    broker.orders = [order()]
    assert runner.reconcile(now=NOW).updated == 1
    assert runner.journal()[0].status == "open"
    assert runner.execute(intent(), now=NOW).duplicate
    assert len(broker.calls) == 1


def test_partial_fills_are_deduplicated_preserved_and_fully_verified(tmp_path):
    runner, broker = executor(tmp_path)
    runner.execute(intent(), now=NOW)
    broker.orders = [order(status="partial")]
    broker.fills = [fill(quantity=30)]
    first = runner.reconcile(now=NOW)
    assert first.fills_added == 1
    assert (runner.journal()[0].status, runner.journal()[0].filled_quantity) == ("partial", 30)
    assert runner.reconcile(now=NOW).fills_added == 0
    broker.orders = [order(status="filled")]
    assert runner.reconcile(now=NOW).updated == 1
    assert runner.journal()[0].status == "unknown"
    broker.fills.append(fill(quantity=70, fill_id="fill-2"))
    assert runner.reconcile(now=NOW).fills_added == 1
    assert (runner.journal()[0].status, runner.journal()[0].filled_quantity) == ("filled", 100)
    assert len(broker.calls) == 1


def test_fill_only_provider_can_confirm_known_order_id(tmp_path):
    runner, broker = executor(tmp_path)
    runner.execute(intent(), now=NOW)
    broker.orders = []
    broker.fills = [fill(quantity=30)]
    assert runner.reconcile(now=NOW).fills_added == 1
    assert runner.journal()[0].status == "partial"
    broker.fills.append(fill(quantity=70, fill_id="fill-2"))
    runner.reconcile(now=NOW)
    assert runner.journal()[0].status == "filled"


def test_date_aware_reconciliation_recovers_only_unfinished_execution_dates(tmp_path):
    runner, broker = executor(tmp_path)
    runner.execute(intent(), now=NOW)
    queried = []
    def dated(dates):
        queried.append(dates)
        return [fill()] if dates else []
    broker.query_fills_for_dates = dated
    broker.query_fills = lambda: pytest.fail("Undated query would lose a prior-day fill")
    assert runner.reconcile(now=NOW + timedelta(days=1)).fills_added == 1
    assert queried == [(NOW.date(),)]
    assert runner.journal()[0].status == "filled"
    runner.reconcile(now=NOW + timedelta(days=2))
    assert queried[-1] == ()


@pytest.mark.parametrize("offset,precision,expected", [
    (0, "seconds", True),
    (-1, "seconds", False),
    (0, "microseconds", False),
])
def test_declared_second_precision_accepts_same_second_not_earlier(tmp_path, offset, precision, expected):
    runner, broker = executor(tmp_path)
    created = NOW.replace(microsecond=500000)
    assert runner.execute(intent(), now=created).status == "accepted"
    broker.fills = [fill(timestamp=NOW + timedelta(seconds=offset), timestamp_precision=precision)]
    result = runner.reconcile(now=NOW + timedelta(seconds=1))
    assert (runner.journal()[0].status == "filled") is expected
    if not expected:
        assert result.issues == ("invalid_fill_evidence",)


def test_unknown_without_order_id_cannot_be_linked_by_a_similar_fill(tmp_path):
    runner, broker = executor(tmp_path)
    broker.receipt = SubmitReceipt(ACCOUNT, None)
    runner.execute(intent(), now=NOW)
    broker.fills = [fill()]
    assert runner.reconcile(now=NOW).fills_added == 0
    assert runner.journal()[0].status == "unknown"


def test_partial_cancel_releases_only_unfilled_quantity(tmp_path):
    runner, broker = executor(tmp_path, FakeBroker(cash=1000))
    runner.execute(intent(), now=NOW)
    broker.orders = [order(status="cancelled")]
    broker.fills = [fill(quantity=30)]
    runner.reconcile(now=NOW)
    assert (runner.journal()[0].status, runner.journal()[0].filled_quantity) == ("cancelled", 30)
    broker.state = replace(broker.state, available_cash=700)
    result = runner.execute(intent(intent_id="new", limit_price=7), now=NOW)
    assert result.status == "accepted"


def test_cancelled_partial_order_cannot_reopen_on_repeated_fill_only_evidence(tmp_path):
    runner, broker = executor(tmp_path)
    runner.execute(intent(), now=NOW)
    broker.orders = [order(status="cancelled")]
    broker.fills = [fill(quantity=30)]
    runner.reconcile(now=NOW)
    broker.orders = []
    runner.reconcile(now=NOW)
    assert runner.journal()[0].status == "cancelled"
    assert runner.journal()[0].filled_quantity == 30
    broker.orders = [order(status="open", asof=NOW + timedelta(seconds=1))]
    runner.reconcile(now=NOW + timedelta(seconds=1))
    assert runner.journal()[0].status == "cancelled"


def test_duplicate_provider_order_id_is_reserved_as_unknown(tmp_path):
    runner, broker = executor(tmp_path)
    broker.receipt = SubmitReceipt(ACCOUNT, "single-order")
    assert runner.execute(intent(), now=NOW).status == "accepted"
    second = runner.execute(intent(intent_id="second"), now=NOW)
    assert (second.status, second.order_id, second.reason_code) == (
        "unknown", None, "duplicate_broker_order_id")
    assert runner.execute(intent(intent_id="second"), now=NOW).duplicate
    assert len(broker.calls) == 2


@pytest.mark.parametrize("bad_fill", [
    fill(quantity=101),
    fill(price=10.01),
    fill(price=float("nan")),
    fill(timestamp=NOW + timedelta(seconds=1)),
    fill(quantity=-1),
    fill(fee=-1),
])
def test_bad_fill_evidence_never_changes_holdings(tmp_path, bad_fill):
    runner, broker = executor(tmp_path)
    runner.execute(intent(), now=NOW)
    broker.orders = [order(status="filled")]
    broker.fills = [bad_fill]
    assert "invalid_fill_evidence" in runner.reconcile(now=NOW).issues
    assert runner.journal()[0].filled_quantity == 0


def test_conflicting_fill_id_rolls_back_the_entire_order_update(tmp_path):
    runner, broker = executor(tmp_path)
    runner.execute(intent(), now=NOW)
    broker.fills = [fill(quantity=30)]
    runner.reconcile(now=NOW)
    broker.fills = [fill(quantity=70, fill_id="fill-2"), fill(quantity=40)]
    assert runner.reconcile(now=NOW).issues == ("invalid_fill_evidence",)
    assert runner.journal()[0].filled_quantity == 30
    with sqlite3.connect(runner.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM fills").fetchone()[0] == 1


def test_wrong_order_and_ambiguous_matches_do_not_resolve_unknown(tmp_path):
    runner, broker = executor(tmp_path)
    broker.error = TimeoutError()
    runner.execute(intent(), now=NOW)
    broker.orders = [order(quantity=200)]
    assert runner.reconcile(now=NOW).issues == ("order_evidence_mismatch",)
    broker.orders = [order(), order(order_id="another")]
    assert runner.reconcile(now=NOW).issues == ("ambiguous_order_match",)
    assert runner.journal()[0].status == "unknown"


def test_snapshot_must_reflect_known_fills_before_new_order(tmp_path):
    runner, broker = executor(tmp_path)
    runner.execute(intent(), now=NOW)
    later = NOW + timedelta(seconds=10)
    broker.orders = [order(status="filled", asof=later)]
    broker.fills = [fill(timestamp=later)]
    runner.reconcile(now=later)
    result = runner.execute(intent(intent_id="next"), now=later)
    assert result.reason_code == "snapshot_precedes_known_fill"


def _concurrent_worker(db_path, marker_path, decision, queue):
    """Spawned separately, so SQLite rather than an in-process lock protects us."""
    broker = FakeBroker(cash=1500)
    original = broker.submit

    def submit(value):
        with Path(marker_path).open("a", encoding="utf-8") as stream:
            stream.write(value.intent_id + "\n")
        return original(value)

    broker.submit = submit
    runner = SimulationExecutor(db_path, broker, config())
    result = runner.execute(intent(intent_id=decision), now=NOW)
    queue.put((result.status, result.reason_code, result.duplicate))


@pytest.mark.parametrize("decisions", [("same", "same"), ("first", "second")])
def test_separate_processes_cannot_duplicate_or_overspend(tmp_path, decisions):
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    db_path, marker = tmp_path / "concurrent.sqlite", tmp_path / "remote_calls.txt"
    processes = [ctx.Process(target=_concurrent_worker, args=(db_path, marker, value, queue))
                 for value in decisions]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0
    results = [queue.get(timeout=5) for _ in processes]
    queue.close()
    assert len(marker.read_text().splitlines()) == 1
    if decisions[0] == decisions[1]:
        assert sum(r[2] for r in results) == 1
    else:
        assert sum(r[1] == "insufficient_available_cash" for r in results) == 1
