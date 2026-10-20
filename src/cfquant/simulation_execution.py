"""Fail-closed, provider-neutral execution for *simulated* accounts only.

An adapter must positively identify a simulation account.  This module does not
log in, know any provider URL, or turn a research backtest into an order.
``available_shares`` means shares currently sellable (including T+1 restrictions).
Available cash may already exclude broker holds; subtracting local reservations
again is deliberately conservative until a provider can reconcile its holds.

SQLite commits ``submitting`` and a reservation BEFORE calling the broker.
Every subsequent invocation deduplicates that intent, including after a process
crash.  An uncertain submission is resolved only by querying identifiable orders
and fills; absence from a query is never permission to send it again.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import time as clock
from typing import Iterator, Mapping, Protocol, Sequence
from zoneinfo import ZoneInfo


SHANGHAI = ZoneInfo("Asia/Shanghai")
_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_ASSET = re.compile(
    r"(?:(?:600|601|603|605|510|511|512|513|515|516|517|518|520|560|561|562|563|588)\d{3}\.SH"
    r"|(?:000|001|002|003|159)\d{3}\.SZ)\Z"
)
_PENDING = ("submitting", "unknown", "accepted", "open", "partial")
_ORDER_STATES = frozenset(("accepted", "open", "partial", "filled", "cancelled", "rejected"))


@dataclass(frozen=True)
class OrderIntent:
    intent_id: str
    strategy_id: str
    account_binding: str
    asset: str
    side: str
    quantity: int
    limit_price: float
    signal_asof: date
    execution_date: date


@dataclass(frozen=True)
class BrokerSnapshot:
    account_binding: str
    asof: datetime
    available_cash: float
    available_shares: Mapping[str, int]
    is_simulation: bool


@dataclass(frozen=True)
class SubmitReceipt:
    account_binding: str
    order_id: str | None
    status: str = "accepted"  # Only accepted or definitively rejected.


@dataclass(frozen=True)
class BrokerOrder:
    order_id: str
    intent_id: str
    account_binding: str
    asset: str
    side: str
    quantity: int
    limit_price: float
    status: str
    asof: datetime


@dataclass(frozen=True)
class BrokerFill:
    fill_id: str
    order_id: str
    account_binding: str
    quantity: int
    price: float
    timestamp: datetime
    fee: float | None = None
    timestamp_precision: str = "microseconds"


class SimulationBroker(Protocol):
    simulation_only: bool

    def snapshot(self) -> BrokerSnapshot: ...
    def submit(self, intent: OrderIntent) -> SubmitReceipt: ...
    def query_orders(self) -> Sequence[BrokerOrder]: ...
    def query_fills(self) -> Sequence[BrokerFill]: ...


@dataclass(frozen=True)
class ExecutionConfig:
    account_binding: str
    allowed_assets: frozenset[str]
    trading_days: tuple[date, ...]
    enabled: bool = False
    max_order_notional: float = 20_000.0
    max_daily_notional: float = 100_000.0
    snapshot_max_age_seconds: float = 30.0
    fee_buffer_bps: float = 25.0

    def __post_init__(self) -> None:
        if not isinstance(self.account_binding, str) or not _ID.fullmatch(self.account_binding):
            raise ValueError("Invalid simulation account binding.")
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be an explicit boolean.")
        if not self.allowed_assets or any(not isinstance(a, str) or not _ASSET.fullmatch(a)
                                          for a in self.allowed_assets):
            raise ValueError("Only explicitly allowed main-board shares and supported ETFs are permitted.")
        if (not self.trading_days
                or any(type(d) is not date for d in self.trading_days)
                or tuple(sorted(set(self.trading_days))) != self.trading_days):
            raise ValueError("An explicit sorted unique trading calendar is required.")
        for value in (self.max_order_notional, self.max_daily_notional, self.snapshot_max_age_seconds):
            if not _finite(value) or value <= 0:
                raise ValueError("Execution limits must be finite and positive.")
        if not _finite(self.fee_buffer_bps) or not 0 <= self.fee_buffer_bps <= 1000:
            raise ValueError("Invalid fee buffer.")


@dataclass(frozen=True)
class ExecutionResult:
    intent_id: str
    status: str
    order_id: str | None = None
    filled_quantity: int = 0
    reason_code: str = ""
    duplicate: bool = False


@dataclass(frozen=True)
class ReconciliationResult:
    updated: int = 0
    fills_added: int = 0
    issues: tuple[str, ...] = field(default_factory=tuple)


def _finite(value: object) -> bool:
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def _aware(value: object) -> bool:
    return isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None


def _stamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _fingerprint(intent: OrderIntent) -> str:
    payload = asdict(intent)
    payload["signal_asof"] = intent.signal_asof.isoformat()
    payload["execution_date"] = intent.execution_date.isoformat()
    # Normalize 10 and 10.0 without weakening equality of any economic fields.
    payload["limit_price"] = float(intent.limit_price)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


class SimulationExecutor:
    """One configured account, durable journal, no hidden retries or fill guesses.

    Use distinct intent IDs for distinct decisions, but never change an ID after
    a timeout.  IDs are caller-supplied stable identifiers, not generated on rerun.
    A new instance defaults to disabled unless the caller explicitly arms it.
    """

    def __init__(self, db_path: str | Path, broker: SimulationBroker, config: ExecutionConfig):
        self.db_path = Path(db_path)
        self.broker = broker
        self.config = config
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS intents (
                    account TEXT NOT NULL, intent_id TEXT NOT NULL, fingerprint TEXT NOT NULL,
                    strategy_id TEXT NOT NULL, asset TEXT NOT NULL, side TEXT NOT NULL,
                    quantity INTEGER NOT NULL, limit_price REAL NOT NULL,
                    signal_asof TEXT NOT NULL, execution_date TEXT NOT NULL,
                    status TEXT NOT NULL, order_id TEXT, filled_quantity INTEGER NOT NULL DEFAULT 0,
                    reason_code TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL, broker_asof TEXT,
                    PRIMARY KEY(account, intent_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS account_order ON intents(account, order_id)
                    WHERE order_id IS NOT NULL;
                CREATE TABLE IF NOT EXISTS fills (
                    account TEXT NOT NULL, fill_id TEXT NOT NULL, intent_id TEXT NOT NULL,
                    order_id TEXT NOT NULL, quantity INTEGER NOT NULL, price REAL NOT NULL,
                    timestamp TEXT NOT NULL, fee REAL,
                    PRIMARY KEY(account, fill_id)
                );
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, account TEXT NOT NULL,
                    intent_id TEXT NOT NULL, status TEXT NOT NULL, reason_code TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );
            """)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=30000")
            # Changing journal mode can return BUSY without observing SQLite's
            # busy_timeout when two processes initialize the file together.
            # This bounded retry is local setup only, never an order retry.
            deadline = clock.monotonic() + 30
            while True:
                try:
                    if conn.execute("PRAGMA journal_mode").fetchone()[0] != "wal":
                        conn.execute("PRAGMA journal_mode=WAL")
                    break
                except sqlite3.OperationalError as exc:
                    if (getattr(exc, "sqlite_errorcode", None) not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
                            or clock.monotonic() >= deadline):
                        raise
                    clock.sleep(0.02)
            conn.execute("PRAGMA synchronous=FULL")
            yield conn
        finally:
            conn.close()

    def _reason(self, intent: OrderIntent, now: datetime) -> str:
        if not self.config.enabled:
            return "execution_disabled"
        if getattr(self.broker, "simulation_only", None) is not True:
            return "simulation_adapter_required"
        if intent.account_binding != self.config.account_binding:
            return "account_mismatch"
        if not _aware(now):
            return "timezone_required"
        if any(not isinstance(v, str) or not _ID.fullmatch(v)
               for v in (intent.intent_id, intent.strategy_id)):
            return "invalid_identifier"
        if (not isinstance(intent.asset, str) or intent.asset not in self.config.allowed_assets
                or not _ASSET.fullmatch(intent.asset)):
            return "asset_not_allowed"
        if intent.side not in ("buy", "sell"):
            return "invalid_side"
        if (type(intent.quantity) is not int or not 0 < intent.quantity <= 2_147_483_600
                or intent.quantity % 100):
            return "invalid_lot_size"
        if not _finite(intent.limit_price) or intent.limit_price <= 0:
            return "invalid_limit_price"
        if type(intent.signal_asof) is not date or type(intent.execution_date) is not date:
            return "invalid_signal_date"
        local = now.astimezone(SHANGHAI)
        if intent.execution_date != local.date():
            return "wrong_execution_date"
        days = self.config.trading_days
        if intent.execution_date not in days:
            return "not_trading_day"
        index = days.index(intent.execution_date)
        if index == 0 or intent.signal_asof != days[index - 1]:
            return "stale_or_future_signal"
        local_time = local.time().replace(tzinfo=None)
        if not (time(9, 30) <= local_time < time(11, 30) or time(13) <= local_time < time(15)):
            return "outside_continuous_session"
        notional = intent.quantity * intent.limit_price
        if not math.isfinite(notional) or notional > self.config.max_order_notional:
            return "order_notional_limit"
        return ""

    def _snapshot_reason(self, snapshot: BrokerSnapshot, now: datetime) -> str:
        if not isinstance(snapshot, BrokerSnapshot):
            return "invalid_snapshot"
        if snapshot.is_simulation is not True:
            return "simulation_account_required"
        if snapshot.account_binding != self.config.account_binding:
            return "snapshot_account_mismatch"
        if not _aware(snapshot.asof):
            return "invalid_snapshot_time"
        age = (now - snapshot.asof).total_seconds()
        if age < 0 or age > self.config.snapshot_max_age_seconds:
            return "stale_snapshot"
        if not _finite(snapshot.available_cash) or snapshot.available_cash < 0:
            return "invalid_available_cash"
        if not isinstance(snapshot.available_shares, Mapping) or any(
            not isinstance(asset, str) or type(qty) is not int or qty < 0
            for asset, qty in snapshot.available_shares.items()
        ):
            return "invalid_available_shares"
        return ""

    def execute(self, intent: OrderIntent, *, now: datetime | None = None) -> ExecutionResult:
        live_clock = now is None
        now = now or datetime.now(timezone.utc)
        reason = self._reason(intent, now)
        if reason:
            return ExecutionResult(intent.intent_id, "blocked", reason_code=reason)
        fingerprint = _fingerprint(intent)
        # BEGIN IMMEDIATE serializes snapshot checks and reservations across
        # processes. The remote call happens only after the reservation commits.
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            old = conn.execute("SELECT * FROM intents WHERE account=? AND intent_id=?",
                               (self.config.account_binding, intent.intent_id)).fetchone()
            if old is not None:
                conn.commit()
                if old["fingerprint"] != fingerprint:
                    return ExecutionResult(intent.intent_id, "blocked", reason_code="intent_id_conflict",
                                           duplicate=True)
                return self._result(old, duplicate=True)
            try:
                snapshot = self.broker.snapshot()
            except Exception:
                conn.rollback()
                return ExecutionResult(intent.intent_id, "blocked", reason_code="snapshot_unavailable")
            if live_clock:
                now = datetime.now(timezone.utc)
            reason = self._reason(intent, now) or self._snapshot_reason(snapshot, now)
            if not reason:
                latest_fill = conn.execute("SELECT MAX(timestamp) FROM fills WHERE account=?",
                                           (self.config.account_binding,)).fetchone()[0]
                if latest_fill and _stamp(snapshot.asof) < latest_fill:
                    reason = "snapshot_precedes_known_fill"
            rows = conn.execute("SELECT * FROM intents WHERE account=?",
                                (self.config.account_binding,)).fetchall()
            if not reason:
                day_used = sum(r["quantity"] * r["limit_price"] for r in rows
                               if r["execution_date"] == intent.execution_date.isoformat()
                               and r["status"] != "rejected")
                if day_used + intent.quantity * intent.limit_price > self.config.max_daily_notional:
                    reason = "daily_notional_limit"
            if not reason and intent.side == "buy":
                multiplier = 1 + self.config.fee_buffer_bps / 10000
                held = sum((r["quantity"] - r["filled_quantity"]) * r["limit_price"] * multiplier
                           for r in rows if r["status"] in _PENDING and r["side"] == "buy")
                if held + intent.quantity * intent.limit_price * multiplier > snapshot.available_cash:
                    reason = "insufficient_available_cash"
            if not reason and intent.side == "sell":
                held = sum(r["quantity"] - r["filled_quantity"] for r in rows
                           if r["status"] in _PENDING and r["side"] == "sell" and r["asset"] == intent.asset)
                if held + intent.quantity > snapshot.available_shares.get(intent.asset, 0):
                    reason = "insufficient_sellable_shares"
            if reason:
                conn.rollback()
                return ExecutionResult(intent.intent_id, "blocked", reason_code=reason)
            conn.execute("""INSERT INTO intents
                (account,intent_id,fingerprint,strategy_id,asset,side,quantity,limit_price,
                 signal_asof,execution_date,status,created_at,updated_at)
                 VALUES (?,?,?,?,?,?,?,?,?,?,'submitting',?,?)""",
                         (intent.account_binding, intent.intent_id, fingerprint, intent.strategy_id,
                          intent.asset, intent.side, intent.quantity, intent.limit_price,
                          intent.signal_asof.isoformat(), intent.execution_date.isoformat(),
                          _stamp(now), _stamp(now)))
            self._event(conn, intent.intent_id, "submitting", "", now)
            conn.commit()
        try:
            receipt = self.broker.submit(intent)
        except Exception:
            return self._finish_submission(intent, None, "unknown", "submission_outcome_unknown", now)
        if (not isinstance(receipt, SubmitReceipt)
                or receipt.account_binding != intent.account_binding
                or receipt.status not in ("accepted", "rejected")
                or (receipt.status == "accepted" and not self._valid_id(receipt.order_id))
                or (receipt.order_id is not None and not self._valid_id(receipt.order_id))):
            return self._finish_submission(intent, None, "unknown", "invalid_submission_receipt", now)
        return self._finish_submission(intent, receipt.order_id, receipt.status, "", now)

    def _finish_submission(self, intent: OrderIntent, order_id: str | None,
                           status: str, reason: str, now: datetime) -> ExecutionResult:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute("SELECT * FROM intents WHERE account=? AND intent_id=?",
                                   (self.config.account_binding, intent.intent_id)).fetchone()
            # A concurrent reconciliation may already have more authoritative
            # fill evidence than the submit receipt; never overwrite it.
            if current["status"] not in ("submitting", "unknown"):
                conn.commit()
                return self._result(current)
            if order_id is not None and conn.execute(
                "SELECT 1 FROM intents WHERE account=? AND order_id=? AND intent_id<>?",
                (self.config.account_binding, order_id, intent.intent_id)
            ).fetchone():
                status, reason, order_id = "unknown", "duplicate_broker_order_id", None
            conn.execute("""UPDATE intents SET status=?,order_id=COALESCE(?,order_id),
                reason_code=?,updated_at=? WHERE account=? AND intent_id=?""",
                         (status, order_id, reason, _stamp(now),
                          self.config.account_binding, intent.intent_id))
            self._event(conn, intent.intent_id, status, reason, now)
            conn.commit()
            return self._result(conn.execute("SELECT * FROM intents WHERE account=? AND intent_id=?",
                                            (self.config.account_binding, intent.intent_id)).fetchone())

    @staticmethod
    def _valid_id(value: object) -> bool:
        return isinstance(value, str) and bool(_ID.fullmatch(value))

    def _event(self, conn: sqlite3.Connection, intent_id: str, status: str,
               reason: str, now: datetime) -> None:
        conn.execute("INSERT INTO events(account,intent_id,status,reason_code,timestamp) VALUES(?,?,?,?,?)",
                     (self.config.account_binding, intent_id, status, reason, _stamp(now)))

    @staticmethod
    def _result(row: sqlite3.Row, duplicate: bool = False) -> ExecutionResult:
        pending = row["status"] == "submitting"
        return ExecutionResult(row["intent_id"], "unknown" if pending else row["status"],
                               row["order_id"], row["filled_quantity"],
                               "submission_unconfirmed" if pending else row["reason_code"], duplicate)

    def journal(self) -> list[ExecutionResult]:
        """Return local status; a durable unfinished submission is always unknown."""
        with self._connect() as conn:
            return [self._result(row) for row in conn.execute(
                "SELECT * FROM intents WHERE account=? ORDER BY created_at,intent_id",
                (self.config.account_binding,)).fetchall()]

    def reconcile(self, *, now: datetime | None = None) -> ReconciliationResult:
        """Query only; never re-send, infer fills, or clear an absent unknown order."""
        now = now or datetime.now(timezone.utc)
        if not _aware(now) or getattr(self.broker, "simulation_only", None) is not True:
            return ReconciliationResult(issues=("simulation_adapter_or_timezone_required",))
        try:
            orders = list(self.broker.query_orders())
            dated_query = getattr(self.broker, "query_fills_for_dates", None)
            if callable(dated_query):
                with self._connect() as conn:
                    unresolved = conn.execute(
                        "SELECT DISTINCT execution_date FROM intents WHERE account=? "
                        "AND status IN ('submitting','unknown','accepted','open','partial') "
                        "ORDER BY execution_date", (self.config.account_binding,)).fetchall()
                dates = tuple(date.fromisoformat(row[0]) for row in unresolved)
                fills = list(dated_query(dates))
            else:
                fills = list(self.broker.query_fills())
        except Exception:
            return ReconciliationResult(issues=("reconciliation_unavailable",))
        issues: list[str] = []
        updated = added = 0
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute("SELECT * FROM intents WHERE account=?",
                                (self.config.account_binding,)).fetchall()
            for row in rows:
                matches = [o for o in orders if isinstance(o, BrokerOrder)
                           and o.account_binding == self.config.account_binding
                           and o.intent_id == row["intent_id"]]
                if len(matches) != 1:
                    if matches:
                        issues.append("ambiguous_order_match")
                        continue
                    # Some official simulation APIs expose fills but no order
                    # query. A known submit order ID still lets actual fills
                    # confirm execution without manufacturing an order report.
                    if not row["order_id"]:
                        continue
                    order = None
                    order_id, evidence_asof = row["order_id"], now
                else:
                    order = matches[0]
                    if not self._order_matches(order, row, now):
                        issues.append("order_evidence_mismatch")
                        continue
                    if row["broker_asof"] and _stamp(order.asof) < row["broker_asof"]:
                        continue
                    order_id, evidence_asof = order.order_id, order.asof
                collision = conn.execute(
                    "SELECT 1 FROM intents WHERE account=? AND order_id=? AND intent_id<>?",
                    (self.config.account_binding, order_id, row["intent_id"])
                ).fetchone()
                if collision:
                    issues.append("duplicate_broker_order_id")
                    continue
                matched_fills = [f for f in fills if isinstance(f, BrokerFill)
                                 and f.account_binding == self.config.account_binding
                                 and f.order_id == order_id]
                if order is None and not matched_fills:
                    continue
                conn.execute("SAVEPOINT order_fills")
                valid = True
                fresh = 0
                for fill in matched_fills:
                    if not self._fill_matches(fill, row, evidence_asof, now):
                        valid = False
                        break
                    values = (self.config.account_binding, fill.fill_id, row["intent_id"], order_id,
                              fill.quantity, float(fill.price), _stamp(fill.timestamp), fill.fee)
                    existing = conn.execute("SELECT * FROM fills WHERE account=? AND fill_id=?",
                                            (self.config.account_binding, fill.fill_id)).fetchone()
                    if existing is not None:
                        if tuple(existing) != values:
                            valid = False
                            break
                    else:
                        conn.execute("INSERT INTO fills VALUES(?,?,?,?,?,?,?,?)", values)
                        fresh += 1
                quantity = conn.execute(
                    "SELECT COALESCE(SUM(quantity),0) FROM fills WHERE account=? AND intent_id=?",
                    (self.config.account_binding, row["intent_id"])
                ).fetchone()[0]
                if not valid or quantity > row["quantity"]:
                    conn.execute("ROLLBACK TO order_fills")
                    conn.execute("RELEASE order_fills")
                    issues.append("invalid_fill_evidence")
                    continue
                conn.execute("RELEASE order_fills")
                reported_state = order.status if order is not None else row["status"]
                state, reason = reported_state, ""
                if reported_state == "filled" and quantity != row["quantity"]:
                    state, reason = "unknown", "filled_status_without_full_fills"
                elif reported_state == "rejected" and quantity:
                    state, reason = "unknown", "rejected_status_with_fills"
                elif quantity == row["quantity"]:
                    state = "filled"
                elif quantity and reported_state in _PENDING:
                    state = "partial"
                elif reported_state == "partial" and not quantity:
                    state, reason = "unknown", "partial_status_without_fills"
                # Filled evidence remains immutable; stale or contradictory
                # status messages cannot erase recorded trades.
                if row["status"] == "filled" and state != "filled":
                    state, reason = "filled", "terminal_state_preserved"
                elif row["status"] == "cancelled" and state in _PENDING:
                    state, reason = "cancelled", "terminal_state_preserved"
                conn.execute("""UPDATE intents SET status=?,order_id=?,filled_quantity=?,reason_code=?,
                    broker_asof=?,updated_at=? WHERE account=? AND intent_id=?""",
                             (state, order_id, quantity, reason,
                              _stamp(order.asof) if order else row["broker_asof"],
                              _stamp(now), self.config.account_binding, row["intent_id"]))
                self._event(conn, row["intent_id"], state, reason, now)
                updated += 1
                added += fresh
            conn.commit()
        return ReconciliationResult(updated, added, tuple(dict.fromkeys(issues)))

    def _order_matches(self, order: BrokerOrder, row: sqlite3.Row, now: datetime) -> bool:
        return (
            self._valid_id(order.order_id)
            and (row["order_id"] is None or order.order_id == row["order_id"])
            and order.asset == row["asset"] and order.side == row["side"]
            and type(order.quantity) is int and order.quantity == row["quantity"]
            and _finite(order.limit_price) and order.limit_price == row["limit_price"]
            and order.status in _ORDER_STATES
            and _aware(order.asof) and order.asof <= now
            and _stamp(order.asof) >= row["created_at"]
        )

    @staticmethod
    def _fill_matches(fill: BrokerFill, row: sqlite3.Row, evidence_asof: datetime, now: datetime) -> bool:
        # A provider declaring second precision cannot report the fractional
        # part of an immediate fill. Only that explicitly declared precision
        # permits flooring the submission bound; an earlier second still fails.
        created = datetime.fromisoformat(row["created_at"])
        seconds = fill.timestamp_precision == "seconds"
        lower_bound = created.replace(microsecond=0) if seconds else created
        return (
            SimulationExecutor._valid_id(fill.fill_id)
            and type(fill.quantity) is int and fill.quantity > 0
            and _finite(fill.price) and fill.price > 0
            and (fill.fee is None or (_finite(fill.fee) and fill.fee >= 0))
            and fill.timestamp_precision in {"seconds", "microseconds"}
            and _aware(fill.timestamp) and fill.timestamp <= min(now, evidence_asof)
            and (not seconds or fill.timestamp.microsecond == 0)
            and fill.timestamp >= lower_bound
            and ((row["side"] == "buy" and fill.price <= row["limit_price"])
                 or (row["side"] == "sell" and fill.price >= row["limit_price"]))
        )
