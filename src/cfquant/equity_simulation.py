"""Pure executable-lot adapters for explicitly dated V3 and CogAlpha experiments.

These are forward simulation adaptations. Frozen financial/industry references,
lot sizes, live quote limits and actual-cash execution differ from the historical
research ledgers. There are no broker calls or state writes in this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
import hashlib
import json
import math
import re
from typing import Mapping, Sequence
import urllib.request

import numpy as np
import pandas as pd

from .benchmark_research import POLICIES
from .engine import _affordable_notional
from .risk import constrained_weights
from .simulation_execution import OrderIntent
from .simulation_signals import SHANGHAI


V3_ID = "v3_managed"
COG_ID = "cogalpha_rank"
FORWARD_START = date(2026, 10, 21)
SCORE_COLUMNS = {V3_ID: "v3_lightgbm_frozen_reference",
                 COG_ID: "cogalpha_731_rank_frozen_reference"}
_ASSET = re.compile(r"(?:(?:600|601|603|605)\d{3}\.SH|(?:000|001|002|003)\d{3}\.SZ)\Z")
_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
ADAPTATION = (
    "Forward simulation adaptation; financial/industry reference frozen at 2026-09-18. "
    "Integer lots, current quote limits, actual cash and broker fills differ from research ledgers."
)


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _now(now: datetime | None) -> datetime:
    value = datetime.now(SHANGHAI) if now is None else now
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("A timezone-aware clock is required.")
    return value.astimezone(SHANGHAI)


def _sessions(signal_asof: date, execution_date: date, trading_days: Sequence[date]) -> tuple[date, ...]:
    if type(signal_asof) is not date or type(execution_date) is not date:
        raise ValueError("Signal and execution dates must be date objects.")
    days = tuple(trading_days)
    if not days or any(type(day) is not date for day in days) or tuple(sorted(set(days))) != days:
        raise ValueError("Use sorted unique official trading dates.")
    if signal_asof not in days or execution_date not in days:
        raise ValueError("Signal and execution dates must be official sessions.")
    if days.index(execution_date) != days.index(signal_asof) + 1:
        raise ValueError("Execution must be the strictly next official session.")
    return days


def _score_rows(scores: pd.DataFrame, strategy_id: str, signal_asof: date) -> pd.DataFrame:
    required = {"date", "asset", "eligible", SCORE_COLUMNS[strategy_id]}
    if not isinstance(scores, pd.DataFrame) or not required <= set(scores) or scores.empty:
        raise ValueError("A nonempty current dated score table is required.")
    data = scores.copy()
    dates = pd.to_datetime(data.date, errors="raise")
    if (dates.isna().any() or dates.dt.tz is not None
            or not dates.eq(pd.Timestamp(signal_asof)).all() or data.asset.duplicated().any()):
        raise ValueError("Scores must have unique assets and exactly the requested signal date.")
    if not data.eligible.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("Eligibility must contain explicit Boolean values.")
    data = data.set_index("asset").sort_index()
    numeric = pd.to_numeric(data[SCORE_COLUMNS[strategy_id]], errors="raise")
    allowed = data.index.map(lambda asset: isinstance(asset, str) and bool(_ASSET.fullmatch(asset)))
    data["usable_score"] = numeric.where(data.eligible & allowed & np.isfinite(numeric))
    return data


@dataclass(frozen=True)
class EquityQuote:
    asset: str
    price: float
    asof: datetime
    up_limit: float
    down_limit: float
    price_basis: str = "raw"
    source: str = "tencent_public"


def fetch_equity_quotes(assets: Sequence[str], *, now: datetime | None = None,
                        max_age_seconds: float = 30.0, opener=None) -> dict[str, EquityQuote]:
    """One bounded HTTPS request for 1..100 explicitly named main-board shares.

    Source contract: easyquotation/tencent.py, fields 2/3/30/47/48, at
    https://github.com/shidenggui/easyquotation/blob/master/easyquotation/tencent.py
    Public Tencent data are a separate research feed, not the THS broker feed.
    """
    assets = tuple(assets)
    if (not 1 <= len(assets) <= 100 or len(set(assets)) != len(assets)
            or any(not isinstance(asset, str) or not _ASSET.fullmatch(asset) for asset in assets)):
        raise ValueError("Request 1..100 unique supported main-board assets.")
    if not _finite(max_age_seconds) or max_age_seconds <= 0:
        raise ValueError("Quote age limit must be positive.")
    codes = {asset[-2:].lower() + asset[:6]: asset for asset in assets}
    request = urllib.request.Request("https://qt.gtimg.cn/q=" + ",".join(sorted(codes)),
                                     headers={"User-Agent": "Qingxu-simulation/2.5"})
    try:
        with (opener or urllib.request.urlopen)(request, timeout=12) as response:
            raw = response.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("oversize")
        quotes = {}
        for row in raw.decode("gb18030").splitlines():
            if not row.strip():
                continue
            match = re.fullmatch(r'\s*v_((?:sh|sz)\d{6})="([^"]+)";\s*', row)
            if not match or match.group(1) not in codes:
                raise ValueError("unexpected code")
            asset = codes[match.group(1)]
            fields = match.group(2).split("~")
            if (len(fields) <= 49 or fields[2] != asset[:6] or asset in quotes
                    or not re.fullmatch(r"\d{14}", fields[30])):
                raise ValueError("duplicate or malformed")
            stamp = datetime.strptime(fields[30], "%Y%m%d%H%M%S").replace(tzinfo=SHANGHAI)
            quotes[asset] = EquityQuote(asset, float(fields[3]), stamp,
                                         float(fields[47]), float(fields[48]))
        if set(quotes) != set(assets):
            raise ValueError("missing")
    except Exception:
        raise RuntimeError("Public equity quote request/format failed; no retries or fallback.") from None
    local = _now(now)
    for asset, quote in quotes.items():
        reason = _quote_reason(quote, asset, local, max_age_seconds)
        if reason:
            raise ValueError(f"Public equity quote not executable for {asset}: {reason}.")
    return quotes


def fetch_equity_quote(asset: str, **kwargs) -> EquityQuote:
    return fetch_equity_quotes([asset], **kwargs)[asset]


def _quote_reason(quote: EquityQuote | None, asset: str, now: datetime, max_age: float) -> str:
    if not isinstance(quote, EquityQuote) or quote.asset != asset or quote.price_basis != "raw":
        return "missing_raw_quote"
    if (not all(_finite(value) and value > 0 for value in
                (quote.price, quote.up_limit, quote.down_limit))
            or not quote.down_limit < quote.up_limit):
        return "missing_current_price_limits"
    if quote.price < quote.down_limit - 1e-8 or quote.price > quote.up_limit + 1e-8:
        return "price_outside_current_limits"
    if abs(quote.price * 100 - round(quote.price * 100)) > 1e-7:
        return "invalid_stock_price_tick"
    if not isinstance(quote.asof, datetime) or quote.asof.tzinfo is None:
        return "missing_quote_timezone"
    age = (now - quote.asof).total_seconds()
    if quote.asof.astimezone(SHANGHAI).date() != now.date() or not 0 <= age <= max_age:
        return "stale_or_future_quote"
    return ""


@dataclass(frozen=True)
class EquityTargets:
    strategy_id: str
    signal_asof: date
    execution_date: date
    weights: dict[str, float]
    rebalance_due: bool
    metadata: dict


def build_v3_targets(scores: pd.DataFrame, market_tail: pd.DataFrame,
                     index_close: pd.Series, *, signal_asof: date, execution_date: date,
                     trading_days: Sequence[date], previous_target_constituents: Sequence[str] = (),
                     start: date = FORWARD_START) -> EquityTargets:
    """Managed V3's 50/20 buffer, 60-day risk budget and 120-day index trend.

    Rebalances anchor to the new forward start, every 20 official sessions.
    Missing selected-stock returns contribute zero only inside the risk estimate,
    matching risk.build_targets; executable quotes never receive that treatment.
    """
    days = _sessions(signal_asof, execution_date, trading_days)
    if type(start) is not date or start not in days or execution_date < start:
        raise ValueError("Forward start must be an official session on/before execution.")
    policy = POLICIES["managed"]
    due = (days.index(execution_date) - days.index(start)) % policy.rebalance_every == 0
    rows = _score_rows(scores, V3_ID, signal_asof)
    metadata = {"adaptation": ADAPTATION, "policy": asdict(policy), "forward_start": str(start),
                "signal_asof": str(signal_asof), "execution_date": str(execution_date),
                "rebalance_due": due}
    if not due:
        return EquityTargets(V3_ID, signal_asof, execution_date, {}, False, metadata)
    if not {"volatility", "industry"} <= set(rows):
        raise ValueError("V3 requires formation-date volatility and industry.")
    if (not isinstance(index_close, pd.Series) or not isinstance(index_close.index, pd.DatetimeIndex)
            or index_close.index.tz is not None or not index_close.index.is_unique
            or not index_close.index.is_monotonic_increasing or len(index_close) < policy.trend_window
            or not index_close.index.equals(index_close.index.normalize())
            or index_close.index[-1].date() != signal_asof):
        raise ValueError("Need a current, ordered, complete 120-session CSI300 close history.")
    expected = pd.DatetimeIndex([day for day in days if day <= signal_asof])
    tail_dates = expected[-policy.trend_window:]
    if not index_close.index[-policy.trend_window:].equals(tail_dates):
        raise ValueError("CSI300 history misses official sessions.")
    benchmark = pd.to_numeric(index_close.iloc[-policy.trend_window:], errors="raise")
    if not np.isfinite(benchmark).all() or (benchmark <= 0).any():
        raise ValueError("Invalid CSI300 history.")
    required = {"date", "asset", "close"}
    if not required <= set(market_tail) or market_tail.empty:
        raise ValueError("Stock adjusted-close history is required for the risk estimate.")
    market = market_tail.loc[:, ["date", "asset", "close"]].copy()
    market["date"] = pd.to_datetime(market.date, errors="raise")
    if (market.date.dt.tz is not None or market.date.isna().any()
            or not market.date.eq(market.date.dt.normalize()).all()
            or market.date.max().date() != signal_asof
            or market.duplicated(["date", "asset"]).any()):
        raise ValueError("Stock history contains future/missing dates or duplicate keys.")
    if not market.date.isin(expected).all():
        raise ValueError("Stock history contains unofficial sessions.")
    closes = market.pivot(index="date", columns="asset", values="close")
    if len(expected) < 61:
        raise ValueError("Insufficient risk-estimation calendar.")
    risk_dates = expected[-61:]
    if not closes.index.intersection(risk_dates).equals(risk_dates):
        raise ValueError("Stock history misses full risk-estimation sessions.")
    closes = closes.reindex(risk_dates).apply(pd.to_numeric, errors="raise")
    if np.isinf(closes.to_numpy()).any() or (closes <= 0).any().any():
        raise ValueError("Nonpositive or infinite stock risk-estimation prices.")
    valid_scores = rows.usable_score
    volatility = pd.to_numeric(rows.volatility, errors="raise")
    previous = tuple(dict.fromkeys(previous_target_constituents))
    seed = constrained_weights(valid_scores, volatility, rows.industry, 1.0, policy, previous)
    selected = seed[seed > 0]
    missing_fraction = 0.0
    if not selected.empty:
        returns = closes.pct_change(fill_method=None).iloc[1:].reindex(columns=selected.index)
        missing_fraction = float(returns.isna().to_numpy().mean())
        hist = returns.fillna(0).dot(selected / selected.sum())
        sigma = float(hist.std(ddof=1) * np.sqrt(252))
        scale = min(1.0, policy.target_volatility / sigma) if sigma > 1e-8 else 0.0
    else:
        sigma, scale = 0.0, 0.0
    defensive = bool(benchmark.iloc[-1] < benchmark.mean())
    exposure = policy.max_exposure * scale * (policy.defensive_scale if defensive else 1.0)
    weights = constrained_weights(valid_scores, volatility, rows.industry, exposure, policy, previous)
    weights = {asset: float(weight) for asset, weight in weights.items() if weight > 0}
    metadata.update({"estimated_annual_volatility": sigma, "defensive": defensive,
                     "target_exposure": sum(weights.values()),
                     "risk_estimate_missing_return_fraction": missing_fraction,
                     "previous_target_constituents": list(previous),
                     "target_constituents": list(weights)})
    return EquityTargets(V3_ID, signal_asof, execution_date, weights, True, metadata)


@dataclass(frozen=True)
class CogSelection:
    signal_asof: date
    execution_date: date
    original_holdings: tuple[str, ...]
    sell_candidates: tuple[str, ...]
    buy_candidates: tuple[str, ...]
    selection_id: str

    def __post_init__(self):
        if (type(self.signal_asof) is not date or type(self.execution_date) is not date
                or self.signal_asof >= self.execution_date
                or len(self.original_holdings) > 50 or len(self.sell_candidates) > 5
                or len(self.buy_candidates) > 55):
            raise ValueError("Invalid frozen CogAlpha selection dimensions.")
        for names in (self.original_holdings, self.sell_candidates, self.buy_candidates):
            if (len(names) != len(set(names))
                    or any(not isinstance(asset, str) or not _ASSET.fullmatch(asset) for asset in names)):
                raise ValueError("Frozen selection contains invalid or duplicate assets.")
        if (not set(self.sell_candidates) <= set(self.original_holdings)
                or set(self.buy_candidates).intersection(self.original_holdings)):
            raise ValueError("Frozen CogAlpha entrants/exits do not match original holdings.")

    def _payload(self) -> dict:
        return {"signal_asof": str(self.signal_asof), "execution_date": str(self.execution_date),
                "original_holdings": list(self.original_holdings),
                "sell_candidates": list(self.sell_candidates), "buy_candidates": list(self.buy_candidates),
                "holdings": 50, "dropout": 5, "strategy_id": COG_ID}

    def to_dict(self) -> dict:
        return {**self._payload(), "selection_id": self.selection_id}

    @classmethod
    def from_dict(cls, payload: dict) -> "CogSelection":
        result = cls(date.fromisoformat(payload["signal_asof"]),
                     date.fromisoformat(payload["execution_date"]),
                     tuple(payload["original_holdings"]), tuple(payload["sell_candidates"]),
                     tuple(payload["buy_candidates"]), payload["selection_id"])
        if result.to_dict() != payload or _hash(result._payload()) != result.selection_id:
            raise ValueError("Frozen CogAlpha selection identity changed.")
        return result


def build_cogalpha_selection(scores: pd.DataFrame, *, signal_asof: date, execution_date: date,
                             trading_days: Sequence[date], holdings: Mapping[str, int]) -> CogSelection:
    """Freeze the engine's 50/5 ranking exactly once before any daily execution."""
    _sessions(signal_asof, execution_date, trading_days)
    rows = _score_rows(scores, COG_ID, signal_asof)
    _validate_holdings(holdings, holdings)
    held = {asset for asset, shares in holdings.items() if shares > 0}
    if len(held) > 50:
        raise ValueError("CogAlpha account already exceeds 50 holdings.")

    def ranking(asset):
        value = rows.usable_score.get(asset, np.nan)
        return (-float(value) if np.isfinite(value) else math.inf, asset)

    available = sorted((asset for asset, score in rows.usable_score.items()
                        if asset not in held and np.isfinite(score)), key=ranking)
    today = available[:5 + max(0, 50 - len(held))]
    combined = sorted(held.union(today), key=ranking)
    replace_count = min(5, max(0, len(combined) - 50))
    sell = tuple(asset for asset in reversed(combined[-replace_count:]) if asset in held) if replace_count else ()
    result = CogSelection(signal_asof, execution_date, tuple(sorted(held)), sell, tuple(today), "")
    return CogSelection(signal_asof, execution_date, result.original_holdings, sell, tuple(today),
                        _hash(result._payload()))


@dataclass(frozen=True)
class EquityPlan:
    intents: tuple[OrderIntent, ...]
    blocked: tuple[dict, ...]
    reserved_cash: float
    metadata: dict
    selection: CogSelection | None = None


def _validate_holdings(holdings, available):
    if not isinstance(holdings, Mapping) or not isinstance(available, Mapping):
        raise ValueError("Holdings and available shares must be mappings.")
    for asset in set(holdings).union(available):
        total, sellable = holdings.get(asset, 0), available.get(asset, 0)
        if (not isinstance(asset, str) or not _ASSET.fullmatch(asset)
                or type(total) is not int or type(sellable) is not int
                or min(total, sellable) < 0 or sellable > total or total % 100 or sellable % 100):
            raise ValueError("Use supported assets and valid 100-share total/sellable lots.")


class _Planner:
    def __init__(self, strategy_id, signal_asof, execution_date, *, account_binding,
                 account_nav, available_cash, holdings, available_shares, quotes, now,
                 max_order_notional, fee_bps, minimum_fee, quote_max_age_seconds):
        self.now = _now(now)
        if type(execution_date) is not date or self.now.date() != execution_date or signal_asof >= execution_date:
            raise ValueError("The dated signal must execute today after formation.")
        if not isinstance(account_binding, str) or not _ID.fullmatch(account_binding):
            raise ValueError("Invalid account binding.")
        _validate_holdings(holdings, available_shares)
        if (not _finite(account_nav) or account_nav <= 0 or not _finite(available_cash)
                or not 0 <= available_cash <= account_nav + 1e-8):
            raise ValueError("Current broker NAV and cash must be finite and consistent.")
        if (not _finite(max_order_notional) or max_order_notional <= 0
                or not _finite(fee_bps) or not 0 <= fee_bps <= 1000
                or not _finite(minimum_fee) or minimum_fee < 5
                or not _finite(quote_max_age_seconds) or quote_max_age_seconds <= 0):
            raise ValueError("Invalid lot-order, fee, or freshness limits.")
        self.strategy_id, self.signal_asof, self.execution_date = strategy_id, signal_asof, execution_date
        self.account_binding, self.nav, self.initial_cash = account_binding, account_nav, available_cash
        self.cash = available_cash
        self.holdings, self.available, self.quotes = holdings, available_shares, quotes
        self.cap, self.rate, self.minimum_fee = max_order_notional, fee_bps / 10000, minimum_fee
        self.max_age = quote_max_age_seconds
        self.intents, self.blocked = [], []

    def block(self, asset, side, reason):
        self.blocked.append({"asset": asset, "side": side, "reason": reason})

    def quote(self, asset, side):
        quote = self.quotes.get(asset)
        reason = _quote_reason(quote, asset, self.now, self.max_age)
        if not reason:
            if side == "buy" and quote.price >= quote.up_limit - 1e-8:
                reason = "upper_limit"
            elif side == "sell" and quote.price <= quote.down_limit + 1e-8:
                reason = "lower_limit"
        if reason:
            self.block(asset, side, reason)
            return None
        return quote

    def order(self, asset, side, requested, quote, budget=None):
        quantity = min(requested, math.floor(self.cap / quote.price / 100) * 100)
        if side == "sell":
            quantity = min(quantity, self.available.get(asset, 0))
            if quantity == 0:
                self.block(asset, side, "no_sellable_lots")
                return False
        else:
            available = self.cash if budget is None else min(self.cash, budget)
            affordable = math.floor(_affordable_notional(available, self.rate, self.minimum_fee)
                                    / quote.price / 100) * 100
            quantity = min(quantity, affordable)
        if quantity <= 0:
            self.block(asset, side, "no_affordable_lot")
            return False
        notional = quantity * quote.price
        fee = max(notional * self.rate, self.minimum_fee)
        if side == "buy":
            # Reserve current cash, never proceeds from planned sells.
            self.cash -= notional + fee
        elif fee > notional:
            # Very low-price lots can cost more in minimum fees than proceeds.
            if fee - notional > self.cash:
                self.block(asset, side, "insufficient_cash_for_sell_fee")
                return False
            self.cash -= fee - notional
        identity = _hash({"strategy_id": self.strategy_id, "account": self.account_binding,
                          "execution_date": str(self.execution_date), "asset": asset})
        self.intents.append(OrderIntent("equity:" + identity, self.strategy_id, self.account_binding,
                                        asset, side, int(quantity), quote.price,
                                        self.signal_asof, self.execution_date))
        return True

    def result(self, metadata, selection=None):
        return EquityPlan(tuple(self.intents), tuple(self.blocked), self.initial_cash - self.cash,
                          {**metadata, "adaptation": ADAPTATION,
                           "planned_at": self.now.isoformat(), "fee_bps": self.rate * 10000,
                           "minimum_fee": self.minimum_fee, "max_order_notional": self.cap,
                           "sell_proceeds_assumed": 0.0, "planned_sells_release_slots": False},
                          selection)


def plan_v3_orders(targets: EquityTargets, *, account_binding: str, account_nav: float,
                   available_cash: float, holdings: Mapping[str, int],
                   available_shares: Mapping[str, int], quotes: Mapping[str, EquityQuote],
                   now: datetime | None = None, max_order_notional: float = 20_000.0,
                   fee_bps: float = 25.0, minimum_fee: float = 5.0,
                   quote_max_age_seconds: float = 30.0) -> EquityPlan:
    """Lot deltas against current NAV; failed exits retain slots and consume NAV."""
    if not isinstance(targets, EquityTargets) or targets.strategy_id != V3_ID:
        raise ValueError("Use a current V3 target bundle.")
    if (len(targets.weights) > 50 or any(not _ASSET.fullmatch(asset) or not _finite(weight)
                                        or not 0 < weight <= POLICIES["managed"].max_stock_weight + 1e-8
                                        for asset, weight in targets.weights.items())
            or sum(targets.weights.values()) > POLICIES["managed"].max_exposure + 1e-8):
        raise ValueError("V3 target limits violated.")
    planner = _Planner(V3_ID, targets.signal_asof, targets.execution_date,
                       account_binding=account_binding, account_nav=account_nav,
                       available_cash=available_cash, holdings=holdings, available_shares=available_shares,
                       quotes=quotes, now=now, max_order_notional=max_order_notional,
                       fee_bps=fee_bps, minimum_fee=minimum_fee,
                       quote_max_age_seconds=quote_max_age_seconds)
    if not targets.rebalance_due:
        return planner.result(targets.metadata)
    held = {asset for asset, shares in holdings.items() if shares > 0}
    vacancies = max(0, 50 - len(held))
    cash_reserve = account_nav * (1.0 - sum(targets.weights.values()))
    buys = []
    for asset in sorted(held.union(targets.weights)):
        quote = quotes.get(asset)
        reason = _quote_reason(quote, asset, planner.now, planner.max_age)
        if reason:
            planner.block(asset, "rebalance", reason)
            continue
        desired = math.floor(account_nav * targets.weights.get(asset, 0) / quote.price / 100) * 100
        delta = desired - holdings.get(asset, 0)
        if delta < 0:
            quote = planner.quote(asset, "sell")
            if quote:
                planner.order(asset, "sell", -delta, quote)
        elif delta > 0:
            buys.append((asset, delta))
    for asset, delta in sorted(buys, key=lambda item: (-targets.weights[item[0]], item[0])):
        if asset not in held and vacancies <= 0:
            planner.block(asset, "buy", "no_actual_holding_slot")
            continue
        quote = planner.quote(asset, "buy")
        budget = max(0.0, planner.cash - cash_reserve)
        if quote and planner.order(asset, "buy", delta, quote, budget=budget) and asset not in held:
            vacancies -= 1
    return planner.result({**targets.metadata, "target_weights": targets.weights,
                            "minimum_cash_reserve": cash_reserve,
                            "actual_holding_slots_before_orders": len(held)})


def plan_cogalpha_orders(scores: pd.DataFrame, *, signal_asof: date, execution_date: date,
                         trading_days: Sequence[date], account_binding: str, account_nav: float,
                         available_cash: float, holdings: Mapping[str, int],
                         available_shares: Mapping[str, int], quotes: Mapping[str, EquityQuote],
                         selection: CogSelection | None = None, now: datetime | None = None,
                         max_order_notional: float = 20_000.0, fee_bps: float = 25.0,
                         minimum_fee: float = 5.0, quote_max_age_seconds: float = 30.0) -> EquityPlan:
    """Daily 50/5 dropout with unchanged retained shares and frozen daily selection.

    Persist ``result.selection`` before sending anything, and reuse it after
    reconciled fills. Rebuilding it after sells can cause repeated five-name
    replacements on the same day. Pending orders/cash reservations must already
    be excluded or block a new planning pass at the caller.
    """
    _sessions(signal_asof, execution_date, trading_days)
    _score_rows(scores, COG_ID, signal_asof)
    if selection is None:
        selection = build_cogalpha_selection(scores, signal_asof=signal_asof, execution_date=execution_date,
                                             trading_days=trading_days, holdings=holdings)
    else:
        selection = CogSelection.from_dict(selection.to_dict())
        if selection.signal_asof != signal_asof or selection.execution_date != execution_date:
            raise ValueError("Frozen selection belongs to a different session.")
    held = {asset for asset, shares in holdings.items() if shares > 0}
    if len(held) > 50 or not held <= set(selection.original_holdings).union(selection.buy_candidates):
        raise ValueError("Actual holdings differ from the frozen selection's permitted assets.")
    planner = _Planner(COG_ID, signal_asof, execution_date, account_binding=account_binding,
                       account_nav=account_nav, available_cash=available_cash, holdings=holdings,
                       available_shares=available_shares, quotes=quotes, now=now,
                       max_order_notional=max_order_notional, fee_bps=fee_bps,
                       minimum_fee=minimum_fee, quote_max_age_seconds=quote_max_age_seconds)
    for asset in selection.sell_candidates:
        if asset not in held:
            continue
        quote = planner.quote(asset, "sell")
        if quote:
            planner.order(asset, "sell", holdings[asset], quote)
    vacancies = max(0, 50 - len(held))
    candidates = [asset for asset in selection.buy_candidates if asset not in held]
    buy = candidates[:vacancies]
    budget = available_cash / len(buy) if buy else 0.0
    for asset in buy:
        quote = planner.quote(asset, "buy")
        if quote:
            requested = math.floor(_affordable_notional(budget, planner.rate, planner.minimum_fee)
                                   / quote.price / 100) * 100
            planner.order(asset, "buy", requested, quote, budget=budget)
    if not vacancies and candidates:
        for asset in candidates:
            planner.block(asset, "buy", "no_actual_holding_slot")
    return planner.result({"holdings": 50, "dropout": 5,
                            "retained_shares_resized": False, "selection_id": selection.selection_id,
                            "actual_holding_slots_before_orders": len(held)}, selection)
