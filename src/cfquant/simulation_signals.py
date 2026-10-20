"""Fresh, causal RSRS targets for dedicated simulation accounts.

The signal is formed on the CSI 300 price index; orders use the *raw* price of
510300.SH.  This is an explicitly different ETF implementation, not a claim to
replicate the research index ledger.  Market acquisition is read-only.  Nothing
in this module submits an order or changes an execution permission.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any
import urllib.request
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .data import TushareProvider, digest
from .rsrs import SPECS, strategy
from .simulation_execution import OrderIntent


SHANGHAI = ZoneInfo("Asia/Shanghai")
SIGNAL_ASSET = "000300.SH"
EXECUTION_ASSET = "510300.SH"
HISTORY_START = date(2019, 1, 1)
STRATEGY_IDS = ("rsrs_standard_ma", "rsrs_right_ma", "etf_buy_hold")
RESEARCH_STRATEGIES = dict(zip(STRATEGY_IDS, ("standard_ma", "right_ma", "buy_hold")))
_STRATEGY_ALIASES = {value: key for key, value in RESEARCH_STRATEGIES.items()}
SCHEMA_VERSION = 1
MAPPING_NOTICE = (
    "CSI 300 price-index close signal -> 510300.SH raw-price ETF execution; "
    "ETF tracking error, distributions, trading costs and lot rounding differ "
    "from the historical index research proxy."
)
_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _aware(value: datetime) -> bool:
    return isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None


def _now(value: datetime | None) -> datetime:
    value = value if value is not None else datetime.now(timezone.utc)
    if not _aware(value):
        raise ValueError("An explicit timezone is required.")
    return value.astimezone(SHANGHAI)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def strategy_config(strategy_id: str) -> dict:
    strategy_id = _STRATEGY_ALIASES.get(strategy_id, strategy_id)
    if strategy_id not in STRATEGY_IDS:
        raise ValueError("Only frozen standard_ma, right_ma and buy_hold strategies are supported.")
    research_id = RESEARCH_STRATEGIES[strategy_id]
    spec = next((item for item in SPECS if item.name == research_id), None)
    return {
        "strategy_id": strategy_id, "research_strategy_id": research_id, "signal_asset": SIGNAL_ASSET,
        "execution_asset": EXECUTION_ASSET, "history_start": HISTORY_START.isoformat(),
        "initial_signal_state": 0, "maximum_target_weight": 0.95,
        "calendar": "Tushare trade_cal SSE", "close_ready_local_time": "15:30",
        "n": spec.n if spec else None, "m": spec.m if spec else None,
        "score": spec.score if spec else "buy_hold", "upper": spec.upper if spec else None,
        "lower": spec.lower if spec else None, "trend_filter": spec.trend_filter if spec else False,
        "trend_lag": spec.trend_lag if spec else None,
        "mapping_notice": MAPPING_NOTICE,
    }


def normalize_calendar(calendar: pd.DataFrame) -> pd.DataFrame:
    """Require every calendar date, including closed dates, from an official feed.

    An open-days-only list cannot prove that today is a holiday or that the next
    known open day is really the next session, so it is deliberately rejected.
    """
    if not isinstance(calendar, pd.DataFrame) or not {"cal_date", "is_open"} <= set(calendar):
        raise ValueError("Official trade_cal requires cal_date and is_open, including closed days.")
    dates = pd.to_datetime(calendar.cal_date.astype(str), errors="raise")
    if (dates.isna().any() or dates.duplicated().any() or dates.empty
            or dates.dt.tz is not None or not dates.equals(dates.dt.normalize())):
        raise ValueError("Calendar dates must be unique timezone-naive daily dates.")
    states = pd.to_numeric(calendar.is_open, errors="raise")
    if not states.isin([0, 1]).all():
        raise ValueError("Calendar is_open must be zero or one.")
    result = pd.DataFrame({"cal_date": dates.to_numpy(), "is_open": states.to_numpy(dtype=int)})
    result = result.sort_values("cal_date").reset_index(drop=True)
    expected = pd.date_range(result.cal_date.iloc[0], result.cal_date.iloc[-1], freq="D")
    if not pd.DatetimeIndex(result.cal_date).equals(expected):
        raise ValueError("Official calendar has missing dates; weekday guesses are forbidden.")
    return result


def session_dates(calendar: pd.DataFrame, *, now: datetime | None = None) -> tuple[date, date]:
    """Last completed session and its strictly next official session."""
    local = _now(now)
    cal = normalize_calendar(calendar)
    today = pd.Timestamp(local.date())
    if cal.cal_date.iloc[0] > today or cal.cal_date.iloc[-1] < today:
        raise ValueError("Official calendar does not cover today.")
    days = pd.DatetimeIndex(cal.loc[cal.is_open.eq(1), "cal_date"])
    ready_today = local.time().replace(tzinfo=None) >= time(15, 30)
    completed = days[days <= today] if ready_today else days[days < today]
    if completed.empty:
        raise ValueError("Calendar lacks the previous completed trading session.")
    signal_day = completed[-1]
    following = days[days > signal_day]
    if following.empty:
        raise ValueError("Calendar lacks the next official trading session.")
    return signal_day.date(), following[0].date()


def _prices(frame: pd.DataFrame, *, asset: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"Missing prices for {asset}.")
    result = frame.copy()
    if "ts_code" in result and not result.ts_code.astype(str).eq(asset).all():
        raise ValueError(f"Unexpected asset in {asset} prices.")
    if "date" in result:
        result.index = pd.to_datetime(result.pop("date").astype(str), errors="raise")
    elif "trade_date" in result:
        result.index = pd.to_datetime(result.pop("trade_date").astype(str), errors="raise")
    if (not isinstance(result.index, pd.DatetimeIndex) or result.index.tz is not None
            or result.index.hasnans or not result.index.is_unique
            or not result.index.equals(result.index.normalize())):
        raise ValueError("Prices need unique timezone-naive daily dates.")
    required = ["open", "high", "low", "close"]
    if not set(required) <= set(result):
        raise ValueError("Unadjusted OHLC fields are required.")
    result = result.loc[:, required].apply(pd.to_numeric, errors="raise").sort_index()
    if (not np.isfinite(result.to_numpy()).all() or (result <= 0).any().any()
            or (result.high < result.low).any()
            or (result.high < result[["open", "close"]].max(axis=1)).any()
            or (result.low > result[["open", "close"]].min(axis=1)).any()):
        raise ValueError("Invalid OHLC; missing prices cannot be filled.")
    result.index.name = "date"
    return result


def _price_records(prices: pd.DataFrame) -> list[dict]:
    rows = prices.reset_index()
    rows["date"] = rows.date.dt.strftime("%Y-%m-%d")
    return rows.to_dict("records")


@dataclass(frozen=True)
class SignalBundle:
    strategy_id: str
    signal_asof: date
    execution_date: date
    target_weight: float
    created_at: datetime
    score: float | None
    config_sha256: str
    data_sha256: str
    calendar_sha256: str
    history_rows: int
    signal_asset: str = SIGNAL_ASSET
    execution_asset: str = EXECUTION_ASSET
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        config = strategy_config(self.strategy_id)
        if self.strategy_id not in STRATEGY_IDS or self.config_sha256 != _hash(config):
            raise ValueError("Signal configuration identity does not match the frozen strategy.")
        if (self.signal_asset != SIGNAL_ASSET or self.execution_asset != EXECUTION_ASSET
                or self.schema_version != SCHEMA_VERSION):
            raise ValueError("Unsupported index-to-ETF mapping or schema.")
        if (type(self.signal_asof) is not date or type(self.execution_date) is not date
                or self.signal_asof >= self.execution_date or not _aware(self.created_at)
                or self.created_at.astimezone(SHANGHAI).date() < self.signal_asof):
            raise ValueError("Invalid or future signal dates.")
        if not _finite(self.target_weight) or self.target_weight not in (0.0, 0.95):
            raise ValueError("Frozen strategies use only zero or 95% ETF target weights.")
        if any(not isinstance(value, str) or not _HASH.fullmatch(value)
               for value in (self.config_sha256, self.data_sha256, self.calendar_sha256)):
            raise ValueError("Missing input identity.")
        if type(self.history_rows) is not int or self.history_rows < 620:
            raise ValueError("A complete anchored history of at least 620 bars is required.")
        if self.strategy_id != "etf_buy_hold" and not _finite(self.score):
            raise ValueError("RSRS score must be finite after strict warmup.")
        if self.strategy_id == "etf_buy_hold" and (self.score is not None or self.target_weight != 0.95):
            raise ValueError("Buy-and-hold is the same 95% ETF exposure benchmark.")

    @property
    def target_weights(self) -> dict[str, float]:
        return {self.execution_asset: self.target_weight}

    def _payload(self) -> dict:
        return {
            "schema_version": self.schema_version, "strategy_id": self.strategy_id,
            "signal_asof": self.signal_asof.isoformat(), "execution_date": self.execution_date.isoformat(),
            "created_at": self.created_at.astimezone(timezone.utc).isoformat(),
            "target_weights": self.target_weights, "score": self.score,
            "signal_asset": self.signal_asset, "execution_asset": self.execution_asset,
            "config": strategy_config(self.strategy_id), "config_sha256": self.config_sha256,
            "data_sha256": self.data_sha256, "calendar_sha256": self.calendar_sha256,
            "history_rows": self.history_rows, "mapping_notice": MAPPING_NOTICE,
        }

    @property
    def bundle_id(self) -> str:
        return _hash(self._payload())

    def to_dict(self) -> dict:
        return {**self._payload(), "bundle_id": self.bundle_id}


def prepare_signal_bundle(strategy_id: str, index_prices: pd.DataFrame,
                          calendar: pd.DataFrame, *, now: datetime | None = None,
                          etf_daily: pd.DataFrame | None = None) -> SignalBundle:
    """Freeze a latest-completed-close target; incomplete/future data block it."""
    local = _now(now)
    config = strategy_config(strategy_id)
    strategy_id = config["strategy_id"]
    cal = normalize_calendar(calendar)
    signal_day, execution_day = session_dates(cal, now=local)
    history = _prices(index_prices, asset=SIGNAL_ASSET)
    if history.index[-1].date() != signal_day:
        raise ValueError("Index prices are stale or contain future/incomplete session data.")
    history = history.loc[pd.Timestamp(HISTORY_START):]
    if cal.cal_date.iloc[0].date() > HISTORY_START:
        raise ValueError("Calendar must cover the fixed 2019 history anchor.")
    expected = pd.DatetimeIndex(cal.loc[
        cal.is_open.eq(1) & cal.cal_date.ge(pd.Timestamp(HISTORY_START))
        & cal.cal_date.le(pd.Timestamp(signal_day)), "cal_date"
    ])
    if not history.index.equals(expected):
        raise ValueError("Index history has missing/extra trading sessions; no automatic filling.")
    if len(history) < 620:
        raise ValueError("At least 620 complete anchored bars are required for strict warmup.")
    # A daily ETF close proves recent data access only; it is never an order price.
    etf = _prices(etf_daily, asset=EXECUTION_ASSET)
    if etf.index[-1].date() != signal_day:
        raise ValueError("Raw ETF daily data are stale or contain future/incomplete data.")
    if not etf.index.isin(expected).all():
        raise ValueError("ETF daily dates must be official completed trading sessions.")
    values, decision = strategy(history, config["research_strategy_id"])
    score = None if strategy_id == "etf_buy_hold" else float(values[config["score"]].iloc[-1])
    if strategy_id != "etf_buy_hold" and not math.isfinite(score):
        raise ValueError("RSRS score is unavailable; strict warmup or degenerate data.")
    calendar_records = cal.copy()
    calendar_records["cal_date"] = calendar_records.cal_date.dt.strftime("%Y-%m-%d")
    return SignalBundle(
        strategy_id=strategy_id, signal_asof=signal_day, execution_date=execution_day,
        target_weight=0.95 if int(decision.iloc[-1]) else 0.0, created_at=local, score=score,
        config_sha256=_hash(config),
        data_sha256=_hash({"index": _price_records(history), "raw_etf": _price_records(etf)}),
        calendar_sha256=_hash(calendar_records.to_dict("records")), history_rows=len(history),
    )


def save_signal_bundle(bundle: SignalBundle, directory: str | Path) -> Path:
    """Write once, content-addressed. An existing different file is never replaced."""
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{bundle.execution_date}_{bundle.strategy_id}_{bundle.bundle_id}.json"
    contents = _json_bytes(bundle.to_dict()) + b"\n"
    try:
        with path.open("xb") as stream:
            stream.write(contents)
    except FileExistsError:
        if path.read_bytes() != contents:
            raise ValueError("Existing immutable signal bundle differs; refusing overwrite.") from None
    return path


def load_signal_bundle(path: str | Path) -> SignalBundle:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Signal bundle must be a JSON object.")
    try:
        bundle = SignalBundle(
            strategy_id=payload["strategy_id"], signal_asof=date.fromisoformat(payload["signal_asof"]),
            execution_date=date.fromisoformat(payload["execution_date"]),
            target_weight=payload["target_weights"][EXECUTION_ASSET],
            created_at=datetime.fromisoformat(payload["created_at"]), score=payload["score"],
            config_sha256=payload["config_sha256"], data_sha256=payload["data_sha256"],
            calendar_sha256=payload["calendar_sha256"], history_rows=payload["history_rows"],
            signal_asset=payload["signal_asset"], execution_asset=payload["execution_asset"],
            schema_version=payload["schema_version"],
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("Incomplete or malformed signal bundle.") from exc
    if payload != bundle.to_dict():
        raise ValueError("Signal bundle integrity check failed.")
    return bundle


@dataclass(frozen=True)
class RawQuote:
    asset: str
    price: float
    asof: datetime
    price_basis: str  # Must explicitly be "raw"; adjusted research prices are not executable.
    source: str = "provided"


def fetch_raw_quote(*, token_file: str | None = None, now: datetime | None = None,
                    max_age_seconds: float = 30.0, quote_source: str = "rt_etf_k",
                    opener=None) -> RawQuote:
    """One uncached, bounded request to the official ETF realtime endpoint.

    Source: https://tushare.pro/document/2?doc_id=400.  It requires separately
    enabled ``rt_etf_k`` access; this function never purchases access, retries,
    or relabels a daily close as a live quote.  An unavailable or undated
    ``trade_time`` fails closed, even if a numeric latest price is returned.
    """
    if not _finite(max_age_seconds) or max_age_seconds <= 0:
        raise ValueError("Quote freshness limit must be positive.")
    if quote_source == "tencent_public":
        return _tencent_public_quote(now=now, max_age_seconds=max_age_seconds, opener=opener)
    if quote_source != "rt_etf_k":
        raise ValueError("Select rt_etf_k or tencent_public explicitly; there is no hidden fallback.")
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token and token_file:
        token = Path(token_file).read_text(encoding="utf-8-sig").strip()
    if not token:
        raise ValueError("Set TUSHARE_TOKEN or provide a local token file.")
    request = urllib.request.Request(
        "https://api.tushare.pro",
        data=_json_bytes({"api_name": "rt_etf_k", "token": token,
                          "params": {"ts_code": EXECUTION_ASSET, "topic": "HQ_FND_TICK"},
                          "fields": "ts_code,close,trade_time"}),
        headers={"Content-Type": "application/json"},
    )
    try:
        with (opener or urllib.request.urlopen)(request, timeout=12) as response:
            contents = response.read(2 * 1024 * 1024 + 1)
        if len(contents) > 2 * 1024 * 1024:
            raise ValueError("oversize")
        payload = json.loads(contents)
    except Exception:
        raise RuntimeError("Realtime ETF quote request failed; no retry or stale-price fallback.") from None
    if not isinstance(payload, dict) or payload.get("code") != 0:
        raise RuntimeError("Realtime ETF quote unavailable; check existing rt_etf_k access. No retry.")
    try:
        data = payload["data"]
        frame = pd.DataFrame(data["items"], columns=data["fields"])
        if len(frame) != 1 or not {"ts_code", "close", "trade_time"} <= set(frame):
            raise ValueError("shape")
        row = frame.iloc[0]
        if row.ts_code != EXECUTION_ASSET or not _finite(float(row.close)) or float(row.close) <= 0:
            raise ValueError("price")
        # A time-only response cannot establish that the quote is from today.
        stamp = str(row.trade_time).strip()
        if re.fullmatch(r"\d{14}", stamp):
            stamp = datetime.strptime(stamp, "%Y%m%d%H%M%S").isoformat()
        elif not re.match(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}", stamp):
            raise ValueError("undated")
        asof = datetime.fromisoformat(stamp)
        asof = asof.replace(tzinfo=SHANGHAI) if not _aware(asof) else asof
        quote = RawQuote(EXECUTION_ASSET, float(row.close), asof, "raw", "rt_etf_k")
        local = _now(now)  # Evaluate at receipt time, not at the start of a slow request.
        age = (local - asof).total_seconds()
        if not 0 <= age <= max_age_seconds or asof.astimezone(SHANGHAI).date() != local.date():
            raise ValueError("stale")
        return quote
    except (KeyError, ValueError, TypeError, OverflowError):
        raise ValueError("Realtime ETF quote lacks a valid fresh provider timestamp or price.") from None


def _tencent_public_quote(*, now: datetime | None, max_age_seconds: float, opener) -> RawQuote:
    """Narrow raw quote adapter following easyquotation's Tencent field contract.

    Primary implementation: https://github.com/shidenggui/easyquotation/blob/master/easyquotation/tencent.py
    The source maps fields 2/3/30 to code/latest price/full quote timestamp.
    This opt-in public research feed is not the THS broker's own feed and has no
    latency guarantee.  Its provider timestamp, never the fetch time, gates use.
    """
    request = urllib.request.Request("https://qt.gtimg.cn/q=sh510300",
                                     headers={"User-Agent": "Qingxu-simulation/2.5"})
    try:
        with (opener or urllib.request.urlopen)(request, timeout=12) as response:
            contents = response.read(64 * 1024 + 1)
        if len(contents) > 64 * 1024:
            raise ValueError("oversize")
        match = re.fullmatch(r'\s*v_sh510300="([^"]+)";\s*', contents.decode("gb18030"))
        if not match:
            raise ValueError("format")
        fields = match.group(1).split("~")
        if len(fields) <= 49 or fields[2] != "510300":
            raise ValueError("code")
        price = float(fields[3])
        if not math.isfinite(price) or price <= 0:
            raise ValueError("price")
        if not re.fullmatch(r"\d{14}", fields[30]):
            raise ValueError("timestamp")
        stamp = datetime.strptime(fields[30], "%Y%m%d%H%M%S").replace(tzinfo=SHANGHAI)
    except Exception:
        raise RuntimeError("Public ETF quote request or format failed; no retries or price fallback.") from None
    local = _now(now)
    if (stamp.date() != local.date()
            or not 0 <= (local - stamp).total_seconds() <= max_age_seconds):
        raise ValueError("Public ETF quote is stale or future-dated; provider timestamp retained.")
    return RawQuote(EXECUTION_ASSET, price, stamp, "raw", "tencent_public")


def plan_order(bundle: SignalBundle, *, account_binding: str, total_shares: int,
               available_shares: int, available_cash: float, quote: RawQuote,
               now: datetime | None = None, fee_buffer_bps: float = 25.0,
               quote_max_age_seconds: float = 30.0,
               max_order_notional: float | None = None) -> OrderIntent | None:
    """Plan one 100-share-lot delta for an otherwise ETF-and-cash-only account.

    The caller verifies that the account has no other assets/pending orders.
    This pure planner cannot submit and cannot override the executor's calendar,
    simulation-account, session, risk-limit or duplicate-order checks.
    """
    local = _now(now)
    if not isinstance(bundle, SignalBundle):
        raise ValueError("A verified SignalBundle is required.")
    if local.date() != bundle.execution_date:
        raise ValueError("Signal is not for today's execution session.")
    if bundle.created_at > local:
        raise ValueError("Signal creation timestamp is in the future.")
    if not isinstance(account_binding, str) or not _IDENTIFIER.fullmatch(account_binding):
        raise ValueError("Invalid simulation account binding.")
    if (not isinstance(quote, RawQuote) or quote.asset != bundle.execution_asset
            or quote.price_basis != "raw" or not _finite(quote.price) or quote.price <= 0
            or not _aware(quote.asof)):
        raise ValueError("A current unadjusted quote for the execution ETF is required.")
    if not _finite(quote_max_age_seconds) or quote_max_age_seconds <= 0:
        raise ValueError("Quote freshness limit must be positive.")
    age = (local - quote.asof).total_seconds()
    if not 0 <= age <= quote_max_age_seconds or quote.asof.astimezone(SHANGHAI).date() != local.date():
        raise ValueError("ETF quote is stale or future-dated.")
    if abs(quote.price * 1000 - round(quote.price * 1000)) > 1e-7:
        raise ValueError("ETF limit price must use the 0.001 yuan tick.")
    if (type(total_shares) is not int or type(available_shares) is not int
            or min(total_shares, available_shares) < 0 or available_shares > total_shares
            or total_shares % 100 or available_shares % 100):
        raise ValueError("Holdings and sellable shares must be nonnegative whole 100-share lots.")
    if not _finite(available_cash) or available_cash < 0:
        raise ValueError("Available cash must be finite and nonnegative.")
    if not _finite(fee_buffer_bps) or not 0 <= fee_buffer_bps <= 1000:
        raise ValueError("Invalid fee buffer.")
    if max_order_notional is not None and (not _finite(max_order_notional) or max_order_notional <= 0):
        raise ValueError("Order notional cap must be finite and positive.")
    price = float(quote.price)
    nav = available_cash + total_shares * price
    if not math.isfinite(nav):
        raise ValueError("Account valuation overflow.")
    desired = math.floor(nav * bundle.target_weight / price / 100) * 100
    delta = desired - total_shares
    if delta > 0:
        affordable = math.floor(available_cash / (price * (1 + fee_buffer_bps / 10000)) / 100) * 100
        quantity, side = min(delta, affordable), "buy"
    else:
        quantity, side = min(-delta, available_shares), "sell"
    if max_order_notional is not None:
        quantity = min(quantity, math.floor(max_order_notional / price / 100) * 100)
    if quantity == 0:
        return None
    # Price, quantity, refreshed bundle hashes and side are deliberately absent.
    # A rerun can never invent a new intent after a timeout or partial fill.
    identity = _hash({
        "strategy_id": bundle.strategy_id, "execution_date": bundle.execution_date.isoformat(),
        "asset": bundle.execution_asset, "account_binding": account_binding,
    })
    return OrderIntent(
        intent_id=f"rsrs:{identity}", strategy_id=bundle.strategy_id,
        account_binding=account_binding, asset=bundle.execution_asset, side=side,
        quantity=int(quantity), limit_price=price,
        signal_asof=bundle.signal_asof, execution_date=bundle.execution_date,
    )


@dataclass(frozen=True)
class MarketSnapshot:
    directory: Path
    index_prices: pd.DataFrame
    etf_daily: pd.DataFrame
    calendar: pd.DataFrame
    manifest: dict


def refresh_market_data(output_root: str | Path, *, token_file: str | None = None,
                        now: datetime | None = None,
                        preload_path: str | Path | None =
                        "D:/Desktop/cf2026-quant-platform/data/private/project2_rsrs/sh000300.csv",
                        provider_factory=TushareProvider) -> MarketSnapshot:
    """Three bounded read-only Tushare queries, with retained acquisition evidence.

    Existing Tencent index history is only used after an overlapping Tushare bar
    agrees within 0.02 index points for every OHLC field.  It is never rewritten.
    With no preload, request a homogeneous Tushare series from the 2019 anchor.
    Each invocation has its own cache so an earlier empty successful response
    cannot mask a later daily publication.
    """
    local = _now(now)
    folder = Path(output_root) / local.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True, exist_ok=False)
    provider = provider_factory(folder / "raw", token_file, delay=0.4)
    end_calendar = local.date() + timedelta(days=21)
    if (end_calendar - HISTORY_START).days >= 8000:
        raise ValueError("Calendar range exceeds the bounded acquisition budget.")
    calendar = normalize_calendar(provider.query(
        "trade_cal", {"exchange": "SSE", "start_date": HISTORY_START.strftime("%Y%m%d"),
                      "end_date": end_calendar.strftime("%Y%m%d")}, "cal_date,is_open"))
    signal_day, execution_day = session_dates(calendar, now=local)
    preload = None
    source = "Tushare Pro index_daily, homogeneous fixed 2019 anchor"
    preload_sha256 = None
    if preload_path is not None and Path(preload_path).is_file():
        preload = _prices(pd.read_csv(preload_path), asset=SIGNAL_ASSET)
        preload = preload.loc[pd.Timestamp(HISTORY_START):pd.Timestamp(signal_day)]
        if preload.empty:
            raise ValueError("Preload has no usable anchored history.")
        preload_sha256 = digest(Path(preload_path))
        start_index = preload.index[-1].date()
    else:
        start_index = HISTORY_START
    params = {"ts_code": SIGNAL_ASSET, "start_date": start_index.strftime("%Y%m%d"),
              "end_date": signal_day.strftime("%Y%m%d")}
    acquired = _prices(provider.query("index_daily", params, "ts_code,trade_date,open,high,low,close"),
                       asset=SIGNAL_ASSET)
    if len(acquired) >= 8000:
        raise ValueError("Index response may be truncated at the row limit.")
    if preload is not None:
        overlap = acquired.index.intersection(preload.index)
        if (overlap.empty or np.abs(acquired.loc[overlap].to_numpy()
                                   - preload.loc[overlap].to_numpy()).max() > 0.0200001):
            raise ValueError("Vendor overlap differs; refusing to silently splice index histories.")
        history = pd.concat([preload, acquired.loc[acquired.index > preload.index[-1]]])
        source = "Historical preload plus Tushare index_daily; overlapping OHLC agree within 0.02 points"
    else:
        history = acquired
    etf = _prices(provider.query(
        "fund_daily", {"ts_code": EXECUTION_ASSET, "start_date": signal_day.strftime("%Y%m%d"),
                       "end_date": signal_day.strftime("%Y%m%d")},
        "ts_code,trade_date,open,high,low,close"), asset=EXECUTION_ASSET)
    # Run exactly the same readiness gate as consumers before publishing a ready manifest.
    for strategy_id in STRATEGY_IDS:
        prepare_signal_bundle(strategy_id, history, calendar, now=local, etf_daily=etf)
    history.to_csv(folder / "index_daily.csv", index=True)
    etf.to_csv(folder / "etf_daily_raw.csv", index=True)
    calendar.to_csv(folder / "trade_cal.csv", index=False, date_format="%Y-%m-%d")
    manifest = {
        "schema_version": 1, "ready": True, "created_at": local.astimezone(timezone.utc).isoformat(),
        "signal_asof": signal_day.isoformat(), "execution_date": execution_day.isoformat(),
        "signal_asset": SIGNAL_ASSET, "execution_asset": EXECUTION_ASSET,
        "index_source": source, "preload_sha256": preload_sha256,
        "index_rows": len(history), "etf_rows": len(etf), "mapping_notice": MAPPING_NOTICE,
        "files": {name: digest(folder / name)
                  for name in ("index_daily.csv", "etf_daily_raw.csv", "trade_cal.csv")},
    }
    (folder / "manifest.json").write_bytes(_json_bytes(manifest) + b"\n")
    return MarketSnapshot(folder, history, etf, calendar, manifest)
