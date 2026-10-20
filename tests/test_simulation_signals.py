"""Meaningful execution-readiness tests, with no network or actual orders."""
from dataclasses import replace
from datetime import date, datetime, timedelta
import io
import json

import numpy as np
import pandas as pd
import pytest

from cfquant.simulation_signals import (
    EXECUTION_ASSET, SHANGHAI, STRATEGY_IDS, RawQuote, fetch_raw_quote,
    load_signal_bundle, normalize_calendar, plan_order, prepare_signal_bundle,
    refresh_market_data, save_signal_bundle, session_dates,
)


@pytest.fixture
def inputs():
    # Explicit *synthetic* official-feed shape; holiday Monday is closed even
    # though it is a weekday. Production only consumes Tushare trade_cal.
    calendar = pd.DataFrame({"cal_date": pd.date_range("2019-01-01", "2022-01-21")})
    calendar["is_open"] = (calendar.cal_date.dt.dayofweek < 5).astype(int)
    calendar.loc[calendar.cal_date.eq("2022-01-17"), "is_open"] = 0
    dates = pd.DatetimeIndex(calendar.loc[
        calendar.is_open.eq(1) & calendar.cal_date.le("2022-01-14"), "cal_date"])
    t = np.arange(len(dates), dtype=float)
    low = 100 + 0.04 * t + 4 * np.sin(t / 17)
    high = low + 1.2 + 0.4 * np.cos(t / 13)
    frame = pd.DataFrame({"low": low, "high": high, "open": (high + low) / 2,
                          "close": (high + low) / 2 + 0.1}, index=dates)
    etf = pd.DataFrame({"open": [4.0], "high": [4.2], "low": [3.9], "close": [4.1]},
                       index=pd.DatetimeIndex(["2022-01-14"]))
    now = datetime(2022, 1, 14, 16, tzinfo=SHANGHAI)
    return frame, calendar, etf, now


def bundle(inputs, strategy="buy_hold"):
    frame, calendar, etf, now = inputs
    return prepare_signal_bundle(strategy, frame, calendar, etf_daily=etf, now=now)


def planning(inputs, strategy="buy_hold", **kwargs):
    signal = bundle(inputs, strategy)
    now = datetime(2022, 1, 18, 10, tzinfo=SHANGHAI)
    args = dict(account_binding="sim-account", total_shares=0, available_shares=0,
                available_cash=200_000.0, quote=RawQuote(EXECUTION_ASSET, 4.0, now, "raw"), now=now)
    args.update(kwargs)
    return plan_order(signal, **args)


@pytest.mark.parametrize("strategy", STRATEGY_IDS)
def test_causal_bundle_and_explicit_etf_mapping(inputs, strategy):
    signal = bundle(inputs, strategy)
    assert signal.signal_asof == date(2022, 1, 14)
    assert signal.execution_date == date(2022, 1, 18)  # Explicit Monday holiday.
    assert signal.target_weights.keys() == {EXECUTION_ASSET}
    assert "000300.SH" not in signal.target_weights
    assert signal.history_rows > 620
    assert len(signal.data_sha256) == 64
    assert "tracking error" in signal.to_dict()["mapping_notice"]


def test_close_cutoff_and_official_calendar(inputs):
    _, calendar, _, _ = inputs
    before = datetime(2022, 1, 18, 15, 29, tzinfo=SHANGHAI)
    after = before + timedelta(minutes=1)
    assert session_dates(calendar, now=before) == (date(2022, 1, 14), date(2022, 1, 18))
    assert session_dates(calendar, now=after) == (date(2022, 1, 18), date(2022, 1, 19))
    with pytest.raises(ValueError, match="missing dates"):
        normalize_calendar(calendar[calendar.is_open.eq(1)])
    with pytest.raises(ValueError, match="next official"):
        session_dates(calendar[calendar.cal_date.le("2022-01-14")],
                      now=datetime(2022, 1, 14, 16, tzinfo=SHANGHAI))
    with pytest.raises(ValueError, match="cover today"):
        session_dates(calendar, now=datetime(2022, 2, 1, 10, tzinfo=SHANGHAI))


@pytest.mark.parametrize("change", ["stale", "future", "missing", "warmup", "degenerate"])
def test_no_stale_future_missing_or_unwarmed_signal(inputs, change):
    prices, calendar, etf, now = inputs
    if change == "stale":
        prices = prices.iloc[:-1]
    elif change == "future":
        prices = pd.concat([prices, prices.iloc[[-1]].set_axis(pd.DatetimeIndex(["2022-01-18"]))])
    elif change == "missing":
        prices = prices.drop(prices.index[100])
    elif change == "warmup":
        # Explicit complete history shorter than required, not a missing-date case.
        end = prices.index[500]
        prices = prices.loc[:end]
        etf = etf.set_axis(pd.DatetimeIndex([end]))
        now = end.to_pydatetime().replace(hour=16, tzinfo=SHANGHAI)
    else:
        prices = prices.assign(open=100, close=100, low=99, high=101)
    with pytest.raises(ValueError):
        prepare_signal_bundle("standard_ma", prices, calendar, etf_daily=etf, now=now)


def test_missing_or_wrong_etf_data_blocks(inputs):
    prices, calendar, etf, now = inputs
    with pytest.raises(ValueError, match="Missing prices"):
        prepare_signal_bundle("buy_hold", prices, calendar, now=now)
    with pytest.raises(ValueError, match="stale"):
        prepare_signal_bundle("buy_hold", prices, calendar, now=now,
                              etf_daily=etf.set_axis(pd.DatetimeIndex(["2022-01-13"])))
    with pytest.raises(ValueError, match="Unexpected asset"):
        prepare_signal_bundle("buy_hold", prices, calendar, now=now,
                              etf_daily=etf.assign(ts_code="000300.SH"))


def test_bundle_write_once_roundtrip_and_tamper_detection(inputs, tmp_path):
    signal = bundle(inputs)
    path = save_signal_bundle(signal, tmp_path)
    before = path.read_bytes()
    assert save_signal_bundle(signal, tmp_path) == path
    assert load_signal_bundle(path) == signal
    assert path.read_bytes() == before
    payload = json.loads(before)
    payload["execution_asset"] = "000300.SH"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_signal_bundle(path)
    with pytest.raises(ValueError, match="refusing overwrite"):
        save_signal_bundle(signal, tmp_path)


def test_only_raw_lot_rounded_affordable_etf_order(inputs):
    order = planning(inputs)
    assert order.asset == "510300.SH"
    assert order.quantity == 47_500
    assert order.quantity % 100 == 0
    assert order.quantity * order.limit_price * 1.0025 <= 200_000
    assert order.signal_asof == date(2022, 1, 14)
    assert order.execution_date == date(2022, 1, 18)
    capped = planning(inputs, max_order_notional=20_000.0)
    assert capped.quantity == 5000
    assert planning(inputs, available_cash=100.0) is None
    assert planning(inputs, max_order_notional=200.0) is None


def test_intent_identity_survives_quote_and_quantity_changes(inputs):
    first = planning(inputs)
    now = datetime(2022, 1, 18, 10, tzinfo=SHANGHAI)
    second = planning(inputs, quote=RawQuote(EXECUTION_ASSET, 4.2, now, "raw"))
    assert first.quantity != second.quantity
    assert first.intent_id == second.intent_id
    assert first.intent_id != planning(inputs, account_binding="another-sim-account").intent_id


def test_exit_respects_t_plus_one_available_shares(inputs):
    signal = replace(bundle(inputs, "standard_ma"), target_weight=0.0)
    now = datetime(2022, 1, 18, 10, tzinfo=SHANGHAI)
    order = plan_order(signal, account_binding="sim-account", total_shares=1000,
                       available_shares=300, available_cash=100.0,
                       quote=RawQuote(EXECUTION_ASSET, 4.0, now, "raw"), now=now)
    assert (order.side, order.quantity) == ("sell", 300)


@pytest.mark.parametrize("change", ["adjusted", "index", "stale", "future", "old_signal", "fractional"])
def test_order_rejects_nonexecutable_inputs(inputs, change):
    now = datetime(2022, 1, 18, 10, tzinfo=SHANGHAI)
    args = {}
    if change == "adjusted":
        args["quote"] = RawQuote(EXECUTION_ASSET, 4.0, now, "adjusted")
    elif change == "index":
        args["quote"] = RawQuote("000300.SH", 4000.0, now, "raw")
    elif change == "stale":
        args["quote"] = RawQuote(EXECUTION_ASSET, 4.0, now - timedelta(seconds=31), "raw")
    elif change == "future":
        args["quote"] = RawQuote(EXECUTION_ASSET, 4.0, now + timedelta(seconds=1), "raw")
    elif change == "old_signal":
        args["now"] = now + timedelta(days=1)
    else:
        args["total_shares"] = 100.5
    with pytest.raises(ValueError):
        planning(inputs, **args)


def test_refresh_is_bounded_preserves_preload_and_checks_splice(inputs, tmp_path):
    prices, calendar, etf, now = inputs
    preload = tmp_path / "historical.csv"
    prices.iloc[:-2].rename_axis("date").to_csv(preload)
    original = preload.read_bytes()
    calls = []

    class Provider:
        def __init__(self, *args, **kwargs):
            pass

        def query(self, api, params, fields):
            calls.append((api, params, fields))
            if api == "trade_cal":
                return calendar
            frame = prices.iloc[-3:] if api == "index_daily" else etf
            asset = "000300.SH" if api == "index_daily" else "510300.SH"
            return frame.rename_axis("trade_date").reset_index().assign(ts_code=asset)

    market = refresh_market_data(tmp_path / "market", now=now, preload_path=preload,
                                 provider_factory=Provider)
    assert len(calls) == 3
    assert [item[0] for item in calls] == ["trade_cal", "index_daily", "fund_daily"]
    assert preload.read_bytes() == original
    assert market.manifest["ready"] is True
    assert market.index_prices.index.equals(prices.index)
    assert (market.directory / "manifest.json").is_file()


def test_refresh_refuses_vendor_discontinuity(inputs, tmp_path):
    prices, calendar, etf, now = inputs
    preload = tmp_path / "historical.csv"
    prices.iloc[:-2].rename_axis("date").to_csv(preload)

    class Provider:
        def __init__(self, *args, **kwargs):
            pass

        def query(self, api, params, fields):
            if api == "trade_cal":
                return calendar
            return (prices.iloc[-3:] + 2).rename_axis("trade_date").reset_index().assign(ts_code="000300.SH")

    with pytest.raises(ValueError, match="Vendor overlap differs"):
        refresh_market_data(tmp_path / "market", now=now, preload_path=preload,
                            provider_factory=Provider)
    assert not list((tmp_path / "market").rglob("manifest.json"))


@pytest.mark.parametrize("stamp", ["2022-01-18 10:00:00", "20220118100000"])
def test_realtime_quote_uses_provider_time_and_single_request(monkeypatch, stamp):
    monkeypatch.setenv("TUSHARE_TOKEN", "synthetic-test-value")
    calls = []

    def opener(request, timeout):
        calls.append((json.loads(request.data), timeout))
        return io.BytesIO(json.dumps({"code": 0, "data": {
            "fields": ["ts_code", "close", "trade_time"], "items": [["510300.SH", 4.0, stamp]]
        }}).encode())

    now = datetime(2022, 1, 18, 10, tzinfo=SHANGHAI)
    quote = fetch_raw_quote(now=now, opener=opener)
    assert quote.asof == now
    assert quote.price_basis == "raw"
    assert len(calls) == 1
    assert calls[0][0]["api_name"] == "rt_etf_k"
    assert calls[0][0]["params"] == {"ts_code": "510300.SH", "topic": "HQ_FND_TICK"}
    assert calls[0][1] == 12


@pytest.mark.parametrize("stamp", ["10:00:00", "2022-01-14 15:00:00", "2022-01-18 10:00:01", None])
def test_realtime_quote_never_fakes_date_or_freshness(monkeypatch, stamp):
    monkeypatch.setenv("TUSHARE_TOKEN", "synthetic-test-value")

    def opener(request, timeout):
        return io.BytesIO(json.dumps({"code": 0, "data": {
            "fields": ["ts_code", "close", "trade_time"], "items": [["510300.SH", 4.0, stamp]]
        }}).encode())

    with pytest.raises(ValueError, match="fresh provider timestamp"):
        fetch_raw_quote(now=datetime(2022, 1, 18, 10, tzinfo=SHANGHAI), opener=opener)


def test_quote_permission_error_not_retried_or_leaked(monkeypatch):
    monkeypatch.setenv("TUSHARE_TOKEN", "synthetic-test-value")
    calls = []

    def opener(request, timeout):
        calls.append(1)
        return io.BytesIO(b'{"code":-2002,"msg":"PRIVATE_SERVER_DETAILS"}')

    with pytest.raises(RuntimeError, match="existing rt_etf_k access") as error:
        fetch_raw_quote(opener=opener)
    assert len(calls) == 1
    assert "PRIVATE_SERVER_DETAILS" not in str(error.value)


def test_explicit_public_quote_source_needs_no_token(monkeypatch):
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    calls = []
    fields = ["0"] * 52
    fields[2], fields[3], fields[30] = "510300", "4.123", "20220118100000"

    def opener(request, timeout):
        calls.append(request.full_url)
        return io.BytesIO(('v_sh510300="' + "~".join(fields) + '";').encode("gb18030"))

    quote = fetch_raw_quote(quote_source="tencent_public", opener=opener,
                            now=datetime(2022, 1, 18, 10, tzinfo=SHANGHAI))
    assert quote.price == 4.123
    assert quote.source == "tencent_public"
    assert calls == ["https://qt.gtimg.cn/q=sh510300"]
    with pytest.raises(ValueError, match="stale"):
        fetch_raw_quote(quote_source="tencent_public", opener=opener,
                        now=datetime(2022, 1, 18, 10, 1, tzinfo=SHANGHAI))


def test_public_quote_rejects_asset_mismatch():
    fields = ["0"] * 52
    fields[2], fields[3], fields[30] = "000300", "4000", "20220118100000"

    def opener(request, timeout):
        return io.BytesIO(('v_sh510300="' + "~".join(fields) + '";').encode())

    with pytest.raises(RuntimeError, match="format failed"):
        fetch_raw_quote(quote_source="tencent_public", opener=opener,
                        now=datetime(2022, 1, 18, 10, tzinfo=SHANGHAI))
