import json
from datetime import date
from pathlib import Path

import pytest

from cfquant.ths_simulation import (
    ProviderError, THSSimulationClient, THSSimulationBroker, provision_account, rows,
)


class OfficialFixture:
    def __init__(self):
        self.calls = []

    def __call__(self, path, params):
        self.calls.append((path, params))
        if path == "/pt_add_user":
            return {"errorcode": 0, "errormsg": "123456"}
        if path == "/pt_qry_stkaccount_dklc":
            return {"errorcode": 0, "result": [
                {"usrid": "123456", "scdm": "1", "gdzh": "SYNTHETIC_SZ"},
                {"usrid": "123456", "scdm": "2", "gdzh": "SYNTHETIC_SH"},
            ]}
        if path == "/pt_qry_fund_t":
            return {"errorcode": 0, "result": {}, "list": [
                {"usrid": "123456", "zjye": "200000", "kyje": "199000", "djje": "1000"}]}
        if path == "/pt_stk_weituo_dklc":
            return {"errorcode": 0, "result": [{"entrust_no": "SYNTHETIC_ORDER_1"}]}
        return {"errorcode": 0, "result": [], "list": []}


def test_provision_once_and_inspect_current_observed_schema(tmp_path):
    transport = OfficialFixture()
    first = provision_account(tmp_path, "rsrs_standard_ma", transport=transport)
    second = provision_account(tmp_path, "rsrs_standard_ma", transport=transport)
    assert first == second and len(transport.calls) == 2
    client = THSSimulationClient(tmp_path, "rsrs_standard_ma", transport=transport)
    snapshot, path = client.save_observation()
    assert snapshot["nav"] == 200000 and snapshot["available_cash"] == 199000
    assert snapshot["frozen_cash"] == 1000 and snapshot["trades_today"] == []
    assert snapshot["mobile_account_linked"] is False
    assert path.is_file()
    assert not any(path == "/pt_stk_weituo_dklc" for path, _ in transport.calls)


def test_creation_timeout_cannot_repeat_remote_mutation(tmp_path):
    calls = []
    def fail(path, params):
        calls.append(path)
        raise ProviderError("timeout")
    with pytest.raises(ProviderError):
        provision_account(tmp_path, "trial", transport=fail)
    with pytest.raises(ValueError, match="重复开户"):
        provision_account(tmp_path, "trial", transport=fail)
    assert len(calls) == 1


def test_created_account_recovers_shareholder_query_without_second_creation(tmp_path):
    fixture = OfficialFixture()
    def partial(path, params):
        if path == "/pt_qry_stkaccount_dklc":
            raise ProviderError("query failed")
        return fixture(path, params)
    with pytest.raises(ProviderError):
        provision_account(tmp_path, "trial", transport=partial)
    provision_account(tmp_path, "trial", transport=fixture)
    assert sum(path == "/pt_add_user" for path, _ in fixture.calls) == 1


def test_orders_use_documented_simulation_fields_and_require_receipt(tmp_path):
    fixture = OfficialFixture()
    provision_account(tmp_path, "trial", transport=fixture)
    client = THSSimulationClient(tmp_path, "trial", transport=fixture)
    order = client.submit_limit(asset="510300.SH", side="buy", quantity=100, price=4.521)
    assert order == "SYNTHETIC_ORDER_1"
    path, params = fixture.calls[-1]
    assert path == "/pt_stk_weituo_dklc"
    assert params["scdm"] == "2" and params["yybd"] == "997376" and params["wtsl"] == "100"
    client.transport = lambda *args: {"errorcode": 0, "result": [{}]}
    with pytest.raises(ProviderError, match="禁止自动重试"):
        client.submit_limit(asset="510300.SH", side="buy", quantity=100, price=4.521)


def test_account_mismatch_missing_cash_and_unidentified_fill_fail_closed(tmp_path):
    fixture = OfficialFixture()
    provision_account(tmp_path, "trial", transport=fixture)
    client = THSSimulationClient(tmp_path, "trial", transport=fixture)
    client.transport = lambda *args: {"errorcode": 0, "list": [{"usrid": "OTHER", "zjye": "200000"}]}
    with pytest.raises(ProviderError, match="唯一匹配"):
        client.inspect()
    client.transport = lambda *args: {"errorcode": 0, "list": [{"zqdm": "510300", "cjsl": "100"}]}
    with pytest.raises(ProviderError, match="自动对账暂停"):
        THSSimulationBroker(client).query_fills()


def test_envelopes_are_not_silently_coerced():
    assert rows({"result": {}, "list": False}) == []
    assert rows({"result": [{"x": 1}], "list": [{"x": 1}]}) == [{"x": 1}]
    with pytest.raises(ProviderError):
        rows({"result": [{"x": 1}], "list": [{"x": 2}]})
    with pytest.raises(ProviderError):
        rows({"errorcode": 0})


@pytest.mark.parametrize("quantity,price", [(True, 10), (1.5, 10), (99, 10), (100, float("nan")), (100, -1)])
def test_invalid_order_never_sends(tmp_path, quantity, price):
    fixture = OfficialFixture()
    provision_account(tmp_path, "trial", transport=fixture)
    client = THSSimulationClient(tmp_path, "trial", transport=fixture)
    before = len(fixture.calls)
    with pytest.raises((ValueError, ProviderError)):
        client.submit_limit(asset="510300.SH", side="buy", quantity=quantity, price=price)
    assert len(fixture.calls) == before


def test_documented_nonempty_position_fields_are_valued_without_replacing_sellable_qty(tmp_path):
    fixture = OfficialFixture()
    provision_account(tmp_path, "trial", transport=fixture)
    def transport(path, params):
        if path == "/pt_web_qry_stock":
            return {"errorcode": 0, "data": [{"zqdm": "510300", "gpsl": "200", "kysl": "100",
                    "gpz": "800", "gpcb": "3.9", "fdyk": "20"}]}
        return fixture(path, params)
    state = THSSimulationClient(tmp_path, "trial", transport=transport).inspect()
    assert state["positions"] == {"510300.SH": 200}
    assert state["available_shares"] == {"510300.SH": 100}
    assert state["market_value"] == 800 and state["nav"] == 200800


def test_date_scoped_history_uses_official_script_fields_and_real_ids(tmp_path):
    fixture = OfficialFixture()
    provision_account(tmp_path, "trial", transport=fixture)
    calls = []
    def transport(path, params):
        calls.append((path, params))
        assert path == "/pt_qry_busin1"
        return {"errorcode": 0, "list": [{"wth": "same-order", "cjbh": "same-fill",
                "cjsl": "100", "cjj": "4.0", "cjsj": "10:00:00", "fee": "5"}]}
    broker = THSSimulationBroker(THSSimulationClient(tmp_path, "trial", transport=transport))
    dates = (date(2026, 10, 19), date(2026, 10, 20))
    fills = broker.query_fills_for_dates(dates, today=date(2026, 10, 21))
    assert [(c[1]["start"], c[1]["end"]) for c in calls] == [("20261019", "20261019"), ("20261020", "20261020")]
    assert all("usrname" in p and p["yhbId"] == "997376" for _, p in calls)
    assert fills[0].fill_id != fills[1].fill_id
    assert fills[0].order_id != fills[1].order_id
    assert [f.timestamp.date() for f in fills] == list(dates)
    assert all(f.timestamp_precision == "seconds" for f in fills)
    assert broker.query_fills_for_dates((), today=date(2026, 10, 21)) == []
    assert len(calls) == 2


def test_current_day_incomplete_fill_can_use_same_day_history_but_not_fabricate_ids(tmp_path):
    fixture = OfficialFixture()
    provision_account(tmp_path, "trial", transport=fixture)
    paths = []
    def transport(path, params):
        paths.append(path)
        trade = {"entrust_no": "order-1", "business_no": "fill-1", "cjg": "4",
                 "cje": "400", "cjsj": "2026-10-20 10:00:00"}
        if path == "/pt_qry_busin1":
            trade["cjsl"] = "100"
        return {"errorcode": 0, "list": [trade]}
    client = THSSimulationClient(tmp_path, "trial", transport=transport)
    broker = THSSimulationBroker(client)
    fills = broker.query_fills_for_dates((date(2026, 10, 20),), today=date(2026, 10, 20))
    assert len(fills) == 1 and fills[0].quantity == 100
    assert paths == ["/pt_qry_busin_nocache", "/pt_qry_busin1"]
    client.transport = lambda *args: {"errorcode": 0, "list": [
        {"zqdm": "510300", "cjj": "4", "cje": "400", "cjsl": "100", "cjsj": "10:00:00"}]}
    with pytest.raises(ProviderError, match="可靠编号"):
        broker.query_fills_for_dates((date(2026, 10, 20),), today=date(2026, 10, 20))


def test_provider_fill_full_date_must_match_the_requested_history_day(tmp_path):
    fixture = OfficialFixture()
    provision_account(tmp_path, "trial", transport=fixture)
    client = THSSimulationClient(tmp_path, "trial", transport=lambda *args: {
        "errorcode": 0, "list": [{"entrust_no": "1", "business_no": "2", "cjsl": "100",
                                "cjj": "4", "cjsj": "2026-10-19 10:00:00"}]})
    with pytest.raises(ProviderError, match="查询范围"):
        THSSimulationBroker(client).query_fills_for_dates((date(2026, 10, 20),), today=date(2026, 10, 21))


def test_submit_receipt_and_fill_share_the_execution_day_namespace(tmp_path):
    from cfquant.simulation_execution import OrderIntent
    fixture = OfficialFixture()
    provision_account(tmp_path, "trial", transport=fixture)
    client = THSSimulationClient(tmp_path, "trial", transport=fixture)
    broker = THSSimulationBroker(client)
    intent = OrderIntent("synthetic-intent", "trial", client.account_binding,
                         "510300.SH", "buy", 100, 4., date(2026, 10, 19), date(2026, 10, 20))
    receipt = broker.submit(intent)
    client.transport = lambda *args: {"errorcode": 0, "list": [
        {"entrust_no": "SYNTHETIC_ORDER_1", "business_no": "synthetic-fill", "cjsl": "100",
         "cjj": "4", "cjsj": "10:00:00"}]}
    recovered = broker.query_fills_for_dates((date(2026, 10, 20),), today=date(2026, 10, 21))
    assert receipt.order_id == recovered[0].order_id == "20261020:SYNTHETIC_ORDER_1"
