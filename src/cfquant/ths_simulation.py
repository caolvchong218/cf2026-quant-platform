"""Dedicated THS SkillHub *simulation* service, never a brokerage gateway.

Protocol provenance and known documentation drift: docs/THS_SIMULATION_RESEARCH.md.
The provider publishes HTTP only. No phone password, cookie, API key, real
brokerage account or configurable endpoint is accepted by this adapter.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request

BASE_URL = "http://trade.10jqka.com.cn:8088"
DEPARTMENT = "997376"
PROVIDER = "ths_skillhub_simulation"
PATHS = frozenset({
    "/pt_add_user", "/pt_qry_stkaccount_dklc", "/pt_stk_weituo_dklc",
    "/pt_web_qry_stock", "/pt_qry_userinfo_v1", "/pt_qry_fund_t",
    "/pt_qry_busin_nocache", "/pt_qry_busin1", "/pt_qry_gainstat",
})


class ProviderError(RuntimeError):
    """Sanitized failure: never include request URLs or account identifiers."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ProviderError("模拟服务发生重定向，已停止请求。")


def request(path: str, params: dict, *, timeout: float = 12) -> dict:
    """One attempt only: even a GET can be a mutating simulated order."""
    if path not in PATHS:
        raise ValueError("Unsupported simulation endpoint")
    query = urllib.parse.urlencode({**params, "datatype": "json"})
    req = urllib.request.Request(
        BASE_URL + path + "?" + query,
        headers={"User-Agent": "Mozilla/5.0", "Cache-Control": "no-cache"},
    )
    try:
        with urllib.request.build_opener(_NoRedirect()).open(req, timeout=timeout) as response:
            raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ProviderError("模拟服务返回过大，已停止处理。")
        data = json.loads(raw)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise ProviderError("模拟服务请求失败；下单请求不得自动重试。") from None
    if not isinstance(data, dict):
        raise ProviderError("模拟服务返回格式不符。")
    code = data.get("errorcode", data.get("error_no", data.get("code")))
    if code is None and isinstance(data.get("ret"), dict):
        code = data["ret"].get("code")
    if str(code) != "0":
        raise ProviderError("模拟服务未确认成功，请核对账户或接口状态。")
    return data


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pending")
    temporary.write_bytes((json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())
    os.replace(temporary, path)


def account_path(root: Path, strategy_id: str) -> Path:
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", strategy_id):
        raise ValueError("Invalid strategy identifier")
    return Path(root) / "runs/simulation_auto/accounts" / f"{strategy_id}.json"


def load_account(root: Path, strategy_id: str) -> dict:
    data = json.loads(account_path(root, strategy_id).read_text(encoding="utf-8"))
    if (data.get("provider") != PROVIDER or data.get("department") != DEPARTMENT
            or data.get("strategy_id") != strategy_id or data.get("simulation_only") is not True
            or not re.fullmatch(r"skill_\d{13}", data.get("username", ""))):
        raise ValueError("不是本平台创建的专用模拟账户。")
    if data.get("status") != "ready" or not str(data.get("account_id", "")).isdigit():
        raise ValueError("模拟开户尚未完成；请先核查已有开户结果，不要重复开户。")
    return data


def provision_account(root: Path, strategy_id: str, *, transport=request) -> dict:
    """Provision once; preserve the username before any remote mutation.

    Incomplete requests are deliberately not retried. Only newly created
    dedicated simulation identities may enter this adapter.
    """
    path = account_path(root, strategy_id)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "created":
            return load_account(root, strategy_id)
        if (data.get("provider") != PROVIDER or data.get("department") != DEPARTMENT
                or data.get("strategy_id") != strategy_id
                or data.get("simulation_only") is not True
                or not re.fullmatch(r"skill_\d{13}", data.get("username", ""))
                or not str(data.get("account_id", "")).isdigit()):
            raise ValueError("Invalid pending simulation identity")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "provider": PROVIDER, "simulation_only": True, "department": DEPARTMENT,
            "strategy_id": strategy_id, "username": f"skill_{int(time.time()*1000)}",
            "created_at": datetime.now(timezone.utc).isoformat(), "status": "creating",
            "mobile_account_linked": False,
        }
        with path.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
        result = transport("/pt_add_user", {"usrname": data["username"], "yybid": DEPARTMENT})
        account_id = str(result.get("errormsg", ""))
        if not account_id.isdigit():
            raise ProviderError("开户结果缺少资金账号；已保留开户请求，请勿重复创建。")
        data.update(account_id=account_id, status="created")
        _write(path, data)
    account_id = data["account_id"]
    result = transport("/pt_qry_stkaccount_dklc", {"usrid": account_id, "yybid": DEPARTMENT})
    shareholders = {}
    for row in result.get("result", []):
        if str(row.get("usrid")) != account_id:
            raise ProviderError("股东信息账户不一致。")
        shareholder = row.get("gdzh", row.get("gddm"))
        if str(row.get("scdm")) in {"1", "2"} and shareholder:
            shareholders[str(row["scdm"])] = str(shareholder)
    if not shareholders:
        raise ProviderError("未取得模拟股东信息。")
    data.update(shareholders=shareholders, status="ready")
    _write(path, data)
    return data


def rows(data: dict) -> list[dict]:
    """Accept the documented alternate list envelopes; reject ambiguity."""
    found = []
    for key in ("list", "result", "data"):
        value = data.get(key)
        if isinstance(value, list):
            found.append(value)
        elif isinstance(value, dict) and isinstance(value.get("list"), list):
            found.append(value["list"])
    if isinstance(data.get("ret"), dict) and isinstance(data["ret"].get("item"), list):
        found.append(data["ret"]["item"])
    if not found:
        if any(key in data and data[key] in (False, []) for key in ("list", "result", "data")):
            return []
        raise ProviderError("模拟查询未返回可识别的记录列表。")
    nonempty = [value for value in found if value]
    if len(nonempty) > 1 and any(value != nonempty[0] for value in nonempty[1:]):
        raise ProviderError("模拟查询含有相互冲突的记录列表。")
    result = nonempty[0] if nonempty else []
    if any(not isinstance(row, dict) for row in result):
        raise ProviderError("模拟查询的记录格式不符。")
    return result


def number(value, *, nonnegative=True) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ProviderError("模拟账户数值缺失或格式不符。") from None
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ProviderError("模拟账户数值无效。")
    return value


class THSSimulationClient:
    """Queries and raw submission for an already provisioned isolated account."""

    simulation_only = True

    def __init__(self, root: Path, strategy_id: str, *, transport=request):
        self.root, self.strategy_id = Path(root), strategy_id
        self.account = load_account(root, strategy_id)
        self.account_binding = hashlib.sha256(
            (PROVIDER + ":" + self.account["account_id"]).encode()
        ).hexdigest()[:24]
        self.transport = transport

    def inspect(self) -> dict:
        account_id = self.account["account_id"]
        funds = rows(self.transport("/pt_qry_fund_t", {"usrid": account_id}))
        if len(funds) != 1 or str(funds[0].get("usrid")) != account_id:
            raise ProviderError("资金查询没有返回唯一匹配的模拟账户。")
        fund = funds[0]
        positions = rows(self.transport("/pt_web_qry_stock", {
            "name": account_id, "yybid": DEPARTMENT, "type": "1"}))
        available, quantities, market_value = {}, {}, 0.0
        from .simulation_bridge import normalize_code
        for row in positions:
            code, market, _ = normalize_code(str(row.get("zqdm", "")))
            asset = code + "." + market
            if asset in quantities:
                raise ProviderError("持仓返回重复证券，不能自动合并。")
            quantity, usable = number(row.get("gpsl")), number(row.get("kysl"))
            if quantity != int(quantity) or usable != int(usable) or usable > quantity:
                raise ProviderError("模拟持仓数量或可卖数量无效。")
            quantities[asset], available[asset] = int(quantity), int(usable)
            market_value += number(row.get("gpz"))
        cash, usable_cash = number(fund.get("zjye")), number(fund.get("kyje"))
        frozen = number(fund.get("djje", fund.get("dje")))
        return {
            "provider": PROVIDER, "simulation_only": True,
            "account_binding": self.account_binding, "strategy_id": self.strategy_id,
            "asof": datetime.now(timezone.utc).isoformat(),
            "cash": cash, "available_cash": usable_cash, "frozen_cash": frozen,
            "positions": quantities, "available_shares": available,
            "market_value": market_value, "nav": cash + market_value,
            "valuation_basis": "provider_reported_cash_plus_position_market_value",
            "mobile_account_linked": False,
        }

    def today_trades(self) -> list[dict]:
        return rows(self.transport("/pt_qry_busin_nocache", {
            "usrname": self.account["account_id"], "kind": "1"}))

    def history_trades(self, trade_date: date) -> list[dict]:
        """Query exactly one unresolved execution date using the official script."""
        if type(trade_date) is not date:
            raise ValueError("Historical trade query needs an explicit date")
        return rows(self.transport("/pt_qry_busin1", {
            "usrname": self.account["account_id"], "start": trade_date.strftime("%Y%m%d"),
            "end": trade_date.strftime("%Y%m%d"), "yhbId": DEPARTMENT, "kind": "1"}))

    def submit_limit(self, *, asset: str, side: str, quantity: int, price: float) -> str:
        if side not in {"buy", "sell"} or not re.fullmatch(r"\d{6}\.(SH|SZ)", asset):
            raise ValueError("Unsupported order")
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0 or quantity % 100:
            raise ValueError("Order quantity must be positive 100-share lots")
        number(price)
        if price <= 0:
            raise ValueError("Invalid limit price")
        code, market = asset.split(".")
        market_code = "2" if market == "SH" else "1"
        if market_code not in self.account["shareholders"]:
            raise ProviderError("模拟账户没有该市场股东信息。")
        data = self.transport("/pt_stk_weituo_dklc", {
            "usrid": self.account["account_id"], "zqdm": code,
            "gddh": self.account["shareholders"][market_code], "scdm": market_code,
            "yybd": DEPARTMENT, "wtjg": str(price), "wtsl": str(quantity),
            "mmlb": "B" if side == "buy" else "S",
        })
        result = data.get("result")
        if isinstance(result, list) and len(result) == 1:
            result = result[0]
        if not isinstance(result, dict) or not result.get("entrust_no"):
            raise ProviderError("请求已发出但未取得委托编号；结果待核对，禁止自动重试。")
        if "userid" in result and str(result["userid"]) != self.account["account_id"]:
            raise ProviderError("委托回执账户不一致；结果待核对，禁止自动重试。")
        return str(result["entrust_no"])

    def save_observation(self) -> tuple[dict, Path]:
        snapshot = self.inspect()
        snapshot["trades_today"] = self.today_trades()
        directory = self.root / "runs/simulation_auto/observations" / self.strategy_id
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = directory / (timestamp + ".json")
        _write(path, snapshot)
        return snapshot, path


class THSSimulationBroker:
    """Execution-core adapter. Unknown order/fill identifiers stay unresolved."""

    simulation_only = True

    def __init__(self, client: THSSimulationClient):
        self.client = client

    def snapshot(self):
        from .simulation_execution import BrokerSnapshot
        state = self.client.inspect()
        return BrokerSnapshot(state["account_binding"], datetime.fromisoformat(state["asof"]),
                              state["available_cash"], state["available_shares"], True)

    def submit(self, intent):
        from .simulation_execution import SubmitReceipt
        if (intent.account_binding != self.client.account_binding
                or intent.strategy_id != self.client.strategy_id):
            raise ValueError("Strategy or simulated account does not match")
        order_id = self.client.submit_limit(
            asset=intent.asset, side=intent.side, quantity=intent.quantity, price=intent.limit_price)
        return SubmitReceipt(self.client.account_binding, self._dated_id(order_id, intent.execution_date), "accepted")

    def query_orders(self):
        # The official package supplies no open-order/cancel endpoint. Do not
        # invent either an endpoint or a terminal state based on an empty query.
        return []

    def query_fills(self):
        from zoneinfo import ZoneInfo
        return self.query_fills_for_dates((datetime.now(ZoneInfo("Asia/Shanghai")).date(),))

    @staticmethod
    def _dated_id(value, trade_date):
        # Provider identifiers may restart each trading day. Namespacing is
        # based on actual IDs, never inferred from a price/quantity combination.
        value = str(value) if value is not None else ""
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,100}", value):
            raise ProviderError("成交或委托缺少可靠编号，自动对账暂停。")
        return trade_date.strftime("%Y%m%d") + ":" + value

    def query_fills_for_dates(self, execution_dates, *, today=None):
        """Only unresolved dates are read; no order is submitted or inferred."""
        from zoneinfo import ZoneInfo
        today = today or datetime.now(ZoneInfo("Asia/Shanghai")).date()
        dates = tuple(execution_dates)
        if (type(today) is not date or len(dates) > 31
                or any(type(day) is not date or day > today for day in dates)):
            raise ProviderError("待核验交易日范围不符，自动对账暂停。")
        dates = sorted(set(dates))
        output = []
        for day in dates:
            records = self.client.today_trades() if day == today else self.client.history_trades(day)
            try:
                output.extend(self._parse_fills(records, day))
            except ProviderError:
                if day != today:
                    raise
                # The API specification lists quantity only for history.
                # Its same-date query may supply missing fields; empty history
                # never erases a nonempty but unidentifiable current-day result.
                historical = self.client.history_trades(day)
                if not historical:
                    raise
                output.extend(self._parse_fills(historical, day))
        return output

    def _parse_fills(self, records, trade_date):
        from .simulation_execution import BrokerFill
        from zoneinfo import ZoneInfo
        result = []
        for row in records:
            order_id = row.get("entrust_no", row.get("wth"))
            fill_id = row.get("business_no", row.get("cjbh"))
            stamp = str(row.get("cjsj", "")).strip()
            # Identifier aliases are explicit. Without IDs a fill can be shown
            # to the user, but cannot settle a durable automatic-order intent.
            order_id = self._dated_id(order_id, trade_date)
            fill_id = self._dated_id(fill_id, trade_date)
            quantity = number(row.get("cjsl"))
            if quantity != int(quantity) or quantity <= 0:
                raise ProviderError("模拟成交数量无效。")
            # A time-only record is dated by a single-day provider query, not by
            # a guess from a local order. Full dates must match that query.
            if re.fullmatch(r"\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?", stamp):
                stamp = trade_date.isoformat() + " " + stamp
            if not re.fullmatch(r"\d{4}[-/]\d{2}[-/]\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})?", stamp):
                raise ProviderError("模拟成交日期格式不符，自动对账暂停。")
            try:
                instant = datetime.fromisoformat(stamp.replace("/", "-"))
            except ValueError:
                raise ProviderError("模拟成交日期格式不符。") from None
            if instant.tzinfo is None:
                instant = instant.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
            instant = instant.astimezone(ZoneInfo("Asia/Shanghai"))
            if instant.date() != trade_date:
                raise ProviderError("模拟成交日期与查询范围不符，自动对账暂停。")
            result.append(BrokerFill(
                fill_id, order_id, self.client.account_binding, int(quantity),
                number(row.get("cjg", row.get("cjj"))), instant,
                number(row["fee"]) if "fee" in row else None,
                "microseconds" if "." in stamp else "seconds"))
        return result
