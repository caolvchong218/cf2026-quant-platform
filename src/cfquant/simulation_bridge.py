"""A file-only bridge for simulated trades; no login or order submission.

Historical adjusted holdings are exported as reference weights, never shares.
Uploaded executions retain every row, including duplicates, for reconciliation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile

import pandas as pd

from .experiment_catalog import Experiment, read_ledger


OFFICIAL_SOURCES = {
    "同花顺模拟炒股": "https://moni.10jqka.com.cn/index.shtml",
    "SuperMind 模拟交易与明细导出": "https://quant.10jqka.com.cn/view/help/10",
    "SuperMind 常见问题": "https://quant.10jqka.com.cn/view/help/16",
    "SuperMind 研究环境": "https://quant.10jqka.com.cn/view/help/14",
}
ALIASES = {
    "timestamp": ("交易时间", "成交时间", "成交日期时间", "交易日期时间", "timestamp", "datetime", "trade_time"),
    "date": ("交易日期", "成交日期", "日期", "date", "trade_date"),
    "code": ("证券代码", "股票代码", "代码", "证券", "code", "asset", "symbol", "security"),
    "name": ("证券名称", "股票名称", "名称", "name", "security_name"),
    "side": ("操作", "买卖方向", "买卖标志", "交易方向", "方向", "委托类别", "side", "operation"),
    "price": ("交易价格", "成交价格", "成交均价", "价格", "price", "trade_price"),
    "quantity": ("交易数量", "成交数量", "成交股数", "数量", "quantity", "volume", "trade_volume"),
    "amount": ("交易金额", "成交金额", "成交额", "金额", "amount", "notional", "trade_amount"),
    "trade_id": ("成交编号", "成交序号", "成交号", "交易编号", "trade_id", "execution_id"),
    "exchange": ("交易市场", "市场", "交易所", "exchange", "market"),
}
REQUIRED = ("code", "side", "price", "quantity", "amount")
MAX_BYTES = 10 * 1024 * 1024
MAX_ROWS = 50_000
SIDES = {"买入": "buy", "证券买入": "buy", "买": "buy", "买进": "buy", "buy": "buy", "b": "buy",
         "卖出": "sell", "证券卖出": "sell", "卖": "sell", "卖出成交": "sell", "sell": "sell", "s": "sell"}
EXCHANGES = {"SH": "SH", "XSHG": "SH", "SSE": "SH", "上海": "SH", "沪市": "SH", "上海A股": "SH",
             "SZ": "SZ", "XSHE": "SZ", "SZSE": "SZ", "深圳": "SZ", "深市": "SZ", "深圳A股": "SZ",
             "BJ": "BJ", "BSE": "BJ", "北京": "BJ", "北交所": "BJ"}


@dataclass(frozen=True)
class ImportResult:
    normalized: pd.DataFrame
    issues: pd.DataFrame
    column_mapping: dict[str, str]
    encoding: str
    source_sha256: str
    source_kind: str
    amount_tolerance: float

    @property
    def valid(self) -> bool:
        return not self.normalized.empty and not (self.issues.severity == "error").any()

    @property
    def duplicate_rows(self) -> int:
        return int(self.normalized.get("duplicate_flag", pd.Series(dtype=bool)).sum())


def _header(value: str) -> str:
    value = str(value).strip().lstrip("\ufeff")
    value = re.sub(r"[（(][^）)]*[）)]", "", value)
    return re.sub(r"[\s_\-]+", "", value).casefold()


def _decode(raw: bytes) -> tuple[str, str]:
    if len(raw) > MAX_BYTES:
        raise ValueError("成交文件不能超过 10 MB；请按日期拆分后导入。")
    if not raw.strip():
        raise ValueError("文件为空。")
    encodings = ("utf-16",) if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else ("utf-8-sig", "gb18030")
    for encoding in encodings:
        try:
            text = raw.decode(encoding)
            if "\x00" in text:
                raise ValueError("文件不是可读取的 CSV 文本。")
            return text, encoding
        except UnicodeDecodeError:
            continue
    raise ValueError("无法识别文件编码，请另存为 UTF-8 或 GB18030 CSV。")


def normalize_code(value: str, exchange: str = "") -> tuple[str, str, list[str]]:
    """Normalize six-digit securities without numeric coercion or lost zeroes."""
    raw = str(value).strip().strip("'").upper()
    warnings: list[str] = []
    # Accept Excel's exact text wrapper, never arbitrary formula expressions.
    quoted = re.fullmatch(r'="(\d{1,6})"', raw)
    if quoted:
        raw = quoted.group(1)
    explicit = ""
    suffix = re.fullmatch(r"(\d{1,6})[.\-](SH|SZ|BJ|XSHG|XSHE|SSE|SZSE|BSE)", raw)
    prefix = re.fullmatch(r"(SH|SZ|BJ)[.\-]?(\d{1,6})", raw)
    if suffix:
        raw, explicit = suffix.group(1), EXCHANGES[suffix.group(2)]
    elif prefix:
        raw, explicit = prefix.group(2), EXCHANGES[prefix.group(1)]
    if re.fullmatch(r"\d{1,6}\.0+", raw):
        raw = raw.split(".")[0]
        warnings.append("代码原为数字格式，已恢复为六位文本；请核对原证券。")
    if not re.fullmatch(r"\d{1,6}", raw):
        raise ValueError("证券代码须为六位数字，可附 SH/SZ/BJ 市场后缀。")
    if len(raw) < 6:
        warnings.append("代码不足六位，已在左侧补零；请核对原证券。")
    code = raw.zfill(6)
    market_text = str(exchange).strip()
    market = EXCHANGES.get(market_text.upper(), EXCHANGES.get(market_text, "")) if market_text else ""
    if market_text and not market:
        raise ValueError("无法识别交易市场。")
    if explicit and market and explicit != market:
        raise ValueError("代码后缀与交易市场不一致。")
    market = explicit or market
    if not market:
        market = ("BJ" if code.startswith(("4", "8", "92")) else
                  "SH" if code.startswith(("5", "6", "9")) else
                  "SZ" if code.startswith(("0", "1", "2", "3")) else "")
        if not market:
            warnings.append("无法根据代码识别市场，仅保留六位代码。")
    return code, market, warnings


def _number(value: str, label: str) -> Decimal:
    text = str(value).strip().replace(",", "").replace("，", "")
    text = text.removeprefix("￥").removeprefix("¥")
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"{label}必须为有效数值。") from exc
    if not number.is_finite() or number <= 0:
        raise ValueError(f"{label}必须为有限正数。")
    if number > Decimal("100000000000000"):
        raise ValueError(f"{label}超出可接受范围，请核查单位。")
    return number


def _timestamp(value: str, date: str, now: pd.Timestamp) -> tuple[pd.Timestamp, str]:
    text = str(value).strip()
    date = str(date).strip()
    warning = ""
    if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?", text):
        if not date:
            raise ValueError("只有时分秒，缺少交易日期。")
        text = f"{date} {text}"
    elif not text:
        text = date
    if not re.search(r"\d{4}[-/年]?\d{2}[-/月]?\d{2}", text):
        raise ValueError("时间须包含完整年月日，例如 2026-09-18 09:35:00。")
    try:
        if re.fullmatch(r"\d{14}", text):
            stamp = pd.to_datetime(text, format="%Y%m%d%H%M%S")
        else:
            stamp = pd.Timestamp(text)
    except (ValueError, TypeError) as exc:
        raise ValueError("交易日期或时间无效。") from exc
    if pd.isna(stamp):
        raise ValueError("交易时间为空。")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("Asia/Shanghai").tz_localize(None)
    if stamp.year < 1990 or stamp > now + pd.Timedelta(minutes=5):
        raise ValueError("交易时间早于 1990 年或晚于当前时间。")
    if len(text) <= 10:
        warning = "原文件仅含日期，未补造成交时分秒。"
    if stamp.dayofweek >= 5:
        warning = (warning + " 周末成交日期，请核对模拟平台原记录。").strip()
    return stamp, warning


def parse_simulated_trades(raw: bytes, *, source_kind: str = "uploaded_simulation",
                           now: pd.Timestamp | str | None = None, amount_tolerance: float = .05) -> ImportResult:
    """Validate every row; errors block saving, duplicates remain visible."""
    if source_kind not in ("uploaded_simulation", "synthetic_demo"):
        raise ValueError("未知回执来源类型。")
    if not 0 <= amount_tolerance <= 1:
        raise ValueError("金额核对允差须在 0 至 1 元之间。")
    text, encoding = _decode(raw)
    try:
        frame = pd.read_csv(io.StringIO(text), sep=None, engine="python", dtype=str,
                            keep_default_na=False, nrows=MAX_ROWS+1, on_bad_lines="error")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise ValueError("CSV 结构无法解析，请检查分隔符、表头和引号。") from exc
    if len(frame) > MAX_ROWS:
        raise ValueError("成交记录超过 50,000 行，请按日期拆分。")
    if frame.empty:
        raise ValueError("文件只有表头，没有成交记录。")
    mapping: dict[str, str] = {}
    for field, aliases in ALIASES.items():
        matches = [col for col in frame.columns if _header(col) in {_header(alias) for alias in aliases}]
        if field in matches:
            matches = [field]
        if len(matches) > 1:
            raise ValueError(f"字段 {field} 匹配了多个列，请保留一个明确列名：{'、'.join(matches)}")
        if matches:
            mapping[field] = matches[0]
    missing = [field for field in REQUIRED if field not in mapping]
    if "timestamp" not in mapping and "date" not in mapping:
        missing.append("timestamp / date")
    if missing:
        raise ValueError("缺少必要字段：" + "、".join(missing) + "。可下载示例 CSV 对照列名。")
    current = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None) if now is None else pd.Timestamp(now)
    if current.tzinfo is not None:
        current = current.tz_convert("Asia/Shanghai").tz_localize(None)
    issues: list[dict] = []
    normalized: list[dict] = []

    def issue(row: int, severity: str, field: str, message: str):
        issues.append({"source_row": row, "severity": severity, "field": field, "message": message})

    for row_number, (_, row) in enumerate(frame.iterrows(), start=1):
        values = {field: str(row[column]).strip() for field, column in mapping.items()}
        item: dict = {"source_row": row_number, "timestamp": "", "code": "", "exchange": "", "asset": "",
                      "name": values.get("name", ""), "side": "", "price": None, "quantity": None,
                      "amount": None, "expected_amount": None, "amount_difference": None,
                      "trade_id": values.get("trade_id", ""), "time_precision": ""}
        try:
            code, market, warnings = normalize_code(values["code"], values.get("exchange", ""))
            item.update(code=code, exchange=market, asset=f"{code}.{market}" if market else code)
            for warning in warnings:
                issue(row_number, "warning", "code", warning)
        except ValueError as exc:
            issue(row_number, "error", "code", str(exc))
        try:
            stamp, warning = _timestamp(values.get("timestamp", ""), values.get("date", ""), current)
            date_only = warning.startswith("原文件仅含日期")
            item["timestamp"] = stamp.date().isoformat() if date_only else stamp.isoformat(sep=" ")
            item["time_precision"] = "date" if date_only else "datetime"
            if warning:
                issue(row_number, "warning", "timestamp", warning)
        except ValueError as exc:
            issue(row_number, "error", "timestamp", str(exc))
        side = SIDES.get(values["side"].casefold())
        if side is None:
            issue(row_number, "error", "side", "操作须为买入或卖出；撤单、申购和信用交易等不能混作股票成交。")
        else:
            item["side"] = side
        numbers: dict[str, Decimal] = {}
        for field, label in (("price", "成交价格"), ("quantity", "成交数量"), ("amount", "成交金额")):
            try:
                number = _number(values[field], label)
                if field == "quantity" and number != number.to_integral_value():
                    raise ValueError("成交数量须为整数股；复权单位不能当作真实成交股数。")
                numbers[field] = number
                item[field] = int(number) if field == "quantity" else float(number)
            except ValueError as exc:
                issue(row_number, "error", field, str(exc))
        if len(numbers) == 3:
            expected = numbers["price"] * numbers["quantity"]
            difference = numbers["amount"] - expected
            item["expected_amount"], item["amount_difference"] = float(expected), float(difference)
            if abs(difference) > Decimal(str(amount_tolerance)):
                issue(row_number, "error", "amount", f"金额与价格 × 数量相差 {difference:.4f} 元，超过 {amount_tolerance:.2f} 元允差；请确认金额不含费用及价格精度。")
        fingerprint = {key: item[key] for key in ("timestamp", "asset", "side", "price", "quantity", "amount")}
        item["fingerprint"] = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:24]
        normalized.append(item)
    result = pd.DataFrame(normalized)
    duplicates = result.duplicated("fingerprint", keep=False)
    nonempty_ids = result.trade_id.ne("")
    duplicates |= nonempty_ids & result.trade_id.duplicated(keep=False)
    result["duplicate_flag"] = duplicates
    for row_number in result.loc[duplicates, "source_row"]:
        issue(int(row_number), "warning", "duplicate", "与其他记录的成交编号或关键成交字段重复；记录全部保留，请对照原文件确认。")
    for row_number in result.loc[result.name.str.match(r"^[=+@\-\t\r]", na=False), "source_row"]:
        issue(int(row_number), "warning", "name", "名称可能被表格软件解释为公式；CSV 导出时将作为纯文本处理。")
    diagnostics = pd.DataFrame(issues, columns=["source_row", "severity", "field", "message"])
    return ImportResult(result, diagnostics, mapping, encoding, hashlib.sha256(raw).hexdigest(), source_kind, amount_tolerance)


def reconcile_by_security(result: ImportResult) -> pd.DataFrame:
    """Gross trade cash flow and quantity changes, never investment returns."""
    if not result.valid:
        raise ValueError("请先修正所有错误，再进行成交金额对账。")
    frame = result.normalized
    rows = []
    for asset, group in frame.groupby("asset", sort=True):
        buys, sells = group[group.side == "buy"], group[group.side == "sell"]
        buy_amount, sell_amount = float(buys.amount.sum()), float(sells.amount.sum())
        rows.append({"asset": asset, "code": str(group.iloc[0].code), "name": str(group.iloc[0]["name"]),
                     "buy_quantity": int(buys.quantity.sum()), "sell_quantity": int(sells.quantity.sum()),
                     "net_quantity_change": int(buys.quantity.sum()-sells.quantity.sum()),
                     "buy_amount": buy_amount, "sell_amount": sell_amount,
                     "gross_cash_flow": sell_amount-buy_amount, "record_count": len(group),
                     "duplicate_rows": int(group.duplicate_flag.sum())})
    return pd.DataFrame(rows)


def validation_summary(result: ImportResult) -> dict:
    frame = result.normalized
    return {"valid": result.valid, "source_kind": result.source_kind, "encoding": result.encoding,
            "source_sha256": result.source_sha256, "rows": len(frame),
            "error_count": int((result.issues.severity == "error").sum()),
            "warning_count": int((result.issues.severity == "warning").sum()),
            "duplicate_rows": result.duplicate_rows, "duplicates_removed": False,
            "column_mapping": result.column_mapping, "amount_tolerance_cny": result.amount_tolerance,
            "account_connected": False, "orders_submitted": 0,
            "profit_or_return_computed": False,
            "interpretation": "仅核对已导入成交的买卖金额与数量变化；缺少初始资金、初始持仓、费用和市值，不计算账户收益。"}


def safe_csv(frame: pd.DataFrame) -> bytes:
    """Preserve code strings and prevent uploaded names becoming spreadsheet formulas."""
    output = frame.copy()
    for column in output.select_dtypes(include=["object", "string"]).columns:
        output[column] = output[column].map(lambda value: "'" + value if isinstance(value, str)
                                           and re.match(r"^[=+@\-\t\r]", value) else value)
    return output.to_csv(index=False).encode("utf-8-sig")


def historical_watchlist(root: Path, record: Experiment, as_of: str) -> tuple[pd.DataFrame, dict]:
    """Export actual historical weights; intentionally omit adjusted units/prices."""
    if not record.has_details:
        raise ValueError("公开聚合快照没有逐股持仓，无法导出真实观察组合。")
    date = pd.Timestamp(as_of).strftime("%Y-%m-%d")
    holdings, count = read_ledger(root, record, "positions", start=date, end=date)
    daily, _ = read_ledger(root, record, "daily", start=date, end=date)
    if holdings.empty or daily.empty:
        raise ValueError("该日期没有完整的持仓截面与资金账本。")
    if count != len(holdings):
        raise ValueError("持仓截面未完整读取，不能导出部分组合。")
    if not {"asset", "weight"} <= set(holdings.columns):
        raise ValueError("持仓账本缺少股票或权重字段。")
    weights = pd.to_numeric(holdings.weight, errors="coerce")
    if weights.isna().any() or (~weights.between(0, 1)).any() or weights.sum() > 1.000001:
        raise ValueError("持仓权重无效，请先核对原始账本。")
    rows = []
    for asset, weight in zip(holdings.asset, weights):
        code, exchange, _ = normalize_code(str(asset))
        rows.append({"as_of": date, "code": code, "exchange": exchange, "asset": str(asset),
                     "reference_weight": float(weight), "usage": "historical_watchlist_only"})
    watch = pd.DataFrame(rows).sort_values("reference_weight", ascending=False).reset_index(drop=True)
    row = daily.iloc[0]
    meta = {"as_of": date, "experiment_id": record.id, "strategy": record.strategy,
            "source": record.relative_path, "securities": len(watch), "stock_weight": float(weights.sum()),
            "cash_weight": float(row.cash/row.nav), "historical_only": True,
            "order_file": False, "share_quantity_exported": False,
            "weight_semantics": "当日收盘实际持仓占总净值的权重，不是今日信号或调仓指令。",
            "adjusted_units_warning": "回测 units 是复权单位，不能作为同花顺真实股数，因此本文件不导出 units。"}
    return watch, meta


def synthetic_example_csv() -> bytes:
    """Small explicitly synthetic executions for an offline classroom demo."""
    return ("交易时间,证券代码,证券名称,操作,交易价格,交易数量,交易金额\n"
            "2026-09-18 09:35:00,000001,合成示例甲,买入,10.00,100,1000.00\n"
            "2026-09-18 10:15:00,600000,合成示例乙,买入,8.00,200,1600.00\n"
            "2026-09-18 14:10:00,000001,合成示例甲,卖出,10.20,100,1020.00\n").encode("utf-8-sig")


def save_simulated_receipt(root: Path, result: ImportResult, *, label: str = "") -> tuple[Path, bool]:
    """Save a validated normalized receipt atomically under a content-derived ID."""
    if not result.valid:
        raise ValueError("仍有错误，不能保存为已验证模拟回执。")
    if not re.fullmatch(r"[a-f0-9]{64}", result.source_sha256):
        raise ValueError("源文件摘要无效。")
    if len(label) > 160:
        raise ValueError("回执名称最多 160 字。")
    root = Path(root).resolve()
    folder = (root / "runs/paper_imports").resolve()
    if not folder.is_relative_to(root):
        raise ValueError("模拟回执目录超出项目范围。")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{result.source_sha256[:24]}.json"
    if not path.resolve().is_relative_to(folder):
        raise ValueError("模拟回执路径无效。")
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("validation", {}).get("source_sha256") != result.source_sha256:
            raise ValueError("已有回执的内容标识不匹配，未覆盖文件。")
        return path, False
    payload = {"schema_version": 1, "saved_at_utc": datetime.now(timezone.utc).isoformat(),
               "label": label.strip(), "validation": validation_summary(result),
               "trades": result.normalized.to_dict("records"), "issues": result.issues.to_dict("records"),
               "by_security": reconcile_by_security(result).to_dict("records")}
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=folder, prefix=".receipt-", suffix=".tmp", delete=False) as temp:
        json.dump(payload, temp, ensure_ascii=False, indent=2, allow_nan=False)
        temp.flush()
        os.fsync(temp.fileno())
        temporary = temp.name
    os.replace(temporary, path)
    return path, True


def connection_status() -> dict:
    return {"schema_version": 1, "product": "同花顺 / SuperMind 模拟交易文件桥",
            "account_connected": False, "automated_trade_api_connected": False,
            "ordinary_simulation_public_trade_api_verified": False,
            "official_documentation_checked": True,
            "file_bridge_available": True, "login_performed": False, "orders_submitted": 0,
            "live_trading_supported": False, "sources": OFFICIAL_SOURCES,
            "modes": [
                {"capability": "历史观察组合与代码清单", "status": "可用", "detail": "导出历史权重，不含委托股数。"},
                {"capability": "模拟成交 CSV 导入与金额核对", "status": "可用", "detail": "文件校验、重复提示、规范化导出和本地回执。"},
                {"capability": "同花顺普通模拟炒股账号", "status": "未连接", "detail": "官方入口已核查，未登录，未核实公开交易 API。"},
                {"capability": "SuperMind Python 模拟策略", "status": "官方支持 / 本项目未连接", "detail": "需用户账号与环境；当前仅衔接官方成交明细导出。"},
                {"capability": "自动下单 / 实盘交易", "status": "未提供", "detail": "本模块仅处理文件，不登录、不发送订单。"},
            ]}
