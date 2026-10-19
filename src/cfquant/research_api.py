"""Bounded DeepSeek research requests with a durable, conservative budget ledger.

Budget scope is one shared cache directory. Before any HTTP request, its worst
case cost is fsynced to a hash-chained journal. Failed or interrupted calls keep
that reservation. A valid success replaces it with conservative token cost;
cache hits never issue another request. There are no automatic retries.

Pricing verified 2026-10-19 at the official DeepSeek pricing page: Flash peak
cache-miss input CNY 2 / million tokens, output CNY 8 / million tokens. This
client supports only that explicitly priced model and no tool/search calls.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid
from urllib import error as urlerror, request as urlrequest


ENDPOINT = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"
MAX_INPUT_BYTES = 24_000
MAX_RESPONSE_BYTES = 128 * 1024
MAX_OUTPUT_TOKENS = 4096
MAX_REQUESTS = 400
MAX_BUDGET_MICRO_CNY = 50_000_000
INPUT_MICRO_CNY_PER_TOKEN = 2
OUTPUT_MICRO_CNY_PER_TOKEN = 8
FRAMING_TOKEN_ALLOWANCE = 2048
_JSON_INSTRUCTION = "Return one valid JSON object only. Do not include Markdown fences."
_SECRET_FIELDS = {"api_key", "apikey", "authorization", "password", "secret", "access_token",
                  "refresh_token", "credentials", "token"}
_SAFE_ERRORS = {
    "busy", "journal_corrupt", "storage_failed", "credentials_missing", "credentials_invalid",
    "credential_in_prompt", "budget_exhausted", "call_limit_exhausted", "previous_failure",
    "reserved_unknown", "cache_corrupt", "transport_failed", "http_error", "redirect_rejected",
    "response_too_large", "invalid_response", "invalid_usage", "usage_bound_exceeded",
    "response_truncated", "invalid_json_content", "budget_halted",
}


class ResearchAPIError(RuntimeError):
    """Only a fixed status and optional numeric HTTP code are exposed."""
    def __init__(self, code: str, http_status: int | None = None):
        self.code = code if code in _SAFE_ERRORS else "transport_failed"
        self.http_status = http_status if type(http_status) is int and 100 <= http_status <= 599 else None
        super().__init__(self.code + (f" (HTTP {self.http_status})" if self.http_status is not None else ""))


def _encoded(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _money(units: int) -> float:
    return units / 1_000_000


def _sanitize(value, credential: str = "", depth=0):
    if depth > 30:
        raise ResearchAPIError("invalid_response")
    if isinstance(value, str):
        if credential:
            value = value.replace(credential, "[REDACTED]")
        value = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}", "[REDACTED]", value)
        value = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", value)
        # Prompts can contain JSON as text; redact recognized credential fields
        # there too instead of assuming every secret is a top-level dictionary.
        value = re.sub(r'(?i)("(?:api[_-]?key|authorization|password|secret|access_token|refresh_token)"\s*:\s*)"(?:\\.|[^"\\])*"',
                       r'\1"[REDACTED]"', value)
        value = re.sub(r"(?i)\b(api[_-]?key|authorization|password|secret|access_token|refresh_token)\s*=\s*[^\s,;]+",
                       r"\1=[REDACTED]", value)
        return value
    if isinstance(value, list):
        return [_sanitize(item, credential, depth + 1) for item in value]
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            key = _sanitize(str(key), credential, depth + 1)
            normalized = key.lower().replace("-", "_").replace(" ", "_")
            cleaned[key] = ("[REDACTED]" if normalized in _SECRET_FIELDS or normalized.endswith("_api_key")
                            else _sanitize(item, credential, depth + 1))
        return cleaned
    return value


class _NoRedirect(urlrequest.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ResearchAPIError("redirect_rejected", code) from None


def _official_transport(*, url: str, headers: dict, body: dict, timeout: float) -> dict:
    """The sole network boundary; no redirects, proxies, response-body errors."""
    if url != ENDPOINT:
        raise ResearchAPIError("redirect_rejected")
    request = urlrequest.Request(url, data=_encoded(body), headers=headers, method="POST")
    try:
        opener = urlrequest.build_opener(urlrequest.ProxyHandler({}), _NoRedirect())
        with opener.open(request, timeout=timeout) as response:
            if response.geturl() != ENDPOINT:
                raise ResearchAPIError("redirect_rejected")
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urlerror.HTTPError as exc:
        code = int(exc.code)
        exc.close()
        raise ResearchAPIError("redirect_rejected" if 300 <= code < 400 else "http_error", code) from None
    except ResearchAPIError:
        raise
    except Exception:
        raise ResearchAPIError("transport_failed") from None
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ResearchAPIError("response_too_large")
    try:
        value = json.loads(raw)
    except Exception:
        raise ResearchAPIError("invalid_response") from None
    if not isinstance(value, dict):
        raise ResearchAPIError("invalid_response")
    return value


@contextmanager
def _serial_lock(path: Path):
    """OS releases this persistent-file lock on process death; no stale PID lock."""
    try:
        handle = path.open("a+b")
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
    except OSError:
        raise ResearchAPIError("storage_failed") from None
    acquired = False
    try:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError:
            raise ResearchAPIError("busy") from None
        yield
    finally:
        if acquired:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


class ResearchAPI:
    """One budget pool shared by all instances/processes using cache_dir.

    Credential loading is deferred until an uncached, budget-approved request.
    key_file takes priority when present, otherwise DEEPSEEK_API_KEY is used.
    Different namespaces explicitly request independent samples; within each
    namespace identical request-body hashes reuse the durable cached response.
    """
    def __init__(self, cache_dir: str | Path, key_file: str | Path | None = None, *,
                 budget_cny=50, max_calls: int = 400, model: str = MODEL,
                 transport=None, timeout: float = 60):
        try:
            amount = Decimal(str(budget_cny))
            if not amount.is_finite() or not 0 < amount <= 50:
                raise ValueError
            self._budget = int(amount * 1_000_000)
        except (InvalidOperation, ValueError, TypeError):
            raise ValueError("budget_cny must be positive and at most 50") from None
        if self._budget < 1:
            raise ValueError("budget_cny is below one micro CNY")
        if type(max_calls) is not int or not 1 <= max_calls <= MAX_REQUESTS:
            raise ValueError("max_calls must be an integer in [1, 400]")
        if model != MODEL:
            raise ValueError("Only the explicitly priced deepseek-flash model is supported")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 120:
            raise ValueError("timeout must be in (0, 120]")
        self.cache_dir = Path(cache_dir)
        self.key_file = Path(key_file) if key_file is not None else None
        self.max_calls, self.model, self.timeout = max_calls, model, float(timeout)
        self._transport = transport or _official_transport
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.journal_path = self.cache_dir / "receipts.jsonl"
        self._lock_path = self.cache_dir / "request.lock"

    def _fresh_state(self):
        return {"tail": "0" * 64, "policy": None, "calls": {}, "by_cache": {}, "halted": False}

    def _read_state(self):
        state = self._fresh_state()
        if not self.journal_path.exists():
            return state
        try:
            if self.journal_path.stat().st_size > 128 * 1024 * 1024:
                raise ValueError
            with self.journal_path.open("rb") as handle:
                for index, line in enumerate(handle):
                    if index > 2 * MAX_REQUESTS or len(line) > 300_000 or not line.endswith(b"\n"):
                        raise ValueError
                    event = json.loads(line)
                    digest = event.pop("event_hash")
                    if event["previous_hash"] != state["tail"] or hashlib.sha256(_encoded(event)).hexdigest() != digest:
                        raise ValueError
                    state["tail"] = digest
                    kind = event["event"]
                    if kind == "policy":
                        if index != 0 or state["policy"] is not None:
                            raise ValueError
                        if (event["version"] != 1 or event["model"] != MODEL
                                or type(event["budget_micro_cny"]) is not int or not 0 < event["budget_micro_cny"] <= MAX_BUDGET_MICRO_CNY
                                or type(event["max_calls"]) is not int or not 1 <= event["max_calls"] <= MAX_REQUESTS
                                or event["input_micro_cny_per_token"] != 2 or event["output_micro_cny_per_token"] != 8):
                            raise ValueError
                        state["policy"] = event
                    elif kind == "reserved":
                        if state["policy"] is None or event["call_id"] != len(state["calls"]) + 1:
                            raise ValueError
                        expected = event["input_token_bound"] * 2 + event["max_tokens"] * 8
                        if (event["input_token_bound"] != event["input_bytes"] + FRAMING_TOKEN_ALLOWANCE
                                or event["estimated_max_micro_cny"] != expected
                                or not 0 < event["input_bytes"] <= MAX_INPUT_BYTES
                                or not 1 <= event["max_tokens"] <= MAX_OUTPUT_TOKENS
                                or event["cache_key"] in state["by_cache"]):
                            raise ValueError
                        state["calls"][event["call_id"]] = {"reservation": event, "completion": None}
                        state["by_cache"][event["cache_key"]] = event["call_id"]
                    elif kind == "completed":
                        call = state["calls"][event["call_id"]]
                        reservation = call["reservation"]
                        if call["completion"] is not None or event["request_hash"] != reservation["request_hash"]:
                            raise ValueError
                        if event["status"] == "success":
                            usage = event["result"]["usage"]
                            if (type(usage["prompt_tokens"]) is not int or type(usage["completion_tokens"]) is not int
                                    or not 0 <= usage["prompt_tokens"] <= reservation["input_token_bound"]
                                    or not 0 <= usage["completion_tokens"] <= reservation["max_tokens"]
                                    or event["charged_micro_cny"] != usage["prompt_tokens"] * 2 + usage["completion_tokens"] * 8):
                                raise ValueError
                            if hashlib.sha256(_encoded(event["result"])).hexdigest() != event["result_hash"]:
                                raise ValueError
                        elif event["status"] not in _SAFE_ERRORS or event["charged_micro_cny"] != reservation["estimated_max_micro_cny"]:
                            raise ValueError
                        call["completion"] = event
                        if event["status"] == "usage_bound_exceeded":
                            state["halted"] = True
                    else:
                        raise ValueError
        except Exception:
            raise ResearchAPIError("journal_corrupt") from None
        if state["policy"] is None:
            raise ResearchAPIError("journal_corrupt")
        used = sum((call["completion"]["charged_micro_cny"] if call["completion"] is not None else
                    call["reservation"]["estimated_max_micro_cny"]) for call in state["calls"].values())
        if len(state["calls"]) > state["policy"]["max_calls"] or used > state["policy"]["budget_micro_cny"]:
            raise ResearchAPIError("journal_corrupt")
        return state

    def _append(self, state, event):
        event = {**event, "previous_hash": state["tail"]}
        event["event_hash"] = hashlib.sha256(_encoded(event)).hexdigest()
        try:
            with self.journal_path.open("ab") as handle:
                handle.write(_encoded(event) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            raise ResearchAPIError("storage_failed") from None
        state["tail"] = event["event_hash"]

    def _usage_summary(self, state):
        policy = state["policy"]
        ceiling = min(self._budget, policy["budget_micro_cny"]) if policy else self._budget
        call_limit = min(self.max_calls, policy["max_calls"]) if policy else self.max_calls
        charged = successful = reserved = 0
        for call in state["calls"].values():
            completion = call["completion"]
            if completion is not None and completion["status"] == "success":
                successful += 1
                charged += completion["charged_micro_cny"]
            else:
                reserved += call["reservation"]["estimated_max_micro_cny"]
        return {"budget_cny": _money(ceiling), "calls": len(state["calls"]),
                "max_calls": call_limit, "successful_calls": successful,
                "successful_cost_cny": _money(charged), "uncertain_reserved_cny": _money(reserved),
                "reserved_or_spent_cny": _money(charged + reserved),
                "remaining_cny": _money(max(0, ceiling - charged - reserved)),
                "remaining_calls": max(0, call_limit - len(state["calls"])),
                "halted": state["halted"], "pricing_basis": "peak cache-miss input 2; output 8 CNY per million tokens",
                "_used_micro_cny": charged + reserved, "_budget_micro_cny": ceiling}

    def summary(self) -> dict:
        with _serial_lock(self._lock_path):
            result = self._usage_summary(self._read_state())
        return {key: value for key, value in result.items() if not key.startswith("_")}

    def _credential(self) -> str:
        try:
            if self.key_file is not None and self.key_file.exists():
                if self.key_file.stat().st_size > 4096:
                    raise ResearchAPIError("credentials_invalid")
                value = self.key_file.read_text(encoding="utf-8-sig").strip()
            else:
                value = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        except ResearchAPIError:
            raise
        except Exception:
            raise ResearchAPIError("credentials_invalid") from None
        if not value:
            raise ResearchAPIError("credentials_missing")
        if not re.fullmatch(r"[A-Za-z0-9._-]{8,256}", value):
            raise ResearchAPIError("credentials_invalid")
        return value

    def _cache_path(self, namespace, request_hash):
        return self.cache_dir / "responses" / hashlib.sha256(namespace.encode()).hexdigest() / f"{request_hash}.json"

    def _write_cache(self, path: Path, result: dict):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
            with temporary.open("xb") as handle:
                handle.write(_encoded(result))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except OSError:
            raise ResearchAPIError("storage_failed") from None

    def _cached(self, state, cache_key, namespace, request_hash):
        call_id = state["by_cache"].get(cache_key)
        if call_id is None:
            return None
        completion = state["calls"][call_id]["completion"]
        if completion is None:
            raise ResearchAPIError("reserved_unknown")
        if completion["status"] != "success":
            raise ResearchAPIError("previous_failure", completion.get("http_status"))
        result = completion["result"]
        path = self._cache_path(namespace, request_hash)
        try:
            if path.exists():
                if path.stat().st_size > 300_000 or hashlib.sha256(path.read_bytes()).hexdigest() != completion["result_hash"]:
                    raise ResearchAPIError("cache_corrupt")
            else:
                # A crash after the durable success receipt but before cache
                # persistence is recoverable without issuing another HTTP call.
                self._write_cache(path, result)
        except OSError:
            raise ResearchAPIError("cache_corrupt") from None
        return {**result, "cached": True}

    def _response(self, envelope, reservation, credential):
        try:
            if not isinstance(envelope, dict):
                raise ResearchAPIError("invalid_response")
            if len(_encoded(envelope)) > MAX_RESPONSE_BYTES:
                raise ResearchAPIError("response_too_large")
            choice = envelope["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ResearchAPIError("response_truncated")
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise ResearchAPIError("invalid_json_content")
            try:
                parsed = json.loads(content, parse_constant=lambda value: (_ for _ in ()).throw(ValueError()))
            except (ValueError, RecursionError):
                raise ResearchAPIError("invalid_json_content") from None
            if not isinstance(parsed, dict):
                raise ResearchAPIError("invalid_json_content")
            usage = envelope["usage"]
            counts = {name: usage[name] for name in ("prompt_tokens", "completion_tokens")}
            if any(type(value) is not int or value < 0 for value in counts.values()):
                raise ResearchAPIError("invalid_usage")
            if (counts["prompt_tokens"] > reservation["input_token_bound"]
                    or counts["completion_tokens"] > reservation["max_tokens"]):
                raise ResearchAPIError("usage_bound_exceeded")
            counts["total_tokens"] = counts["prompt_tokens"] + counts["completion_tokens"]
            if "total_tokens" in usage and usage["total_tokens"] != counts["total_tokens"]:
                raise ResearchAPIError("invalid_usage")
            model = envelope.get("model", MODEL)
            if not isinstance(model, str) or not model or len(model) > 200 or any(ord(c) < 32 for c in model):
                raise ResearchAPIError("invalid_response")
            clean = _sanitize(parsed, credential)
            return {"content": _encoded(clean).decode("utf-8"), "json": clean, "usage": counts,
                    "model": _sanitize(model, credential), "finish_reason": "stop"}
        except ResearchAPIError:
            raise
        except Exception:
            raise ResearchAPIError("invalid_response") from None

    def request(self, messages, *, temperature=.8, max_tokens=2048, cache_namespace="default") -> dict:
        if not isinstance(messages, list) or not 1 <= len(messages) <= 63:
            raise ValueError("messages must contain 1 to 63 text messages")
        checked = []
        for message in messages:
            if (not isinstance(message, dict) or set(message) != {"role", "content"}
                    or message["role"] not in {"system", "user", "assistant"}
                    or not isinstance(message["content"], str)):
                raise ValueError("Messages require only role and string content")
            checked.append({"role": message["role"], "content": message["content"]})
        if type(max_tokens) is not int or not 1 <= max_tokens <= MAX_OUTPUT_TOKENS:
            raise ValueError("max_tokens must be an integer in [1, 4096]")
        if (isinstance(temperature, bool) or not isinstance(temperature, (float, int))
                or not math.isfinite(temperature) or not 0 <= temperature <= 2):
            raise ValueError("temperature must be in [0, 2]")
        if not isinstance(cache_namespace, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,120}", cache_namespace):
            raise ValueError("cache_namespace must be a bounded alphanumeric experiment identifier")
        body = {"model": self.model, "messages": [{"role": "system", "content": _JSON_INSTRUCTION}] + checked,
                "temperature": float(temperature), "max_tokens": max_tokens,
                "response_format": {"type": "json_object"}, "thinking": {"type": "disabled"}, "stream": False}
        try:
            encoded = _encoded(body)
        except (ValueError, UnicodeError):
            raise ValueError("Request text must be valid UTF-8") from None
        if len(encoded) > MAX_INPUT_BYTES:
            raise ValueError("Request exceeds 24000 UTF-8 bytes")
        request_hash = hashlib.sha256(encoded).hexdigest()
        cache_key = hashlib.sha256((cache_namespace + "\0" + request_hash).encode()).hexdigest()
        estimated = (len(encoded) + FRAMING_TOKEN_ALLOWANCE) * 2 + max_tokens * 8
        with _serial_lock(self._lock_path):
            state = self._read_state()
            cached = self._cached(state, cache_key, cache_namespace, request_hash)
            if cached is not None:
                return cached
            summary = self._usage_summary(state)
            if state["halted"]:
                raise ResearchAPIError("budget_halted")
            if summary["remaining_calls"] <= 0:
                raise ResearchAPIError("call_limit_exhausted")
            if summary["_used_micro_cny"] + estimated > summary["_budget_micro_cny"]:
                raise ResearchAPIError("budget_exhausted")
            credential = self._credential()
            if credential in encoded.decode("utf-8"):
                raise ResearchAPIError("credential_in_prompt")
            if state["policy"] is None:
                policy = {"event": "policy", "version": 1, "created_at": _now(), "model": MODEL,
                          "budget_micro_cny": self._budget, "max_calls": self.max_calls,
                          "input_micro_cny_per_token": 2, "output_micro_cny_per_token": 8,
                          "pricing_verified_date": "2026-10-19"}
                self._append(state, policy)
                state["policy"] = policy
            reservation = {"event": "reserved", "call_id": len(state["calls"]) + 1,
                           "request_hash": request_hash, "cache_key": cache_key,
                           "cache_namespace": cache_namespace, "requested_model": self.model,
                           "created_at": _now(), "prompt": _sanitize(body, credential),
                           "input_bytes": len(encoded), "input_token_bound": len(encoded) + FRAMING_TOKEN_ALLOWANCE,
                           "max_tokens": max_tokens, "estimated_max_micro_cny": estimated,
                           "estimated_max_cost_cny": _money(estimated), "status": "reserved"}
            self._append(state, reservation)
            try:
                response = self._transport(url=ENDPOINT,
                    headers={"Content-Type": "application/json", "Authorization": "Bearer " + credential},
                    body=body, timeout=self.timeout)
                parsed = self._response(response, reservation, credential)
                cost = parsed["usage"]["prompt_tokens"] * 2 + parsed["usage"]["completion_tokens"] * 8
            except Exception as exc:
                status = exc.code if isinstance(exc, ResearchAPIError) else "transport_failed"
                http_status = exc.http_status if isinstance(exc, ResearchAPIError) else None
                self._append(state, {"event": "completed", "call_id": reservation["call_id"],
                                    "request_hash": request_hash, "completed_at": _now(), "status": status,
                                    "http_status": http_status, "charged_micro_cny": estimated,
                                    "charged_cost_cny": _money(estimated), "reservation_retained": True})
                raise ResearchAPIError(status, http_status) from None
            result = {"status": "success", **parsed, "requested_model": self.model,
                      "request_hash": request_hash, "cache_namespace": cache_namespace,
                      "call_id": reservation["call_id"], "created_at": reservation["created_at"],
                      "completed_at": _now(), "estimated_max_cost_cny": _money(estimated),
                      "charged_cost_cny": _money(cost), "cached": False}
            completion = {"event": "completed", "call_id": reservation["call_id"],
                          "request_hash": request_hash, "completed_at": result["completed_at"],
                          "status": "success", "charged_micro_cny": cost, "charged_cost_cny": _money(cost),
                          "result": result, "result_hash": hashlib.sha256(_encoded(result)).hexdigest()}
            self._append(state, completion)
            self._write_cache(self._cache_path(cache_namespace, request_hash), result)
            return result
