"""Local mocked transports only: no real endpoint or user credentials are read."""
import json
from urllib import error as urlerror

import pytest

from cfquant import research_api as module
from cfquant.research_api import ENDPOINT, ResearchAPI, ResearchAPIError


MESSAGES = [{"role": "user", "content": "Propose one causal factor as JSON."}]
FAKE_KEY = "sk-local-fixture-123456789"


@pytest.fixture(autouse=True)
def no_user_key_environment(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)


@pytest.fixture
def key_file(tmp_path):
    path = tmp_path / "fixture_key.txt"
    path.write_text(FAKE_KEY, encoding="utf-8")
    return path


def envelope(content=None, *, prompt=100, output=50, finish="stop"):
    return {"model": "DeepSeek-V4.1-Flash-1017", "choices": [{"finish_reason": finish,
             "message": {"content": json.dumps(content if content is not None else {"factor": "returns(close, 20)"})}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": output, "total_tokens": prompt + output}}


def events(api):
    return [json.loads(line) for line in api.journal_path.read_text(encoding="utf-8").splitlines()]


def test_bounded_request_contract_durable_reservation_and_conservative_usage(tmp_path, key_file):
    calls = []
    api = None
    def transport(**kwargs):
        calls.append(kwargs)
        recorded = events(api)
        assert recorded[-1]["event"] == "reserved"  # Durable before network.
        assert len(recorded) == 2
        return envelope()
    api = ResearchAPI(tmp_path / "cache", key_file, transport=transport)
    result = api.request(MESSAGES)
    assert calls[0]["url"] == ENDPOINT
    assert calls[0]["headers"]["Authorization"] == "Bearer " + FAKE_KEY
    body = calls[0]["body"]
    assert body["response_format"] == {"type": "json_object"}
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_tokens"] == 2048 and body["temperature"] == .8
    assert body["stream"] is False
    assert result["content"] == '{"factor":"returns(close, 20)"}'
    assert result["usage"]["total_tokens"] == 150
    assert result["model"] == "DeepSeek-V4.1-Flash-1017"
    assert result["charged_cost_cny"] == pytest.approx((100 * 2 + 50 * 8) / 1_000_000)
    assert result["charged_cost_cny"] < result["estimated_max_cost_cny"]
    receipt = events(api)[1]
    assert receipt["input_token_bound"] == len(module._encoded(body)) + 2048
    assert receipt["estimated_max_micro_cny"] == receipt["input_token_bound"] * 2 + 2048 * 8
    summary = api.summary()
    assert summary["calls"] == summary["successful_calls"] == 1
    assert summary["uncertain_reserved_cny"] == 0
    assert summary["remaining_cny"] == pytest.approx(50 - result["charged_cost_cny"])
    assert FAKE_KEY not in api.journal_path.read_text(encoding="utf-8")
    assert FAKE_KEY not in json.dumps(result)
    assert FAKE_KEY not in repr(api)


def test_cache_hit_after_restart_needs_no_credential_or_second_charge(tmp_path, key_file):
    calls = []
    def transport(**kwargs):
        calls.append(kwargs)
        return envelope()
    directory = tmp_path / "cache"
    original = ResearchAPI(directory, key_file, transport=transport)
    first = original.request(MESSAGES, cache_namespace="seed:17")
    second = ResearchAPI(directory, transport=lambda **kwargs: pytest.fail("Cache made a request"))
    hit = second.request(MESSAGES, cache_namespace="seed:17")
    assert hit["cached"] is True and first["cached"] is False
    assert hit["request_hash"] == first["request_hash"]
    assert len(calls) == 1 and second.summary()["calls"] == 1
    assert len(events(second)) == 3
    assert second.summary()["reserved_or_spent_cny"] == first["charged_cost_cny"]


def test_explicit_namespace_requests_independent_sample_but_body_hash_stays_exact(tmp_path, key_file):
    calls = []
    api = ResearchAPI(tmp_path / "cache", key_file, transport=lambda **kwargs: calls.append(kwargs) or envelope())
    a = api.request(MESSAGES, cache_namespace="seed_1")
    b = api.request(MESSAGES, cache_namespace="seed_2")
    assert a["request_hash"] == b["request_hash"]
    assert a["call_id"] != b["call_id"]
    assert len(calls) == 2
    assert len(list((tmp_path / "cache/responses").rglob("*.json"))) == 2


def test_no_auto_retry_and_failure_retains_reservation_across_restart(tmp_path, key_file):
    calls = []
    def failure(**kwargs):
        calls.append(kwargs)
        raise TimeoutError("Do not expose: " + FAKE_KEY + str(kwargs["headers"]))
    directory = tmp_path / "cache"
    api = ResearchAPI(directory, key_file, transport=failure)
    with pytest.raises(ResearchAPIError, match="transport_failed") as raised:
        api.request(MESSAGES)
    assert FAKE_KEY not in str(raised.value)
    assert len(calls) == 1
    reserved = events(api)[1]["estimated_max_cost_cny"]
    assert api.summary()["uncertain_reserved_cny"] == reserved
    again = ResearchAPI(directory, key_file, transport=lambda **kwargs: pytest.fail("Retried failed call"))
    with pytest.raises(ResearchAPIError, match="previous_failure"):
        again.request(MESSAGES)
    assert again.summary()["calls"] == 1
    assert again.summary()["reserved_or_spent_cny"] == reserved
    assert events(api)[-1]["reservation_retained"] is True
    assert FAKE_KEY not in api.journal_path.read_text(encoding="utf-8")


def test_interruption_leaves_reserved_unknown_and_os_lock_releases(tmp_path, key_file):
    def interrupt(**kwargs):
        raise KeyboardInterrupt
    directory = tmp_path / "cache"
    api = ResearchAPI(directory, key_file, transport=interrupt)
    with pytest.raises(KeyboardInterrupt):
        api.request(MESSAGES)
    restarted = ResearchAPI(directory, key_file, transport=lambda **kwargs: pytest.fail("Unknown state was retried"))
    assert restarted.summary()["calls"] == 1
    assert restarted.summary()["uncertain_reserved_cny"] > 0
    with pytest.raises(ResearchAPIError, match="reserved_unknown"):
        restarted.request(MESSAGES)


def test_budget_rejects_before_network_and_bounds_cannot_be_relaxed_on_restart(tmp_path, key_file):
    directory = tmp_path / "cache"
    tiny = ResearchAPI(directory, key_file, budget_cny=.000001,
                       transport=lambda **kwargs: pytest.fail("Over-budget request"))
    with pytest.raises(ResearchAPIError, match="budget_exhausted"):
        tiny.request(MESSAGES)
    assert tiny.summary()["calls"] == 0
    api = ResearchAPI(directory, key_file, budget_cny=.03,
                      transport=lambda **kwargs: (_ for _ in ()).throw(OSError("unknown")))
    with pytest.raises(ResearchAPIError):
        api.request(MESSAGES)
    restarted = ResearchAPI(directory, key_file, budget_cny=50,
                            transport=lambda **kwargs: pytest.fail("Raised persisted budget"))
    assert restarted.summary()["budget_cny"] == .03
    with pytest.raises(ResearchAPIError, match="budget_exhausted"):
        restarted.request(MESSAGES, cache_namespace="new_sample")


def test_call_ceiling_counts_failed_calls_and_cache_still_works(tmp_path, key_file):
    directory = tmp_path / "cache"
    api = ResearchAPI(directory, key_file, max_calls=1, transport=lambda **kwargs: envelope())
    api.request(MESSAGES)
    restarted = ResearchAPI(directory, key_file, max_calls=400, transport=lambda **kwargs: pytest.fail("Call ceiling exceeded"))
    assert restarted.request(MESSAGES)["cached"]
    with pytest.raises(ResearchAPIError, match="call_limit_exhausted"):
        restarted.request(MESSAGES, cache_namespace="seed_2")
    assert restarted.summary()["max_calls"] == 1


def test_input_and_output_limits_are_enforced_without_leaking_response(tmp_path, key_file):
    api = ResearchAPI(tmp_path / "cache", key_file,
                      transport=lambda **kwargs: envelope({"blob": "x" * (module.MAX_RESPONSE_BYTES + 1)}))
    with pytest.raises(ValueError, match="24000"):
        api.request([{"role": "user", "content": "数" * 9000}])
    assert api.summary()["calls"] == 0
    with pytest.raises(ResearchAPIError, match="response_too_large"):
        api.request(MESSAGES)
    assert api.summary()["uncertain_reserved_cny"] > 0


def test_credential_echo_is_redacted_in_content_metadata_cache_and_receipts(tmp_path, key_file):
    malicious = envelope({"explanation": "Echo " + FAKE_KEY, "authorization": "another-secret",
                          "nested": {"api_key": "unrelated-private-value", "proposal": "returns(close,20)"}})
    malicious["model"] = FAKE_KEY
    api = ResearchAPI(tmp_path / "cache", key_file, transport=lambda **kwargs: malicious)
    result = api.request(MESSAGES)
    serialized = json.dumps(result)
    assert FAKE_KEY not in serialized and "another-secret" not in serialized
    assert "unrelated-private-value" not in serialized
    assert result["json"]["authorization"] == "[REDACTED]"
    for path in (tmp_path / "cache").rglob("*"):
        if path.is_file():
            assert FAKE_KEY.encode() not in path.read_bytes()
    with pytest.raises(ResearchAPIError, match="credential_in_prompt"):
        api.request([{"role": "user", "content": "unsafe " + FAKE_KEY}])
    assert api.summary()["calls"] == 1


def test_embedded_json_credential_fields_are_sanitized_in_saved_prompt(tmp_path, key_file):
    api = ResearchAPI(tmp_path / "cache", key_file, transport=lambda **kwargs: envelope())
    api.request([{"role": "user", "content": 'Example configuration: {"api_key": "unrelated-private-value"}'}])
    saved = api.journal_path.read_text(encoding="utf-8")
    assert "unrelated-private-value" not in saved
    assert "[REDACTED]" in saved


def test_environment_credential_is_lazy_and_file_takes_priority(tmp_path, key_file, monkeypatch):
    calls = []
    api = ResearchAPI(tmp_path / "cache", key_file, transport=lambda **kwargs: calls.append(kwargs) or envelope())
    assert api.summary()["calls"] == 0
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-environment-fixture-123")
    api.request(MESSAGES)
    assert calls[0]["headers"]["Authorization"] == "Bearer " + FAKE_KEY
    env_api = ResearchAPI(tmp_path / "env_cache", transport=lambda **kwargs: calls.append(kwargs) or envelope())
    env_api.request(MESSAGES)
    assert calls[1]["headers"]["Authorization"] == "Bearer sk-environment-fixture-123"


def test_missing_key_does_not_reserve_or_call(tmp_path):
    api = ResearchAPI(tmp_path / "cache", tmp_path / "absent.txt",
                      transport=lambda **kwargs: pytest.fail("Missing credentials reached transport"))
    with pytest.raises(ResearchAPIError, match="credentials_missing"):
        api.request(MESSAGES)
    assert api.summary()["calls"] == 0


def test_redirects_rejected_before_forwarding_authorization(tmp_path, key_file, monkeypatch):
    seen = []
    class Opener:
        def open(self, request, timeout):
            seen.append(request)
            raise urlerror.HTTPError(ENDPOINT, 302, "secret " + FAKE_KEY,
                                     {"Location": "https://untrusted.invalid/"}, None)
    monkeypatch.setattr(module.urlrequest, "build_opener", lambda *handlers: Opener())
    api = ResearchAPI(tmp_path / "cache", key_file)
    with pytest.raises(ResearchAPIError, match="redirect_rejected") as raised:
        api.request(MESSAGES)
    assert raised.value.http_status == 302
    assert len(seen) == 1 and seen[0].full_url == ENDPOINT
    assert FAKE_KEY not in str(raised.value)
    handler = module._NoRedirect()
    with pytest.raises(ResearchAPIError, match="redirect_rejected"):
        handler.redirect_request(seen[0], None, 307, "ignored", {}, "https://untrusted.invalid/")
    with pytest.raises(ResearchAPIError, match="redirect_rejected"):
        module._official_transport(url="https://untrusted.invalid/", headers={}, body={}, timeout=1)


def test_busy_lock_prevents_concurrent_reservations(tmp_path, key_file):
    api = ResearchAPI(tmp_path / "cache", key_file, transport=lambda **kwargs: envelope())
    second = ResearchAPI(tmp_path / "cache", key_file, transport=lambda **kwargs: pytest.fail("Concurrent request"))
    with module._serial_lock(api._lock_path):
        with pytest.raises(ResearchAPIError, match="busy"):
            second.request(MESSAGES)
    assert api.summary()["calls"] == 0


def test_corrupt_or_partial_journal_fails_closed(tmp_path, key_file):
    api = ResearchAPI(tmp_path / "cache", key_file, transport=lambda **kwargs: envelope())
    api.request(MESSAGES)
    original = api.journal_path.read_bytes()
    api.journal_path.write_bytes(original + b'{"event":"reserved"')
    with pytest.raises(ResearchAPIError, match="journal_corrupt"):
        api.request(MESSAGES, cache_namespace="new")
    api.journal_path.write_bytes(original.replace(b'"charged_micro_cny":600', b'"charged_micro_cny":0'))
    with pytest.raises(ResearchAPIError, match="journal_corrupt"):
        api.summary()


def test_cache_corruption_does_not_trigger_paid_retry(tmp_path, key_file):
    api = ResearchAPI(tmp_path / "cache", key_file, transport=lambda **kwargs: envelope())
    result = api.request(MESSAGES)
    cache = api._cache_path("default", result["request_hash"])
    cache.write_text('{"content":"tampered"}', encoding="utf-8")
    with pytest.raises(ResearchAPIError, match="cache_corrupt"):
        api.request(MESSAGES)
    assert api.summary()["calls"] == 1


def test_success_journal_recovers_interrupted_cache_write_without_network(tmp_path, key_file, monkeypatch):
    directory = tmp_path / "cache"
    api = ResearchAPI(directory, key_file, transport=lambda **kwargs: envelope())
    monkeypatch.setattr(api, "_write_cache", lambda *args: (_ for _ in ()).throw(ResearchAPIError("storage_failed")))
    with pytest.raises(ResearchAPIError, match="storage_failed"):
        api.request(MESSAGES)
    restarted = ResearchAPI(directory, transport=lambda **kwargs: pytest.fail("Durable success was charged twice"))
    hit = restarted.request(MESSAGES)
    assert hit["cached"] and restarted.summary()["successful_calls"] == 1


def test_provider_exceeding_usage_bound_halts_future_calls(tmp_path, key_file):
    api = ResearchAPI(tmp_path / "cache", key_file, transport=lambda **kwargs: envelope(prompt=1_000_000))
    with pytest.raises(ResearchAPIError, match="usage_bound_exceeded"):
        api.request(MESSAGES)
    assert api.summary()["halted"] is True
    with pytest.raises(ResearchAPIError, match="budget_halted"):
        api.request(MESSAGES, cache_namespace="new")


@pytest.mark.parametrize("response", [
    envelope(finish="length"), {"choices": [], "usage": {}},
    {"choices": [{"finish_reason": "stop", "message": {"content": "not JSON"}}]},
    envelope([], prompt=1), envelope(prompt=True),
])
def test_invalid_response_keeps_full_reservation(tmp_path, key_file, response):
    api = ResearchAPI(tmp_path / "cache", key_file, transport=lambda **kwargs: response)
    with pytest.raises(ResearchAPIError):
        api.request(MESSAGES)
    receipt = events(api)[1]
    assert api.summary()["reserved_or_spent_cny"] == receipt["estimated_max_cost_cny"]


@pytest.mark.parametrize("kwargs", [{"budget_cny": 50.01}, {"budget_cny": float("nan")},
                                   {"max_calls": 401}, {"max_calls": True},
                                   {"model": "deepseek-v4-pro"}, {"timeout": 0}])
def test_hard_safety_limits_cannot_be_overridden(tmp_path, kwargs):
    with pytest.raises(ValueError):
        ResearchAPI(tmp_path / "cache", **kwargs)


@pytest.mark.parametrize("kwargs", [{"max_tokens": 4097}, {"max_tokens": True},
                                   {"temperature": float("inf")}, {"cache_namespace": "../unsafe"}])
def test_invalid_request_controls_do_not_charge(tmp_path, key_file, kwargs):
    api = ResearchAPI(tmp_path / "cache", key_file, transport=lambda **kw: pytest.fail("Invalid request"))
    with pytest.raises(ValueError):
        api.request(MESSAGES, **kwargs)
    assert api.summary()["calls"] == 0


def test_invalid_unicode_prompt_has_only_a_safe_validation_error(tmp_path, key_file):
    api = ResearchAPI(tmp_path / "cache", key_file, transport=lambda **kwargs: pytest.fail("Invalid Unicode input"))
    with pytest.raises(ValueError, match="valid UTF-8") as raised:
        api.request([{"role": "user", "content": FAKE_KEY + "\ud800"}])
    assert FAKE_KEY not in str(raised.value)
    assert api.summary()["calls"] == 0
