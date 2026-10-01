"""All model requests use injected local transports; no network/API key required."""
import json
import numpy as np
import pytest

from cfquant.llm_research import (PROMPT_VERSION, ResearchConnection, ResearchRequestError,
                                 explain_holding, request_research)


def valid_proposal(**changes):
    return {"id": "trend", "label": "趋势", "expression": "returns(close, 20)",
            "hypothesis": "趋势可能持续，需要验证。", **changes}


def openai_envelope(content, finish_reason="stop"):
    return {"choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)},
                         "finish_reason": finish_reason}]}


def test_offline_is_explicit_deterministic_and_does_not_call_transport():
    def forbidden(**kwargs):
        pytest.fail("Offline mode called a transport")
    payload = {"objective": "研究动量", "metrics": {"rank_ic": np.float64(.012), "ic_ir": np.nan}}
    first = request_research(ResearchConnection(), payload, transport=forbidden)
    second = request_research(ResearchConnection(), payload, transport=forbidden)
    assert first == second
    assert first["provider"] == "offline" and first["is_model_generated"] is False
    assert "未调用模型" in first["explanation_label"]
    assert first["evidence"]["metrics"] == {"rank_ic": .012, "ic_ir": None}
    assert first["prompt_version"] == PROMPT_VERSION
    assert first["proposals"] and all(p["status"] == "dsl_validated_not_evaluated" for p in first["proposals"])
    assert "rank_ic" not in first  # Only the supplied evidence owns metrics.


def test_openai_compatible_contract_validates_proposals_and_preserves_evidence():
    calls = []
    secret = "sk-transient-test-credential"
    payload = {"metrics": {"ic": .02}, "allowed_fields": ["close", "volume"]}
    def transport(**kwargs):
        calls.append(kwargs)
        return openai_envelope({"explanation": "所附 IC 是研究证据，机制仍需验证。",
                               "metrics": {"ic": .99},
                               "proposals": [valid_proposal(), valid_proposal(id="future", expression="delay(close, -2)"),
                                             valid_proposal(id="python", expression="close.shift(1)")]})
    connection = ResearchConnection("openai-compatible", "https://example.test/v1/", "test-model", 256, 7)
    result = request_research(connection, payload, api_key=secret, transport=transport)
    call = calls[0]
    assert call["url"] == "https://example.test/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer " + secret
    assert call["body"]["max_tokens"] == 256 and call["timeout"] == 7
    assert call["body"]["response_format"] == {"type": "json_object"}
    assert result["evidence"] == payload
    assert result["model"] == "test-model" and "未核验" in result["explanation_label"]
    assert len(result["proposals"]) == 1 and len(result["rejected_proposals"]) == 2
    assert result["proposals"][0]["fields"] == ["close"]
    assert "metrics" not in result
    assert secret not in json.dumps(result, ensure_ascii=False)
    assert secret not in json.dumps(call["body"], ensure_ascii=False)


def test_local_ollama_contract_uses_bounded_nonstreaming_json():
    calls = []
    def transport(**kwargs):
        calls.append(kwargs)
        return {"message": {"content": json.dumps({"explanation": "仅根据所附信息解释。", "proposals": []})}, "done": True}
    result = request_research(ResearchConnection("ollama", "http://127.0.0.1:11434", "qwen", 300),
                              {"objective": "价量研究"}, transport=transport)
    call = calls[0]
    assert call["url"] == "http://127.0.0.1:11434/api/chat"
    assert call["body"]["format"] == "json" and call["body"]["stream"] is False
    assert call["body"]["options"] == {"temperature": 0, "num_predict": 300}
    assert "Authorization" not in call["headers"]
    assert result["provider"] == "ollama" and result["proposals"] == []


def test_echoed_credential_is_redacted_and_transport_exception_is_safe():
    secret = "sk-secret-must-not-return"
    def echo(**kwargs):
        return openai_envelope({"explanation": "echo: " + secret,
                               "proposals": [valid_proposal(hypothesis="echo " + secret)]})
    result = request_research(ResearchConnection("openai", model="test"), {}, api_key=secret, transport=echo)
    assert secret not in json.dumps(result)
    assert "[REDACTED]" in result["explanation"]
    def fails(**kwargs):
        raise RuntimeError(secret + str(kwargs["headers"]))
    with pytest.raises(ResearchRequestError) as exc:
        request_research(ResearchConnection("openai", model="test"), {}, api_key=secret, transport=fails)
    assert secret not in str(exc.value)


@pytest.mark.parametrize("payload", [{"api_key": "hidden"}, {"nested": {"Authorization": "hidden"}},
                                      {"x": "x" * 24001}, {"allowed_fields": ["close", "_future"]}])
def test_secret_payloads_and_excessive_or_invalid_inputs_are_rejected_before_transport(payload):
    def forbidden(**kwargs):
        pytest.fail("Invalid input called a transport")
    with pytest.raises(ValueError):
        request_research(ResearchConnection("openai", model="test"), payload, transport=forbidden)


@pytest.mark.parametrize("envelope", [
    {"choices": []},
    openai_envelope({"explanation": "partial", "proposals": []}, finish_reason="length"),
    openai_envelope({"explanation": "", "proposals": []}),
    openai_envelope({"explanation": "text", "proposals": [valid_proposal()] * 9}),
    {"choices": [{"message": {"content": "not JSON"}}]},
    {"choices": [{"message": {"content": "x" * 66000}}]},
])
def test_malformed_incomplete_and_oversized_model_responses_are_rejected(envelope):
    with pytest.raises(ResearchRequestError):
        request_research(ResearchConnection("openai", model="test"), {}, transport=lambda **kwargs: envelope)


def test_duplicate_and_unavailable_field_proposals_cannot_enter_candidate_set():
    def transport(**kwargs):
        return openai_envelope({"explanation": "待验证。", "proposals": [valid_proposal(), valid_proposal(),
            valid_proposal(id="missing", expression="rolling_mean(volume, 5)")]})
    result = request_research(ResearchConnection("openai", model="test"), {"allowed_fields": ["close"]}, transport=transport)
    assert len(result["proposals"]) == 1 and len(result["rejected_proposals"]) == 2


def test_holding_explanation_reconstructs_supplied_additive_contributions():
    holding = {"asset": "600000", "date": "2025-02-03", "score": .3, "contributions": [
        {"factor": "trend", "value": .02, "normalized_value": 1., "weight": .5, "contribution": .5},
        {"factor": "risk", "normalized_value": -.5, "weight": .4},
    ]}
    result = explain_holding(holding)
    evidence = result["evidence"]
    assert evidence["contribution_total"] == pytest.approx(.3)
    assert evidence["score_minus_contribution_total"] == pytest.approx(0)
    assert evidence["contributions_complete"] is True
    assert evidence["contributions"][1]["contribution"] == pytest.approx(-.2)
    assert "trend：+0.5" in result["explanation"] and "risk：-0.2" in result["explanation"]
    assert "contribution" not in holding["contributions"][1]  # Caller is unchanged.
    assert result["proposals"] == [] and result["is_model_generated"] is False


def test_holding_missing_contributions_remain_unknown_and_discrepancy_is_visible():
    incomplete = explain_holding({"asset": "A", "score": 1., "contributions": [
        {"factor": "one", "contribution": .3}, {"factor": "missing", "contribution": None}]})
    assert incomplete["evidence"]["contribution_total"] is None
    assert incomplete["evidence"]["contributions_complete"] is False
    assert "不完整" in incomplete["explanation"]
    mismatch = explain_holding({"asset": "A", "score": 1., "contributions": [{"factor": "one", "contribution": .3}]})
    assert mismatch["evidence"]["score_minus_contribution_total"] == pytest.approx(.7)
    assert "相差" in mismatch["explanation"]


def test_optional_model_holding_explanation_receives_actual_contributions():
    calls = []
    def transport(**kwargs):
        calls.append(kwargs)
        return openai_envelope({"explanation": "已记录的 trend 贡献为 0.2。", "proposals": []})
    result = explain_holding({"asset": "A", "score": .2, "contributions": [{"factor": "trend", "contribution": .2}]},
                             ResearchConnection("openai", model="test"), transport=transport)
    sent = json.loads(calls[0]["body"]["messages"][1]["content"])
    assert sent["kind"] == "explain_holding" and sent["contribution_total"] == .2
    assert "deterministic_explanation" in result
    assert result["evidence"]["contributions"][0]["contribution"] == .2


@pytest.mark.parametrize("kwargs", [{"provider": "unknown"}, {"timeout": 0}, {"max_output_tokens": 5000},
                                    {"base_url": "https://user:secret@example.test/v1"},
                                    {"base_url": "https://example.test/v1?api_key=secret"}])
def test_connection_rejects_invalid_limits_and_embedded_credentials(kwargs):
    with pytest.raises(ValueError):
        ResearchConnection(**kwargs)
