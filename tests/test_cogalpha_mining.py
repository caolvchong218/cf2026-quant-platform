"""Deterministic in-memory API mocks; never load keys or contact a provider."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cfquant.cogalpha_mining import LAYERS, ROLES, TrainingOnlyError, mine, validate_card
from cfquant.research_api import ResearchAPIError


@pytest.fixture
def config():
    return json.loads((Path(__file__).resolve().parents[1] / "configs/project2_cogalpha.json").read_text(encoding="utf-8"))


def proposal(expression="returns(close, 20)"):
    return {"id": "candidate", "label": "Momentum", "expression": expression,
            "hypothesis": "A historical momentum proxy; no guarantee of future performance."}


class FakeAPI:
    def __init__(self, hook=None):
        self.calls = []
        self.cache = {}
        self.hook = hook

    def request(self, messages, **kwargs):
        namespace = kwargs["cache_namespace"]
        key = json.dumps([messages, kwargs], sort_keys=True)
        if key in self.cache:
            return {**copy.deepcopy(self.cache[key]), "cached": True}
        self.calls.append((copy.deepcopy(messages), kwargs))
        value = self.hook(messages, kwargs, len(self.calls)) if self.hook else proposal(f"returns(close, 20) + {len(self.calls)} * rank(volume)")
        result = {"json": value, "request_hash": hashlib.sha256(key.encode()).hexdigest(),
                  "call_id": len(self.calls), "model": "fixture-version", "cached": False,
                  "usage": {"total_tokens": 100}, "charged_cost_cny": .001}
        self.cache[key] = copy.deepcopy(result)
        return result


def evaluation(card):
    assert card["evaluation_cache_key"]
    return {"quality_passed": True, "metrics": {"ic": .02, "rank_ic": .03, "ic_ir": .2,
            "rank_ic_ir": .3, "mi": .04, "coverage": .9, "valid_days": 100}}


def test_exact_budget_roles_real_parents_and_formula_free_control(tmp_path, config):
    api = FakeAPI()
    result = mine(api, evaluation, tmp_path, config=config)
    assert len(ROLES) == 21 and len(LAYERS) == 7
    assert result["counts"] == {"slots": 77, "accepted": 77, "rejected": 0,
                               "proposal_attempts": 77, "repair_slots": 0}
    assert [len(result["arms"][x]) for x in ("initial", "fresh", "evolved")] == [21, 49, 49]
    assert result["arms"]["fresh"][:21] == result["arms"]["evolved"][:21] == result["arms"]["initial"]
    evaluated_ids = set()
    for card in result["cards"].values():
        assert card["provider"] == "DeepSeek official API"
        assert card["attempts"][0]["provider_version"] == "fixture-version"
        for parent in card["parent_ids"]:
            assert parent in evaluated_ids and parent in result["arms"]["evolved"]
        evaluated_ids.add(card["id"])
    for messages, kwargs in api.calls:
        payload = json.loads(messages[1]["content"])
        if payload["operation"] == "fresh":
            assert "parents" not in payload
            assert "expression" not in json.dumps(payload["training_feedback"])
        assert kwargs["max_tokens"] == 2048
    json.dumps(result, allow_nan=False)


def test_completed_resume_has_zero_calls_and_zero_evaluations(tmp_path, config):
    api = FakeAPI()
    first = mine(api, evaluation, tmp_path, config=config)
    def forbidden(*args, **kwargs):
        pytest.fail("Completed resume must not request or evaluate")
    api.request = forbidden
    second = mine(api, forbidden, tmp_path, config=config)
    assert first == second


def test_interrupted_callback_uses_api_cache_and_stable_evaluation_key(tmp_path, config):
    api, keys = FakeAPI(), []
    def stop(card):
        keys.append(card["evaluation_cache_key"])
        raise RuntimeError("fixture interruption")
    with pytest.raises(RuntimeError, match="fixture interruption"):
        mine(api, stop, tmp_path, config=config)
    assert len(api.calls) == 1
    def resumed(card):
        if not len(keys) > 1:
            keys.append(card["evaluation_cache_key"])
        return evaluation(card)
    result = mine(api, resumed, tmp_path, config=config)
    assert keys[0] == keys[1]
    assert len(api.calls) == 77
    assert result["cards"][result["arms"]["initial"][0]]["attempts"][0]["cached"]


@pytest.mark.parametrize("expression", ["rolling_mean(close,121)", "delay(close,121)",
                                          "returns(close,121)", "__import__('os')", "close.shift(-1)"])
def test_dsl_rejects_unbounded_or_executable_expressions(expression):
    with pytest.raises(ValueError):
        validate_card(proposal(expression))


def test_tanh_and_semantic_numeric_spelling_canonicalized():
    first = validate_card(proposal("tanh(returns(close, 120)) + 1"))
    second = validate_card(proposal("(tanh(returns(close,120.0)))+1.0"))
    assert first["expression_hash"] == second["expression_hash"]


def test_invalid_dsl_once_repaired_before_callback(tmp_path, config):
    def hook(messages, kwargs, count):
        if count == 1:
            return proposal("rolling_mean(close, 121)")
        return proposal(f"returns(close,20) + {count} * rank(volume)")
    api = FakeAPI(hook)
    seen = []
    def callback(card):
        assert "121" not in card["expression"]
        seen.append(card)
        return evaluation(card)
    result = mine(api, callback, tmp_path, config=config)
    first = result["cards"][result["arms"]["initial"][0]]
    assert first["repair"] and len(first["attempts"]) == 2
    assert first["attempts"][1]["namespace"] == first["attempts"][0]["namespace"] + ".repair"
    assert len(api.calls) == 78 and len(seen) == 77


def test_repeated_bad_schema_rejected_without_replacement(tmp_path, config):
    api = FakeAPI(lambda *args: {"factors": [proposal()]})
    result = mine(api, lambda _: pytest.fail("Invalid proposals cannot evaluate"), tmp_path, config=config)
    assert result["counts"]["slots"] == 77
    assert result["counts"]["accepted"] == 0
    assert result["counts"]["proposal_attempts"] == 98  # 21+28 fresh each repaired; no evolution parents.
    assert all(result["cards"][i]["reject_reason"] == "no_positive_training_parent"
               for i in result["arms"]["evolved"][21:])


def test_duplicate_expressions_consume_slot_no_repair_or_callback(tmp_path, config):
    api = FakeAPI(lambda *args: proposal("rank(close)"))
    calls = []
    def callback(card):
        calls.append(card)
        return evaluation(card)
    result = mine(api, callback, tmp_path, config=config)
    assert len(calls) == 1 and len(api.calls) == 77
    assert result["counts"]["accepted"] == 1 and result["counts"]["rejected"] == 76
    assert result["counts"]["repair_slots"] == 0
    assert result["cards"][result["arms"]["evolved"][21]]["parent_reused"]


def test_mi_failure_is_explicit_engineering_fallback_not_qualified(tmp_path, config):
    def callback(card):
        result = evaluation(card)
        result["metrics"]["mi"] = .001
        return result
    result = mine(FakeAPI(), callback, tmp_path, config=config)
    assert all(result["cards"][i]["parent_policy"] == "fallback_engineering_positive_rankic"
               for i in result["arms"]["evolved"][21:])
    assert not any(q["qualified"] for q in result["qualifications"]["evolved"]["factors"].values())


@pytest.mark.parametrize("leak", [{"test_rank_ic": .9}, {"nested": {"validation": .9}},
                                  {"split": "test"}, {"testRankIC": .9}, {"oos": .9}])
def test_nontraining_callback_fields_fail_closed(tmp_path, config, leak):
    def callback(card):
        return {**evaluation(card), **leak}
    api = FakeAPI()
    with pytest.raises(TrainingOnlyError):
        mine(api, callback, tmp_path, config=config)
    assert len(api.calls) == 1
    assert not (tmp_path / "mining_result.json").exists()


def test_nan_metrics_serialized_null_never_qualify(tmp_path, config):
    def callback(card):
        result = evaluation(card)
        result["metrics"]["ic"] = np.nan
        return result
    result = mine(FakeAPI(), callback, tmp_path, config=config)
    json.dumps(result, allow_nan=False)
    assert all(c["train_metrics"]["ic"] is None for c in result["cards"].values())
    assert "NaN" not in (tmp_path / "mining_result.json").read_text(encoding="utf-8")


def test_changed_protocol_cannot_silently_resume(tmp_path, config):
    mine(FakeAPI(), evaluation, tmp_path, config=config)
    config["model"]["temperature"] = .6
    with pytest.raises(ValueError, match="different frozen protocol"):
        mine(FakeAPI(), evaluation, tmp_path, config=config)


def test_quality_failure_repair_and_low_coverage_cannot_override_qc(tmp_path, config):
    def callback(card):
        result = evaluation(card)
        result["metrics"]["coverage"] = .1
        return result
    result = mine(FakeAPI(), callback, tmp_path, config=config)
    assert result["counts"]["accepted"] == 0
    assert result["counts"]["repair_slots"] == 49


def test_seed_determinism_for_parent_selection(tmp_path, config):
    first = mine(FakeAPI(), evaluation, tmp_path / "first", config=config)
    second = mine(FakeAPI(), evaluation, tmp_path / "second", config=config)
    assert [x["parent_ids"] for x in first["cards"].values()] == [x["parent_ids"] for x in second["cards"].values()]
    assert first["api_sampling_seed_controlled"] is False


def test_api_failed_slot_retained_and_not_retried(tmp_path, config):
    def hook(messages, kwargs, count):
        if count == 1:
            raise ResearchAPIError("invalid_json_content")
        return proposal(f"rank(close) + {count} * rank(volume)")
    api = FakeAPI(hook)
    result = mine(api, evaluation, tmp_path, config=config)
    first = result["cards"][result["arms"]["initial"][0]]
    assert first["status"] == "rejected_api_error" and first["completed"]
    assert first["attempts"][0]["api_error"] == "invalid_json_content"
    assert not first["repair"] and len(api.calls) == 77
    assert result["counts"]["accepted"] == 76
    assert first["id"] not in result["accepted_ids"]["initial"]
    def forbidden(*args, **kwargs):
        pytest.fail("Known failed slot must not retry on resume")
    api.request = forbidden
    assert mine(api, forbidden, tmp_path, config=config)["counts"] == result["counts"]


@pytest.mark.parametrize("code,http_status", [("budget_exhausted", None), ("reserved_unknown", None),
                                               ("credentials_missing", None), ("http_error", 401)])
def test_fatal_api_error_persisted_and_stops_entire_run(tmp_path, config, code, http_status):
    def hook(*args):
        raise ResearchAPIError(code, http_status)
    api = FakeAPI(hook)
    with pytest.raises(ResearchAPIError):
        mine(api, evaluation, tmp_path, config=config)
    assert len(api.calls) == 1
    record = json.loads(next((tmp_path / "slots").glob("*.json")).read_text(encoding="utf-8"))
    assert record["status"] == "halted_api_error" and not record["completed"]
    assert record["attempts"][0]["api_error"] == code


def test_schema_repair_does_not_echo_unbounded_response(tmp_path, config):
    def hook(messages, kwargs, count):
        if count == 1:
            return {"bogus": "x" * 100_000}
        if count == 2:
            assert len(json.dumps(messages).encode()) < 24_000
            assert "bogus" not in messages[2]["content"]
        return proposal(f"rank(close) + {count} * rank(volume)")
    result = mine(FakeAPI(hook), evaluation, tmp_path, config=config)
    assert result["counts"]["accepted"] == 77
