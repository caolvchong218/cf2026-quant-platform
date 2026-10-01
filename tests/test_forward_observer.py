import json

import pytest

from cfquant.forward_observer import (append_observation, check_observations, freeze_plan,
                                      list_plans, load_plan, observation_summary)


def frozen(root):
    return freeze_plan(root, {"factor": "momentum", "holdings": 10}, "2025-12-31",
                       config={"rebalance_every": 5}, code_identity={"commit": "fixture"},
                       data_identity={"sha256": "fixture_snapshot"}, last_known_trade_date="2025-12-31",
                       frozen_at="2026-01-03T08:00:00+00:00")


def test_freeze_identities_and_idempotent_append_preserve_prior_bytes(tmp_path):
    plan = frozen(tmp_path)
    assert frozen(tmp_path) == plan
    assert list_plans(tmp_path) == [plan]
    assert load_plan(tmp_path, plan["plan_id"]) == plan
    path = tmp_path / plan["plan_id"] / "plan.json"
    original_plan = path.read_bytes()
    record = append_observation(tmp_path, plan["plan_id"], "2026-01-02", "2025-12-31",
                                fills=[{"date": "2026-01-02", "side": "buy", "units": 100}],
                                metrics={"nav": 9995}, observed_at="2026-01-03T09:00:00+00:00")
    accepted_path = tmp_path / plan["plan_id"] / "observations" / "2026-01-02.json"
    original_observation = accepted_path.read_bytes()
    retry = append_observation(tmp_path, plan["plan_id"], "2026-01-02", "2025-12-31",
                               fills=[{"date": "2026-01-02", "side": "buy", "units": 100}],
                               metrics={"nav": 9995}, observed_at="2026-01-04T09:00:00+00:00")
    assert retry == record
    assert path.read_bytes() == original_plan
    assert accepted_path.read_bytes() == original_observation
    with pytest.raises(ValueError, match="conflict"):
        append_observation(tmp_path, plan["plan_id"], "2026-01-02", "2025-12-31", metrics={"nav": 9999})
    assert accepted_path.read_bytes() == original_observation


def test_observer_rejects_pre_cutoff_signals_and_future_unobserved_dates(tmp_path):
    plan = frozen(tmp_path)
    with pytest.raises(ValueError, match="after data_cutoff"):
        append_observation(tmp_path, plan["plan_id"], "2025-12-31", "2025-12-30")
    with pytest.raises(ValueError, match="strictly precede"):
        append_observation(tmp_path, plan["plan_id"], "2026-01-02", "2026-01-02")
    with pytest.raises(ValueError, match="follow signal_asof"):
        append_observation(tmp_path, plan["plan_id"], "2026-01-05", "2026-01-02", fills=[{"date": "2026-01-02"}])
    with pytest.raises(ValueError, match="future date"):
        append_observation(tmp_path, plan["plan_id"], "2026-01-05", "2026-01-02", observed_at="2026-01-03T10:00:00+00:00")
    assert check_observations(tmp_path, plan["plan_id"])["new_days"] == 0


def test_observation_chain_integrity_and_nonchronological_append(tmp_path):
    plan = frozen(tmp_path)
    append_observation(tmp_path, plan["plan_id"], "2026-01-05", "2026-01-02", observed_at="2026-01-05T10:00:00+00:00")
    with pytest.raises(ValueError, match="last recorded"):
        append_observation(tmp_path, plan["plan_id"], "2026-01-02", "2025-12-31")
    append_observation(tmp_path, plan["plan_id"], "2026-01-06", "2026-01-05", observed_at="2026-01-06T10:00:00+00:00")
    checked = check_observations(tmp_path, plan["plan_id"])
    assert checked["new_days"] == 2
    records = checked["observations"]
    assert records[1]["previous_sha256"] == records[0]["record_sha256"]
    path = tmp_path / plan["plan_id"] / "observations" / "2026-01-05.json"
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["payload"]["metrics"] = {"rewritten": True}
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        check_observations(tmp_path, plan["plan_id"])


def test_read_only_summary_distinguishes_historical_and_forward_paper_records(tmp_path):
    plan = frozen(tmp_path)
    before = observation_summary(tmp_path, plan["plan_id"], available_dates=["2025-12-31", "2026-01-02"])
    assert before["status"] == "awaiting_new_sessions"
    assert before["available_new_days"] == 1
    append_observation(tmp_path, plan["plan_id"], "2026-01-02", "2025-12-31", observed_at="2026-01-03T10:00:00+00:00")
    historical = observation_summary(tmp_path, plan["plan_id"])
    assert historical["historical_post_freeze_days"] == 1
    assert historical["status"] == "historical_post_freeze_replay"
    append_observation(tmp_path, plan["plan_id"], "2026-01-05", "2026-01-02", observed_at="2026-01-05T10:00:00+00:00")
    summary = observation_summary(tmp_path, plan["plan_id"], available_dates=["2026-01-05", "2026-01-06"])
    assert summary["new_days"] == 2
    assert summary["historical_post_freeze_days"] == 1
    assert summary["forward_candidate_days"] == 1
    assert summary["available_new_days"] == 1
    assert not summary["genuine_forward_verified"]


def test_plan_rewrite_and_path_traversal_rejected(tmp_path):
    plan = frozen(tmp_path)
    path = tmp_path / plan["plan_id"] / "plan.json"
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["strategy_definition"]["holdings"] = 20
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        load_plan(tmp_path, plan["plan_id"])
    with pytest.raises(ValueError, match="plan_id"):
        load_plan(tmp_path, "../outside")


def test_unspecified_identity_is_reported_and_time_must_be_zoned(tmp_path):
    plan = freeze_plan(tmp_path, {"factor": "fixed"}, "2025-12-31")
    assert not observation_summary(tmp_path, plan["plan_id"])["data_identity_supplied"]
    with pytest.raises(ValueError, match="timezone"):
        freeze_plan(tmp_path, {"factor": "fixed"}, "2025-12-31", frozen_at="2026-01-03")
