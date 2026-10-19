"""Audit stored CogAlpha evidence without API calls, model fits, or backtests.

Default: read-only readiness, with passed=null. --final requires all expected
output and writes one aggregate report (default: tmp/cogalpha_release_audit.json).
Use --data-root and --private-output for private inputs outside this checkout.
--self-test checks synthetic ledgers and deliberate tampering, without research
data, API calls, or formal experiment reruns. It is not a release certification.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_audit_helpers(root):
    """Import only pure audit helpers, from the checkout actually being audited."""
    import importlib
    global performance, select_diverse, _train_only
    sys.path.insert(0, str(root / "src"))
    helpers = {}
    for name, function in (("analytics", "performance"), ("cogalpha", "select_diverse"),
                           ("cogalpha_mining", "_train_only")):
        module = importlib.import_module("cfquant." + name)
        require(Path(module.__file__).resolve() == (root / "src/cfquant" / (name + ".py")).resolve(),
                "audit helper imported from another checkout")
        helpers[function] = getattr(module, function)
    performance, select_diverse, _train_only = (helpers[name] for name in ("performance", "select_diverse", "_train_only"))

SEEDS = (731, 732, 733)
MODELS = ("rank", "ridge", "lightgbm")
ARMS = ("llm_initial21", "llm_fresh49", "llm_evolved49", "llm_evolved_diverse")
PHASES = ("validation", "test")
SOURCE_FILES = tuple("src/cfquant/" + name for name in (
    "cogalpha.py", "cogalpha_models.py", "cogalpha_mining.py", "cogalpha_study.py",
    "factor_lab.py", "engine.py", "analytics.py", "research_api.py"))
EVALUATOR_FILES = ("cogalpha.py", "factor_lab.py", "factors.py", "cogalpha_study.py")
REQUIRED = ("manifest.json", "protocol.json", "data_quality.json", "budget.json", "lineage.json",
            "proposals.json", "selection.json", "model_metadata.json", "validation.json",
            "performance.csv", "comparison.csv", "factor_metrics.csv", "pool_summary.csv",
            "qualification.csv", "source_reference.json")


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_calculation_sources(root, manifest_hashes, frozen_commit):
    """Byte-exact manifest checks are mandatory even in a .git-free source zip."""
    require(set(manifest_hashes) == set(SOURCE_FILES), "calculation manifest file set changed")
    for name, expected in manifest_hashes.items():
        require(digest(root / name) == expected, "source manifest hash mismatch: " + name)
    git = shutil.which("git")
    if not (root / ".git").exists() or git is None:
        return {"manifest_sha256_checked": True, "files_hashed": len(manifest_hashes),
                "git_comparison_executed": False, "frozen_commit": None,
                "requested_frozen_commit": frozen_commit,
                "git_skip_reason": "source archive has no .git" if not (root / ".git").exists() else "Git executable unavailable",
                "source_comparison": "Manifest SHA256 is byte-exact; frozen Git history was NOT checked."}
    command = [git, "-c", f"safe.directory={root.as_posix()}", "-C", str(root)]
    frozen = subprocess.check_output(command + ["rev-parse", "--verify", frozen_commit + "^{commit}"],
                                     text=True, stderr=subprocess.PIPE).strip()
    for name in (*SOURCE_FILES, "src/cfquant/factors.py", "src/cfquant/config.py", "configs/project2_cogalpha.json"):
        current = (root / name).read_bytes()
        old = subprocess.check_output(command + ["show", frozen + ":" + name], stderr=subprocess.PIPE)
        require(current.replace(b"\r\n", b"\n") == old.replace(b"\r\n", b"\n"), "source differs from frozen commit: " + name)
    return {"manifest_sha256_checked": True, "files_hashed": len(manifest_hashes),
            "git_comparison_executed": True, "frozen_commit": frozen, "git_skip_reason": None,
            "source_comparison": "Git bytes normalized only for CRLF/LF; manifest SHA256 remains byte-exact."}


class AuditError(ValueError):
    pass


def require(condition, message):
    if not bool(condition):
        raise AuditError(message)


def contained(base, name):
    base = Path(base).resolve()
    target = (base / name).resolve()
    require(target.is_relative_to(base), "artifact path escapes its declared folder")
    return target


def close(actual, expected, label, *, atol=1e-10, rtol=1e-10):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(a.shape == b.shape, label + ": shape mismatch")
    require(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True), label + ": numeric mismatch")
    finite = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.


def same(a, b, name):
    if isinstance(a, str) or isinstance(b, str):
        require(str(a) == str(b), name + ": text mismatch")
    else:
        close(np.nan if a is None else a, np.nan if b is None else b, name)


def expected_cases():
    cases = {(0, "fixed16", model, phase, "paper") for model in MODELS for phase in PHASES}
    cases |= {(0, "literal_" + str(i), "rank", phase, "paper") for i in (1, 2, 3) for phase in PHASES}
    cases |= {(seed, arm, model, phase, "paper") for seed in SEEDS for arm in ARMS for model in MODELS for phase in PHASES}
    cases |= {(seed, arm, "lightgbm", "test", "zero") for seed in SEEDS for arm in ("llm_evolved49", "llm_fresh49")}
    require(len(cases) == 90, "audit case definition is inconsistent")
    return cases


def case_key(case):
    seed, arm, model, phase, cost = case
    return f"s{seed}__{arm}__{model}__{phase}__{cost}"


def readiness(public, private):
    missing = ["public/" + name for name in REQUIRED if not (public / name).is_file()]
    completed, counts = {}, {}
    for seed in SEEDS:
        folder = private / "mining" / str(seed)
        counts[str(seed)] = len(list((folder / "slots").glob("*.json")))
        path = folder / "mining_result.json"
        completed[str(seed)] = False
        if path.is_file():
            try:
                result = read_json(path)
                completed[str(seed)] = bool(result.get("completed")) and len(result.get("cards", {})) == 77 and all(
                    slot.get("completed") for slot in result.get("cards", {}).values())
            except (OSError, ValueError):
                pass  # Active/incomplete writes are readiness gaps, not research defects.
        else:
            missing.append(f"private/mining/{seed}/mining_result.json")
    rows = None
    if (public / "performance.csv").is_file():
        try:
            rows = len(pd.read_csv(public / "performance.csv"))
        except (OSError, ValueError, pd.errors.EmptyDataError):
            pass
    ledgers = sum((private / "backtests" / case_key(case) / "record.json").is_file() for case in expected_cases())
    ready = not missing and all(completed.values()) and rows == ledgers == 90
    return {"status": "ready_for_final_audit" if ready else "not_ready", "passed": None, "audit_executed": False,
            "expected_slots": 231, "completed_seeds": completed, "observed_slot_files": counts,
            "expected_portfolio_runs": 90, "observed_performance_rows": rows, "observed_ledger_records": ledgers,
            "missing_artifacts": missing, "network_calls": 0,
            "meaning": "Readiness only. Unfinished work is not an audit failure; no success is certified."}


class ReleaseAudit:
    def __init__(self, root, public, private, api_dir, data_root, frozen_commit):
        self.root, self.public, self.private = root, public, private
        self.api_dir, self.data_root, self.frozen_commit = api_dir, data_root, frozen_commit
        self.protocol = read_json(root / "configs/project2_cogalpha.json")
        self.manifest = read_json(public / "manifest.json")
        self.identity = read_json(private / "identity.json")
        self.metrics = pd.read_csv(public / "performance.csv")
        self.lineage = [read_json(private / "mining" / str(seed) / "mining_result.json") for seed in SEEDS]
        self.selection = read_json(public / "selection.json")
        self.metadata = read_json(public / "model_metadata.json")
        self.budget = read_json(public / "budget.json")
        self.evaluator_hashes = {name: digest(root / "src/cfquant" / name) for name in EVALUATOR_FILES}
        self.checks, self.details = [], {}

    def section(self, name, function):
        started = time.perf_counter()
        try:
            self.details[name] = function()
            self.checks.append({"section": name, "passed": True, "seconds": time.perf_counter() - started})
        except Exception as exc:
            # Never echo private records, prompts, response bodies, or headers.
            reason = str(exc) if isinstance(exc, AuditError) else "audit could not complete: " + type(exc).__name__
            self.checks.append({"section": name, "passed": False, "seconds": time.perf_counter() - started, "reason": reason})

    def identities(self):
        config = self.root / "configs/project2_cogalpha.json"
        require(read_json(self.public / "protocol.json") == self.protocol, "public/frozen protocol mismatch")
        require(tuple(self.protocol["experiments"]["replicate_seeds"]) == SEEDS, "replicate seeds changed")
        require(self.manifest["protocol_sha256"] == self.identity["protocol_sha256"] == digest(config), "protocol byte hash changed")
        basis = {"protocol_sha256": self.identity["protocol_sha256"], "input_sha256": self.identity["input_sha256"]}
        identity = hashlib.sha256(json.dumps(basis, sort_keys=True).encode()).hexdigest()
        require(identity == self.identity["identity"] == self.manifest["identity"], "study identity mismatch")
        require(self.manifest["input_sha256"] == self.identity["input_sha256"], "input identities differ")
        for name, expected in self.identity["input_sha256"].items():
            require(digest(contained(self.data_root, self.protocol["data"][name])) == expected, "input file changed: " + name)
        sources = verify_calculation_sources(self.root, self.manifest["calculation_sha256"], self.frozen_commit)
        files = {p.relative_to(self.public).as_posix() for p in self.public.rglob("*") if p.is_file() and p.name != "manifest.json"}
        require(files == set(self.manifest["files"]), "public manifest must cover every release file exactly")
        for name, expected in self.manifest["files"].items():
            require(digest(contained(self.public, name)) == expected, "public artifact hash mismatch: " + name)
        erratum_path = self.root / "configs/project2_cogalpha_source_erratum.json"
        erratum, source = read_json(erratum_path), read_json(self.public / "source_reference.json")
        require(self.manifest["source_metadata_erratum_sha256"] == digest(erratum_path), "source erratum hash mismatch")
        require(erratum["frozen_config_sha256"] == digest(config), "erratum does not bind frozen configuration")
        require(erratum["incorrect_frozen_value"] == self.protocol["paper_sha256"], "erratum old value mismatch")
        require(erratum["correct_pdf_sha256"] == source["pdf_sha256"] and len(source["pdf_sha256"]) == 64, "corrected PDF identity mismatch")
        require(erratum["requires_new_paid_run"] is False, "erratum scope changed")
        require(self.manifest["independent_blind_test"] is False and self.manifest["exact_paper_table_replication"] is False, "evidence overclaims replication level")
        require(self.manifest["no_test_feedback"] is True and self.protocol["test_feedback_to_llm"] is False, "feedback declaration changed")
        return {**sources, "public_files_hashed": len(files), "input_files_hashed": 3, "study_identity": identity}

    @lru_cache(maxsize=512)
    def factor_receipt(self, factor_id):
        record = read_json(self.private / "factors" / (factor_id + ".json"))
        require(record["identity"] == self.identity["identity"], "factor belongs to another study")
        require(record.get("evaluator_sha256") == self.evaluator_hashes, "factor not replayed with frozen evaluator")
        path = contained(self.private / "factors", record["data_file"])
        require(digest(path) == record["factor_sha256"], "factor parquet hash mismatch")
        require(digest(path.with_suffix(".ic.csv")) == record["daily_ic_sha256"], "training IC hash mismatch")
        dates = pd.read_csv(path.with_suffix(".ic.csv"), usecols=["date"], parse_dates=["date"]).date
        start, end = map(pd.Timestamp, self.protocol["split"]["train"])
        require(dates.ge(start).all() and dates.le(end).all(), "selection IC includes nontraining dates")
        return record

    def slots(self):
        require(read_json(self.public / "lineage.json") == self.lineage, "public lineage differs from retained mining results")
        proposals = read_json(self.public / "proposals.json")
        public = {card["id"]: card for card in proposals}
        require(len(public) == len(proposals), "duplicate public proposal identifiers")
        config_hash = hashlib.sha256(encoded(self.protocol)).hexdigest()
        identifiers, rejected, calls, attempts = set(), Counter(), {}, 0
        for seed, result in zip(SEEDS, self.lineage):
            require(result["seed"] == seed and result["config_hash"] == config_hash, "mining seed/configuration mismatch")
            require(result["completed"] is True and len(result["cards"]) == 77, "incomplete mining slots")
            cards, arms = result["cards"], result["arms"]
            require(len(arms["initial"]) == 21 and len(arms["fresh"]) == len(arms["evolved"]) == 49, "proposal arm budgets changed")
            require(arms["initial"] == arms["fresh"][:21] == arms["evolved"][:21], "initial pool not shared")
            require(set(arms["fresh"][21:]).isdisjoint(arms["evolved"][21:]), "nonshared slots overlap")
            require(set(cards) == set(arms["fresh"]) | set(arms["evolved"]), "dropped/unaccounted mining slots")
            require(identifiers.isdisjoint(cards), "replicate slot identifiers overlap")
            identifiers.update(cards)
            folder = self.private / "mining" / str(seed) / "slots"
            require({p.stem for p in folder.glob("*.json")} == set(cards), "slot files missing or unexpected")
            for slot_id, slot in cards.items():
                require(read_json(folder / (slot_id + ".json")) == slot, "slot receipt differs from lineage")
                require(slot["completed"] is True and slot["id"] == slot_id, "unfinished/misidentified slot")
                require(slot_id in public and public[slot_id]["quality_status"] == slot["status"], "public acceptance/rejection missing")
                require(len(slot["attempts"]) <= 2, "more than one repair")
                attempts += len(slot["attempts"])
                if slot["status"] != "accepted":
                    require(bool(slot["reject_reason"]), "rejected slot lost its reason")
                    rejected[slot["reject_reason"]] += 1
                else:
                    _train_only(slot["train_metrics"])
                    receipt = self.factor_receipt(slot_id)
                    require(receipt["evaluation"]["quality_passed"] is True, "accepted factor quality receipt failed")
                    for metric, expected in slot["train_metrics"].items():
                        same(receipt["evaluation"]["metrics"][metric], expected, slot_id + "/train " + metric)
                for attempt in slot["attempts"]:
                    call_id = attempt.get("call_id")
                    if call_id is not None:
                        require(call_id not in calls, "billed call attributed to multiple attempts")
                        calls[call_id] = attempt
                for parent in slot["parent_ids"]:
                    require(parent in cards and cards[parent]["status"] == "accepted" and parent in arms["evolved"], "invalid/nontraining parent")
            for arm in ("initial", "fresh", "evolved"):
                require(result["accepted_ids"][arm] == [key for key in arms[arm] if cards[key]["status"] == "accepted"], "accepted list changes arm membership")
            require(result["counts"]["slots"] == 77 and result["counts"]["accepted"] == sum(s["status"] == "accepted" for s in cards.values()), "accepted counts mismatch")
            require(result["counts"]["rejected"] + result["counts"]["accepted"] == 77, "rejections omitted from counts")
        require(len(identifiers) == 231, "expected 231 unique retained slots")
        self.attempts_by_call = calls
        return {"slots": 231, "replicates": 3, "proposal_attempts": attempts,
                "rejected_slots": sum(rejected.values()), "rejections_by_reason": dict(rejected), "config_hash": config_hash}

    @lru_cache(maxsize=2)
    def score(self, seed, arm, model):
        return pd.read_parquet(self.private / "models" / f"{seed}_{arm}" / (model + "_scores.parquet"))

    def selections_and_models(self):
        pools = {(0, "fixed16")} | {(0, "literal_" + str(i)) for i in (1, 2, 3)}
        pools |= {(seed, arm) for seed in SEEDS for arm in ARMS}
        selected = {(int(row["replicate"]), row["arm"]): row for row in self.selection}
        metadata = {(int(row["replicate"]), row["arm"]): row for row in self.metadata}
        require(len(self.selection) == len(selected) == 16 and set(selected) == pools, "selection table must retain all 16 pools")
        require(len(self.metadata) == len(metadata) == 16 and set(metadata) == pools, "model metadata pools differ")
        candidates = {(0, "fixed16"): [c["id"] for c in read_json(self.public / "proposals.json") if c.get("source") == "deterministic_reference"]}
        require(len(candidates[(0, "fixed16")]) == 16, "fixed reference proposal count changed")
        for number in (1, 2, 3):
            candidates[(0, "literal_" + str(number))] = ["paper_listing" + str(number)]
        for seed, result in zip(SEEDS, self.lineage):
            for public_arm, source_arm in (("llm_initial21", "initial"), ("llm_fresh49", "fresh"),
                                           ("llm_evolved49", "evolved"), ("llm_evolved_diverse", "evolved")):
                candidates[(seed, public_arm)] = result["accepted_ids"][source_arm]
        matrix_count = 0
        for key in sorted(pools):
            seed, arm = key
            require(selected[key]["training_only"] is True, "selection not declared training-only")
            metrics, histories = [], {}
            for factor_id in candidates[key]:
                receipt = self.factor_receipt(factor_id)
                if receipt["evaluation"]["quality_passed"]:
                    metrics.append({"id": factor_id, **receipt["evaluation"]["metrics"]})
                    if arm == "llm_evolved_diverse":
                        path = contained(self.private / "factors", receipt["data_file"]).with_suffix(".ic.csv")
                        histories[factor_id] = pd.read_csv(path, parse_dates=["date"]).set_index("date").rank_ic
            table = pd.DataFrame(metrics) if metrics else pd.DataFrame(columns=["id", "rank_ic"])
            history = pd.concat(histories, axis=1) if histories else None
            expected = table.id.tolist() if arm.startswith("literal_") else select_diverse(table, history, max_factors=20, max_abs_correlation=.8)
            require(selected[key]["selected"] == expected, "selection differs from frozen training-only rule: " + str(key))
            private_selection = read_json(self.private / f"selection_{seed}_{arm}.json")
            require(private_selection["selected"] == expected, "public/private selection differs")
            folder = self.private / "models" / f"{seed}_{arm}"
            private_meta = read_json(folder / "metadata.json")
            projection = {k: v for k, v in private_meta.items() if k not in {"coefficients", "intercept"}}
            require(metadata[key] == {"replicate": seed, "arm": arm, **projection}, "public metadata differs from approved private projection")
            require("coefficients" not in metadata[key] and "intercept" not in metadata[key], "public metadata contains fitted coefficients")
            require(private_meta["selected"] == expected, "model used a different factor pool")
            literal = arm.startswith("literal_")
            if not literal:
                require(private_meta["annual_retraining"] is False and private_meta["validation_tuning"] is False, "model used later refits/tuning")
                require(private_meta["training_fit_count_per_model"] == int(bool(expected)), "unexpected model fit count")
                require(pd.Timestamp(private_meta["last_label_exit"]) < pd.Timestamp(private_meta["cutoff"]), "training label exits cross cutoff")
                if expected:
                    require(pd.Timestamp(private_meta["last_used_label_exit"]) < pd.Timestamp(private_meta["cutoff"]), "used label exits cross cutoff")
                    require(len(private_meta["training_sha256"]) == 64, "training design identity missing")
                require(private_meta["parameters"]["ridge"]["alpha"] == self.protocol["combination"]["ridge_alpha"], "ridge parameter changed")
                lgb = private_meta["parameters"]["lightgbm"]
                require(lgb["n_estimators"] == 200 and lgb["num_leaves"] == 15 and lgb["max_depth"] == 5
                        and lgb["random_state"] == 20261019 + seed, "LightGBM parameters changed")
            models = ("rank",) if literal else MODELS
            require({p.name for p in folder.glob("*_scores.parquet")} == {model + "_scores.parquet" for model in models}, "prediction artifact set differs")
            for model in models:
                score = self.score(seed, arm, model)
                require(all(dtype == np.dtype("float64") for dtype in score.dtypes), "saved predictions are not float64")
                require(score.index.is_unique and score.index.is_monotonic_increasing and score.columns.is_unique, "prediction axes invalid")
                require(score.loc[score.index < pd.Timestamp(self.protocol["split"]["validation"][0])].isna().all().all(), "prediction emitted before frozen prediction period")
                require(not np.isinf(score.to_numpy()).any(), "infinite saved predictions")
                if not literal:
                    require(int(np.isfinite(score.to_numpy()).sum()) == private_meta["prediction_rows"], "prediction row count differs from metadata")
                matrix_count += 1
        return {"pools": 16, "prediction_matrices": matrix_count, "training_only_selection_recomputed": True,
                "limitation": "Verifies released selection rules/artifacts, not unrecorded human decisions."}

    def budget_receipts(self):
        path = self.api_dir / "receipts.jsonl"
        tail, policy, calls, cache_keys = "0" * 64, None, {}, set()
        for number, line in enumerate(path.read_bytes().splitlines(keepends=True)):
            require(line.endswith(b"\n"), "API journal ends with a partial write")
            event = json.loads(line)
            event_hash = event.pop("event_hash")
            require(event["previous_hash"] == tail and hashlib.sha256(encoded(event)).hexdigest() == event_hash, "API receipt hash chain broken")
            tail = event_hash
            kind = event["event"]
            if kind == "policy":
                require(number == 0 and policy is None, "API policy missing or repeated")
                policy = event
                require(event["budget_micro_cny"] <= 50_000_000 and event["max_calls"] <= 400, "API authorization ceiling exceeded")
                require(event["input_micro_cny_per_token"] == 2 and event["output_micro_cny_per_token"] == 8, "API pricing policy changed")
            elif kind == "reserved":
                require(policy is not None and event["call_id"] == len(calls) + 1, "API reservation sequence invalid")
                require(event["cache_key"] not in cache_keys, "same cached request reserved twice")
                cache_keys.add(event["cache_key"])
                require(event["input_token_bound"] == event["input_bytes"] + 2048, "API input upper bound invalid")
                require(event["estimated_max_micro_cny"] == 2 * event["input_token_bound"] + 8 * event["max_tokens"], "API worst-case reserve invalid")
                require(0 < event["input_bytes"] <= 24000 and 0 < event["max_tokens"] <= 4096, "API request bounds exceeded")
                calls[event["call_id"]] = [event, None]
                body = event["prompt"]
                require(body["model"] == "deepseek-flash" and body["thinking"] == {"type": "disabled"}, "unregistered model/thinking mode")
                if str(event["cache_namespace"]).startswith("cogalpha."):
                    payload = json.loads(body["messages"][1]["content"])
                    _train_only(payload)
                    require(payload["train_dates"] == self.protocol["split"]["train"], "LLM prompt training window changed")
            elif kind == "completed":
                reservation, previous = calls[event["call_id"]]
                require(previous is None and event["request_hash"] == reservation["request_hash"], "API completion/reservation mismatch")
                if event["status"] == "success":
                    result = event["result"]
                    require(hashlib.sha256(encoded(result)).hexdigest() == event["result_hash"], "API response receipt hash mismatch")
                    usage = result["usage"]
                    require(0 <= usage["prompt_tokens"] <= reservation["input_token_bound"] and
                            0 <= usage["completion_tokens"] <= reservation["max_tokens"], "API actual usage exceeds bound")
                    require(event["charged_micro_cny"] == usage["prompt_tokens"] * 2 + usage["completion_tokens"] * 8, "API charge differs from usage")
                else:
                    require(event["charged_micro_cny"] == reservation["estimated_max_micro_cny"], "failed API request released reserve")
                calls[event["call_id"]][1] = event
            else:
                raise AuditError("unknown API journal event")
        require(policy is not None, "API journal missing policy")
        successful = [completion for _, completion in calls.values() if completion is not None and completion["status"] == "success"]
        uncertain = [reservation for reservation, completion in calls.values() if completion is None or completion["status"] != "success"]
        spent = sum(event["charged_micro_cny"] for event in successful)
        reserved = sum(event["estimated_max_micro_cny"] for event in uncertain)
        require(len(calls) <= 400 and spent + reserved <= 50_000_000, "aggregate API budget exceeded")
        require(reserved == 0 and not uncertain, "uncertain API reservation remains")
        require(self.budget["calls"] == len(calls) and self.budget["successful_calls"] == len(successful), "public API counts differ from journal")
        close(self.budget["successful_cost_cny"], spent / 1_000_000, "API successful cost", atol=1e-9)
        close(self.budget["reserved_or_spent_cny"], (spent + reserved) / 1_000_000, "API total cost", atol=1e-9)
        require(self.budget["uncertain_reserved_cny"] == 0 and self.budget["halted"] is False, "public budget has unresolved state")
        attempts = getattr(self, "attempts_by_call", None)
        require(attempts is not None, "slot audit must pass before proposal/API linkage is certified")
        for call_id, attempt in attempts.items():
            require(call_id in calls, "proposal has no durable API receipt")
            reservation, completion = calls[call_id]
            require(attempt["call_hash"] == reservation["request_hash"] and attempt["namespace"] == reservation["cache_namespace"], "proposal points to another API request")
            require(completion is not None and completion["status"] == "success", "proposal lacks successful API completion")
            close(attempt["charged_cost_cny"], completion["charged_cost_cny"], "proposal charge")
        return {"calls": len(calls), "successful_calls": len(successful), "spent_cny": spent / 1_000_000,
                "uncertain_reserved_cny": 0., "unattributed_preflight_calls": len(set(calls) - set(attempts)),
                "journal_sha256": digest(path), "journal_final_hash": tail}

    def ledger(self, key, daily, trades, positions, orders, score, initial, buy, sell, floor):
        dates = pd.DatetimeIndex(daily.date)
        require((daily.cash >= -1e-7).all() and (daily.nav > 0).all(), key + ": negative cash/nonpositive NAV")
        require(not positions.duplicated(["date", "asset"]).any() and positions.units.ge(0).all(), key + ": duplicate/negative positions")
        previous_dates = pd.Series(score.index, index=score.index).shift(1)
        if len(trades):
            require(trades.side.isin(["buy", "sell"]).all() and trades.units.gt(0).all() and trades.notional.gt(0).all(), key + ": invalid fills")
            require(pd.DatetimeIndex(trades.signal_date).equals(pd.DatetimeIndex(previous_dates.reindex(trades.date))), key + ": fills not based on prior-session signal")
            close(trades.notional, trades.units * trades.adjusted_price, key + "/fill notional", atol=1e-7)
            costs = np.maximum(np.where(trades.side.eq("buy"), buy, sell) * trades.notional.to_numpy(), floor)
            close(trades.cost, costs, key + "/fill commission", atol=1e-8)
            sign = np.where(trades.side.eq("buy"), 1., -1.)
            cash_delta = pd.Series(sign * trades.notional.to_numpy() + trades.cost.to_numpy(), index=trades.date).groupby(level=0).sum().reindex(dates, fill_value=0.)
            change = pd.DataFrame({"date": trades.date, "asset": trades.asset, "change": sign * trades.units.to_numpy()})
            balances = change.pivot_table(index="date", columns="asset", values="change", aggfunc="sum", fill_value=0.).reindex(dates, fill_value=0.).cumsum()
        else:
            cash_delta, balances = pd.Series(0., index=dates), pd.DataFrame(index=dates)
        assets = balances.columns.union(pd.Index(positions.asset.unique()))
        units = positions.pivot(index="date", columns="asset", values="units").reindex(index=dates, columns=assets).fillna(0.)
        balances = balances.reindex(columns=assets, fill_value=0.)
        units_error = close(units, balances, key + "/unit balances", atol=1e-7, rtol=1e-10)
        cash_error = close(daily.cash.to_numpy(), initial - cash_delta.cumsum().to_numpy(), key + "/cash balances", atol=1e-6, rtol=1e-10)
        prices_checked = hasattr(self, "market_prices")
        if prices_checked:
            # Market inputs are independently authenticated by the identities section.
            close(trades.adjusted_price, self.quote_values(trades, "open"), key + "/actual opening fill prices")
            close(trades.raw_price, self.quote_values(trades, "raw_open"), key + "/actual raw fill prices")
            close(positions.mark_price, self.quote_values(positions, "mark"), key + "/actual closing marks")
            close(positions.stale_mark, self.quote_values(positions, "stale"), key + "/stale flags")
            if len(trades):
                raw = self.quote_values(trades, "raw_open")
                allowed = np.where(trades.side.eq("buy"), raw < self.quote_values(trades, "up_limit") - 1e-8,
                                   raw > self.quote_values(trades, "down_limit") + 1e-8)
                require(allowed.all(), key + ": a recorded fill violates opening price limits")
            row_numbers = self.market_dates.get_indexer(dates)
            column_numbers = self.market_assets.get_indexer(units.columns)
            require((row_numbers >= 0).all() and (column_numbers >= 0).all(), key + ": held asset/date absent from market")
            opening_marks = self.market_prices["opening_mark"][np.ix_(row_numbers, column_numbers)]
            prior_units = units.shift(1, fill_value=0.).to_numpy()
            prior_cash = np.r_[initial, daily.cash.to_numpy()[:-1]]
            close(daily.opening_nav, prior_cash + np.nansum(prior_units * opening_marks, axis=1),
                  key + "/actual opening valuation", atol=1e-6)
        close(positions.value, positions.units * positions.mark_price, key + "/position value", atol=1e-7)
        market_value = positions.groupby("date").value.sum().reindex(dates, fill_value=0.)
        close(daily.market_value.to_numpy(), market_value.to_numpy(), key + "/market value", atol=1e-6)
        close(daily.nav, daily.cash.to_numpy() + market_value.to_numpy(), key + "/NAV", atol=1e-6)
        fees = trades.groupby("date").cost.sum().reindex(dates, fill_value=0.)
        close(daily.cost.to_numpy(), fees.to_numpy(), key + "/daily fees", atol=1e-8)
        close(daily.cumulative_cost.to_numpy(), fees.cumsum().to_numpy(), key + "/cumulative fees", atol=1e-7)
        close(daily.gross_attribution_nav, daily.nav + daily.cumulative_cost, key + "/gross attribution", atol=1e-6)
        for side, column in (("buy", "buy_notional"), ("sell", "sell_notional")):
            notional = trades.loc[trades.side.eq(side)].groupby("date").notional.sum().reindex(dates, fill_value=0.)
            close(daily[column].to_numpy(), notional.to_numpy(), key + "/" + column, atol=1e-7)
        close(daily.turnover, (daily.buy_notional + daily.sell_notional) / daily.opening_nav, key + "/turnover")
        close(daily["return"], np.diff(np.r_[initial, daily.nav]) / np.r_[initial, daily.nav][:-1], key + "/returns")
        require(daily.holdings_count.le(50).all(), key + ": more than 50 held names")
        close(daily.holdings_count, positions.groupby("date").asset.count().reindex(dates, fill_value=0.).to_numpy(), key + "/holding count")
        if len(orders):
            require(pd.DatetimeIndex(orders.signal_date).equals(pd.DatetimeIndex(previous_dates.reindex(orders.date))), key + ": orders not based on prior-session signal")
        # Verify decisions from saved scores and actual prior holdings, not a new backtest.
        before = units.shift(1, fill_value=0.)
        order_groups = {date: group for date, group in orders.groupby("date")}
        fill_groups = {date: group for date, group in trades.groupby("date")}
        for index, date in enumerate(dates):
            held = set(before.columns[before.iloc[index].to_numpy() > 1e-10])
            signal_date = previous_dates.loc[date]
            values = score.loc[signal_date] if pd.notna(signal_date) else pd.Series(np.nan, index=score.columns)
            ranking = lambda asset: (-float(values[asset]) if np.isfinite(values[asset]) else np.inf, str(asset))
            entrants = sorted([asset for asset in score.columns if asset not in held and np.isfinite(values[asset])],
                              key=ranking)[:5 + max(0, 50 - len(held))]
            combined = sorted(held.union(entrants), key=ranking)
            replace = min(5, max(0, len(combined) - 50))
            sell_names = set(combined[-replace:]).intersection(held) if replace else set()
            day_orders = order_groups.get(date, orders.iloc[:0])
            day_fills = fill_groups.get(date, trades.iloc[:0])
            require(set(day_orders.loc[day_orders.side.eq("sell"), "asset"]) == sell_names, key + ": sell requests disagree with saved ranking")
            sold = set(day_fills.loc[day_fills.side.eq("sell"), "asset"])
            bought = set(day_fills.loc[day_fills.side.eq("buy"), "asset"])
            require(sold.issubset(sell_names) and len(sold) <= 5 and bought.isdisjoint(held), key + ": replacement bound/retained-unit rule violated")
            # The registered protocol has no participation cap: sales are full or blocked.
            allowed_buys = entrants[:max(0, 50 - len(held - sold))]
            require(set(day_orders.loc[day_orders.side.eq("buy"), "asset"]) == set(allowed_buys), key + ": buy requests disagree with actual vacancies")
            require(bought.issubset(allowed_buys), key + ": filled entrant absent from saved prediction ranking")
        return {"cash_error_cny": cash_error, "units_error": units_error, "ranking_and_timing_checked": True,
                "fill_and_mark_prices_checked_against_frozen_market": prices_checked}

    def prepare_market_prices(self):
        fields = ("open", "close", "raw_open", "up_limit", "down_limit")
        market = pd.read_csv(contained(self.data_root, self.protocol["data"]["market"]),
                             usecols=["date", "asset", *fields], dtype={"asset": str}, parse_dates=["date"])
        require(not market.duplicated(["date", "asset"]).any(), "duplicate market date/asset")
        quotes = market.pivot(index="date", columns="asset", values=list(fields)).sort_index()
        self.market_dates = pd.DatetimeIndex(quotes.index)
        self.market_assets = pd.Index(sorted(market.asset.unique()))
        self.market_prices = {field: quotes[field].reindex(columns=self.market_assets).to_numpy() for field in fields}
        closes = pd.DataFrame(self.market_prices["close"], index=self.market_dates)
        # Frozen shared Config default: mark missing closes stale for 20 sessions,
        # then write down; no changes to the execution or research implementation.
        marks = closes.ffill(limit=20).fillna(0.)
        self.market_prices["mark"] = marks.to_numpy()
        self.market_prices["stale"] = closes.isna().to_numpy()
        self.market_prices["opening_mark"] = np.where(np.isfinite(self.market_prices["open"]),
                                                      self.market_prices["open"], marks.shift(1).to_numpy())

    def quote_values(self, frame, field):
        rows = self.market_dates.get_indexer(pd.DatetimeIndex(frame.date))
        columns = self.market_assets.get_indexer(frame.asset)
        require((rows >= 0).all() and (columns >= 0).all(), "ledger quote refers to unknown asset/date")
        return self.market_prices[field][rows, columns]

    def prediction_summary(self, key, row, daily, score):
        start, end = self.protocol["split"][row.phase]
        exits = pd.Series(score.index, index=score.index).shift(-11)
        requested = (score.index >= start) & (score.index <= end)
        realized = exits.notna() & (exits < pd.Timestamp(end) + pd.Timedelta(days=1))
        expected = score.index[requested & realized.to_numpy()]
        require(pd.DatetimeIndex(daily.date).equals(expected), key + ": predictive metrics include crossing/unrealized labels")
        for metric in ("ic", "rank_ic", "mi"):
            values = daily[metric].replace([np.inf, -np.inf], np.nan).dropna()
            same(getattr(row, metric), float(values.mean()) if len(values) else np.nan, key + "/prediction " + metric)
            if metric != "mi":
                deviation = values.std(ddof=1)
                ratio = values.mean() / deviation if np.isfinite(deviation) and deviation > 1e-12 else np.nan
                same(getattr(row, metric + "_ir"), ratio, key + "/prediction IR")
        denominator = daily.eligible_count.sum()
        same(row.coverage, daily.n_pairs.sum() / denominator if denominator else np.nan, key + "/prediction coverage")
        require(row.valid_days == int((daily.ic.notna() & daily.rank_ic.notna()).sum()), key + ": valid prediction day count differs")
        require(row.observations == int(daily.n_pairs.sum()) and row.n_dates == len(daily), key + ": prediction observation counts differ")

    def portfolios(self):
        columns = ["replicate", "arm", "model", "phase", "cost_case"]
        require(len(self.metrics) == 90 and not self.metrics.duplicated(columns).any(), "performance must contain 90 unique cases")
        cases = set(self.metrics[columns].itertuples(index=False, name=None))
        require(cases == expected_cases(), "portfolio result set differs from frozen experiment matrix")
        validation = read_json(self.public / "validation.json")
        require(validation["portfolio_runs"] == self.manifest["portfolio_runs"] == 90 and validation["all_ledgers_passed"] is True, "release did not finish all ledgers")
        checks = {row["key"]: row for row in validation["checks"]}
        require(len(checks) == 90 and set(checks) == {case_key(case) for case in cases}, "public reconciliation set incomplete")
        config = self.protocol["portfolio"]
        initial = float(config["initial_cash_cny"])
        benchmark = pd.read_csv(contained(self.data_root, self.protocol["data"]["benchmark"]))
        benchmark["date"] = pd.to_datetime(benchmark.trade_date.astype(str))
        benchmark = benchmark.set_index("date").sort_index()
        self.prepare_market_prices()
        summaries = []
        for row in self.metrics.itertuples(index=False):
            key = case_key(tuple(getattr(row, column) for column in columns))
            folder = self.private / "backtests" / key
            record = read_json(folder / "record.json")
            require(record["identity"] == self.identity["identity"] and record["checks"]["passed"] is True, key + ": ledger identity/check mismatch")
            require({"key": key, **record["checks"]} == checks[key], key + ": public/private reconciliation mismatch")
            for name, expected in record["row"].items():
                if hasattr(row, name):
                    same(getattr(row, name), expected, key + "/" + name)
            daily, trades, positions, orders = (pd.read_parquet(folder / (name + ".parquet")) for name in ("daily", "trades", "positions", "orders"))
            score = self.score(int(row.replicate), row.arm, row.model)
            dates = pd.DatetimeIndex(daily.date)
            require(dates.is_unique and dates.is_monotonic_increasing, key + ": invalid ledger calendar")
            start, end = self.protocol["split"][row.phase]
            require(dates.equals(score.index[(score.index >= start) & (score.index <= end)]), key + ": phase dates changed")
            buy, sell, floor = (config["buy_cost"], config["sell_cost"], config["minimum_commission_cny"]) if row.cost_case == "paper" else (0., 0., 0.)
            detail = self.ledger(key, daily, trades, positions, orders, score, initial, buy, sell, floor)
            statistics = performance(daily, initial, trades=trades)
            for name, expected in statistics.items():
                same(getattr(row, name), expected, key + "/recomputed " + name)
            close(row.cagr, statistics["annualized_return"], key + "/cagr")
            close(row.annual_turnover, statistics["mean_turnover"] * 252, key + "/annual turnover")
            public = pd.read_csv(contained(self.public, row.daily_file), parse_dates=["date"])
            require(pd.DatetimeIndex(public.date).equals(dates), key + ": public daily dates differ")
            nav = daily.nav.to_numpy() / initial
            close(public.nav, nav, key + "/public NAV")
            close(public.exposure, daily.market_value / daily.nav, key + "/public exposure")
            close(public.drawdown, 1 - nav / np.maximum.accumulate(np.r_[1., nav])[1:], key + "/drawdown")
            b = benchmark.reindex(dates)
            bnav = b.close.to_numpy() / float(b.open.iloc[0])
            close(public.benchmark_nav, bnav, key + "/benchmark")
            breturns = np.diff(np.r_[1., bnav]) / np.r_[1., bnav][:-1]
            excess = daily["return"].to_numpy() - breturns
            close(row.aer, excess.mean() * 252, key + "/AER")
            ir = excess.mean() / excess.std(ddof=1) * np.sqrt(252) if excess.std(ddof=1) > 1e-15 else np.nan
            close(row.information_ratio, ir, key + "/information ratio")
            close(row.benchmark_total_return, bnav[-1] - 1., key + "/benchmark total return")
            prediction = pd.read_csv(folder / "prediction_daily.csv", parse_dates=["date"])
            self.prediction_summary(key, row, prediction, score)
            summaries.append({"key": key, "sessions": len(daily), "fills": len(trades), **detail})
            print(f"Audited stored ledger {len(summaries)}/90: {key}", flush=True)
        return {"runs": 90, "sessions": sum(r["sessions"] for r in summaries), "fills": sum(r["fills"] for r in summaries),
                "max_cash_error_cny": max(r["cash_error_cny"] for r in summaries),
                "max_units_error": max(r["units_error"] for r in summaries),
                "method": "Independent vectorized cash/units reconstruction and prior-score order audit; no engine rerun",
                "runs_checked": summaries}

    def comparisons(self):
        table = pd.read_csv(self.public / "comparison.csv")
        fields = ["replicate", "improved", "baseline", "model"]
        expected = {(seed, improved, baseline, model) for seed in SEEDS for model in MODELS for improved, baseline in
                    (("llm_evolved49", "llm_fresh49"), ("llm_evolved_diverse", "llm_evolved49"))}
        require(len(table) == 18 and not table.duplicated(fields).any(), "comparison table must have 18 unique rows")
        require(set(table[fields].itertuples(index=False, name=None)) == expected, "comparison omitted registered results")
        lookup = self.metrics.set_index(["replicate", "arm", "model", "phase", "cost_case"])
        for row in table.itertuples(index=False):
            baseline = lookup.loc[(row.replicate, row.baseline, row.model, "test", "paper")]
            improved = lookup.loc[(row.replicate, row.improved, row.model, "test", "paper")]
            for metric in ("rank_ic", "ic", "cagr", "sharpe", "max_drawdown", "aer"):
                same(getattr(row, "baseline_" + metric), baseline[metric], "comparison baseline " + metric)
                same(getattr(row, "improved_" + metric), improved[metric], "comparison improved " + metric)
                same(getattr(row, "delta_" + metric), improved[metric] - baseline[metric], "comparison delta " + metric)
        return {"rows": 18, "all_models_and_replicates_retained": True,
                "negative_returns_retained": int((self.metrics.total_return < 0).sum()),
                "negative_evolution_rank_ic_deltas_retained": int(((table.improved == "llm_evolved49") & (table.delta_rank_ic < 0)).sum())}

    def run(self):
        for name, method in (("identities_and_release_hashes", self.identities), ("proposal_slots_and_failures", self.slots),
                             ("training_selection_and_models", self.selections_and_models),
                             ("api_budget_and_receipts", self.budget_receipts), ("ninety_stored_portfolios", self.portfolios),
                             ("eighteen_comparisons", self.comparisons)):
            self.section(name, method)
        passed = all(check["passed"] for check in self.checks)
        return {"schema": "qingxu.project2.cogalpha.release_audit.v1", "status": "passed" if passed else "failed",
                "passed": passed, "audit_executed": True, "created_utc": datetime.now(timezone.utc).isoformat(),
                "checks": self.checks, "details": self.details, "audit_script_sha256": digest(__file__),
                "network_calls": 0, "credentials_read": False, "backtests_rerun": 0, "models_refit": 0,
                "source_artifacts_modified": False,
                "limits": ["Audits stored evidence and its declared lineage, not future investment performance.",
                           "Prediction IC summaries are reconciled to saved daily metrics; price-label correlations are not recomputed.",
                           "Saved predictions are linked to orders via prior-session ranking without model retraining.",
                           "Frozen rules and retained results cannot prove absence of unrecorded human decisions."]}


def self_test(root):
    """Small synthetic executions only; no formal inputs, credentials or API."""
    from unittest.mock import patch
    import importlib
    sys.path.insert(0, str(root / "src"))
    engine = importlib.import_module("cfquant.engine")
    config_module = importlib.import_module("cfquant.config")
    for module in (engine, config_module):
        expected = root / "src/cfquant" / (module.__name__.split(".")[-1] + ".py")
        require(Path(module.__file__).resolve() == expected.resolve(), "self-test imported another checkout")
    checks = []
    for stale in (False, True):
        name = "stale_write_down" if stale else "normal_top50_drop5"
        calendar = pd.bdate_range("2024-01-02", periods=27 if stale else 6)
        codes = [f"S{i:03}" for i in range(60)]
        market = pd.DataFrame([{"date": date, "asset": code, "open": 100., "close": 100.,
                                "raw_open": 100., "raw_close": 100., "up_limit": 110.,
                                "down_limit": 90., "volume": 1_000_000.}
                               for date in calendar for code in codes])
        scores = pd.DataFrame(np.tile(np.arange(60, 0, -1, dtype=float), (len(calendar), 1)),
                              index=calendar, columns=codes)
        if stale:
            mask = market.asset.eq("S000") & market.date.ge(calendar[2])
            market.loc[mask, ["open", "close", "raw_open", "raw_close", "up_limit", "down_limit"]] = np.nan
        else:
            scores.iloc[1:] = np.arange(60, dtype=float)
        config = config_module.Config(start=str(calendar[1].date()), end=str(calendar[-1].date()),
                                      holdings=50, rebalance_every=1, initial_cash=1_000_000.,
                                      buy_cost=.0005, sell_cost=.0005)
        result = engine.run_backtest(market, calendar, scores, config, dropout=5, minimum_commission=5.)
        audit = ReleaseAudit.__new__(ReleaseAudit)
        audit.data_root, audit.protocol = root, {"data": {"market": "tmp/in_memory_synthetic.csv"}}
        with patch.object(pd, "read_csv", return_value=market.copy()):
            audit.prepare_market_prices()
        arguments = (result.trades, result.positions, result.orders, scores, config.initial_cash,
                     config.buy_cost, config.sell_cost, 5.)
        detail = audit.ledger(name, result.daily, *arguments)
        require(detail["fill_and_mark_prices_checked_against_frozen_market"], "synthetic price checks not executed")
        checks.append({"case": name, "passed": True})
        for tamper in ("cash", "mark_price"):
            daily, positions = result.daily.copy(), result.positions.copy()
            if tamper == "cash":
                daily.loc[daily.index[0], "cash"] += 100.
                expected = "cash balances"
            else:
                positions.loc[positions.index[0], "mark_price"] += .5
                expected = "actual closing marks"
            try:
                audit.ledger(name, daily, result.trades, positions, result.orders, scores,
                             config.initial_cash, config.buy_cost, config.sell_cost, 5.)
            except AuditError as exc:
                require(expected in str(exc), "tampered synthetic input rejected for unexpected reason")
            else:
                raise AuditError("tampered synthetic input was accepted")
            checks.append({"case": name + "_reject_changed_" + tamper, "passed": True})
    hashes = {name: digest(root / name) for name in SOURCE_FILES}
    # Simulate a source zip even when this self-test runs in a live checkout.
    with patch.object(Path, "exists", return_value=False), patch.object(shutil, "which", return_value=None):
        archived = verify_calculation_sources(root, hashes, "synthetic_git_not_present")
        require(archived["manifest_sha256_checked"] and not archived["git_comparison_executed"],
                "source archive incorrectly claims Git verification")
        checks.append({"case": "source_zip_hashes_checked_git_explicitly_unchecked", "passed": True})
        bad = {**hashes, SOURCE_FILES[0]: "0" * 64}
        try:
            verify_calculation_sources(root, bad, "synthetic_git_not_present")
        except AuditError as exc:
            require("source manifest hash mismatch" in str(exc), "changed hash rejected for unexpected reason")
        else:
            raise AuditError("source zip accepted changed source hash")
        checks.append({"case": "source_zip_reject_changed_calculation_hash", "passed": True})
    return {"schema": "qingxu.project2.cogalpha.audit_self_test.v1", "mode": "synthetic_self_test",
            "passed": True, "created_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
            "release_audit_executed": False, "research_data_read": False, "credentials_read": False,
            "network_calls": 0, "synthetic_executions": 2, "formal_backtests_rerun": 0, "models_refit": 0,
            "audit_script_sha256": digest(__file__),
            "meaning": "Synthetic audit self-test only; this does not certify any formal research results."}


def report_path(root, supplied, mode, protected):
    default = "cogalpha_audit_self_test.json" if mode == "synthetic_self_test" else "cogalpha_release_audit.json"
    output = (supplied if supplied is not None else root / "tmp" / default).resolve()
    require(output.suffix.lower() == ".json", "audit output must be an explicit JSON report path")
    require(not any(output.is_relative_to(path.resolve()) for path in protected),
            "audit output must not overwrite source, evidence, or private input directories")
    require(output != Path(__file__).resolve(), "audit output cannot overwrite the script")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--public", type=Path)
    parser.add_argument("--private-output", "--private", dest="private_output", type=Path,
                        help="Private formal-run directory (read only).")
    parser.add_argument("--api-dir", type=Path)
    parser.add_argument("--data-root", type=Path, help="Root containing protocol-relative private data paths; defaults to --root.")
    parser.add_argument("--frozen-commit", default="bde19ff33917edcb0fd24f85d215f666d32d08cc")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--final", action="store_true", help="Audit completed stored evidence; no API, model fit or backtest.")
    mode.add_argument("--self-test", action="store_true", help="Pure synthetic audit checks and tampering rejection; no formal manifest required.")
    parser.add_argument("--output", type=Path, help="One aggregate JSON report at this explicit path; defaults to --root/tmp.")
    args = parser.parse_args()
    root = args.root.resolve()
    public = (args.public or root / "evidence/project2_cogalpha").resolve()
    private = (args.private_output or root / "runs/project2_cogalpha_formal").resolve()
    api_dir = (args.api_dir or root / "runs/project2_cogalpha_api").resolve()
    data_root = (args.data_root or root).resolve()
    protected = [root / name for name in ("src", "configs", "scripts", "tests", ".git", "evidence")]
    protected.extend((public, private, api_dir, data_root / "data"))
    if args.self_test:
        output = report_path(root, args.output, "synthetic_self_test", protected)
        result = self_test(root)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"mode": result["mode"], "passed": result["passed"], "checks": len(result["checks"]),
                          "report": str(output), "release_audit_executed": False}, ensure_ascii=False, indent=2))
        return 0
    state = readiness(public, private)
    if not args.final or state["status"] != "ready_for_final_audit":
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0 if not args.final else 2
    output = report_path(root, args.output, "final", protected)
    load_audit_helpers(root)
    audit = ReleaseAudit(root, public, private, api_dir, data_root, args.frozen_commit)
    result = audit.run()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "passed": result["passed"], "report": str(output),
                      "sections": result["checks"]}, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Do not print exception bodies or traceback: private input could occur in them.
        print(json.dumps({"status": "audit_not_completed", "passed": False,
                          "error_type": type(exc).__name__, "network_calls": 0,
                          "reason": "Audit stopped; no complete result certified."}, ensure_ascii=False))
        raise SystemExit(1)
