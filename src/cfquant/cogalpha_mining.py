"""Bounded CogAlpha proposal/evolution orchestration; no credential or API creation.

The supplied API owns the shared budget and durable cache. Only training metrics
may enter this module. Completed slots are durable; incomplete evaluations may
be retried after interruption, so callbacks receive an evaluation_cache_key.
This is a small DSL-based mechanism replication, not the paper's full runtime.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd

from .cogalpha import ELITE_FLOORS, METRIC_NAMES, QUALIFIED_FLOORS, qualify_factors
from .factor_lab import DEFAULT_FIELDS, validate_expression
from .research_api import ResearchAPIError


PROMPT_VERSION = "qingxu.cogalpha.mining.v1"
LAYERS = (
    ("Market Structure & Cycle", ("AgentMarketCycle", "AgentVolatilityRegime")),
    ("Extreme Risk & Fragility", ("AgentTailRisk", "AgentCrashPredictor")),
    ("Price-Volume Dynamics", ("AgentLiquidity", "AgentOrderImbalance", "AgentPriceVolumeCoherence", "AgentVolumeStructure")),
    ("Price-Volatility Behavior", ("AgentDailyTrend", "AgentReversal", "AgentRangeVol", "AgentLagResponse", "AgentVolAsymmetry")),
    ("Multi-Scale Complexity", ("AgentDrawdown", "AgentFractal")),
    ("Stability & Regime-Gating", ("AgentRegimeGating", "AgentStability")),
    ("Geometric & Fusion", ("AgentBarShape", "AgentCreative", "AgentComposite", "AgentHerding")),
)
ROLES = tuple((layer, role) for layer, roles in LAYERS for role in roles)
TRAIN_METRICS = (*METRIC_NAMES, "coverage", "valid_days")
SYSTEM_PROMPT = """You are a quantitative factor researcher proposing ONE auditable OHLCV factor.
Output exactly a JSON object with four nonempty string fields: id, label, expression, hypothesis.
No markdown, additional candidates, Python functions, imports, attributes, subscripts, or arbitrary code.
Allowed fields: open, high, low, close, volume. Prices are contemporaneously adjusted research prices;
volume is inversely adjusted to preserve turnover. No fundamentals, order book, news, labels or future data.
Expression grammar: numeric constants, + - * /, unary +/-, parentheses, and these positional operators:
delay(x,n), returns(x,n=1), rolling_mean(x,n), rolling_std(x,n), rolling_min(x,n), rolling_max(x,n),
rolling_corr(x,y,n), rank(x), zscore(x), clip(x,lo,hi), safe_divide(x,y), abs(x), tanh(x).
All explicit lags/windows are integer literals <=120 (delay >=0, all others >=1).
rank and zscore are same-day cross-sectional; rolling operators include the current and past rows only.
Use safe_divide where appropriate; windows need full warmup. Do not create a constant-only factor.
Expression length <=2000, AST nodes <=160, depth <=24. Label <=160, id <=100, hypothesis <=2000 characters.
Only training summaries are feedback. Higher factor values should predict higher next-open-to-10-session
forward returns. Do not infer later market outcomes from knowledge or dates, or promise profitability.
Explain a falsifiable economic hypothesis and limitations. OHLCV proxies are not observations of actual
orders, investor identities, fundamentals, or causal mechanisms. Seven layers/21 roles are organizational
perspectives, not evidence that these economic mechanisms are measured. This bounded DSL experiment differs
from the original paper's generated Python, model, dataset and search scale. Return JSON only."""


class TrainingOnlyError(ValueError):
    """A callback attempted to return non-training information."""


def _finite(value):
    if isinstance(value, Mapping):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, np.generic):
        return _finite(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _encoded(value):
    return json.dumps(_finite(value), ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _hash(value):
    return hashlib.sha256(_encoded(value).encode("utf-8")).hexdigest()


def _save(path, value):
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(_encoded(value) + "\n", encoding="utf-8")
    # Windows scanners can briefly retain a handle on newly written receipts.
    for attempt in range(6):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.02 * (attempt + 1))


def _train_only(value, path="callback"):
    if isinstance(value, Mapping):
        for key, item in value.items():
            # valid_days is a required TRAIN statistic; validation/test never are.
            folded = re.sub(r"([a-z])([A-Z])", r"\1_\2", str(key)).lower()
            if re.search(r"(^|[^a-z])(test|validation|holdout|oos)([^a-z]|$)", folded):
                raise TrainingOnlyError("Non-training field rejected: " + path + "." + str(key))
            if folded in {"split", "period", "dataset_split"} and item != "train":
                raise TrainingOnlyError("Only split=train feedback is permitted")
            _train_only(item, path + "." + str(key))
    elif isinstance(value, (list, tuple)):
        for item in value:
            _train_only(item, path)


def validate_card(proposal):
    """Validate the model schema and restricted DSL, returning canonical hashes."""
    if not isinstance(proposal, dict) or set(proposal) != {"id", "label", "expression", "hypothesis"}:
        raise ValueError("schema_requires_exactly_id_label_expression_hypothesis")
    for key, maximum in (("id", 100), ("label", 160), ("expression", 2000), ("hypothesis", 2000)):
        value = proposal[key]
        if not isinstance(value, str) or not value.strip() or len(value) > maximum:
            raise ValueError("invalid_string_" + key)
    expression = proposal["expression"].strip()
    validate_expression(expression, DEFAULT_FIELDS)
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and (node.func.id == "delay" or node.func.id.startswith("rolling_")
                                         or (node.func.id == "returns" and len(node.args) == 2)):
            # validate_expression has already excluded all nonliteral parameters.
            if abs(float(ast.literal_eval(node.args[-1]))) > 120:
                raise ValueError("lag_or_window_exceeds_120")
    # Whitespace, redundant parentheses and int/float spelling do not buy a slot.
    class NumericCanonical(ast.NodeTransformer):
        def visit_Constant(self, node):
            return ast.copy_location(ast.Constant(float(node.value)), node)
    canonical = ast.dump(NumericCanonical().visit(copy.deepcopy(tree)), include_attributes=False)
    return {**proposal, "expression": expression, "canonical_expression": ast.unparse(tree),
            "expression_hash": hashlib.sha256(canonical.encode()).hexdigest()}


def _configuration(config):
    if config is None:
        config = json.loads((Path(__file__).resolve().parents[2] / "configs/project2_cogalpha.json").read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    expected = {"initial_agents": 21, "generations": 2, "children_per_generation": 14,
                "mutation_per_generation": 7, "crossover_per_generation": 7,
                "fresh_controls_per_generation": 14, "proposal_slots_per_final_arm": 49,
                "shared_initial_pool": True}
    if any(config["experiments"].get(k) != v for k, v in expected.items()):
        raise ValueError("This frozen protocol requires 21 shared initial + 2x14 slots per arm")
    if config["model"].get("max_repairs_per_slot") != 1 or config["model"].get("automatic_retries") != 0:
        raise ValueError("Exactly one repair and no automatic transport retries are supported")
    qualification = config["source_qualification"]
    if qualification["qualified_quantile"] != .65 or qualification["elite_quantile"] != .8:
        raise ValueError("The frozen qualification quantiles are .65 and .80")
    if (qualification["metric_order"] != list(METRIC_NAMES)
            or qualification["qualified_absolute_floors"] != list(QUALIFIED_FLOORS.values())
            or qualification["elite_absolute_floors"] != list(ELITE_FLOORS.values())):
        raise ValueError("Configuration must agree with the five-metric source floors")
    return config


def _qualification(cards):
    rows = [{"id": c["id"], **{k: c["train_metrics"].get(k) for k in METRIC_NAMES}}
            for c in cards if c["status"] == "accepted"]
    frame = pd.DataFrame(rows, columns=["id", *METRIC_NAMES])
    for metric in METRIC_NAMES:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    annotated, thresholds = qualify_factors(frame)
    return {r["id"]: _finite(r) for r in annotated.to_dict("records")}, _finite(thresholds)


def _parents(cards, seed, slot_id):
    qualified, thresholds = _qualification(cards)
    positive = [c for c in cards if c["status"] == "accepted"
                and c["train_metrics"]["rank_ic"] is not None and c["train_metrics"]["rank_ic"] > 0]
    candidates = [c for c in positive if qualified[c["id"]]["qualified"]]
    policy = "qualified_elite_first"
    if not candidates:
        candidates, policy = positive, "fallback_engineering_positive_rankic"
    candidates.sort(key=lambda c: (-bool(qualified[c["id"]]["elite"]),
                                  -c["train_metrics"]["rank_ic"], c["id"]))
    if not candidates:
        return [], "unavailable_no_positive_training_rankic", thresholds
    # Local randomness is deterministic; provider/model sampling is not seeded.
    rng = np.random.default_rng(int(_hash([seed, slot_id, "parents"])[:16], 16))
    weights = 1. / (np.arange(len(candidates), dtype=float) + 1.)
    elite = [c for c in candidates if qualified[c["id"]]["elite"]]
    first = elite[0] if elite else candidates[int(rng.choice(len(candidates), p=weights / weights.sum()))]
    remaining = [c for c in candidates if c["id"] != first["id"]]
    second = remaining[int(rng.integers(len(remaining)))] if remaining else first
    return [first, second], policy, thresholds


def _feedback(cards):
    """Aggregates and metric-only best/worst cases, never expressions or names."""
    good = [c for c in cards if c["status"] == "accepted"]
    usable = [c for c in good if c["train_metrics"]["rank_ic"] is not None]
    usable.sort(key=lambda c: (-c["train_metrics"]["rank_ic"], c["id"]))
    q, thresholds = _qualification(cards)
    return {"split": "train", "observed_slots": len(cards), "quality_passed": len(good),
            "qualified_count": sum(bool(x["qualified"]) for x in q.values()),
            "elite_count": sum(bool(x["elite"]) for x in q.values()), "thresholds": thresholds,
            "best_metric_examples": [c["train_metrics"] for c in usable[:2]],
            "weak_metric_examples": [c["train_metrics"] for c in usable[-2:]]}


def _evaluation(value, min_coverage):
    _train_only(value)
    if not isinstance(value, Mapping) or type(value.get("quality_passed")) is not bool or not isinstance(value.get("metrics"), Mapping):
        raise ValueError("callback requires Boolean quality_passed and metrics mapping")
    if not set(TRAIN_METRICS).issubset(value["metrics"]):
        raise ValueError("callback metrics must include " + ", ".join(TRAIN_METRICS))
    metrics = {}
    for name in TRAIN_METRICS:
        number = value["metrics"][name]
        if number is not None and (isinstance(number, (bool, np.bool_)) or not isinstance(number, (int, float, np.number))):
            raise ValueError("callback metrics must be numeric or null")
        metrics[name] = _finite(float(number)) if number is not None else None
    valid_days = metrics["valid_days"]
    if valid_days is not None and (valid_days < 0 or int(valid_days) != valid_days):
        raise ValueError("valid_days must be a nonnegative integer")
    if valid_days is not None:
        metrics["valid_days"] = int(valid_days)
    coverage = metrics["coverage"]
    if coverage is not None and not 0 <= coverage <= 1:
        raise ValueError("coverage must lie in [0,1]")
    passed = bool(value["quality_passed"] and coverage is not None and coverage >= min_coverage
                  and valid_days is not None and valid_days > 0)
    return passed, metrics


def mine(api, evaluate_callback, output_dir, *, replicate_seed=731, config=None):
    """Run/resume 77 distinct slots (21 shared + 28 evolution + 28 fresh).

    callback(card) -> {quality_passed: bool, metrics: {ic, rank_ic, ic_ir,
    rank_ic_ir, mi, coverage, valid_days}}. It must evaluate TRAIN ONLY and may
    use card['evaluation_cache_key'] for durable idempotence. A budget or
    indeterminate API error stops the run; no hidden retries or budget pools.
    Re-running
    uses completed slots and the supplied API's cache. Failed transport/response
    slots are retained as rejected_api_error without retry; budget, credential
    and indeterminate storage/usage errors halt the run. Invalid DSL/schema or
    training QC gets one repair; duplicate expressions are rejected immediately.
    Deduplication is independent within each arm, including its shared initial
    pool, so one arm cannot invalidate a proposal solely seen in the other.
    """
    if isinstance(replicate_seed, bool) or not isinstance(replicate_seed, (int, np.integer)) or replicate_seed < 0:
        raise ValueError("replicate_seed must be a nonnegative integer")
    config = _configuration(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slots_dir = output_dir / "slots"
    slots_dir.mkdir(exist_ok=True)
    identity = {"schema": PROMPT_VERSION, "seed": int(replicate_seed), "config_hash": _hash(config),
                "system_prompt_hash": _hash(SYSTEM_PROMPT), "provider": config["model"]["provider"],
                "requested_model": config["model"]["name"], "api_sampling_seed_controlled": False,
                "deduplication": "canonical AST within each arm including shared initial pool",
                "callback_resume": "completed slots skipped; incomplete callback uses stable evaluation_cache_key",
                "proposal_slots_per_arm": 49, "shared_initial_slots": 21, "unique_slots": 77}
    manifest = output_dir / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != identity:
            raise ValueError("Existing mining output belongs to a different frozen protocol/seed")
    else:
        _save(manifest, identity)
        _save(output_dir / "prompt_contract.json", {"system": SYSTEM_PROMPT, "version": PROMPT_VERSION,
                                                   "roles": ROLES, "train_dates": config["split"]["train"]})
    cards = {}
    initial, evolved, fresh = [], [], []

    def run_slot(arm, generation, kind, index, layer, role, pool):
        slot_id = f"s{replicate_seed}_{arm}_g{generation}_{kind}_{index:02d}"
        path = slots_dir / (slot_id + ".json")
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("completed"):
                _train_only(previous.get("train_metrics", {}))
                cards[slot_id] = previous
                return slot_id
        parents, parent_policy, thresholds = [], "not_applicable", None
        if kind in {"mutation", "crossover"}:
            parents, parent_policy, thresholds = _parents(pool, replicate_seed, slot_id)
        record = {"id": slot_id, "arm": arm, "generation": generation, "kind": kind,
                  "layer": layer, "role": role, "parent_ids": [p["id"] for p in parents],
                  "parent_policy": parent_policy, "parent_reused": len(parents) == 2 and parents[0]["id"] == parents[1]["id"],
                  "qualification_at_selection": thresholds, "provider": config["model"]["provider"],
                  "requested_model": config["model"]["name"], "prompt_version": PROMPT_VERSION,
                  "attempts": [], "repair": False, "completed": False, "status": "pending",
                  "reject_reason": None, "train_metrics": None}
        if kind in {"mutation", "crossover"} and not parents:
            record.update(completed=True, status="rejected", reject_reason="no_positive_training_parent")
            _save(path, record)
            cards[slot_id] = record
            return slot_id
        payload = {"slot": slot_id, "layer": layer, "role": role, "operation": kind,
                   "train_dates": config["split"]["train"], "training_feedback": _feedback(pool),
                   "instruction": {"initial": "Propose a new economically motivated factor from your assigned perspective.",
                                   "fresh": "Propose a new independent idea. No parent expressions are supplied; do not reconstruct a parent from scores.",
                                   "mutation": "Modify the first parent's mechanism or horizon; use the second as a contrast, retain a clear economic hypothesis.",
                                   "crossover": "Combine useful mechanisms from both parents, avoiding complexity without economic justification."}[kind]}
        if parents:
            payload["parents"] = [{"id": p["id"], "expression": p["card"]["expression"],
                                    "hypothesis": p["card"]["hypothesis"][:300], "train_metrics": p["train_metrics"]} for p in parents]
        _train_only(payload)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": _encoded(payload)}]
        seen = {c["expression_hash"] for c in pool if c.get("expression_hash")}
        error = None
        for attempt in range(2):
            namespace = f"cogalpha.{replicate_seed}.{arm}.g{generation}.{kind}.{index:02d}" + (".repair" if attempt else "")
            if attempt:
                record["repair"] = True
                previous = record["attempts"][-1]["proposal"]
                preview = {k: str(previous.get(k, ""))[:limit] for k, limit in
                           (("id", 100), ("label", 160), ("expression", 2000), ("hypothesis", 200))} if isinstance(previous, dict) else {"invalid_output_type": type(previous).__name__}
                messages = messages[:2] + [{"role": "assistant", "content": _encoded(preview)},
                    {"role": "user", "content": "Repair this ONE rejected candidate. Error code: " + str(error) +
                     ". Keep the same four-field JSON schema and economic motivation. No later-period information is available."}]
            item = {"attempt": attempt, "namespace": namespace, "call_hash": None,
                    "call_id": None, "cached": None, "provider_version": None, "usage": None,
                    "charged_cost_cny": None, "proposal": None, "prompt_hash": _hash(messages),
                    "reject_reason": None, "api_error": None}
            record["attempts"].append(item)
            _save(path, record)
            try:
                response = api.request(messages, temperature=config["model"]["temperature"],
                                       max_tokens=config["model"]["max_output_tokens"], cache_namespace=namespace)
            except ResearchAPIError as exc:
                error = "api_" + exc.code
                item.update(api_error=exc.code, http_status=exc.http_status, reject_reason=error)
                record["reject_reason"] = error
                fatal = exc.code in {"budget_halted", "budget_exhausted", "call_limit_exhausted", "credentials_missing",
                    "credentials_invalid", "credential_in_prompt", "reserved_unknown", "journal_corrupt", "cache_corrupt",
                    "storage_failed", "busy", "invalid_usage", "usage_bound_exceeded"} or exc.http_status in {401, 403}
                record["status"] = "halted_api_error" if fatal else "rejected_api_error"
                record["completed"] = not fatal
                _save(path, record)
                if fatal:
                    raise
                break
            proposal = response.get("json")
            item.update({"call_hash": response.get("request_hash"),
                    "call_id": response.get("call_id"), "cached": response.get("cached"),
                    "provider_version": response.get("model"), "usage": response.get("usage"),
                    "charged_cost_cny": response.get("charged_cost_cny"), "proposal": proposal,
                    "reject_reason": None})
            try:
                card = validate_card(proposal)
            except (ValueError, SyntaxError, TypeError) as exc:
                error = str(exc)
                item["reject_reason"] = error
                if attempt == 0:
                    continue
                break
            record["card"] = {**card, "proposed_id": card["id"], "id": slot_id}
            record["expression_hash"] = card["expression_hash"]
            if card["expression_hash"] in seen:
                error = "duplicate_expression_in_arm"
                item["reject_reason"] = error
                break
            cache_key = _hash([identity["config_hash"], replicate_seed, slot_id, attempt, card["expression_hash"]])
            record["card"]["evaluation_cache_key"] = cache_key
            _save(path, record)
            quality_passed, metrics = _evaluation(evaluate_callback(copy.deepcopy(record["card"])),
                                                   config["evaluation"]["quality_minimum_coverage"])
            item.update(quality_passed=quality_passed, train_metrics=metrics)
            record["train_metrics"] = metrics
            if quality_passed:
                error = None
                record["status"] = "accepted"
                break
            error = "training_quality_check_failed"
            item["reject_reason"] = error
        record.update(completed=True, reject_reason=error)
        if record["status"] not in {"accepted", "rejected_api_error"}:
            record["status"] = "rejected"
        _save(path, record)
        cards[slot_id] = record
        return slot_id

    for index, (layer, role) in enumerate(ROLES):
        initial.append(run_slot("shared", 0, "initial", index, layer, role, [cards[x] for x in initial]))
    evolved.extend(initial)
    fresh.extend(initial)
    for generation in (1, 2):
        for kind in ("mutation", "crossover"):
            for index, (layer, roles) in enumerate(LAYERS):
                role = roles[(generation - 1 + (kind == "crossover")) % len(roles)]
                evolved.append(run_slot("evolved", generation, kind, index, layer, role, [cards[x] for x in evolved]))
        for index in range(14):
            layer, role = ROLES[((generation - 1) * 14 + index) % len(ROLES)]
            fresh.append(run_slot("fresh", generation, "fresh", index, layer, role, [cards[x] for x in fresh]))
    qualifications = {arm: {"factors": _qualification([cards[x] for x in ids])[0],
                            "thresholds": _qualification([cards[x] for x in ids])[1]}
                      for arm, ids in (("initial", initial), ("fresh", fresh), ("evolved", evolved))}
    result = {**identity, "completed": True, "arms": {"initial": initial, "fresh": fresh, "evolved": evolved},
              "cards": cards, "qualifications": qualifications,
              "accepted_ids": {arm: [x for x in ids if cards[x]["status"] == "accepted"]
                               for arm, ids in (("initial", initial), ("fresh", fresh), ("evolved", evolved))},
              "counts": {"slots": len(cards), "accepted": sum(c["status"] == "accepted" for c in cards.values()),
                         "rejected": sum(c["status"] != "accepted" for c in cards.values()),
                         "proposal_attempts": sum(len(c["attempts"]) for c in cards.values()),
                         "repair_slots": sum(c["repair"] for c in cards.values())}}
    result = _finite(result)
    _save(output_dir / "mining_result.json", result)
    return result
