"""Reproducible CogAlpha mechanism study, separate from the production strategy.

Proposal generation sees training summaries only. All random draws and rejected
slots are retained. Models are frozen before validation, and no test outcome
selects an arm. Public files contain research aggregates, not vendor panels.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

from .analytics import performance, reconcile
from .benchmark_research import load_inputs
from .cogalpha import causal_quality_check, factor_metrics, paper_literal_candidates, qualify_factors, select_diverse
from .cogalpha_models import fit_predict
from .config import Config
from .engine import run_backtest
from .factor_lab import builtin_candidates, evaluate_expression
from .factors import panel
from .research_api import ResearchAPI


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(pending, path)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class Study:
    def __init__(self, root, data_root, output, private, config_path):
        self.root, self.data_root = Path(root), Path(data_root)
        self.output, self.private = Path(output), Path(private)
        self.config_path = Path(config_path)
        self.protocol = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.evaluator_sha256 = {name: digest(self.root / "src/cfquant" / name)
                                 for name in ("cogalpha.py", "factor_lab.py", "factors.py", "cogalpha_study.py")}
        self.output.mkdir(parents=True, exist_ok=True)
        self.private.mkdir(parents=True, exist_ok=True)
        self.inputs = {n: digest(self.data_root / self.protocol["data"][n]) for n in ("market", "features", "benchmark")}
        identity = {"protocol_sha256": digest(config_path), "input_sha256": self.inputs}
        self.identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        old = self.private / "identity.json"
        if old.exists() and json.loads(old.read_text(encoding="utf-8"))["identity"] != self.identity:
            raise ValueError("Study input/protocol changed. Preserve the old run and use a new private/output pair.")
        write_json(old, {"identity": self.identity, **identity})
        write_json(self.output / "protocol.json", self.protocol)
        print("Loading verified local Tushare snapshot; no credentials used for data", flush=True)
        frame, self.market, self.calendar, self.benchmark, _ = load_inputs(self.data_root)
        self.assets = pd.Index(sorted(self.market.asset.unique()), name="asset")
        self.eligible = frame.pivot(index="date", columns="asset", values="eligible").reindex(index=self.calendar, columns=self.assets).fillna(False).astype(bool)
        del frame
        self.panels = {n: panel(self.market, self.calendar, n).reindex(columns=self.assets) for n in ("open", "high", "low", "close", "volume")}
        adj = panel(self.market, self.calendar, "adj_factor").reindex(columns=self.assets)
        ref = panel(self.market, self.calendar, "adjustment_reference").reindex(columns=self.assets)
        self.panels["volume"] = self.panels["volume"] * ref / adj
        self.labels = self.panels["open"].shift(-11) / self.panels["open"].shift(-1) - 1
        self.labels = self.labels.where(self.eligible)
        self.exits = pd.Series(self.calendar, index=self.calendar).shift(-11)
        start, end = self.protocol["split"]["train"]
        self.cutoff = pd.Timestamp(end) + pd.Timedelta(days=1)
        self.train_dates = self.calendar[(self.calendar >= start) & (self.calendar <= end) & (self.exits < self.cutoff)]
        self.predict_dates = self.calendar[self.calendar >= self.protocol["split"]["validation"][0]]
        self.factor_rows, self.performance_rows, self.model_metadata = [], [], []
        self.pool_rows, self.selection, self.cards = [], [], {}
        self.validations, self.qualifications = [], []
        write_json(self.output / "data_quality.json", {
            "rows": len(self.market), "assets": len(self.assets), "sessions": len(self.calendar),
            "first": self.calendar[0], "last": self.calendar[-1], "input_sha256": self.inputs,
            "label_horizon": 10, "training_dates": len(self.train_dates),
            "last_training_label_exit": self.exits.reindex(self.train_dates).max(),
            "training_cutoff": self.cutoff, "missing_ohlcv_not_forward_filled": True,
            "volume_is_adjusted_shares_preserving_turnover": True,
            "survivorship_and_static_universe": True})

    def _factor_paths(self, factor_id, expression_hash=None):
        if not isinstance(factor_id, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,149}", factor_id):
            raise ValueError("Factor identity must be a bounded public identifier")
        folder = self.private / "factors"
        folder.mkdir(exist_ok=True)
        if expression_hash is not None:
            if not re.fullmatch(r"[0-9a-f]{64}", expression_hash):
                raise ValueError("Invalid expression hash")
            stem = factor_id + "__" + expression_hash
            return folder / (stem + ".parquet"), folder / (stem + ".json")
        receipt = folder / (factor_id + ".json")
        data_name = factor_id + ".parquet"
        if receipt.exists():
            metadata = json.loads(receipt.read_text(encoding="utf-8"))
            data_name = metadata.get("data_file", data_name)
            if Path(data_name).name != data_name or not data_name.endswith(".parquet"):
                raise ValueError("Invalid factor checkpoint path")
        return folder / data_name, receipt

    def evaluate(self, card):
        factor_id = card["id"]
        _, alias = self._factor_paths(factor_id)
        expression_hash = hashlib.sha256(card["expression"].encode()).hexdigest()
        cache_hash = hashlib.sha256(json.dumps([expression_hash, self.evaluator_sha256], sort_keys=True).encode()).hexdigest()
        path, receipt = self._factor_paths(factor_id, cache_hash)
        # A repaired proposal keeps its slot ID, but every expression has its own
        # immutable value/metric checkpoint. The short alias selects the attempt.
        if alias.exists():
            previous = json.loads(alias.read_text(encoding="utf-8"))
            if previous["identity"] != self.identity:
                raise ValueError("Factor checkpoint study identity mismatch")
            if (previous["expression_sha256"] == expression_hash
                    and previous.get("evaluator_sha256") == self.evaluator_sha256):
                old_path, _ = self._factor_paths(factor_id)
                if digest(old_path) != previous["factor_sha256"]:
                    raise ValueError("Factor checkpoint content changed")
                if digest(old_path.with_suffix(".ic.csv")) != previous["daily_ic_sha256"]:
                    raise ValueError("Training IC checkpoint content changed")
                self.cards[factor_id] = card
                return previous["evaluation"]
        if receipt.exists():
            record = json.loads(receipt.read_text(encoding="utf-8"))
            if record["identity"] != self.identity or record["expression_sha256"] != expression_hash:
                raise ValueError("Factor checkpoint identity mismatch")
            if digest(path) != record["factor_sha256"]:
                raise ValueError("Factor checkpoint content changed")
            if digest(path.with_suffix(".ic.csv")) != record["daily_ic_sha256"]:
                raise ValueError("Training IC checkpoint content changed")
            write_json(alias, record)
            self.cards[factor_id] = card
            return record["evaluation"]
        quality = causal_quality_check(card["expression"], self.panels, self.eligible, self.train_dates)
        try:
            values = evaluate_expression(card["expression"], self.panels).astype("float32")
        except (ValueError, TypeError, KeyError, ArithmeticError, RuntimeError) as exc:
            if quality["passed"]:
                raise RuntimeError("Factor evaluation contradicted its quality check") from exc
            values = pd.DataFrame(np.nan, index=self.calendar, columns=self.assets, dtype="float32")
        values.to_parquet(path)
        # Evaluate the persisted dtype, so a replay uses identical factor values.
        metrics = factor_metrics(values, self.labels, self.eligible,
                                 date_start=self.protocol["split"]["train"][0],
                                 date_end=self.protocol["split"]["train"][1], label_exit=self.exits,
                                 split_end_nextday=str(self.cutoff.date()))
        metrics["daily"].to_csv(path.with_suffix(".ic.csv"), index=False)
        result = {"quality_passed": bool(quality["passed"]), "metrics": clean(metrics["summary"]),
                  "quality": clean(quality), "evaluation_scope": "training_only"}
        record = {"identity": self.identity, "expression_sha256": expression_hash,
                            "factor_sha256": digest(path), "storage_dtype": "float32",
                            "evaluator_sha256": self.evaluator_sha256,
                            "daily_ic_sha256": digest(path.with_suffix(".ic.csv")),
                            "data_file": path.name, "evaluation": result, "card": card}
        write_json(receipt, record)
        write_json(alias, record)
        self.cards[factor_id] = card
        print(f"Factor {factor_id}: quality={result['quality_passed']}, train RankIC={result['metrics'].get('rank_ic')}", flush=True)
        return result

    def factor(self, factor_id):
        path, receipt = self._factor_paths(factor_id)
        metadata = json.loads(receipt.read_text(encoding="utf-8"))
        if digest(path) != metadata["factor_sha256"]:
            raise ValueError("Factor cache changed")
        if metadata.get("evaluator_sha256") != self.evaluator_sha256:
            raise ValueError("Factor evaluator source changed; replay the accepted proposals offline")
        return pd.read_parquet(path).reindex(index=self.calendar, columns=self.assets)

    def register_cards(self, cards):
        for card in cards:
            self.cards[card["id"]] = card

    def register_mining(self, result):
        for factor_id, slot in result["cards"].items():
            card = dict(slot.get("card") or {})
            card.update({"id": factor_id, "quality_status": slot["status"],
                         "parents": slot.get("parent_ids", []),
                         "source": "live_llm", "replicate": result.get("seed"),
                         **{key: slot.get(key) for key in ("layer", "role", "generation", "kind", "parent_policy")}})
            self.cards[factor_id] = card

    def evaluate_pool(self, ids, replicate, arm, slots, *, literal=False):
        candidates = []
        for factor_id in ids:
            try:
                _, receipt = self._factor_paths(factor_id)
                record = json.loads(receipt.read_text(encoding="utf-8"))["evaluation"]
            except FileNotFoundError as exc:
                raise ValueError(f"Accepted factor checkpoint missing: {factor_id}") from exc
            if record["quality_passed"]:
                candidates.append({"id": factor_id, **record["metrics"]})
        metrics = pd.DataFrame(candidates)
        if metrics.empty:
            metrics = pd.DataFrame(columns=["id", "ic", "rank_ic", "ic_ir", "rank_ic_ir", "mi", "coverage", "valid_days"])
        qualified, thresholds = qualify_factors(metrics)
        rank_daily = None
        if arm == "llm_evolved_diverse" and len(metrics):
            for factor_id in metrics.id:
                path, receipt = self._factor_paths(factor_id)
                expected = json.loads(receipt.read_text(encoding="utf-8"))["daily_ic_sha256"]
                if digest(path.with_suffix(".ic.csv")) != expected:
                    raise ValueError("Training IC checkpoint content changed")
            rank_daily = pd.concat({factor_id: pd.read_csv(self._factor_paths(factor_id)[0].with_suffix(".ic.csv"), parse_dates=["date"]).set_index("date").rank_ic
                                    for factor_id in metrics.id}, axis=1)
        selected = list(metrics.id) if literal else select_diverse(metrics, rank_daily, max_factors=20, max_abs_correlation=.8)
        self.selection.append({"replicate": replicate, "arm": arm, "selected": selected, "training_only": True,
                               "paper_qualification_applied_separately": True,
                               "final_selection_rule": "literal" if literal else "positive train RankIC; optional training IC redundancy filter"})
        self.pool_rows.append({"replicate": replicate, "arm": arm, "proposal_slots": slots,
                               "valid_candidates": len(metrics), "qualified_candidates": int(qualified.qualified.sum()),
                               "elite_candidates": int(qualified.elite.sum()), "selected_count": len(selected)})
        q = qualified.copy()
        q["replicate"], q["arm"] = replicate, arm
        self.qualifications.append(q)
        write_json(self.private / f"selection_{replicate}_{arm}.json", {"selected": selected, "thresholds": thresholds})
        # Every surviving candidate is reported; later metrics never alter selection.
        for factor_id in metrics.id:
            values = self.factor(factor_id)
            for phase in ("train", "validation", "test"):
                start, end = self.protocol["split"][phase]
                result = factor_metrics(values, self.labels, self.eligible, date_start=start, date_end=end,
                                        label_exit=self.exits, split_end_nextday=str((pd.Timestamp(end) + pd.Timedelta(days=1)).date()))
                self.factor_rows.append({"id": factor_id, "replicate": replicate, "arm": arm, "phase": phase,
                                         "quality_passed": True, "selected": factor_id in selected, **result["summary"]})
        print(f"Pool {replicate}/{arm}: {len(selected)} selected from {len(metrics)} valid", flush=True)
        factors = {factor_id: self.factor(factor_id) for factor_id in selected}
        if literal:
            score = pd.DataFrame(np.nan, index=self.calendar, columns=self.assets)
            if selected:
                value = factors[selected[0]].where(self.eligible).rank(axis=1, pct=True) - .5
                score.loc[self.predict_dates] = value.loc[self.predict_dates]
            scores = {"rank": score}
            metadata = {"status": "literal_untrained_formula", "selected": selected, "training_rows": 0}
        else:
            scores, metadata = fit_predict(factors, selected, self.labels, self.eligible,
                                          self.train_dates, self.predict_dates, label_exit=self.exits,
                                          cutoff=str(self.cutoff.date()), seed=20261019 + int(replicate))
        self.model_metadata.append({"replicate": replicate, "arm": arm, **metadata})
        model_folder = self.private / "models" / f"{replicate}_{arm}"
        model_folder.mkdir(parents=True, exist_ok=True)
        write_json(model_folder / "metadata.json", metadata)
        for model, score in scores.items():
            # Single literal examples are ranks, not newly trained combinations.
            if literal and model != "rank":
                continue
            # Preserve exact ranking precision used by the execution engine.
            score.to_parquet(model_folder / f"{model}_scores.parquet")
            for phase in ("validation", "test"):
                self.backtest(score, replicate, arm, model, phase, "paper")
                # Zero-fee sensitivity on the primary same-budget pair only.
                if phase == "test" and model == "lightgbm" and arm in {"llm_evolved49", "llm_fresh49"}:
                    self.backtest(score, replicate, arm, model, phase, "zero")
        self.flush()

    def backtest(self, scores, replicate, arm, model, phase, cost_case):
        start, end = self.protocol["split"][phase]
        portfolio = self.protocol["portfolio"]
        buy, sell, minimum = (portfolio["buy_cost"], portfolio["sell_cost"], portfolio["minimum_commission_cny"]) if cost_case == "paper" else (0., 0., 0.)
        key = f"s{replicate}__{arm}__{model}__{phase}__{cost_case}"
        local = self.private / "backtests" / key
        local.mkdir(parents=True, exist_ok=True)
        cfg = Config(name=key, start=start, end=end, factor="cogalpha", holdings=50,
                     rebalance_every=1, initial_cash=portfolio["initial_cash_cny"], buy_cost=buy, sell_cost=sell)
        result = run_backtest(self.market, self.calendar, scores, cfg, dropout=5, minimum_commission=minimum)
        checks = reconcile(result, cfg.initial_cash, buy, sell, minimum_commission=minimum)
        if not checks["passed"]:
            raise ArithmeticError(f"Portfolio reconciliation failed for {key}")
        stats = performance(result.daily, cfg.initial_cash, trades=result.trades)
        prediction = factor_metrics(scores, self.labels, self.eligible, date_start=start, date_end=end,
                                    label_exit=self.exits, split_end_nextday=str((pd.Timestamp(end) + pd.Timedelta(days=1)).date()))
        dates = pd.DatetimeIndex(result.daily.date)
        b = self.benchmark.reindex(dates)
        if b.close.isna().any():
            raise ValueError("Benchmark does not cover portfolio calendar")
        # Same first evaluation open; benchmark is a fee-free price-index proxy.
        start_price = float(b.open.iloc[0])
        benchmark_nav = b.close.to_numpy() / start_price
        benchmark_returns = np.diff(np.r_[1., benchmark_nav]) / np.r_[1., benchmark_nav][:-1]
        excess = result.daily["return"].to_numpy() - benchmark_returns
        nav = result.daily.nav.to_numpy() / cfg.initial_cash
        public = pd.DataFrame({"date": dates, "nav": nav, "benchmark_nav": benchmark_nav,
                               "drawdown": 1 - nav / np.maximum.accumulate(np.r_[1., nav])[1:],
                               "exposure": result.daily.market_value.to_numpy() / result.daily.nav.to_numpy()})
        daily_file = f"daily/{key}.csv"
        (self.output / "daily").mkdir(exist_ok=True)
        public.to_csv(self.output / daily_file, index=False, float_format="%.12g")
        for name in ("daily", "trades", "positions", "orders"):
            getattr(result, name).to_parquet(local / f"{name}.parquet", index=False)
        prediction["daily"].to_csv(local / "prediction_daily.csv", index=False)
        row = {"replicate": replicate, "arm": arm, "model": model, "phase": phase, "cost_case": cost_case,
               "start": str(dates[0].date()), "end": str(dates[-1].date()), **stats,
               "cagr": stats["annualized_return"], "annual_turnover": stats["mean_turnover"] * 252,
               **prediction["summary"], "aer": float(excess.mean() * 252),
               "information_ratio": float(excess.mean() / excess.std(ddof=1) * np.sqrt(252)) if excess.std(ddof=1) > 1e-15 else np.nan,
               "benchmark_total_return": benchmark_nav[-1] - 1, "daily_file": daily_file}
        self.performance_rows.append(row)
        self.validations.append({"key": key, **checks})
        write_json(local / "record.json", {"identity": self.identity, "row": row, "checks": checks})
        print(f"Portfolio {key}: return={stats['total_return']:.4f}, RankIC={row['rank_ic']:.5f}", flush=True)

    def flush(self):
        pd.DataFrame(self.factor_rows).to_csv(self.output / "factor_metrics.csv", index=False)
        pd.DataFrame(self.performance_rows).to_csv(self.output / "performance.csv", index=False)
        pd.DataFrame(self.pool_rows).to_csv(self.output / "pool_summary.csv", index=False)
        if self.qualifications:
            pd.concat(self.qualifications, ignore_index=True).to_csv(self.output / "qualification.csv", index=False)
        write_json(self.output / "selection.json", self.selection)
        write_json(self.output / "proposals.json", list(self.cards.values()))
        public_models = [{key: value for key, value in record.items()
                          if key not in {"coefficients", "intercept"}} for record in self.model_metadata]
        write_json(self.output / "model_metadata.json", public_models)
        write_json(self.output / "validation.json", {"all_ledgers_passed": bool(self.validations) and all(x["passed"] for x in self.validations),
                   "portfolio_runs": len(self.validations), "checks": self.validations,
                   "no_test_feedback": True, "models_frozen_before_validation": True})

    def finalize(self, api_summary, mining_results):
        self.flush()
        write_json(self.output / "budget.json", api_summary)
        write_json(self.output / "lineage.json", mining_results)
        p = pd.DataFrame(self.performance_rows)
        rows = []
        for rep in self.protocol["experiments"]["replicate_seeds"]:
            for improved, baseline in [("llm_evolved49", "llm_fresh49"), ("llm_evolved_diverse", "llm_evolved49")]:
                for model in ("rank", "ridge", "lightgbm"):
                    a = p.query("replicate==@rep and arm==@improved and model==@model and phase=='test' and cost_case=='paper'")
                    b = p.query("replicate==@rep and arm==@baseline and model==@model and phase=='test' and cost_case=='paper'")
                    if len(a) != 1 or len(b) != 1:
                        continue
                    row = {"replicate": rep, "improved": improved, "baseline": baseline, "model": model}
                    for metric in ("rank_ic", "ic", "cagr", "sharpe", "max_drawdown", "aer"):
                        row[f"baseline_{metric}"] = b.iloc[0][metric]
                        row[f"improved_{metric}"] = a.iloc[0][metric]
                        row[f"delta_{metric}"] = a.iloc[0][metric] - b.iloc[0][metric]
                    rows.append(row)
        pd.DataFrame(rows).to_csv(self.output / "comparison.csv", index=False)
        calculation_files = ["cogalpha.py", "cogalpha_models.py", "cogalpha_mining.py", "cogalpha_study.py", "factor_lab.py", "engine.py", "analytics.py", "research_api.py"]
        files = {f.relative_to(self.output).as_posix(): digest(f) for f in sorted(self.output.rglob("*")) if f.is_file() and f.name != "manifest.json"}
        write_json(self.output / "manifest.json", {"schema": "qingxu.project2.cogalpha.evidence.v1", "identity": self.identity,
                   "created_utc": datetime.now(timezone.utc).isoformat(), "data_end": self.protocol["data"]["end"],
                   "input_sha256": self.inputs, "protocol_sha256": digest(self.config_path),
                   "source_metadata_erratum_sha256": digest(self.root / "configs/project2_cogalpha_source_erratum.json")
                       if (self.root / "configs/project2_cogalpha_source_erratum.json").exists() else None,
                   "calculation_sha256": {f"src/cfquant/{f}": digest(self.root / "src/cfquant" / f) for f in calculation_files},
                   "llm_called": api_summary.get("successful_calls", 0) > 1, "model": self.protocol["model"],
                   "portfolio_runs": len(self.performance_rows), "files": files,
                   "historical_development": True, "independent_blind_test": False,
                   "exact_paper_table_replication": False, "no_test_feedback": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--stage", choices=["mine", "evaluate", "all"], default="all")
    args = parser.parse_args()
    root = args.root.resolve()
    study = Study(root, args.data_root, args.output or root / "evidence/project2_cogalpha",
                  args.private_output or root / "runs/project2_cogalpha_formal",
                  args.config or root / "configs/project2_cogalpha.json")
    # One budget pool for all replicas, repairs, diagnostics and resumed stages.
    api = None if args.stage == "evaluate" else ResearchAPI(root / "runs/project2_cogalpha_api", args.key_file)
    from .cogalpha_mining import mine
    results = []
    static = []
    for card in builtin_candidates():
        card = {**card, "id": "fixed_" + card["id"], "source": "deterministic_reference", "generation": 0, "parents": []}
        study.evaluate(card)
        static.append(card)
    literals = paper_literal_candidates()
    for card in literals:
        study.evaluate(card)
    for rep in study.protocol["experiments"]["replicate_seeds"]:
        mining_dir = study.private / "mining" / str(rep)
        if args.stage == "evaluate":
            result = json.loads((mining_dir / "mining_result.json").read_text(encoding="utf-8"))
            expected_config = hashlib.sha256(json.dumps(study.protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            if result["config_hash"] != expected_config:
                raise ValueError("Offline mining protocol changed")
            if not all(slot.get("completed") for slot in result["cards"].values()):
                raise ValueError("Offline evaluation requires completed mining slots")
            for slot in result["cards"].values():
                if slot["status"] == "accepted":
                    evaluation = study.evaluate(slot["card"])
                    if not evaluation["quality_passed"]:
                        raise ValueError("Previously accepted proposal failed offline quality replay")
                    for metric, observed in slot["train_metrics"].items():
                        if metric not in evaluation["metrics"] or observed is None:
                            continue
                        if not np.isclose(evaluation["metrics"][metric], observed, rtol=1e-10, atol=1e-12):
                            raise ValueError(f"Offline training metric changed: {slot['id']}/{metric}")
        else:
            result = mine(api, study.evaluate, mining_dir, replicate_seed=rep, config=study.protocol)
        results.append(result)
        study.register_mining(result)
        if api is not None:
            write_json(study.output / "budget.json", api.summary())
    if args.stage == "mine":
        study.flush()
        write_json(study.output / "lineage.json", results)
        return
    study.evaluate_pool([c["id"] for c in static], 0, "fixed16", 16)
    for i, card in enumerate(literals, 1):
        study.evaluate_pool([card["id"]], 0, f"literal_{i}", 1, literal=True)
    for rep, result in zip(study.protocol["experiments"]["replicate_seeds"], results):
        for source, arm in (("initial", "llm_initial21"), ("fresh", "llm_fresh49"), ("evolved", "llm_evolved49")):
            study.evaluate_pool(result["accepted_ids"][source], rep, arm, len(result["arms"][source]))
        study.evaluate_pool(result["accepted_ids"]["evolved"], rep, "llm_evolved_diverse", 49)
    budget = api.summary() if api is not None else json.loads((study.output / "budget.json").read_text(encoding="utf-8"))
    study.finalize(budget, results)
    print("CogAlpha formal study complete", flush=True)


if __name__ == "__main__":
    main()
