"""Versioned strategy catalogue and one replay path for the interactive app."""
from dataclasses import dataclass, replace
from pathlib import Path
from datetime import datetime, timezone
import json
import pandas as pd
from .analytics import performance, reconcile
from .config import Config
from .data import write_json, digest
from .engine import run_backtest
from .experiment import json_safe
from .factors import panel
from .risk import RiskPolicy, build_targets
from .benchmark_research import POLICIES, NAMES, benchmark_metrics, load_inputs


@dataclass(frozen=True)
class StrategyVersion:
    id: str
    label: str
    version: str
    signal: str
    policy: RiskPolicy
    scores_path: str
    evidence_path: str
    status: str


def catalogue(root: Path):
    result = []
    v3 = root/'evidence/research_v3/decision.json'
    if v3.exists() and (root/'evidence/research_v3/test.csv').exists():
        decision = json.loads(v3.read_text(encoding='utf-8'))
        preferred = decision['selected']
        rows = pd.read_csv(root/'evidence/research_v3/test.csv')
        validation = pd.read_csv(root/'evidence/research_v3/validation.csv').set_index('candidate')
        ordered = [preferred] + [n for n in rows.candidate if n != preferred]
        for name in ordered:
            signal, mode = name.split('__')
            row = rows[rows.candidate == name].iloc[0]
            passed = bool(row.beats_benchmark and row.max_drawdown <= .25)
            status = '历史验收达标' if passed else '历史验收未达标'
            if validation.loc[name,'max_drawdown'] > .25:
                status = '验证回撤超标 · 保留对照'
            result.append(StrategyVersion('v3/'+name, f'V3 · {NAMES[signal]} · '+('稳定仓位' if mode=='steady' else '波动预算'),
                'v3', name, POLICIES[mode], 'runs/research_v3/scores.parquet', 'evidence/research_v3', status))
    v2 = root/'evidence/research_v2/test.csv'
    if v2.exists():
        labels = {'multifactor_raw':'标准化多因子', 'multifactor_neutral':'中性化多因子', 'ridge':'Ridge', 'lightgbm':'LightGBM', 'mlp':'小型MLP'}
        for name, label in labels.items():
            result.append(StrategyVersion('v2/'+name, 'V2 · '+label+' · 原风险组合', 'v2', name, RiskPolicy(),
                'data/private/research_v2/scores.parquet', 'evidence/research_v2',
                '旧版CPI目标预选' if name=='multifactor_raw' else '旧版模型对照'))
    return result


def outcome_status(metrics, max_drawdown=.25):
    reasons = []
    if metrics['total_return'] <= 0:
        reasons.append('累计净收益非正')
    if metrics['max_drawdown'] > max_drawdown:
        reasons.append(f'最大回撤超过{max_drawdown:.0%}研究门槛')
    if 'beats_benchmark' in metrics and not metrics['beats_benchmark']:
        reasons.append('扣费后未超过同期沪深300价格指数')
    return not reasons, reasons


def replay(root, strategy, start, end, holdings=50, rebalance_every=20,
           buy_cost=.001, sell_cost=.0015, initial_cash=1e6):
    if pd.Timestamp(start) < pd.Timestamp('2023-01-03'):
        raise ValueError('模型版本只支持2023-01-03以后的逐年/样本外分数；不能对训练期套用未来模型')
    frame, market, calendar, benchmark, _ = load_inputs(root)
    raw = pd.read_parquet(root/strategy.scores_path)
    column = strategy.signal.split('__')[0]
    scores = raw.pivot(index='date', columns='asset', values=column).reindex(index=calendar,
        columns=pd.Index(sorted(market.asset.unique()), name='asset'))
    formation = calendar[calendar < pd.Timestamp(start)][-1]
    if scores.loc[formation].notna().sum() == 0:
        raise ValueError('所选起始日无可用的已训练策略分数')
    policy = replace(strategy.policy, holdings=int(holdings), rebalance_every=int(rebalance_every))
    cfg = Config(name=strategy.id.replace('/', '_'), start=str(start), end=str(end),
                 holdings=int(holdings), rebalance_every=int(rebalance_every),
                 buy_cost=buy_cost, sell_cost=sell_cost, initial_cash=initial_cash,
                 data_path='data/private/mainboard1000_20260918/data/processed/market.csv',
                 calendar_path='data/private/mainboard1000_20260918/data/processed/calendar.csv')
    targets, risk = build_targets(scores, frame, panel(market, calendar, 'close'), benchmark.close, start, policy)
    result = run_backtest(market, calendar, scores, cfg, target_weights=targets, participation_limit=.01)
    checks = reconcile(result, initial_cash, buy_cost, sell_cost)
    if not checks['passed']:
        raise AssertionError(checks)
    metrics = performance(result.daily, initial_cash, trades=result.trades)
    metrics.update(benchmark_metrics(result.daily, benchmark.close, initial_cash))
    folder = root/'runs/interactive_versions'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder.mkdir(parents=True)
    for name in ['daily','trades','positions','orders']:
        getattr(result,name).to_csv(folder/f'{name}.csv', index=False, float_format='%.12g')
    risk.to_csv(folder/'risk.csv',index=False)
    used = {**cfg.to_dict(), 'strategy_id':strategy.id, 'strategy_label':strategy.label,
            'policy':policy.__dict__, 'score_sha256':digest(root/strategy.scores_path)}
    write_json(folder/'config.json',used);write_json(folder/'checks.json',checks)
    write_json(folder/'metrics.json',json_safe(metrics))
    return result, metrics, folder, used
