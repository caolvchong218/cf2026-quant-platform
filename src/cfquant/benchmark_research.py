"""Versioned, annually retrained benchmark-relative research and replay."""
from dataclasses import replace
from pathlib import Path
import json
import subprocess
import time
import numpy as np
import pandas as pd
from .analytics import performance, reconcile
from .config import Config
from .data import load_market, load_calendar, write_json, digest
from .engine import run_backtest
from .experiment import json_safe
from .factors import panel
from .features import FEATURES
from .models import fit_predict
from .risk import RiskPolicy, build_targets
from .research import cpi_comparison

SIGNALS = ['economic', 'rolling_ridge', 'rolling_lightgbm', 'ensemble']
POLICIES = {
    'steady': RiskPolicy(target_volatility=1e6, max_exposure=.95, defensive_scale=1.),
    'managed': RiskPolicy(target_volatility=.20, max_exposure=.95, defensive_scale=.75),
}
NAMES = {'economic': '价值质量主题', 'rolling_ridge': '年度滚动 Ridge',
         'rolling_lightgbm': '年度滚动 LightGBM', 'ensemble': '固定模型融合'}


def annual_windows(calendar, years):
    """First execution consumes the previous close; all labels predate that close."""
    for year in years:
        first = calendar[calendar >= pd.Timestamp(year, 1, 1)][0]
        cutoff = calendar[calendar.get_loc(first) - 1]
        following = calendar[calendar >= pd.Timestamp(year + 1, 1, 1)]
        end = calendar[calendar.get_loc(following[0]) - 1] if len(following) else calendar[-1] + pd.Timedelta(days=1)
        yield year, cutoff, end


def training_mask(frame, eligible, cutoff):
    return (eligible & (frame.date >= cutoff - pd.DateOffset(years=3)) &
            (frame.date < cutoff) & (frame.label_exit < cutoff) & frame.label.notna())


def benchmark_metrics(daily, benchmark, initial_cash=1e6, opening_base=None):
    """Exact same dates, explicit initial price, no interpolation of benchmark gaps."""
    dates = pd.DatetimeIndex(daily.date)
    before = benchmark.index[benchmark.index < dates[0]]
    if not len(before):
        raise ValueError('Benchmark needs a previous-session close')
    base = float(benchmark.loc[before[-1]]) if opening_base is None else float(opening_base)
    prices = benchmark.reindex(dates).to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any() or base <= 0:
        raise ValueError('Missing or invalid benchmark observations')
    bnav = prices / base
    bret = np.diff(np.r_[1., bnav]) / np.r_[1., bnav][:-1]
    snav = daily.nav.to_numpy() / initial_cash
    sret = np.diff(np.r_[1., snav]) / np.r_[1., snav][:-1]
    active = sret - bret
    tracking = float(np.std(active, ddof=1) * np.sqrt(252))
    beta = float(np.cov(sret, bret, ddof=1)[0, 1] / np.var(bret, ddof=1)) if np.var(bret) > 1e-15 else np.nan
    relative = snav / bnav
    return {'benchmark_return': float(bnav[-1] - 1),
            'benchmark_annualized_return': float(bnav[-1] ** (252 / len(dates)) - 1),
            'benchmark_max_drawdown': float(np.max(1 - bnav / np.maximum.accumulate(np.r_[1., bnav])[1:])),
            'excess_return_pp': float(snav[-1] - bnav[-1]),
            'relative_wealth_return': float(relative[-1] - 1),
            'annualized_tracking_error': tracking,
            'information_ratio': float(active.mean() * 252 / tracking) if tracking > 1e-15 else np.nan,
            'beta': beta, 'benchmark_base_price': base,
            'beats_benchmark': bool(snav[-1] > bnav[-1]),
            'active_max_drawdown': float(np.max(1 - relative / np.maximum.accumulate(np.r_[1., relative])[1:]))}


def period_returns(daily, benchmark, frequency='Y'):
    dates = pd.DatetimeIndex(daily.date)
    b = benchmark.reindex(dates).to_numpy() / benchmark.loc[benchmark.index < dates[0]].iloc[-1]
    nav = daily.nav.to_numpy() / 1e6
    group = dates.year.astype(str) if frequency == 'Y' else np.array([f'{d.year} H{1 if d.month <= 6 else 2}' for d in dates])
    rows = []
    for label in pd.unique(group):
        positions = np.flatnonzero(group == label); first, last = positions[0], positions[-1]
        s = nav[last] / (nav[first - 1] if first else 1.) - 1
        r = b[last] / (b[first - 1] if first else 1.) - 1
        rows.append({'period': label, 'start': str(dates[first].date()), 'end': str(dates[last].date()),
                     'strategy_return': s, 'benchmark_return': r, 'excess_return_pp': s-r})
    return pd.DataFrame(rows)


def choose_candidate(table):
    valid = table[(table.total_return > 0) & (table.excess_return_pp > 0) & (table.max_drawdown <= .25)]
    pool = valid if len(valid) else table
    chosen = pool.sort_values(['worst_year_excess', 'excess_return_pp', 'candidate'],
                             ascending=[False, False, True]).iloc[0].candidate
    return chosen, bool(len(valid))


def load_inputs(root):
    extra = root / 'data/private/research_v2'
    market_path = root / 'data/private/mainboard1000_20260918/data/processed/market.csv'
    cache = json.loads((extra / 'features_cache.json').read_text(encoding='utf-8'))
    if cache['output'] != digest(extra / 'features.parquet'):
        raise ValueError('Feature cache hash mismatch; rebuild v2 features first')
    for name, expected in cache['inputs'].items():
        if digest(root / name) != expected:
            raise ValueError(f'Feature input changed: {name}')
    features = pd.read_parquet(extra / 'features.parquet')
    market = load_market(market_path)
    calendar = load_calendar(market_path.parent / 'calendar.csv')
    b = pd.read_csv(extra / 'benchmark.csv')
    b['date'] = pd.to_datetime(b.trade_date.astype(str))
    return features, market, calendar, b.set_index('date').sort_index(), pd.read_csv(extra / 'cpi.csv')


def rolling_scores(frame, calendar, output):
    columns = ['n_' + f for f in FEATURES]
    eligible = frame.eligible & (frame[columns].notna().sum(axis=1) >= 10)
    z = frame[['z_' + f for f in FEATURES]].fillna(0)
    economic = (.5 * z[['z_earnings_yield', 'z_book_yield', 'z_dividend_yield']].mean(axis=1)
                + .25*z.z_quality_roe + .25*z[['z_low_volatility_20', 'z_range_20']].mean(axis=1)).where(eligible)
    scores = frame[['date', 'asset']].copy()
    scores['economic'] = economic
    scores['rolling_ridge'] = np.nan; scores['rolling_lightgbm'] = np.nan
    folds = []
    for year, cutoff, end in annual_windows(calendar, range(2023, 2027)):
        train = training_mask(frame, eligible, cutoff)
        predict = eligible & (frame.date >= cutoff) & (frame.date < end)
        y = (frame.loc[train].groupby('date').label.rank(pct=True) - .5).to_numpy(dtype=np.float32)
        fold = {'year': year, 'cutoff': str(cutoff.date()), 'prediction_end_exclusive': str(end.date()),
                'train_rows': int(train.sum()), 'prediction_rows': int(predict.sum()),
                'max_training_label_exit': str(frame.loc[train, 'label_exit'].max().date()), 'models': {}}
        assert frame.loc[train, 'label_exit'].max() < cutoff
        for model, target, fields in [('ridge', 'rolling_ridge', columns),
                                     ('lightgbm', 'rolling_lightgbm', ['z_'+f for f in FEATURES]+columns)]:
            print(f'Fold {year}: {model}, training {train.sum():,} rows', flush=True)
            x = frame.loc[train, fields].fillna(0).to_numpy(dtype=np.float32)
            xp = frame.loc[predict, fields].fillna(0).to_numpy(dtype=np.float32)
            pred, info = fit_predict(model, x, y, xp)
            scores.loc[predict, target] = pred
            fold['models'][model] = info
        folds.append(fold)
        write_json(output / 'folds.json', json_safe(folds))
    ranks = scores.groupby('date')[['rolling_lightgbm', 'rolling_ridge', 'economic']].rank(pct=True)
    scores['ensemble'] = .5*ranks.rolling_lightgbm + .25*ranks.rolling_ridge + .25*ranks.economic
    scores.to_parquet(output / 'scores.parquet', index=False)
    return scores


def evaluate_candidate(output, name, scores, frame, market, calendar, bench, start, end,
                       policy, cost_scale=1., delay=0):
    folder = output / name
    folder.mkdir(parents=True, exist_ok=True)
    used_scores = scores.shift(delay) if delay else scores
    cfg = Config(name=name, start=start, end=end, holdings=policy.holdings,
                 rebalance_every=policy.rebalance_every, buy_cost=.001*cost_scale, sell_cost=.0015*cost_scale)
    targets, risk = build_targets(used_scores, frame, panel(market, calendar, 'close'), bench.close, start, policy)
    result = run_backtest(market, calendar, used_scores, cfg, target_weights=targets, participation_limit=.01)
    checks = reconcile(result, cfg.initial_cash, cfg.buy_cost, cfg.sell_cost)
    if not checks['passed']:
        raise AssertionError(checks)
    metrics = performance(result.daily, cfg.initial_cash, trades=result.trades)
    metrics.update(benchmark_metrics(result.daily, bench.close))
    metrics['open_base_benchmark_return'] = benchmark_metrics(result.daily, bench.close,
        opening_base=bench.loc[result.daily.date.iloc[0], 'open'])['benchmark_return']
    metrics['mean_cash_weight'] = float((result.daily.cash/result.daily.nav).mean())
    metrics['partial_fill_orders'] = int((result.orders.status == 'liquidity_limited').sum())
    years = period_returns(result.daily, bench.close)
    metrics['worst_year_excess'] = float(years.excess_return_pp.min())
    for what in ['daily', 'trades', 'positions', 'orders']:
        getattr(result, what).to_csv(folder / f'{what}.csv', index=False, float_format='%.12g')
    risk.to_csv(folder / 'risk.csv', index=False)
    years.to_csv(folder / 'years.csv', index=False)
    period_returns(result.daily, bench.close, 'H').to_csv(folder / 'halves.csv', index=False)
    write_json(folder / 'metrics.json', json_safe(metrics)); write_json(folder / 'checks.json', checks)
    write_json(folder / 'config.json', {'backtest': cfg.to_dict(), 'policy': policy.__dict__, 'signal_delay': delay,
                                      'participation_limit': .01})
    return metrics, result


def run_benchmark_research(root):
    started = time.monotonic()
    output = root / 'runs/research_v3'
    if output.exists():
        raise FileExistsError('Preserve existing v3 run; choose a new version before rerunning')
    output.mkdir(parents=True)
    frame, market, calendar, bench, cpi = load_inputs(root)
    source = {p.name: digest(p) for p in sorted((root/'src/cfquant').glob('*.py'))}
    scores = rolling_scores(frame, calendar, output)
    assets = pd.Index(sorted(market.asset.unique()), name='asset')
    panels = {name: scores.pivot(index='date', columns='asset', values=name).reindex(index=calendar, columns=assets)
              for name in SIGNALS}
    rows = []
    for signal in SIGNALS:
        for mode, policy in POLICIES.items():
            candidate = signal+'__'+mode
            print('Validation: '+candidate, flush=True)
            metrics, _ = evaluate_candidate(output, 'validation_'+candidate, panels[signal], frame, market, calendar,
                bench, '2023-01-03', '2024-12-31', policy)
            rows.append({'candidate': candidate, **metrics})
            pd.DataFrame(rows).to_csv(output/'validation.csv', index=False)
    selected, qualified = choose_candidate(pd.DataFrame(rows))
    write_json(output/'selection.json', {'selected': selected, 'validation_gate_passed': qualified,
        'rule': 'positive net and active return, MDD<=25%; worst-year active return then total active return',
        'final_results_evaluated_after_this_selection': True,
        'prior_v2_final_results_already_observed': True, 'candidate_count': 8})
    print('Preselected '+selected+'; now evaluating the fixed 2025–2026 interval.', flush=True)
    rows = []
    for signal in SIGNALS:
        for mode, policy in POLICIES.items():
            candidate = signal+'__'+mode
            print('Historical final check: '+candidate, flush=True)
            metrics, result = evaluate_candidate(output, 'test_'+candidate, panels[signal], frame, market, calendar,
                bench, '2025-01-02', '2026-09-18', policy)
            rows.append({'candidate': candidate, 'selected': candidate==selected, **metrics})
            write_json(output/('test_'+candidate)/'cpi.json', json_safe(cpi_comparison(result.daily, cpi)))
            pd.DataFrame(rows).to_csv(output/'test.csv', index=False)
    signal, mode = selected.split('__')
    for name, cost, delay in [('double_cost', 2., 0), ('delay_one_day', 1., 1)]:
        print('Stress: '+name, flush=True)
        evaluate_candidate(output, 'stress_'+name, panels[signal], frame, market, calendar, bench,
                           '2025-01-02', '2026-09-18', POLICIES[mode], cost, delay)
    write_json(output/'provenance.json', {'git_revision': subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'source_hashes': source, 'protocol_sha256': digest(root/'docs/RESEARCH_PROTOCOL_V3.md'),
        'features_sha256': digest(root/'data/private/research_v2/features.parquet'),
        'scores_sha256': digest(output/'scores.parquet'), 'seconds': time.monotonic()-started})
    write_json(output/'complete.json', {'selected': selected, 'status': 'complete'})
    print('V3 complete: '+selected, flush=True)
