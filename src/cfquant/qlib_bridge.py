"""Optional, genuine Qlib provider and expression bridge; no Qlib dependency at import.

The adapter writes the documented calendar/instrument/float32 file layout. All
expression evaluation is delegated to installed Microsoft pyqlib, in a separate
interpreter. Price data, individual features, and providers remain private.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import hashlib
import json
import re
import sys

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExpressionSpec:
    name: str
    label: str
    expression: str
    lookback: int


EXPRESSIONS = (
    ExpressionSpec('momentum_20', '20日动量', '$close/Ref($close,20)-1', 20),
    ExpressionSpec('reversal_5', '5日反转', '1-$close/Ref($close,5)', 5),
    ExpressionSpec('low_volatility_20', '20日低波动', '0-Std($close/Ref($close,1)-1,20)', 20),
    ExpressionSpec('trend_20', '20日均线趋势', '$close/Mean($close,20)-1', 19),
    ExpressionSpec('range_20', '20日低振幅', '0-Mean(($high-$low)/$close,20)', 19),
    ExpressionSpec('volume_trend', '5/20日量趋势', 'Mean($volume,5)/Mean($volume,20)-1', 19),
    ExpressionSpec('intraday_return', '日内收益', '$close/$open-1', 0),
    ExpressionSpec('close_location', '收盘区间位置', '($close-$low)/($high-$low+1e-12)', 0),
)
MARKET_PATH = Path('data/private/mainboard1000_20260918/data/processed/market.csv')
CALENDAR_PATH = MARKET_PATH.with_name('calendar.csv')
RAW_PATH = MARKET_PATH.parent.parent / 'raw'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _safe_json(value):
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe_json(value), ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def qlib_symbol(asset: str) -> str:
    match = re.fullmatch(r'(\d{6})\.(SH|SZ)', asset)
    if not match:
        raise ValueError('Expected a six-digit SH/SZ asset code')
    return match[2] + match[1]


def validate_request(asset_count=20, start='2020-01-02', end='2026-09-18'):
    if isinstance(asset_count, bool) or not isinstance(asset_count, int) or asset_count not in (10, 20, 40):
        raise ValueError('asset_count must be 10, 20 or 40')
    for date in (start, end):
        if not isinstance(date, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', date):
            raise ValueError('Dates must use YYYY-MM-DD')
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    if not pd.Timestamp('2020-01-02') <= first <= last <= pd.Timestamp('2026-09-18'):
        raise ValueError('Select a date interval within 2020-01-02 to 2026-09-18')
    if (last - first).days < 45:
        raise ValueError('At least 45 calendar days are required for diagnostics')
    return asset_count, start, end


def bridge_command(root: Path, asset_count=20, start='2020-01-02', end='2026-09-18'):
    validate_request(asset_count, start, end)
    executable = root / '.venv-qlib' / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    return [str(executable), str(root/'scripts/run_qlib_bridge.py'), '--assets', str(asset_count),
            '--start', start, '--end', end]


def select_assets(available, count):
    """A fixed, outcome-independent engineering sample, equally split by exchange."""
    chosen = []
    for suffix in ('.SH', '.SZ'):
        options = sorted(a for a in set(available) if a.endswith(suffix))
        chosen.extend(options[:count // 2])
    if len(chosen) != count:
        raise ValueError('Insufficient assets in one of the two exchanges')
    return sorted(chosen)


def prepare_source(root: Path, count):
    assets = select_assets(pd.read_csv(root/MARKET_PATH, usecols=['asset']).asset.unique(), count)
    columns = ['date', 'asset', 'open', 'high', 'low', 'close', 'raw_close', 'volume', 'adj_factor', 'adjustment_reference']
    chunks = [c[c.asset.isin(assets)] for c in pd.read_csv(root/MARKET_PATH, usecols=columns, chunksize=200_000)]
    market = pd.concat(chunks, ignore_index=True)
    market['date'] = pd.to_datetime(market.date)
    market['factor'] = market.adj_factor / market.adjustment_reference
    observed_scale = market.close / market.raw_close
    if not np.allclose(observed_scale, market.factor, rtol=2e-10, atol=2e-10):
        raise ValueError('Adjusted close/raw close disagrees with the fixed-reference adjustment factor')
    # Qlib volume is in adjusted units, preserving adjusted price * volume.
    market['raw_volume_shares'] = market.volume
    market['volume'] = market.volume / market.factor
    calendar = pd.DatetimeIndex(pd.read_csv(root/CALENDAR_PATH, parse_dates=['date']).date)
    # Tushare daily amount is in thousand yuan, vol in hundred-share lots.
    # Read only the selected assets' source payloads; no network or credentials.
    rows, amount_sources = [], {}
    for path in sorted((root/RAW_PATH).glob('daily_*.json')):
        with path.open(encoding='utf-8') as f:
            header = f.read(800)
        match = re.search(r'"ts_code"\s*:\s*"(\d{6}\.(?:SH|SZ))"', header)
        if not match or match[1] not in assets:
            continue
        payload = json.loads(path.read_text(encoding='utf-8'))
        table = pd.DataFrame(payload['data']['items'], columns=payload['data']['fields'])
        table = table[['ts_code', 'trade_date', 'amount', 'vol']].rename(columns={'ts_code': 'asset', 'trade_date': 'date'})
        table['date'] = pd.to_datetime(table.date.astype(str))
        rows.append(table)
        amount_sources[path.name] = sha256(path)
    if rows:
        amounts = pd.concat(rows, ignore_index=True)
        if amounts.duplicated(['date', 'asset']).any():
            raise ValueError('Ambiguous duplicate daily amount sources')
        market = market.merge(amounts, on=['date', 'asset'], how='left', validate='one_to_one')
        available = market.vol.notna()
        if not np.allclose(market.loc[available, 'vol'] * 100, market.loc[available, 'raw_volume_shares'], rtol=1e-8, atol=1):
            raise ValueError('Tushare volume units disagree with the normalized snapshot')
        market['vwap'] = market.amount * 1000 / market.raw_volume_shares.where(market.raw_volume_shares > 0) * market.factor
    else:
        market['vwap'] = np.nan
        market['amount'] = np.nan
    return market, calendar, assets, amount_sources


def write_provider(market: pd.DataFrame, calendar: pd.DatetimeIndex, destination: Path):
    """Write a new local provider. Missing sessions stay NaN, never forward filled."""
    if destination.exists():
        raise ValueError('Provider destination must be new; previous runs are preserved')
    calendar = pd.DatetimeIndex(calendar)
    if calendar.empty or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError('Calendar must be nonempty, unique and sorted')
    if market.duplicated(['date', 'asset']).any() or not market.date.isin(calendar).all():
        raise ValueError('Duplicate market keys or dates outside the calendar')
    destination.mkdir(parents=True)
    (destination/'calendars').mkdir()
    (destination/'instruments').mkdir()
    (destination/'calendars/day.txt').write_text('\n'.join(calendar.strftime('%Y-%m-%d'))+'\n', encoding='utf-8')
    instruments = []
    fields = ['open', 'high', 'low', 'close', 'volume', 'factor', 'vwap']
    for asset, group in market.groupby('asset', sort=True):
        symbol = qlib_symbol(asset)
        aligned = group.set_index('date').reindex(calendar)
        missing = aligned.close.isna() | (aligned.volume <= 0)
        folder = destination/'features'/symbol.lower()
        folder.mkdir(parents=True)
        for field in fields:
            values = aligned[field].where(~missing).to_numpy(dtype='<f4')
            # The first little-endian float32 is the calendar start index.
            np.r_[np.array([0], dtype='<f4'), values].astype('<f4').tofile(folder/f'{field}.day.bin')
        instruments.append(f'{symbol}\t{calendar[0]:%Y-%m-%d}\t{calendar[-1]:%Y-%m-%d}')
    (destination/'instruments/all.txt').write_text('\n'.join(instruments)+'\n', encoding='utf-8')
    return {'assets': len(instruments), 'calendar_sessions': len(calendar), 'fields': fields,
            'format': 'little-endian float32 header=start_calendar_index then observations',
            'files': {p.relative_to(destination).as_posix(): sha256(p) for p in sorted(destination.rglob('*')) if p.is_file()}}


def pandas_reference(market, calendar):
    """Independent pandas formulas, explicitly matching Qlib's partial windows."""
    panels = {c: market.pivot(index='date', columns='asset', values=c).reindex(calendar).astype('float32')
              for c in ['open', 'high', 'low', 'close', 'volume']}
    for c in panels:
        panels[c] = panels[c].where(panels['close'].notna() & (panels['volume'] > 0))
    close, volume = panels['close'], panels['volume']
    references = {
        'momentum_20': close/close.shift(20)-1,
        'reversal_5': 1-close/close.shift(5),
        'low_volatility_20': -(close/close.shift(1)-1).rolling(20, min_periods=1).std(ddof=1),
        'trend_20': close/close.rolling(20, min_periods=1).mean()-1,
        'range_20': -((panels['high']-panels['low'])/close).rolling(20, min_periods=1).mean(),
        'volume_trend': volume.rolling(5, min_periods=1).mean()/volume.rolling(20, min_periods=1).mean()-1,
        'intraday_return': close/panels['open']-1,
        'close_location': (close-panels['low'])/(panels['high']-panels['low']+1e-12),
    }
    return references, panels


def compare_frames(actual, expected, atol=2e-6, rtol=2e-5):
    actual = actual.reindex(index=expected.index, columns=expected.columns)
    a, b = actual.to_numpy(dtype=float), expected.to_numpy(dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    availability_mismatches = int((np.isfinite(a) != np.isfinite(b)).sum())
    errors = np.abs(a[mask]-b[mask])
    matched = bool(mask.any() and not availability_mismatches and np.allclose(a[mask], b[mask], atol=atol, rtol=rtol))
    return {'passed': matched, 'compared_values': int(mask.sum()), 'availability_mismatches': availability_mismatches,
            'max_absolute_error': float(errors.max()) if errors.size else None, 'atol': atol, 'rtol': rtol}


def unit_checks(market):
    observed = market.close / market.raw_close
    valid = market[['vwap', 'volume', 'amount']].notna().all(axis=1)
    amount = market.loc[valid, 'amount'] * 1000
    reconstructed = market.loc[valid, 'vwap'] * market.loc[valid, 'volume']
    return {'price_scale_matches_close_over_raw_close': bool(np.allclose(observed, market.factor, rtol=2e-10, atol=2e-10)),
            'max_price_scale_absolute_error': float(np.max(np.abs(observed-market.factor))),
            'tushare_volume': 'vol in 100-share lots; checked vol*100 against normalized raw shares',
            'tushare_amount': 'amount in thousand CNY; amount*1000 is CNY',
            'provider_volume': 'raw shares divided by the same fixed-reference price adjustment ratio',
            'provider_vwap': 'amount*1000/raw shares multiplied by the same fixed-reference price adjustment ratio',
            'amount_checked_rows': int(valid.sum()),
            'max_price_volume_amount_error': float(np.max(np.abs(reconstructed-amount))) if valid.any() else None,
            'vwap_times_adjusted_volume_equals_amount': bool(np.allclose(reconstructed, amount, rtol=1e-10, atol=1e-5)) if valid.any() else None}


def run_bridge(root: Path, count=20, start='2020-01-02', end='2026-09-18'):
    validate_request(count, start, end)
    module_hash = sha256(Path(__file__))
    import qlib
    from qlib.data import D
    from qlib.constant import REG_CN
    from qlib.contrib.data.loader import Alpha158DL
    from .analytics import diagnostics

    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    private = root/'data/private/qlib_bridge'/run_id
    private.mkdir(parents=True)
    print('Reading the existing Tushare snapshot...', flush=True)
    market, calendar, assets, amount_sources = prepare_source(root, count)
    provider = private/'provider'
    provider_manifest = write_provider(market, calendar, provider)
    write_json(private/'provider_manifest.json', provider_manifest)
    market.to_parquet(private/'source_subset.parquet', index=False)
    print('Evaluating real Qlib D.features expressions...', flush=True)
    qlib.init(provider_uri=str(provider), region=REG_CN, kernels=1, expression_cache=None,
              dataset_cache=None, logging_level=30, exp_manager={'class': 'MLflowExpManager',
              'module_path': 'qlib.workflow.expm', 'kwargs': {'uri': (private/'mlruns').as_uri(), 'default_exp_name': 'bridge'}})
    query_start = str(calendar[0].date())
    symbols = [qlib_symbol(a) for a in assets]
    expressions = [s.expression for s in EXPRESSIONS]
    computed = D.features(symbols, expressions, start_time=query_start, end_time=str(calendar[-1].date()), freq='day')
    computed.columns = [s.name for s in EXPRESSIONS]
    computed.to_parquet(private/'expressions.parquet')
    refs, panels = pandas_reference(market, calendar)
    dates = calendar[(calendar >= start) & (calendar <= end)]
    if len(dates) < 20:
        raise ValueError('Fewer than 20 observed sessions in requested interval')
    mapped = {qlib_symbol(a): a for a in assets}
    checks, summaries, daily_rows, group_rows = {}, [], [], []
    opens = market.pivot(index='date', columns='asset', values='open').reindex(calendar)
    labels = opens.shift(-21) / opens.shift(-1) - 1
    masked_outputs = []
    for spec in EXPRESSIONS:
        scores = computed[spec.name].unstack('instrument').rename(columns=mapped).reindex(index=calendar, columns=assets)
        expected = refs[spec.name].reindex(columns=assets)
        checks[spec.name] = compare_frames(scores, expected)
        # Diagnose only full observable windows on the selected dates. Qlib's
        # native min_periods=1 is documented and is not hidden by the adapter.
        complete = panels['close'].notna().rolling(spec.lookback+1).sum().eq(spec.lookback+1)
        complete &= panels['volume'].notna().rolling(spec.lookback+1).sum().eq(spec.lookback+1)
        valid = scores.where(complete).loc[dates].replace([np.inf, -np.inf], np.nan)
        diag = diagnostics(valid, labels.loc[dates], groups=5, min_assets=10)
        summaries.append({'factor': spec.name, 'label': spec.label, 'expression': spec.expression, **diag['summary']})
        daily_rows.append(diag['daily'].assign(factor=spec.name))
        group_rows.append(diag['groups'].assign(factor=spec.name))
        masked_outputs.append(valid.stack().rename(spec.name))
    pd.concat(masked_outputs, axis=1).to_parquet(private/'diagnostic_features.parquet')
    print('Evaluating the official Alpha158 expression configuration...', flush=True)
    alpha_fields, alpha_names = Alpha158DL.get_feature_config({'kbar': {}, 'price': {'windows': [0], 'feature': ['OPEN', 'HIGH', 'LOW', 'VWAP']}, 'rolling': {}})
    alpha = D.features(symbols, alpha_fields, start_time=start, end_time=end, freq='day')
    alpha.columns = alpha_names
    alpha.to_parquet(private/'alpha158.parquet')
    alpha_summary = pd.DataFrame({'name': alpha_names, 'expression': alpha_fields,
                                  'finite_values': np.isfinite(alpha).sum().to_numpy(),
                                  'coverage': np.isfinite(alpha).mean().to_numpy()})
    units = unit_checks(market)
    passed = all(c['passed'] for c in checks.values()) and units['price_scale_matches_close_over_raw_close'] and units['vwap_times_adjusted_volume_equals_amount'] is not False
    packages = {name: metadata.version(name) for name in ('pyqlib', 'numpy', 'pandas', 'scipy', 'pyarrow')}
    result = {'status': 'verified' if passed else 'failed', 'passed': passed, 'run_id': run_id,
              'created_utc': datetime.now(timezone.utc).isoformat(), 'packages': packages,
              'python': sys.version.split()[0], 'assets': count, 'source_rows': len(market),
              'provider_calendar_sessions': len(calendar), 'provider_start': query_start,
              'provider_end': str(calendar[-1].date()), 'start': start, 'end': end,
              'diagnostic_sessions': len(dates), 'factor_count': len(EXPRESSIONS),
              'selection': 'Equal SH/SZ counts, lexicographically first symbols within the fixed 2019 historical pool; engineering sample, not a strategy universe.',
              'label': 'Next open to open 20 sessions later: open[t+21]/open[t+1]-1',
              'native_windows': 'Qlib Rolling uses min_periods=1; Std uses ddof=1. Numeric checks match these semantics; displayed diagnostics additionally require complete lookback windows.',
              'adjustment': 'Existing fixed-reference adjusted OHLC; factor=adj_factor/adjustment_reference; adjusted volume=raw shares/factor; adjusted VWAP=Tushare amount*1000/raw shares*factor.',
              'unit_checks': units,
              'source_market_sha256': sha256(root/MARKET_PATH), 'source_calendar_sha256': sha256(root/CALENDAR_PATH),
              'amount_source_files': amount_sources, 'checks': checks,
              'alpha158': {'configured': len(alpha_names), 'with_finite_values': int((alpha_summary.finite_values > 0).sum()),
                           'rows': len(alpha), 'source': 'qlib.contrib.data.loader.Alpha158DL.get_feature_config',
                           'vwap_coverage': float(market.vwap.notna().mean()), 'trained_strategy': False},
              'module_sha256': module_hash,
              'limitations': ['Small deterministic integration sample; factor IC is descriptive, not a net strategy return.',
                              'Overlapping 20-day labels and reused historical dates; no independent blind validation.',
                              'Alpha158 expressions were generated and executed; the 8 bridge factors have independent pandas checks, not all 158 formulas.',
                              'No replacement, retraining or winner selection of the existing V3 strategy.']}
    target = root/'evidence/qlib_bridge'
    # Preserve previous public receipts rather than overwrite history silently.
    history = target/'history'/run_id
    history.mkdir(parents=True)
    pd.DataFrame(summaries).to_csv(history/'factor_summary.csv', index=False, lineterminator='\n')
    pd.concat(daily_rows, ignore_index=True).to_csv(history/'daily.csv', index=False, float_format='%.12g', lineterminator='\n')
    pd.concat(group_rows, ignore_index=True).to_csv(history/'groups.csv', index=False, float_format='%.12g', lineterminator='\n')
    alpha_summary.to_csv(history/'alpha158_summary.csv', index=False, float_format='%.12g', lineterminator='\n')
    write_json(history/'result.json', result)
    write_json(private/'result.json', result)
    # Only a fully checked run can replace the dashboard's current receipt.
    if passed:
        for name in ['factor_summary.csv', 'daily.csv', 'groups.csv', 'alpha158_summary.csv', 'result.json']:
            (target/name).write_bytes((history/name).read_bytes())
        files = {p.name: sha256(p) for p in sorted(target.iterdir()) if p.is_file() and p.name != 'manifest.json'}
        write_json(target/'manifest.json', {'run_id': run_id, 'files': files, 'private_payloads_published': False})
    print(json.dumps({'passed': passed, 'run_id': run_id, 'assets': count, 'alpha158': result['alpha158']}, ensure_ascii=False), flush=True)
    if not passed:
        raise AssertionError('Qlib/pandas comparisons failed; previous verified dashboard evidence preserved')
    return result
