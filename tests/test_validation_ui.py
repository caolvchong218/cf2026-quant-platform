"""Validation-page readiness and immutable plans, with no external data calls."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from cfquant.forward_observer import freeze_plan, load_plan
from cfquant.ui_research_validation import (build_freeze_inputs, index_data_readiness,
                                          parse_index_weights_csv, run_index_experiment)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def app_at(root):
    script = 'from pathlib import Path\nfrom cfquant.ui_research_validation import render\nrender(Path(' + repr(str(root)) + '))'
    return AppTest.from_string(script, default_timeout=30).run()


def freeze_fixture(root, version='v4'):
    local = root / 'runs' / ('research_' + version)
    evidence = root / 'evidence' / ('research_' + version)
    candidate = 'mined_rank' if version == 'v4' else 'rolling_lightgbm__managed'
    write_json(root / 'runs/local_data_source.json', {'data_root': str(root)})
    if version == 'v4':
        write_json(root / 'runs/research_v4_location.json', {'output': str(local)})
    write_json(evidence / 'decision.json', {'selected_factors': ['trend'], 'selected': candidate})
    write_json(evidence / 'folds.json', [{'year': 2025, 'directions': {'trend': 1}, 'weights': {'trend': 1.}}])
    write_json(evidence / 'manifest.json', {'data_end': '2025-06-30', 'input_hashes': {'features': 'actual_feature_sha'},
                                           'source_hashes': {'models.py': 'historical_source_sha'},
                                           'output_hashes': {'folds': 'declared_fold_sha'}})
    configuration = {'candidate': candidate, 'backtest': {'name': candidate, 'start': '2025-01-02', 'end': '2025-06-30'},
                     'policy': {'max_exposure': .95}}
    write_json(local / (('historical_' if version == 'v4' else 'test_') + candidate) / 'config.json', configuration)
    if version == 'v4':
        local.mkdir(parents=True,exist_ok=True)
        (local/'scores.parquet').write_bytes(b'fixture-model-scores')
        manifest=json.loads((evidence/'manifest.json').read_text(encoding='utf-8'))
        manifest['output_hashes']['scores']=hashlib.sha256((local/'scores.parquet').read_bytes()).hexdigest()
        write_json(evidence/'manifest.json',manifest)
        write_json(evidence / 'factor_cards.json', [{'id': 'trend', 'expression': 'returns(close, 20)'}])
        write_json(evidence / 'protocol.json', {'rank_direction': 'training-only'})
        write_json(evidence / 'selection.json', {'whitebox_preselected': candidate})
    else:
        write_json(evidence / 'provenance.json', {'source_hashes': {'models.py': 'historical_v3_sha'},
                                                'features_sha256': 'features_sha', 'scores_sha256': 'scores_sha'})
        protocol = root / 'docs/RESEARCH_PROTOCOL_V3.md'
        protocol.parent.mkdir(parents=True)
        protocol.write_text('Annual training with realized labels only.', encoding='utf-8')
    return version + '/' + candidate


@pytest.mark.parametrize('failure',['no_scores','no_declared_hash','stale_local_manifest'])
def test_auxiliary_freeze_blocks_partial_upgrade_or_stale_snapshot(tmp_path,failure):
    strategy=freeze_fixture(tmp_path)
    manifest_path=tmp_path/'evidence/research_v4/manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if failure=='no_scores':
        # Preserve fixture instead of deleting it, as in the user's workflow.
        (tmp_path/'runs/research_v4/scores.parquet').rename(tmp_path/'archived_scores.parquet')
    elif failure=='no_declared_hash':
        manifest['output_hashes'].pop('scores');write_json(manifest_path,manifest)
    else:
        manifest['data_end']='2025-06-29'
        write_json(tmp_path/'runs/research_v4/manifest.json',manifest)
    assert not build_freeze_inputs(tmp_path,strategy)['ready']


def test_empty_validation_page_has_clear_readiness_and_no_fake_performance(tmp_path):
    app = app_at(tmp_path)
    assert not app.exception
    assert app.button(key='forward_freeze').disabled
    assert app.button(key='index_run').disabled
    assert any(tab.label == '指数增强' for tab in app.tabs)
    assert any('尚无实测指数增强' in message.value for message in app.info)
    assert not (tmp_path / 'runs/forward_observations').exists()
    assert not list((tmp_path / 'runs').glob('index_enhancement_*'))


def test_freeze_preserves_formulas_folds_protocol_config_and_code_data_identity(tmp_path):
    selected = freeze_fixture(tmp_path)
    state = build_freeze_inputs(tmp_path, selected)
    assert state['ready'], state['missing']
    plan = freeze_plan(tmp_path / 'plans', state['strategy_definition'], state['data_cutoff'],
                       config=state['config'], code_identity=state['code_identity'], data_identity=state['data_identity'])
    actual = load_plan(tmp_path / 'plans', plan['plan_id'])
    assert actual['strategy_definition']['strategy_id'] == selected
    artifacts = actual['strategy_definition']['artifacts']
    assert artifacts['factor_cards']['content'][0]['expression'] == 'returns(close, 20)'
    assert artifacts['folds']['content'][0]['weights'] == {'trend': 1.}
    assert artifacts['protocol']['content']['rank_direction'] == 'training-only'
    assert actual['config']['policy']['max_exposure'] == .95
    assert actual['code_identity']['current_source_hashes']
    assert actual['code_identity']['research_source_hashes']['models.py'] == 'historical_source_sha'
    assert actual['data_identity']['input_hashes']['features'] == 'actual_feature_sha'
    assert actual['data_identity']['declared_output_hashes']['folds'] == 'declared_fold_sha'
    assert actual['data_identity']['artifact_hashes']['candidate_config']
    write_json(tmp_path / 'evidence/research_v4/folds.json', [])
    assert load_plan(tmp_path / 'plans', plan['plan_id'])['strategy_definition']['artifacts']['folds']['content']


def test_mainline_freeze_uses_actual_markdown_protocol_and_configuration(tmp_path):
    selected = freeze_fixture(tmp_path, 'v3')
    state = build_freeze_inputs(tmp_path, selected)
    assert state['ready'], state['missing']
    assert state['strategy_definition']['protocol_json_available'] is False
    assert state['strategy_definition']['artifacts']['protocol']['path'].endswith('RESEARCH_PROTOCOL_V3.md')
    assert 'Annual training' in state['strategy_definition']['artifacts']['protocol']['content']
    assert state['config']['candidate'] == 'rolling_lightgbm__managed'
    app = app_at(tmp_path)
    assert not app.exception
    assert not app.button(key='forward_freeze').disabled
    app.button(key='forward_freeze').click().run()
    assert not app.exception
    paths = list((tmp_path / 'runs/forward_observations').glob('*/plan.json'))
    assert len(paths) == 1
    frozen = json.loads(paths[0].read_text(encoding='utf-8'))
    assert frozen['config']['candidate'] == 'rolling_lightgbm__managed'
    assert frozen['code_identity']['current_source_hashes']


def test_missing_candidate_configuration_or_inputs_cannot_freeze(tmp_path):
    selected = freeze_fixture(tmp_path)
    (tmp_path / 'runs/research_v4/historical_mined_rank/config.json').rename(tmp_path / 'runs/research_v4/old_config.json')
    state = build_freeze_inputs(tmp_path, selected)
    assert not state['ready'] and '本地候选 config.json' in state['missing']
    assert not build_freeze_inputs(tmp_path, 'v4/blend')['ready']
    with pytest.raises(ValueError):
        build_freeze_inputs(tmp_path, '../unknown')


def test_index_csv_keeps_codes_and_requires_explicit_percent_conversion():
    payload = b'date,asset,weight,known_at\n2025-01-02,000001,50,2025-01-02\n2025-01-02,600000,50,2025-01-02\n'
    with pytest.raises(ValueError):
        parse_index_weights_csv(payload)
    actual = parse_index_weights_csv(payload, weights_in_percent=True)
    assert actual.asset.tolist() == ['000001', '600000']
    assert actual.weight.tolist() == [.5, .5]
    with pytest.raises(ValueError, match='complete'):
        parse_index_weights_csv(b'date,asset,weight,known_at\n2025-01-02,A,0.5,2025-01-02\n')
    with pytest.raises(ValueError):
        parse_index_weights_csv(b'not valid headers')
    with pytest.raises(ValueError, match='8 MB'):
        parse_index_weights_csv(b'x' * 8_000_001)


def index_fixture(root, monkeypatch):
    """Synthetic test ledger only; never accesses the user's real snapshot."""
    dates = pd.bdate_range('2025-01-02', periods=34)
    assets = ['000001', '600000']
    close = pd.DataFrame(np.cumprod(1 + np.resize([.01, -.02, .03], len(dates)))[:, None] * [[10, 20]], index=dates, columns=assets)
    records = []
    for date in dates:
        for asset in assets:
            price = close.loc[date, asset]
            records.append({'date': date, 'asset': asset, 'open': price, 'close': price, 'raw_open': price,
                            'raw_close': price, 'volume': 1e8, 'up_limit': price * 1.2, 'down_limit': price * .8})
    market = pd.DataFrame(records)
    features = pd.DataFrame({'date': np.repeat(dates, 2), 'asset': assets * len(dates), 'industry': ['X', 'Y'] * len(dates)})
    local = root / 'runs/research_v4'
    local.mkdir(parents=True)
    raw = features[['date', 'asset']].assign(mined_rank=np.tile([1., -1.], len(dates)))
    raw.to_parquet(local / 'scores.parquet', index=False)
    sha = hashlib.sha256((local / 'scores.parquet').read_bytes()).hexdigest()
    write_json(local / 'manifest.json', {'status': 'complete', 'source': 'real_tushare_snapshot',
                                       'data_end': str(dates[-1].date()), 'output_hashes': {'scores': sha},
                                       'fixture_scope': 'synthetic test data only'})
    for relative in ['data/private/research_v2/features.parquet', 'data/private/research_v2/features_cache.json',
                     'data/private/research_v2/benchmark.csv', 'data/private/research_v2/cpi.csv',
                     'data/private/mainboard1000_20260918/data/processed/market.csv',
                     'data/private/mainboard1000_20260918/data/processed/calendar.csv']:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('synthetic fixture', encoding='utf-8')
    benchmark = pd.DataFrame({'close': np.arange(100., 100. + len(dates))}, index=dates)
    monkeypatch.setattr('cfquant.benchmark_research.load_inputs', lambda data_root: (features, market, dates, benchmark, pd.DataFrame()))
    weights = pd.DataFrame({'date': dates[0], 'known_at': dates[0], 'asset': assets, 'weight': [.5, .5]})
    return dates, weights


def test_index_run_uses_actual_engine_reconciles_and_saves_independent_evidence(tmp_path, monkeypatch):
    dates, weights = index_fixture(tmp_path, monkeypatch)
    assert index_data_readiness(tmp_path)['ready']
    output, metrics, diagnostics = run_index_experiment(tmp_path, weights, 'mined_rank', str(dates[24].date()),
        str(dates[-1].date()), rebalance_every=2, lookback=3)
    assert output.name.startswith('index_enhancement_')
    for name in ['daily.csv', 'trades.csv', 'positions.csv', 'orders.csv', 'targets.parquet', 'config.json',
                 'checks.json', 'manifest.json', 'allocation_diagnostics.csv', 'attribution_industries.csv', 'attribution_scope.json']:
        assert (output / name).is_file(), name
    assert json.loads((output / 'checks.json').read_text())['passed']
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['actual_execution_reconciled'] is True
    assert manifest['input_hashes']['index_weights']
    assert manifest['output_hashes']['daily.csv']
    assert diagnostics[diagnostics.used_formation].constraints_passed.all()
    assert metrics['total_return'] is not None
    assert len(pd.read_csv(output / 'trades.csv')) > 0
    assert metrics['total_cost'] > 0
    app = app_at(tmp_path)
    assert not app.exception
    assert any('行业标签对应日期' in item.value for item in app.caption)
    assert any('行业资金归因' in item.value for item in app.markdown)


def test_index_blocked_required_date_produces_no_saved_run(tmp_path, monkeypatch):
    dates, weights = index_fixture(tmp_path, monkeypatch)
    weights['known_at'] = dates[-1] + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match='形成日'):
        run_index_experiment(tmp_path, weights, 'mined_rank', str(dates[3].date()), str(dates[-1].date()), lookback=3)
    assert not list((tmp_path / 'runs').glob('index_enhancement_*'))


def test_index_missing_or_modified_score_identity_is_rejected(tmp_path, monkeypatch):
    dates, weights = index_fixture(tmp_path, monkeypatch)
    with (tmp_path / 'runs/research_v4/scores.parquet').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='哈希'):
        run_index_experiment(tmp_path, weights, 'mined_rank', str(dates[3].date()), str(dates[-1].date()), lookback=3)
    assert not list((tmp_path / 'runs').glob('index_enhancement_*'))


def test_index_requested_interval_cannot_extend_beyond_local_snapshot(tmp_path, monkeypatch):
    dates, weights = index_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='超过本地快照'):
        run_index_experiment(tmp_path, weights, 'mined_rank', str(dates[24].date()),
                             str((dates[-1] + pd.Timedelta(days=7)).date()), lookback=3)
    assert not list((tmp_path / 'runs').glob('index_enhancement_*'))
