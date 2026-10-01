"""V4 replay cannot consume stale scores or load private data before identity checks."""
from dataclasses import replace
import hashlib
import json

import pytest

from cfquant.risk import RiskPolicy
from cfquant.strategies import StrategyVersion, replay


def strategy():
    return StrategyVersion('v4/mined_ridge','Ridge','v4','mined_ridge',RiskPolicy(),
                           'runs/research_v4/scores.parquet','evidence/research_v4','historical')


def manifest(root,value):
    path=root/'evidence/research_v4/manifest.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def scores(root,value=b'corrected-score-snapshot'):
    path=root/'runs/research_v4/scores.parquet'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(value)
    return hashlib.sha256(value).hexdigest()


@pytest.mark.parametrize('kind,message',[
    ('no_manifest','manifest'),('missing_hash','哈希'),('invalid_hash','哈希'),
    ('no_scores','scores.parquet'),('old_scores','不一致'),('malformed_manifest','manifest'),
])
def test_invalid_score_identity_blocks_before_private_input_loading(tmp_path,monkeypatch,kind,message):
    def forbidden(data_root):
        pytest.fail('Private inputs loaded before V4 score identity passed')
    monkeypatch.setattr('cfquant.strategies.load_inputs',forbidden)
    expected=hashlib.sha256(b'corrected-score-snapshot').hexdigest()
    if kind not in {'no_manifest','malformed_manifest'}:
        manifest(tmp_path,{'output_hashes':{'scores':expected}})
    if kind!='no_scores':scores(tmp_path,b'old-score-snapshot' if kind=='old_scores' else b'corrected-score-snapshot')
    if kind=='missing_hash':manifest(tmp_path,{'output_hashes':{}})
    if kind=='invalid_hash':manifest(tmp_path,{'output_hashes':{'scores':'unverified'}})
    if kind=='malformed_manifest':
        path=tmp_path/'evidence/research_v4/manifest.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('not-json')
    with pytest.raises(ValueError,match=message):
        replay(tmp_path,strategy(),'2025-01-02','2025-06-30')


def test_matching_published_score_identity_reaches_private_loader_for_ridge_and_blend(tmp_path,monkeypatch):
    expected=scores(tmp_path)
    manifest(tmp_path,{'output_hashes':{'scores':expected}})
    loaded=[]
    class LoaderReached(Exception):pass
    def loader(data_root):
        loaded.append(data_root)
        raise LoaderReached
    monkeypatch.setattr('cfquant.strategies.load_inputs',loader)
    for signal in ('mined_ridge','blend'):
        spec=replace(strategy(),id='v4/'+signal,signal=signal)
        with pytest.raises(LoaderReached):
            replay(tmp_path,spec,'2025-01-02','2025-06-30')
    assert loaded==[tmp_path,tmp_path]
