"""Validate published evidence and private ledgers without changing an experiment."""
import argparse
from pathlib import Path
import json
from cfquant.data import digest,write_json
from cfquant.research_lab import source_root


def validate(root):
    root=Path(root);public=root/'evidence/research_v4'
    declared=json.loads((public/'SHA256.json').read_text(encoding='utf-8'))
    mismatches=[name for name,expected in declared.items() if digest(public/name)!=expected]
    manifest=json.loads((public/'manifest.json').read_text(encoding='utf-8'))
    cards=json.loads((public/'factor_cards.json').read_text(encoding='utf-8'))
    folds=json.loads((public/'folds.json').read_text(encoding='utf-8'))
    cutoff_passed=all(f['max_label_exit']<f['cutoff'] for f in folds)
    local=root/'runs/research_v4'
    ledger_checks={p.parent.name:json.loads(p.read_text(encoding='utf-8'))['passed'] for p in local.glob('*/checks.json')}
    attribution_checks={p.parent.name:json.loads(p.read_text(encoding='utf-8'))['passed'] for p in local.glob('*/attribution_check.json')}
    current_sources={p.name:digest(p) for p in (root/'src/cfquant').glob('*.py')}
    changes={name:{'experiment':expected,'release':current_sources.get(name)} for name,expected in manifest['source_hashes'].items()
             if current_sources.get(name)!=expected}
    output={'public_hashes_passed':not mismatches,'mismatches':mismatches,
            'unique_factor_count':len({c['id'] for c in cards}),'factor_cards_count':len(cards),
            'fold_label_exits_before_cutoff':cutoff_passed,'ledger_checks':ledger_checks,
            'attribution_checks':attribution_checks,'post_run_source_changes':changes,
            'post_run_change_scope':'UI presentation/index input and legacy version-only cache compatibility; research scores and ledgers unchanged',
            'historical_development':True,'independent_blind_test':False,
            'real_llm_called':False,'real_index_enhancement_completed':False,
            'real_forward_return_available':False,'default_mainline_retained':True}
    output['passed']=not mismatches and len(cards)==len({c['id'] for c in cards})==28 and cutoff_passed and len(ledger_checks)==13 and all(ledger_checks.values()) and len(attribution_checks)==10 and all(attribution_checks.values())
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    result=validate(root)
    if args.output:write_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in {'post_run_source_changes','ledger_checks','attribution_checks'}},ensure_ascii=False,indent=2))
    raise SystemExit(0 if result['passed'] else 1)
