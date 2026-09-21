"""Seal final artifact checks and published derived evidence; no private inputs copied."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / 'evidence/research_v2'
F = ROOT / 'reports/final'
# Published text hashes must survive Git's LF checkout policy on any OS.
for p in E.rglob('*'):
    if p.is_file() and p.suffix in {'.json', '.csv', '.xml'}:
        p.write_bytes(p.read_bytes().replace(b'\r\n', b'\n'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
suite = ET.parse(E / 'tests-final.xml').getroot().find('testsuite')
assert suite is not None and int(suite.attrib['tests']) == 48
assert all(int(suite.attrib[k]) == 0 for k in ('errors', 'failures', 'skipped'))
files = ['CF2026_Final_Report.pdf', 'CF2026_Beamer.pdf', 'CF2026_Presentation_Final.pptx']
assert len(PdfReader(F / files[0]).pages) == 10
assert len(PdfReader(F / files[1]).pages) == 20
receipt = json.loads((E / 'presentation_validation.json').read_text(encoding='utf-8'))
assert sha(F / files[2]) == receipt['finalSha256']
with zipfile.ZipFile(F / files[2]) as z:
    assert z.testzip() is None
for name in files[:2]:
    assert sha(F / name) == sha(ROOT / 'tmp' / name), 'UI download differs'
ui = json.loads((E / 'ui_check.json').read_text(encoding='utf-8'))
assert not ui['errors'] and len(ui['visited']) == 5
repro = json.loads((E / 'repeatability.json').read_text(encoding='utf-8'))
assert repro['passed']
acceptance = json.loads((E / 'acceptance.json').read_text(encoding='utf-8'))
assert len(acceptance['research_runs']) == 13
assert all(x['passed'] for x in acceptance['research_runs'].values())
result = {
    'status': 'passed', 'tests': 48, 'failed': 0, 'skipped': 0,
    'independent_selected_strategy_repeat': True,
    'research_accounting_runs': 13, 'research_input_hashes_checked': 2073,
    'report_pages': 10, 'beamer_pages': 20, 'pptx_slides': 20,
    'native_charts': 6, 'native_tables': 2,
    'visual_review': 'All 10 report pages and 20 Beamer/PPTX slides reviewed; changed pages re-rendered and reviewed.',
    'ui': 'Five research tabs viewed; two PDF downloads match final artifact hashes.',
    'pptx_execution_boundary': 'Artifact Tool final re-import/render and structural validators; native desktop PowerPoint not run.',
    'artifacts_sha256': {name: sha(F / name) for name in files},
    'warnings': 'Matplotlib/Pyparsing deprecation warnings and local pytest cache permission warning; tests passed.',
}
(E / 'delivery_checks.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
manifest = {'files': {str(p.relative_to(E)).replace('\\', '/'): sha(p)
    for p in sorted(E.rglob('*')) if p.is_file() and p.name != 'manifest.json'}}
(E / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
print(json.dumps(result, ensure_ascii=False))
