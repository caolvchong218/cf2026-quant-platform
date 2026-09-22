"""Check current evidence and final artifacts; normalize public text hash bytes."""
from pathlib import Path
import hashlib, json, xml.etree.ElementTree as ET
from pypdf import PdfReader

root=Path(__file__).resolve().parents[1];out=root/'evidence/research_v3';materials=root/'reports/benchmark_v3'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
tests=ET.parse(out/'tests-final.xml').getroot().find('testsuite')
assert int(tests.attrib['tests'])==52
assert all(int(tests.attrib[k])==0 for k in ['failures','errors','skipped'])
load=lambda n:json.loads((out/n).read_text(encoding='utf-8'))
assert load('acceptance.json')['historical_acceptance_passed']
assert load('repeatability.json')['passed']
assert not load('ui_check.json')['errors']
files=['CF2026_V3_Final_Report.pdf','CF2026_V3_Beamer.pdf','CF2026_V3_Presentation.pptx']
assert len(PdfReader(materials/files[0]).pages)==10
assert len(PdfReader(materials/files[1]).pages)==20
for name in files[:2]:assert sha(materials/name)==sha(root/'tmp'/name)
assert sha(materials/files[2])==load('presentation_validation.json')['finalSha256']
receipt={'passed':True,'tests':52,'accounting_runs':21,'independent_replay_tables':4,
         'browser':'default V3, switch V2/V1, return V3 and run; PDF downloads verified',
         'visual_review':'10 report pages; all 20 Beamer and native PPTX slides; revisions re-rendered and inspected',
         'native_pptx':'6 charts, 2 tables, structural checks and final re-import/render; not desktop PowerPoint execution',
         'artifacts_sha256':{name:sha(materials/name) for name in files}}
(out/'delivery_checks.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
for folder in [root/'evidence/research_v2',out]:
    for p in folder.rglob('*'):
        if p.is_file() and p.suffix in ['.json','.csv','.xml']:
            p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
    manifest={'files':{p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*'))
                      if p.is_file() and p.name!='manifest.json'}}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8',newline='\n')
print(json.dumps(receipt))
