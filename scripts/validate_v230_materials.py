"""Check the final delivery artifacts; visual acceptance is recorded separately."""
from pathlib import Path
import hashlib
import json
import zipfile
import xml.etree.ElementTree as ET
import pdfplumber


def main():
    root = Path(__file__).resolve().parents[1]
    materials = root / 'reports/platform_v230'
    expected = {
        'CF2026_V230_Final_Report.pdf': 10,
        'CF2026_V230_Beamer.pdf': 20,
        'CF2026_V230_Speaker_Script.pdf': 10,
    }
    documents = {}
    for name, pages in expected.items():
        path = materials / name
        with pdfplumber.open(path) as document:
            text = '\n'.join(page.extract_text() or '' for page in document.pages)
            members = all(value in text for value in ['12412408', '12410819', '12410436'])
            documents[name] = {'pages': len(document.pages), 'expected_pages': pages,
                               'team_ids_present': members,
                               'passed': len(document.pages) == pages and members,
                               'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    presentation = materials / 'CF2026_V230_Presentation.pptx'
    with zipfile.ZipFile(presentation) as package:
        names = package.namelist()
        count = lambda prefix: sum(n.startswith(prefix) and n[len(prefix):].split('.')[0].isdigit()
                                  and n.endswith('.xml') for n in names)
        documents[presentation.name] = {
            'slides': count('ppt/slides/slide'), 'notes': count('ppt/notesSlides/notesSlide'),
            'sha256': hashlib.sha256(presentation.read_bytes()).hexdigest(),
            'passed': package.testzip() is None and count('ppt/slides/slide') == 20
                      and count('ppt/notesSlides/notesSlide') == 20,
        }
    suites = ET.parse(root / 'evidence/ui_v230/tests.xml').getroot()
    suites = list(suites) if suites.tag == 'testsuites' else [suites]
    total = sum(int(s.attrib['tests']) for s in suites)
    skipped = sum(int(s.attrib.get('skipped', 0)) for s in suites)
    failures = sum(int(s.attrib.get('failures', 0)) + int(s.attrib.get('errors', 0)) for s in suites)
    output = {'documents': documents,
              'tests': {'passed': total - skipped - failures, 'skipped': skipped,
                        'failures_or_errors': failures},
              'structural_checks_passed': all(d['passed'] for d in documents.values()) and failures == 0,
              'visual_acceptance': {
                  'pdf': 'All 10 report, 20 Beamer and 10 script pages inspected; repaired report pages 3/6 and script page 8 re-rendered and checked.',
                  'pptx': 'All 20 runtime-rendered slides inspected; final slide 17 rechecked after validation count update.',
                  'native_powerpoint_application_checked': False,
              }}
    path = root / 'evidence/ui_v230/material_validation.json'
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output['structural_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
