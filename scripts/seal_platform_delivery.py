"""Verify V4 delivery bytes and evidence without normalizing existing files.

Run only after the browser download receipt and canonical PPTX are final. The
only writes are delivery_checks.json and this evidence directory's manifest.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'evidence/platform_v4'
MATERIALS = ROOT/'reports/platform_v4'
PDF_PAGES = {'CF2026_V4_Final_Report.pdf': 10, 'CF2026_V4_Beamer.pdf': 20,
             'CF2026_V4_Speaker_Script.pdf': 10}
UI_NAMES = ['overview', 'compare', 'risk', 'drawdown', 'relative', 'cash',
            'factors', 'factor-comparison', 'market', 'experiments', 'qlib',
            'alpha158', 'connections', 'simulation-import']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def file_from(base, name):
    path = (base/name).resolve()
    require(path.is_relative_to(base.resolve()), f'Path escapes evidence directory: {name}')
    return path


def junit(path, passed, skipped):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == 'testsuite' else root.findall('.//testsuite')
    totals = {key: sum(int(s.get(key, 0)) for s in suites)
              for key in ['tests', 'failures', 'errors', 'skipped']}
    require(totals == {'tests': passed+skipped, 'failures': 0, 'errors': 0, 'skipped': skipped},
            f'Unexpected JUnit counts for {path}: {totals}')
    cases = root.findall('.//testcase')
    require(len(cases) == passed+skipped, f'JUnit case count mismatch: {path}')
    expected_optional = 'test_actual_qlib_expression_engine_matches_independent_reference'
    skipped_names = [c.get('name') for c in cases if c.find('skipped') is not None]
    if skipped:
        require(skipped_names == [expected_optional], 'The only main-environment skip must be the optional Qlib engine check')
    else:
        require(any(c.get('name') == expected_optional and c.find('skipped') is None for c in cases),
                'Real Qlib engine check is not present as a passing test')
    return {'passed': passed, 'skipped': skipped, 'failures': 0, 'errors': 0, 'sha256': sha(path)}


def verify_qlib():
    folder = ROOT/'evidence/qlib_bridge'
    result = load(folder/'result.json')
    manifest = load(folder/'manifest.json')
    require(result['passed'] and result['status'] == 'verified', 'Current Qlib result did not pass')
    require(result['run_id'] == manifest['run_id'], 'Qlib current manifest and receipt have different runs')
    for name, expected in manifest['files'].items():
        require(sha(file_from(folder, name)) == expected, f'Qlib manifest mismatch: {name}')
    require(result['module_sha256'] == sha(ROOT/'src/cfquant/qlib_bridge.py'), 'Qlib executed source hash differs from current bridge')
    require(result['packages']['pyqlib'] == '0.9.7', 'Unexpected Qlib runtime version')
    require(result['assets'] == 20 and result['factor_count'] == 8, 'Expected default 20-stock, eight-expression integration')
    require(len(result['checks']) == 8 and all(c['passed'] and c['availability_mismatches'] == 0 for c in result['checks'].values()),
            'Qlib independent formula or missing-value comparisons failed')
    alpha = result['alpha158']
    require(alpha['configured'] == alpha['with_finite_values'] == 158 and alpha['rows'] == 32580,
            'Alpha158 execution dimensions changed')
    require(alpha['vwap_coverage'] == 1 and alpha['trained_strategy'] is False, 'Unexpected Alpha158 coverage or strategy claim')
    require(result['unit_checks']['price_scale_matches_close_over_raw_close'] and
            result['unit_checks']['vwap_times_adjusted_volume_equals_amount'], 'Price/volume/VWAP unit checks failed')
    browser = load(OUT/'qlib_browser_rerun.json')
    require(browser['passed'] and not browser['browser_errors'], 'Qlib browser rerun failed')
    require(browser['new_run'] == result['run_id'] and browser['previous_run'] != browser['new_run'],
            'Browser did not produce the current, new Qlib run')
    require(browser['assets'] == result['assets'] and browser['alpha158'] == alpha, 'Browser Qlib receipt dimensions disagree')
    require((ROOT/'evidence/ui-v4-qlib-rerun.png').is_file(), 'Missing Qlib browser rerun screenshot')
    return {'passed': True, 'run_id': result['run_id'], 'assets': result['assets'], 'source_rows': result['source_rows'],
            'factor_count': 8, 'compared_values': sum(c['compared_values'] for c in result['checks'].values()),
            'max_absolute_error': max(c['max_absolute_error'] for c in result['checks'].values()),
            'alpha158': alpha, 'source_sha256': result['module_sha256'],
            'manifest_sha256': sha(folder/'manifest.json'), 'browser_receipt_sha256': sha(OUT/'qlib_browser_rerun.json')}


def tex_escape(value):
    return str(value).replace('\\', r'\textbackslash{}').replace('&', r'\&').replace('%', r'\%').replace('_', r'\_').replace('#', r'\#')


def compact(value):
    return re.sub(r'\s+', '', value)


def verify_materials():
    validation = load(MATERIALS/'material_validation.json')
    authoring = load(MATERIALS/'authoring_checks.json')
    require(validation['passed'], 'PDF authoring/visual validation failed')
    require(validation['latex_overflow_or_missing_glyph_warnings'] == 0, 'LaTeX overflow or missing-glyph findings remain')
    pdfs = {}
    for name, pages in PDF_PAGES.items():
        path = MATERIALS/name
        entry = validation['pdfs'][name]
        actual_pages = len(PdfReader(path).pages)
        require(actual_pages == pages == entry['pages'], f'PDF page count mismatch: {name}')
        require(sha(path) == entry['sha256'], f'PDF changed since review: {name}')
        require(entry['all_pages_rendered'] and entry['all_pages_visually_reviewed'], f'PDF visual review is incomplete: {name}')
        pdfs[name] = {'pages': pages, 'sha256': sha(path)}
    deck = load(MATERIALS/'deck_content.json')
    markdown_path = ROOT/'docs/PRESENTATION_SCRIPT_V4.md'
    markdown = markdown_path.read_text(encoding='utf-8')
    beamer = (MATERIALS/'beamer.tex').read_text(encoding='utf-8')
    script_tex = (MATERIALS/'speaker_script.tex').read_text(encoding='utf-8')
    require(len(deck) == validation['slides'] == 20, 'Deck must contain 20 slides')
    require(sha(MATERIALS/'deck_content.json') == validation['deck_sha256'], 'Deck source changed since PDF validation')
    require(sha(markdown_path) == validation['markdown_script_sha256'], 'Markdown speaker script changed since review')
    require(sha(ROOT/'scripts/build_platform_materials.py') == validation['source_builder_sha256'], 'Material builder changed since review')
    headings = re.findall(r'^## (\d+)\. (.+)$', markdown, flags=re.M)
    require(headings == [(str(i), slide['title']) for i, slide in enumerate(deck, 1)], 'Slide and Markdown titles do not map one-to-one')
    bodies = re.split(r'^## \d+\. .+$', markdown, flags=re.M)[1:]
    require(len(bodies) == 20, 'Expected 20 Markdown speaker sections')
    for i, (slide, body) in enumerate(zip(deck, bodies), 1):
        require(slide['notes'] in body, f'Markdown notes differ from deck on slide {i}')
        require(r'\begin{frame}{'+tex_escape(slide['title'])+'}' in beamer, f'Beamer title missing on slide {i}')
        require(r'\note{'+tex_escape(slide['notes'])+'}' in beamer, f'Beamer speaker notes differ on slide {i}')
        require(tex_escape(slide['notes']) in script_tex, f'Printable speaker notes differ on slide {i}')
    seconds = sum(slide['seconds'] for slide in deck[:18])
    characters = sum(len(re.findall(r'[\u4e00-\u9fff]', slide['notes'])) for slide in deck)
    require(seconds == validation['talk_seconds'] == authoring['main_talk_seconds'] == 1145, 'Main talk timing differs')
    require(characters == validation['spoken_chinese_characters'] == authoring['spoken_chinese_characters'], 'Speaker text character count differs')
    require(authoring['main_tests'] == {'passed': 102, 'skipped': 1} and authoring['qlib_tests'] == {'passed': 5, 'skipped': 0},
            'Material test statements differ from final evidence')
    return validation, deck, {'pdfs': pdfs, 'slides': 20, 'main_slides': 18, 'backup_slides': 2,
                              'talk_seconds': seconds, 'spoken_chinese_characters': characters,
                              'deck_script_mapping_passed': True, 'material_validation_sha256': sha(MATERIALS/'material_validation.json')}


def verify_pptx(deck):
    path = MATERIALS/'CF2026_V4_Presentation.pptx'
    validation = load(OUT/'presentation_validation.json')
    digest = sha(path)
    require(digest == validation['finalSha256'] == validation['firstPartyImport']['sha256'], 'Canonical PPTX is not the finalized, re-imported file')
    require(validation['packageIntegrity']['status'] == 'pass' and validation['packageIntegrity']['findingCount'] == 0, 'PPTX package validation failed')
    require(validation['firstPartyImport']['passed'] and validation['presentationLayout']['findingCount'] == 0, 'PPTX import/layout validation failed')
    require(validation['nativeTableArithmetic']['native_table_count'] == 3, 'PPTX validation does not record three native tables')
    require(validation['nativeChartValidation']['passed'] and validation['nativeChartValidation']['detectedChartCount'] == 2, 'Native chart validation failed')
    with zipfile.ZipFile(path) as z:
        require(z.testzip() is None, 'PPTX ZIP CRC failed')
        slides = sorted((n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml', n)),
                        key=lambda n: int(re.search(r'slide(\d+)\.xml', n)[1]))
        charts = [n for n in z.namelist() if re.fullmatch(r'ppt/(?:slides/)?charts/chart\d+\.xml', n)]
        tables = 0
        for i, (part, slide) in enumerate(zip(slides, deck), 1):
            xml = ET.fromstring(z.read(part))
            tables += len(xml.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/main}tbl'))
            text = ''.join(xml.itertext())
            require(compact(slide['title']) in compact(text), f'PPTX title differs on slide {i}')
            notes = ET.fromstring(z.read(f'ppt/notesSlides/notesSlide{i}.xml'))
            require(compact(slide['notes']) in compact(''.join(notes.itertext())), f'PPTX notes differ on slide {i}')
        require(len(slides) == 20 and len(charts) == 2 and tables == 3, f'PPTX structure mismatch: {len(slides)} slides, {len(charts)} charts, {tables} tables')
    return {'sha256': digest, 'slides': 20, 'native_charts': 2, 'native_tables': 3,
            'crc_passed': True, 'title_and_notes_mapping_passed': True,
            'validation_sha256': sha(OUT/'presentation_validation.json'),
            'boundary': 'Package/layout/embedded-chart/re-import checks and separate visual review; native desktop PowerPoint execution not claimed.'}


def verify_downloads(path, artifact_hashes):
    """Read the final browser receipt. Exact field schema is intentionally small."""
    value = load(path)
    require(value.get('passed') is True, 'Browser material downloads did not pass')
    require(not value.get('errors', []) and not value.get('exceptions', []) and not value.get('browser_errors', []), 'Browser download errors remain')
    entries = value['items']
    require(len(entries) == len(artifact_hashes), 'Expected exactly four material downloads')
    names = set()
    for entry in entries:
        name = entry['expectedFile']
        require(name in artifact_hashes, f'Unexpected downloaded artifact: {name}')
        require(entry['passed'] and entry['downloadedSha256'] == artifact_hashes[name], f'Browser download hash mismatch: {name}')
        if 'expectedSha256' in entry:
            require(entry['expectedSha256'] == artifact_hashes[name], f'Stale expected download hash: {name}')
        names.add(name)
    require(names == set(artifact_hashes), 'Browser did not download all four current artifacts')
    return {'passed': True, 'files': sorted(names), 'receipt_sha256': sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-receipt', default='evidence/platform_v4/browser_downloads.json')
    args = parser.parse_args()
    download_path = file_from(ROOT, args.download_receipt)
    main_tests = junit(OUT/'tests-final.xml', 102, 1)
    qlib_tests = junit(ROOT/'evidence/qlib_bridge/test-results.xml', 5, 0)
    qlib = verify_qlib()
    _, deck, materials = verify_materials()
    pptx = verify_pptx(deck)
    ui = load(OUT/'ui_check.json')
    require(ui['views'] == 14 and not ui['errors'] and not ui['exceptions'], 'UI view checks failed')
    screenshots = {}
    for name in UI_NAMES + ['materials', 'qlib-rerun']:
        path = ROOT/'evidence'/f'ui-v4-{name}.png'
        require(path.is_file(), f'Missing UI screenshot: {name}')
        screenshots[path.name] = sha(path)
    artifacts = {name: item['sha256'] for name, item in materials['pdfs'].items()}
    artifacts['CF2026_V4_Presentation.pptx'] = pptx['sha256']
    downloads = verify_downloads(download_path, artifacts)
    receipt = {'passed': True, 'release': 'v2.2.0', 'sealed_utc': datetime.now(timezone.utc).isoformat(),
               'main_tests': main_tests, 'qlib_tests': qlib_tests, 'qlib': qlib,
               'materials': materials, 'pptx': pptx, 'browser_downloads': downloads,
               'ui': {'views': 14, 'additional_checks': ['material downloads', 'actual Qlib rerun'],
                      'errors': [], 'exceptions': [], 'screenshot_sha256': screenshots},
               'artifacts_sha256': artifacts, 'normalization_performed': False,
               'sealing_script_sha256': sha(Path(__file__)),
               'scope': 'Local deliverable acceptance; course submission and GitHub CI are separate actions.'}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'delivery_checks.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    manifest_path = OUT/'manifest.json'
    manifest = {'release': 'v2.2.0', 'files': {p.relative_to(OUT).as_posix(): sha(p)
                                             for p in sorted(OUT.rglob('*')) if p.is_file() and p != manifest_path}}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    require(all(sha(file_from(OUT, name)) == digest for name, digest in manifest['files'].items()), 'Sealed manifest readback failed')
    print(json.dumps({'passed': True, 'main_tests': 102, 'main_skips': 1, 'qlib_tests': 5,
                      'pdf_pages': list(PDF_PAGES.values()), 'pptx': [20, 2, 3], 'ui_views': 14,
                      'qlib_run_id': qlib['run_id'], 'manifest_files': len(manifest['files'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
