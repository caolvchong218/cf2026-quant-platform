"""Package committed code, final materials and editable LaTeX without vendor data."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--destination', type=Path, required=True)
args = parser.parse_args()
dest = args.destination.resolve()
dest.mkdir(parents=True, exist_ok=True)
revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['CF2026_Final_Report.pdf', 'CF2026_Beamer.pdf', 'CF2026_Presentation_Final.pptx']:
    shutil.copy2(root / 'reports/final' / name, dest / name)
latex = dest / 'LaTeX_source'
latex.mkdir(exist_ok=True)
for name in ['final_report.tex', 'beamer.tex']:
    shutil.copy2(root / 'reports/final' / name, latex / name)
shutil.copytree(root / 'reports/final/figures', latex / 'figures', dirs_exist_ok=True)
for name in ['PRESENTATION_SCRIPT.md', 'ACCEPTANCE_V2.md', 'REPRODUCE_V2.md', 'FACTOR_CARDS_V2.md']:
    shutil.copy2(root / 'docs' / name, dest / name)
code = dest / 'CF2026_Source_v2.0.0.zip'
subprocess.run(['git', 'archive', '--format=zip', f'--output={code}', 'HEAD'], cwd=root, check=True)
with zipfile.ZipFile(code) as z:
    assert z.testzip() is None
    names = z.namelist()
    assert all(not n.startswith(('data/private/', '.venv', 'tmp/', 'runs/')) for n in names)
    assert 'app.py' in names and 'configs/demo.yaml' in names
    manifest = json.loads(z.read('evidence/research_v2/manifest.json'))
    for name, expected in manifest['files'].items():
        assert hashlib.sha256(z.read('evidence/research_v2/' + name)).hexdigest() == expected, name
    unpack = root / 'tmp' / ('submission-smoke-' + revision[:10])
    unpack.mkdir(parents=True, exist_ok=True)
    z.extractall(unpack)
readme = f'''# 最终提交包：青序量化研究平台 v2.0.0

代码版本：{revision}
公开仓库：https://github.com/caolvchong218/cf2026-quant-platform

1. 报告：CF2026_Final_Report.pdf，10页。
2. 展示：优先使用真正LaTeX Beamer生成的CF2026_Beamer.pdf，20页；另提供可编辑PowerPoint。
3. 讲稿：PRESENTATION_SCRIPT.md，18页正文约19分钟，2页答辩备份。
4. 源码：CF2026_Source_v2.0.0.zip，含代码、测试、依赖、样本和派生证据。
5. LaTeX_source内用XeLaTeX编译final_report.tex和beamer.tex，可直接修改。
6. 本机平台：D:/Desktop/cf2026-quant-platform/launch.cmd；新电脑按复现指南安装。

预选策略最终区间2025-01-02至2026-09-18：累计净收益6.12%，年化3.65%，最大回撤7.01%。共同完整月份2025年2月至2026年8月：策略7.02%，CPI 0.69%。达到本段历史购买力目标，但未战胜同期沪深300价格指数，也不保证未来收益。

提交前在课程系统补充小组姓名、学号，核对老师文件命名要求。仓库及源码包不含Token或厂商原始数据；真实缓存保存在本机。不要把旧版报告误当最终报告。具体评分由教师决定。

校验：48项测试通过、13个研究账本检查通过、独立环境复算一致。PowerPoint经结构与渲染检查，未在原生桌面PowerPoint执行。LaTeX和PDF为推荐的演示主版本。
'''
(dest / '先读我.md').write_text(readme, encoding='utf-8')
receipt = {'git_revision': revision, 'source_archive_crc_passed': True,
           'source_evidence_hashes_passed': True,
           'files': {str(p.relative_to(dest)).replace('\\', '/'): sha(p)
                     for p in sorted(dest.rglob('*')) if p.is_file() and p.name != 'SHA256.json'}}
(dest / 'SHA256.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'destination': str(dest), 'revision': revision, 'unpacked_smoke_root': str(unpack),
                  'archive_bytes': code.stat().st_size, 'files': len(receipt['files'])}, ensure_ascii=False))
