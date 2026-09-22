"""Package current committed source and V3 materials; do not overwrite V2 history."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,zipfile

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,required=True);args=p.parse_args()
dest=args.destination.resolve();dest.mkdir(parents=True,exist_ok=True)
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['CF2026_V3_Final_Report.pdf','CF2026_V3_Beamer.pdf','CF2026_V3_Presentation.pptx']:
    shutil.copy2(root/'reports/benchmark_v3'/name,dest/name)
latex=dest/'LaTeX_source_V3';latex.mkdir(exist_ok=True)
for name in ['final_report.tex','beamer.tex']:shutil.copy2(root/'reports/benchmark_v3'/name,latex/name)
shutil.copytree(root/'reports/benchmark_v3/figures',latex/'figures',dirs_exist_ok=True)
for name in ['PRESENTATION_SCRIPT_V3.md','REPRODUCE_V3.md','RESEARCH_PROTOCOL_V3.md','BENCHMARK_METHOD_SOURCES_V3.md','FACTOR_CARDS_V2.md','ACCEPTANCE_V2.md']:
    shutil.copy2(root/'docs'/name,dest/name)
code=dest/'CF2026_Source_v2.1.0.zip'
subprocess.run(['git','archive','--format=zip',f'--output={code}','HEAD'],cwd=root,check=True)
with zipfile.ZipFile(code) as z:
    assert z.testzip() is None
    assert not any(n.startswith(('data/private/','runs/','.venv','tmp/')) for n in z.namelist())
    for version in ['v2','v3']:
        prefix=f'evidence/research_{version}/'
        manifest=json.loads(z.read(prefix+'manifest.json'))
        for name,h in manifest['files'].items():assert hashlib.sha256(z.read(prefix+name)).hexdigest()==h,name
    unpack=root/'tmp'/('submission-v3-'+revision[:10]);unpack.mkdir(parents=True,exist_ok=True);z.extractall(unpack)
readme=f'''# 最新提交包：V3指数目标研究 / 平台v2.1.0

代码提交：{revision}
GitHub：https://github.com/caolvchong218/cf2026-quant-platform

最新报告：CF2026_V3_Final_Report.pdf（10页）。
演示：CF2026_V3_Beamer.pdf（20页，真正LaTeX Beamer），另附可编辑PPTX。
讲稿：PRESENTATION_SCRIPT_V3.md，18页正文约19分钟＋2页答辩备份。
源码：CF2026_Source_v2.1.0.zip。LaTeX_source_V3含可编辑报告/幻灯片及图片。

本机双击D:/Desktop/cf2026-quant-platform/launch.cmd。回测实验默认V3最新候选，可选择V2、V1及所有V3候选，修改参数并重跑。旧动量不是默认策略，账本通过不代表策略达标。

固定2025-01-02至2026-09-18：净收益25.13%，年化14.51%，最大回撤9.88%；沪深300价格指数14.55%，全收益指数19.94%。双倍费用仍收益22.48%。2023起连续运行44.09%，最大回撤19.70%，2024年度曾落后指数。

必须说明：V3验证期预选价值主题随后失败；当前候选是在本轮历史复核后筛选，不能称独立盲测成功，不保证未来收益。完整8个候选保留。

旧V2材料在“历史版本”目录，当前提交请使用根目录V3文件。提交前在课程系统填写组员姓名、学号并核对命名。Token和原始厂商数据不在公开源码包。PowerPoint经结构与渲染验收，推荐使用Beamer PDF演示。
'''
(dest/'先读我.md').write_text(readme,encoding='utf-8')
receipt={'git_revision':revision,'tag':'v2.1.0','archive_crc_passed':True,'public_evidence_hashes_passed':True,
         'files':{p.relative_to(dest).as_posix():sha(p) for p in sorted(dest.rglob('*'))
                  if p.is_file() and '历史版本' not in p.parts and p.name!='SHA256.json'}}
(dest/'SHA256.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'revision':revision,'archive_bytes':code.stat().st_size,'smoke_root':str(unpack)}))
