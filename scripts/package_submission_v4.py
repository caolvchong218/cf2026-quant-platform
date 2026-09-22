"""Package current committed source and V4 materials; do not overwrite earlier history."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,zipfile

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,required=True);args=p.parse_args()
dest=args.destination.resolve();dest.mkdir(parents=True,exist_ok=True)
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['CF2026_V4_Final_Report.pdf','CF2026_V4_Beamer.pdf','CF2026_V4_Presentation.pptx','CF2026_V4_Speaker_Script.pdf']:
    shutil.copy2(root/'reports/platform_v4'/name,dest/name)
latex=dest/'LaTeX_source_V4';latex.mkdir(exist_ok=True)
for name in ['final_report.tex','beamer.tex','speaker_script.tex']:shutil.copy2(root/'reports/platform_v4'/name,latex/name)
shutil.copytree(root/'reports/platform_v4/figures',latex/'figures',dirs_exist_ok=True)
for name in ['PRESENTATION_SCRIPT_V4.md','REPRODUCE_V4.md','RESEARCH_PROTOCOL_V3.md','BENCHMARK_METHOD_SOURCES_V3.md','FACTOR_CARDS_V2.md','ACCEPTANCE_V4.md','QLIB_INTEGRATION.md','THS_CONNECTION.md']:
    shutil.copy2(root/'docs'/name,dest/name)
code=dest/'CF2026_Source_v2.2.0.zip'
subprocess.run(['git','archive','--format=zip',f'--output={code}','HEAD'],cwd=root,check=True)
with zipfile.ZipFile(code) as z:
    assert z.testzip() is None
    assert not any(n.startswith(('data/private/','runs/','.venv','tmp/')) for n in z.namelist())
    for section in ['research_v2','research_v3','qlib_bridge','platform_v4']:
        prefix=f'evidence/{section}/'
        manifest=json.loads(z.read(prefix+'manifest.json'))
        for name,h in manifest['files'].items():assert hashlib.sha256(z.read(prefix+name)).hexdigest()==h,name
    unpack=root/'tmp'/('submission-v4-'+revision[:10]);unpack.mkdir(parents=True,exist_ok=True);z.extractall(unpack)
readme=f'''# 最新提交包：平台增强版 V4 / release v2.2.0

代码提交：{revision}
GitHub：https://github.com/caolvchong218/cf2026-quant-platform

本次提供10页报告、20页LaTeX Beamer、可编辑PPTX、逐页完整演讲稿(PDF/Markdown)、LaTeX源码、源代码ZIP和复现指南。

运行：本机双击 D:/Desktop/cf2026-quant-platform/launch.cmd。
新功能：策略同区间对比、风险月历与回撤、12因子诊断、行情K线、实验档案与笔记、真实Qlib因子桥、同花顺模拟交易文件桥。

策略仍为 V3 滚动LightGBM波动预算；2025-01-02至2026-09-18净收益25.13%，最大回撤9.88%，同期沪深300全收益指数19.94%。本轮平台增强没有重新挑选历史赢家。当前策略是历史重选候选，尚非独立盲测，也不保证未来收益。

Qlib实际状态与范围以 QLIB_INTEGRATION.md 和平台页面证据为准。同花顺仅文件导入导出桥，未登录或连接外部模拟账户，未向任何账户提交订单。用户回来后可按 THS_CONNECTION.md 进行账户端验证。

公开聚合结果可离线使用，原始行情、账号凭据、模型缓存和个人实验笔记不在公开源码包。历史材料保存在“历史版本”，本次使用根目录V4材料。PPTX保留原生图表/文字，Beamer PDF适合稳定演示。

材料已经准备，尚未代交课程系统。提交前请填写组员信息并核对老师要求的文件命名。
'''
(dest/'先读我.md').write_text(readme,encoding='utf-8')
receipt={'git_revision':revision,'tag':'v2.2.0','archive_crc_passed':True,'public_evidence_hashes_passed':True,
         'files':{p.relative_to(dest).as_posix():sha(p) for p in sorted(dest.rglob('*'))
                  if p.is_file() and '历史版本' not in p.parts and p.name!='SHA256.json'}}
(dest/'SHA256.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'revision':revision,'archive_bytes':code.stat().st_size,'smoke_root':str(unpack)}))
