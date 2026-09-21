"""Two-page work summary from local evidence. Requires optional reportlab."""
from pathlib import Path
from datetime import datetime
from html import escape
import json
import shutil

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/WORK_PROGRESS_20260921.pdf"
TEXT = ROOT / "docs/WORK_PROGRESS_20260921.md"
DELIVERY = Path("D:/Desktop/GPT的输出/CF2026_Project1")
snapshot = ROOT / "data/private/mainboard1000_20260918/data/processed"
progress = json.loads((snapshot / "progress.json").read_text(encoding="utf-8"))
summary_path = ROOT / "evidence/expanded_data.json"
summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
complete = progress.get("status") == "complete"
accepted = "backtest_folder" in summary
ui_checked = (ROOT / "tmp/expanded-ui-check.json").exists()
stamp = datetime.now().strftime("%Y-%m-%d %H:%M（北京时间）")
repo = "https://github.com/caolvchong218/cf2026-quant-platform"
new_status = (f"已完成：{summary.get('assets', 1000):,} 只，{progress['rows']:,} 条；"
              f"最新日期 {progress['actual_end']}。" if complete else
              f"正在下载：{progress['completed_assets']:,} / 1,000 只，已整理 {progress['rows']:,} 条；尚未完成全量验收。")
new_volume = (f"{summary['snapshot_bytes']/1024**3:.3f} GiB（原始缓存及标准文件合计）"
              if summary.get("snapshot_bytes") else "容量上限 3 GiB，完成后统计实际占用")
new_quality = (f"共 {summary['calendar_sessions']:,} 个交易日；最新交易日有 {summary['assets_with_latest_bar']} 只股票返回日线。"
               f"股票×交易日网格缺 {summary['missing_calendar_asset_cells']:,} 条，另有 {summary['missing_limit_rows']} 条缺涨跌停价格；均保留缺失状态，交易引擎据此限制成交。"
               if summary else "")
sections = [
    ("1. 这次完成的是什么", [
        "已经搭建一个可以运行的计算金融研究平台：读取行情，计算因子，生成持仓，模拟交易，再检查收益与账本。采用 Python，便于组员阅读，并复用 pandas、NumPy、SciPy、Plotly 和 Streamlit。",
        "项目按数据、因子、组合、成交、评价、实验记录和界面分模块组织。后续可以增加因子或预测模型，并继续复用已有数据与回测流程。"]),
    ("2. 已经做好的功能", [
        "数据：接入 Tushare；保存原始响应、交易日历和标准行情；检查重复、异常价格及缺失；为文件记录 SHA-256，支持离线复现。",
        "研究：实现动量、短期反转、低波动三个基础因子；提供 IC、Rank IC、分组收益和组合回测；比较不同调仓频率，保留亏损结果。",
        "交易：收盘形成信号，下一交易日开盘执行；计入买卖费用，处理现金不足、缺报价和涨跌停，并独立核对成交与净值账本。",
        "使用：提供中文网页、命令行、启动脚本、说明文档、8 页课程报告及完整验收包。网页有研究总览、回测、因子诊断、数据检查和扩展指南。"]),
    ("3. 数据拉取到什么程度", []),
    ("4. 如何验证、如何管理版本", [
        "本机 38 项自动测试通过，覆盖因子手算、信号时间、成本与现金、缺失行情、泄漏检查、独立 Backtrader 对照、界面流程，以及新增下载参数、缓存和容量保护。GitHub CI 已通过扩展下载功能版本。",
        "仓库已从 private 改为 public，并验证匿名访问成功。公开前扫描了全部可达历史中的 78 个文件对象，没有发现本次 Token 或原始行情目录。凭据、真实数据、缓存和运行结果目录均排除在 Git 外。",
        "Git 已记录初始化、数据与回测、界面、文档和扩展下载等阶段；原始交付标记为 v1.0.0，扩展功能代码已推送。原始数据快照的 186 个文件校验全部一致。"]),
    ("5. 已经得到的研究结果", [
        "原始 60 只股票、2023-2025 年实验：每 5 个交易日调仓，动量累计净收益约 -64.60%，反转约 -49.71%，低波动约 +24.84%；动量改为每 20 个交易日调仓后约 -41.31%。",
        "这些结果只对应原始股票池及既定成本假设。扩展股票池的回测用于验证平台能处理更大数据，不能把原报告的收益直接当成新股票池的收益。"]),
    ("6. 当前进度与后续边界", [
        ("扩展数据下载、文件哈希核验和全区间回测账本检查已完成。" if accepted else
         "扩展数据仍在下载或验收中；完成后会校验文件、统计质量，并运行全区间回测。"),
        ("新旧数据集切换、最新日期显示及扩展数据的界面回测已实际验证。" if ui_checked else
         "扩展数据完成后，再验证界面切换、日期范围和交互回测；原课程数据与报告继续保留。"),
        "当前没有训练神经网络，也没有复现 CogAlpha 的完整系统；论文用于参考研究流程和扩展方向。平台没有接入实盘下单。课程报告中组员姓名仍待填写。",
        "新股票池是在 2019-12-31 按成交额选定的 1,000 只主板股票，研究从 2020 年开始。它不是全市场或动态指数成分，不包含此后新上市公司；有单日流动性选样偏差。"]),
]
rows = [
    ["项目", "原始课程快照", "扩展快照"],
    ["股票规模", "60 只", "1,000 只" if complete else "目标 1,000 只"],
    ["有效记录", "46,759 条", f"{progress['rows']:,} 条" + ("（已完成）" if complete else "（下载中）")],
    ["行情时间", "2022-10-10 至 2025-12-31", (f"{summary['first_date']} 至 {summary['last_date']}" if summary else "请求 2019-10-01 至 2026-09-18")],
    ["研究起点", "2023-01-03", "2020-01-02"],
]

pdfmetrics.registerFont(TTFont("Hei", "C:/Windows/Fonts/simhei.ttf"))
ink = colors.HexColor("#142B43")
accent = colors.HexColor("#168579")
styles = {
    "title": ParagraphStyle("title", fontName="Hei", fontSize=22, leading=29, textColor=ink, spaceAfter=8),
    "meta": ParagraphStyle("meta", fontName="Hei", fontSize=9, leading=14, textColor=colors.HexColor("#5b6e7e"), spaceAfter=9),
    "h": ParagraphStyle("h", fontName="Hei", fontSize=12.5, leading=19, textColor=accent, spaceBefore=10, spaceAfter=5),
    "p": ParagraphStyle("p", fontName="Hei", fontSize=10, leading=16, textColor=ink, spaceAfter=6, wordWrap="CJK"),
    "cell": ParagraphStyle("cell", fontName="Hei", fontSize=9, leading=14, textColor=ink, wordWrap="CJK"),
}
story = [Paragraph("计算金融 Project 1 · 工作进展", styles["title"]),
         Paragraph(escape(stamp) + "｜已完成内容、验证证据与当前状态", styles["meta"])]
markdown = ["# 计算金融 Project 1 工作进展", "", stamp, ""]
for number, (heading, paragraphs) in enumerate(sections):
    if number == 3:
        story.append(PageBreak())
    story.append(Paragraph(heading, styles["h"]))
    markdown += ["## " + heading, ""]
    for text in paragraphs:
        story.append(Paragraph(escape(text), styles["p"]))
        markdown += [text, ""]
    if number == 2:
        table = Table([[Paragraph(escape(x), styles["cell"]) for x in row] for row in rows],
                      colWidths=[62, 160, 277], hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e5f1ef")),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LINEBELOW", (0,0), (-1,0), .6, accent),
            ("LINEBELOW", (0,1), (-1,-1), .3, colors.HexColor("#d9e3e9")),
            ("TOPPADDING", (0,0), (-1,-1), 7), ("BOTTOMPADDING", (0,0), (-1,-1), 7)]))
        story += [table, Spacer(1,9), Paragraph(escape(new_status), styles["p"]),
                  Paragraph(escape(new_volume), styles["meta"])]
        if new_quality:
            story.append(Paragraph(escape(new_quality), styles["meta"]))
        markdown += ["| " + " | ".join(rows[0]) + " |", "|---|---|---|"]
        markdown += ["| " + " | ".join(r) + " |" for r in rows[1:]]
        markdown += ["", new_status, "", new_volume, ""]
        if new_quality:
            markdown += [new_quality, ""]
story += [Spacer(1,8), Paragraph(f'<link href="{repo}" color="#168579">GitHub：caolvchong218/cf2026-quant-platform</link>', styles["p"]),
          Paragraph("本机项目：D:/Desktop/cf2026-quant-platform；启动：双击 launch.cmd。", styles["meta"])]
markdown += [f"仓库：{repo}", "", "本机项目：D:/Desktop/cf2026-quant-platform；双击 launch.cmd 启动。", ""]

def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor("#d9e3e9"))
    canvas.line(48, 39, A4[0]-48, 39)
    canvas.setFont("Hei", 8)
    canvas.setFillColor(colors.HexColor("#5b6e7e"))
    canvas.drawString(48, 26, "青序 QUANT / 项目工作进展")
    canvas.drawRightString(A4[0]-48, 26, str(doc.page))

SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=48, leftMargin=48,
                  topMargin=40, bottomMargin=50, title="计算金融 Project 1 工作进展", author="CF2026 Project Team").build(story, onFirstPage=footer, onLaterPages=footer)
TEXT.write_text("\n".join(markdown), encoding="utf-8")
DELIVERY.mkdir(parents=True, exist_ok=True)
shutil.copy2(OUT, DELIVERY/"工作进展报告_20260921.pdf")
shutil.copy2(TEXT, DELIVERY/"工作进展报告_20260921.md")
print(str(OUT))
print(new_status)
