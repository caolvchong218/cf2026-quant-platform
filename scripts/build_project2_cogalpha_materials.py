"""Build independent CogAlpha materials from finalized, real study evidence.

--prepare-sources extracts attributed paper figures and creates method diagrams;
it does not fabricate an experiment or write a final report. The normal build
fails closed unless all three declared replicates and ledger checks are present.
Compilation/rendering/visual review are explicit subsequent release operations.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "evidence/project2_cogalpha"
OUT = ROOT / "reports/project2_cogalpha"
FIG = OUT / "figures"
REF = E / "reference"
MEMBERS = "12412408 胡博文；12410819 谢明飞；12410436 王天希"
SECONDS = [20,45,65,90,60,50,75,65,75,55,100,60,95,45,0,0]
ARM = {"fixed16": "固定16因子", "llm_initial21": "初代21槽", "llm_fresh49": "独立新提案49槽",
       "llm_evolved49": "进化49槽", "llm_evolved_diverse": "进化+去冗余", "literal_1": "Listing 1",
       "literal_2": "Listing 2", "literal_3": "Listing 3"}
COLORS = ["#168579", "#4669A1", "#B37A50", "#929CA7"]
PAPER = {"ic": .0591, "rank_ic": .0814, "ic_ir": .3410, "rank_ic_ir": .4350, "aer": .1639, "information_ratio": 1.8999}
MODEL_SOURCE = {"url":"https://api-docs.deepseek.com/quick_start/pricing", "page_title":"Your First API Call",
                "verified_date":"2026-10-19", "requested_alias":"deepseek-flash",
                "official_same_day_mapping":"DeepSeek-V4.1-Flash-1017",
                "evidence_location":"Model table footnote (1); live page verified on 2026-10-19",
                "meaning":"The configured observed_version comes from the official alias mapping, not an immutable checkpoint independently attested by the API response.",
                "limitation":"Server weights and API sampling seed cannot be pinned; preserve original responses and frozen formulas for offline reevaluation."}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def dump(path, value):
    write(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tex(value):
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "$": r"\$",
               "{": r"\{", "}": r"\}", "β": r"\ensuremath{\beta}", "α": r"\ensuremath{\alpha}",
               "ε": r"\ensuremath{\varepsilon}", "Δ": r"\ensuremath{\Delta}", "σ": r"\ensuremath{\sigma}",
               "→": r"\ensuremath{\to}", "≠": r"\ensuremath{\ne}", "≤": r"\ensuremath{\le}", "≥": r"\ensuremath{\ge}",
               "ρ":r"\ensuremath{\rho}", "λ":r"\ensuremath{\lambda}", "μ":r"\ensuremath{\mu}", "θ":r"\ensuremath{\theta}"}
    return "".join(mapping.get(c, c) for c in str(value))


def number(value, digits=4):
    return "不可得" if value is None or not np.isfinite(float(value)) else f"{float(value):.{digits}f}"


def percent(value):
    return "不可得" if value is None or not np.isfinite(float(value)) else f"{100*float(value):.2f}%"


def points(value):
    return "不可得" if value is None or not np.isfinite(float(value)) else f"{100*float(value):+.2f}"


def signed(value, digits=4):
    return "不可得" if value is None or not np.isfinite(float(value)) else f"{float(value):+.{digits}f}"


def table(headers, rows, spec=None, size="small"):
    spec = spec or ("X" + "r" * (len(headers)-1))
    return "\n".join([rf"\begin{{center}}\{size}\begin{{tabularx}}{{\textwidth}}{{{spec}}}\toprule",
                      " & ".join(map(tex, headers)) + r"\\\midrule",
                      *(" & ".join(map(tex, row)) + r"\\" for row in rows),
                      r"\bottomrule\end{tabularx}\end{center}"])


def img(name, height="76mm", width=r".97\textwidth"):
    return rf"\begin{{center}}\includegraphics[width={width},height={height},keepaspectratio]{{figures/{name}}}\end{{center}}"


def pair(left, right, height="68mm"):
    return (r"\begin{center}\begin{minipage}[t]{.49\textwidth}\centering" +
            rf"\includegraphics[width=\linewidth,height={height},keepaspectratio]{{figures/{left}}}" +
            r"\end{minipage}\hfill\begin{minipage}[t]{.49\textwidth}\centering" +
            rf"\includegraphics[width=\linewidth,height={height},keepaspectratio]{{figures/{right}}}" +
            r"\end{minipage}\end{center}")


def savefig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{extension}", dpi=190, bbox_inches="tight")
    plt.close(fig)


def style():
    plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 10, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.labelcolor": "#142B43",
                         "text.color": "#142B43", "axes.titleweight": "bold", "figure.facecolor": "white"})


def prepare_sources(source_pdf):
    source_pdf = Path(source_pdf)
    config = read(ROOT / "configs/project2_cogalpha.json")
    erratum_path=ROOT/"configs/project2_cogalpha_source_erratum.json"
    expected=config["paper_sha256"].lower()
    if erratum_path.exists():
        erratum=read(erratum_path)
        if (erratum["frozen_config_sha256"] != sha(ROOT/"configs/project2_cogalpha.json")
                or erratum["incorrect_frozen_value"] != expected):
            raise ValueError("Source erratum does not identify this frozen configuration")
        expected=erratum["correct_pdf_sha256"]
    if sha(source_pdf) != expected:
        raise ValueError("Paper hash differs from the audited arXiv v4")
    temporary = ROOT / "tmp/project2_cogalpha_source"
    temporary.mkdir(parents=True, exist_ok=True)
    REF.mkdir(parents=True, exist_ok=True)
    for page in (5, 7, 8, 12, 16, 17, 19):
        subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page), "-r", "150", "-png",
                        str(source_pdf), str(temporary / "page")], check=True, capture_output=True)
    regions = {"source_figure1.png": (5, (140,145,1100,757)), "source_figure2.png": (12, (140,140,1100,679)),
               "source_table1.png": (7, (140,140,1100,735)), "source_table3.png": (16,(140,140,1100,431)),
               "source_table5.png": (19, (140,140,1100,982)),
               "source_listing1.png": (8, (114,451,607,697)), "source_listing2.png": (17, (608,665,1099,859)),
               "source_listing3.png": (17, (608,1208,1099,1550))}
    entries = []
    for name, (page, box) in regions.items():
        with Image.open(temporary / f"page-{page:02d}.png") as source:
            source.crop(box).save(REF / name)
        entries.append({"file": name, "pdf_page_1based": page, "dpi": 150,
                        "crop_xyxy": list(box), "sha256": sha(REF/name), "caption_retained": True})
    # Clearly declared, noncontiguous Table 1 excerpt retains original header and
    # all LLM/CogAlpha rows; the full original table is included in the report.
    with Image.open(temporary / "page-07.png") as page:
        header, tail = page.crop((140,140,1100,212)), page.crop((140,516,1100,735))
        focus = Image.new("RGB", (960, header.height + tail.height + 15), "white")
        focus.paste(header, (0,0)); focus.paste(tail, (0,header.height+15))
        focus.save(REF / "source_table1_focus.png")
    entries.append({"file": "source_table1_focus.png", "pdf_page_1based": 7, "dpi": 150,
                    "crop_regions_xyxy": [[140,140,1100,212],[140,516,1100,735]], "gap_pixels": 15,
                    "description": "Noncontiguous excerpt: original column headings + all LLM and CogAlpha rows + original caption; middle rows omitted, full table supplied separately.",
                    "sha256": sha(REF / "source_table1_focus.png"), "caption_retained": True})
    dump(E / "source_reference.json", {"paper": "Cognitive Alpha Mining via LLM-Driven Code-Based Evolution",
        "authors": "Fengyuan Liu et al.", "arxiv": "https://arxiv.org/abs/2511.18850v4",
        "pdf_sha256": sha(source_pdf), "source_identity_erratum": "configs/project2_cogalpha_source_erratum.json",
        "frozen_config_sha256": sha(ROOT/"configs/project2_cogalpha.json"), "page_count": 35, "render": "Poppler pdftoppm 150dpi",
        "use": "Attributed excerpts for critical comparison; no third-party runtime or private data copied", "excerpts": entries,
        "model_identity_source": MODEL_SOURCE})


def method_figures():
    style()
    fig, ax = plt.subplots(figsize=(9.5,5)); ax.axis("off"); ax.set(xlim=(-.15,10.1),ylim=(0,5))
    nodes = [(0.1,3.6,"七层21角色\n只接收训练信息"),(3.6,3.6,"受限DSL提案\n静态检查+数值质量"),(7.1,3.6,"五指标评分\n保留失败与血缘"),
             (0.1,1.0,"同初始池21槽\n演化/独立新提案"),(3.6,1.0,"每臂最终49槽\n固定预算、三次重复"),(7.1,1.0,"冻结因子与模型\n才看后续区间")]
    for x,y,label in nodes:
        ax.add_patch(FancyBboxPatch((x,y),2.7,1,boxstyle="round,pad=.14",facecolor="#EAF3F2",edgecolor="#168579"))
        ax.text(x+1.35,y+.5,label,ha="center",va="center",fontsize=12)
    for a,b in [((2.95,4.1),(3.4,4.1)),((6.45,4.1),(6.95,4.1)),((8.45,3.4),(1.45,2.2)),((2.95,1.5),(3.4,1.5)),((6.45,1.5),(6.95,1.5))]:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=16,color="#142B43"))
    savefig(fig,"local_workflow")
    fig, ax = plt.subplots(figsize=(10,3.2)); ax.axis("off"); ax.set(xlim=(-.15,10.1),ylim=(0,3))
    for x,w,color,label in [(0,4,COLORS[0],"训练 2020—2022\n提案、筛选、模型拟合"),(4.2,2.7,COLORS[1],"验证 2023—2024\n仅报告，不重训"),(7.1,2.8,COLORS[2],"测试 2025—2026.09\n固定主比较")]:
        ax.add_patch(FancyBboxPatch((x,.9),w,1.2,boxstyle="round,pad=.05",facecolor=color,edgecolor="none")); ax.text(x+w/2,1.5,label,color="white",ha="center",va="center",fontsize=11)
    ax.text(5,.35,"信号 t 收盘可得 → t+1 开盘进入 → t+11 开盘退出；边界标签严格剔除",ha="center",fontsize=11)
    savefig(fig,"split_timeline")
    fig, ax = plt.subplots(figsize=(10,4)); ax.axis("off"); ax.set(xlim=(-.15,10.1),ylim=(0,4))
    descriptions=[(.1,1.4,2.3,"共享初代\n21角色 × 1提案"),(3.4,2.4,3.0,"演化臂：2代\n每代7变异 + 7交叉"),(3.4,.4,3.0,"新提案对照：2代\n每代14个新想法"),(7.5,2.4,2.2,"21 + 28 = 49槽\n有父公式与血缘"),(7.5,.4,2.2,"21 + 28 = 49槽\n无父公式")]
    for x,y,w,text in descriptions:
        ax.add_patch(FancyBboxPatch((x,y),w,1,boxstyle="round,pad=.1",facecolor="#EFF3F7",edgecolor="#4669A1")); ax.text(x+w/2,y+.5,text,ha="center",va="center",fontsize=11)
    for a,b in [((2.55,2),(3.2,2.85)),((2.55,1.7),(3.2,.95)),((6.55,2.9),(7.3,2.9)),((6.55,.9),(7.3,.9))]:ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=15,color="#168579"))
    ax.text(5,3.8,"每次重复77个独立槽；失败不补位，修复最多1次且计费",ha="center",fontsize=11)
    savefig(fig,"matched_budget")


def load():
    required = ["manifest.json","protocol.json","performance.csv","comparison.csv","pool_summary.csv","qualification.csv",
                "factor_metrics.csv","selection.json","lineage.json","budget.json","validation.json","data_quality.json","proposals.json","source_reference.json"]
    missing=[x for x in required if not (E/x).exists()]
    if missing: raise FileNotFoundError("Formal evidence not ready; no fabricated values: " + ", ".join(missing))
    d={Path(n).stem: pd.read_csv(E/n) if n.endswith(".csv") else read(E/n) for n in required}
    if not d["validation"].get("all_ledgers_passed") or not d["validation"].get("no_test_feedback"):
        raise ValueError("Finalized ledger and training-only acceptance are required")
    if d["manifest"].get("exact_paper_table_replication") is not False:
        raise ValueError("Mechanism replication boundary must remain explicit")
    d["seeds"]=d["protocol"]["experiments"]["replicate_seeds"]
    if len(d["seeds"]) != 3 or len(set(d["seeds"])) != 3: raise ValueError("All three declared replicates are required")
    p=d["performance"]
    if len(p)!=90 or d["validation"].get("portfolio_runs")!=90:
        raise ValueError("All 90 finalized portfolio results are required")
    if p.duplicated(["replicate","arm","model","phase","cost_case"]).any(): raise ValueError("Ambiguous portfolio identities")
    d["lineage_by_seed"]={int(r["seed"]):r for r in d["lineage"]}
    for seed in d["seeds"]:
        if seed not in d["lineage_by_seed"] or not d["lineage_by_seed"][seed].get("completed"):
            raise ValueError("Incomplete proposal replicate")
        for arm in ("llm_fresh49","llm_evolved49","llm_evolved_diverse"):
            pick(d,seed,arm)
    for name,expected in d["manifest"].get("files",{}).items():
        if not (E/name).is_file() or sha(E/name) != expected:
            raise ValueError("Evidence hash changed: " + name)
    for name,expected in d["manifest"].get("calculation_sha256",{}).items():
        if not (ROOT/name).is_file() or sha(ROOT/name) != expected:
            raise ValueError("Calculation source differs from finalized evidence: " + name)
    erratum=read(ROOT/"configs/project2_cogalpha_source_erratum.json")
    if d["source_reference"]["pdf_sha256"] != erratum["correct_pdf_sha256"]:
        raise ValueError("Source excerpts must identify the corrected audited PDF")
    FIG.mkdir(parents=True,exist_ok=True)
    for path in REF.glob("*.png"): shutil.copy2(path,FIG/path.name)
    return d


def pick(d,seed,arm,model="lightgbm",phase="test",cost="paper"):
    p=d["performance"]
    q=p[(p.replicate==seed)&(p.arm==arm)&(p.model==model)&(p.phase==phase)&(p.cost_case==cost)]
    if len(q)!=1:raise ValueError(f"Expected one real portfolio row: {seed}/{arm}/{model}/{phase}/{cost}")
    return q.iloc[0]


def delta_rows(d,improved="llm_evolved49",baseline="llm_fresh49",model="lightgbm"):
    rows=[]
    for seed in d["seeds"]:
        a,b=pick(d,seed,improved,model),pick(d,seed,baseline,model)
        rows.append({"seed":seed,"base":b,"new":a,**{k:float(a[k])-float(b[k]) for k in ("rank_ic","cagr","sharpe","max_drawdown")}})
    return rows


def summary(d,improved="llm_evolved49",baseline="llm_fresh49"):
    rows=delta_rows(d,improved,baseline)
    values=np.array([r["rank_ic"] for r in rows]); finite=values[np.isfinite(values)]
    if len(finite)!=3:return "至少一次重复的预测指标不可得；不能给出完整的三次重复均值结论。"
    wins=int((values>0).sum())
    return f"三次重复中{wins}/3次测试RankIC提高，平均变化{signed(values.mean())}，范围{signed(values.min())}至{signed(values.max())}。"


def experiment_figures(d):
    style(); seeds=d["seeds"]; x=np.arange(3)
    for filename,arms in [("primary_rankic",["llm_fresh49","llm_evolved49"]),("diverse_rankic",["llm_evolved49","llm_evolved_diverse"])]:
        fig,ax=plt.subplots(figsize=(9,4.1))
        for j,arm in enumerate(arms):ax.bar(x+(j-.5)*.32,[pick(d,s,arm).rank_ic for s in seeds],.30,label=ARM[arm],color=COLORS[j])
        ax.axhline(0,color="#9AA5AE",lw=.8);ax.set(xticks=x,xticklabels=[f"重复 {s}" for s in seeds],ylabel="测试期平均 RankIC",title="固定 LightGBM；全部三次重复")
        ax.legend(frameon=False);ax.grid(axis="y",alpha=.2);savefig(fig,filename)
    fig,axes=plt.subplots(3,1,figsize=(10,9.5),sharex=True)
    for ax,seed in zip(axes,seeds):
        for j,arm in enumerate(("llm_fresh49","llm_evolved49","llm_evolved_diverse")):
            row=pick(d,seed,arm);frame=pd.read_csv(E/row.daily_file,parse_dates=["date"])
            ax.plot(frame.date,frame.nav,label=ARM[arm],color=COLORS[j],lw=1.2)
        ax.plot(frame.date,frame.benchmark_nav,label="沪深300价格指数",color=COLORS[3],lw=1,ls="--")
        ax.set(ylabel="净值",title=f"重复 {seed} · 测试期 · 指定费用");ax.grid(alpha=.18)
    axes[0].legend(ncol=2,frameon=False,fontsize=9);savefig(fig,"portfolio_all_seeds")
    # One declared first-seed illustration for a legible slide; the report shows all three.
    fig,ax=plt.subplots(figsize=(10,4.1))
    for j,arm in enumerate(("llm_fresh49","llm_evolved49","llm_evolved_diverse")):
        row=pick(d,seeds[0],arm);frame=pd.read_csv(E/row.daily_file,parse_dates=["date"])
        ax.plot(frame.date,frame.nav,label=ARM[arm],color=COLORS[j])
    ax.plot(frame.date,frame.benchmark_nav,label="沪深300价格指数",color=COLORS[3],ls="--")
    ax.set(ylabel="净值",title=f"预先固定展示第一重复 {seeds[0]}；非最优曲线筛选");ax.legend(ncol=2,fontsize=9,frameon=False);ax.grid(alpha=.2);savefig(fig,"portfolio_first_seed")
    fig,ax=plt.subplots(figsize=(8,4))
    vals=[pick(d,s,"llm_evolved49").rank_ic for s in seeds]
    ax.bar([str(s) for s in seeds],vals,color=COLORS[0]);ax.axhline(0,color="gray",lw=.6)
    ax.set(ylabel="本地测试 RankIC",title="三个重复均报告；不与原表求直接误差");ax.grid(axis="y",alpha=.2);savefig(fig,"local_table_comparison")


def manual_checks():
    sys.path.insert(0,str(ROOT/"src"))
    from cfquant.cogalpha import paper_literal_candidates
    from cfquant.factor_lab import evaluate_expression
    values={"open":10.,"high":12.,"low":9.,"close":11.,"volume":100000.}
    panels={k:pd.DataFrame([[v]],index=pd.DatetimeIndex(["2020-01-02"]),columns=["TOY"]) for k,v in values.items()}
    cards=paper_literal_candidates();eps=float(cards[0]["denominator_epsilon"])
    expected=[1/(100000+eps),3/(100000+eps),np.tanh(1/(1100000+eps))]
    rows=[]
    for c,e in zip(cards,expected):
        actual=float(evaluate_expression(c["expression"],panels).iloc[0,0])
        if not np.isclose(actual,e,rtol=1e-12,atol=1e-15):raise ValueError("Literal formula differs from hand computation")
        rows.append({"id":c["id"],"expression":c["expression"],"epsilon":eps,"expected":float(e),"actual":actual,"passed":True})
    result={"toy_only_not_market_data":True,"inputs":values,"paper_epsilon":1e-9,"implementation_epsilon":eps,"checks":rows}
    dump(OUT/"manual_formula_checks.json",result)
    return result


def report(d,checks,test_claim,ui_claim):
    rows=delta_rows(d);diverse=delta_rows(d,"llm_evolved_diverse","llm_evolved49")
    p=d["protocol"];budget=d["budget"]
    out=[r"""\documentclass[UTF8,a4paper,10pt]{ctexart}
\usepackage[margin=19mm,top=20mm,bottom=20mm]{geometry}
\usepackage{amsmath,amssymb,booktabs,tabularx,graphicx,xcolor,fancyhdr,enumitem}
\usepackage[colorlinks=true,urlcolor=teal,linkcolor=teal]{hyperref}
\definecolor{ink}{HTML}{142B43}\definecolor{accent}{HTML}{168579}\definecolor{teal}{HTML}{168579}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small CF2026 Project 2 · CogAlpha 独立研究}\fancyhead[R]{\small 青序量化平台}
\fancyfoot[L]{\scriptsize 机制复现 / 同预算消融 / 独立去冗余改进}\fancyfoot[R]{\thepage}
\setlength{\headheight}{15pt}\setlength{\parskip}{5pt}\setlength{\parindent}{2em}\setlength{\emergencystretch}{2em}
\setlist{nosep,leftmargin=1.7em}\ctexset{section={format=\Large\bfseries\color{ink}},subsection={format=\normalsize\bfseries\color{accent}}}
\newcommand{\lead}[1]{\par\noindent{\bfseries\color{accent}#1}\par}
\newcommand{\page}[1]{\newpage\section*{#1}}
\begin{document}
\begin{center}{\small\color{accent}COMPUTATIONAL FINANCE / PROJECT 2}\\[9pt]
{\Huge\bfseries\color{ink}CogAlpha 因子研发研究}\\[7pt]
{\Large 受限机制复现、同预算进化对照与去冗余改进}\\[8pt]
2026年10月19日\quad 截止：2026年11月1日
\end{center}
""",tex(MEMBERS)+r"\par",r"\section*{1\quad 摘要与研究结论}",
    "本项目独立研究 Cognitive Alpha Mining via LLM-Driven Code-Based Evolution。我们把七层21角色、因子质量检查、五指标资格判定和变异/交叉机制接入青序平台，采用真实模型提案和已有Tushare快照进行有限规模实验。RSRS路线继续保留；两条研究共享数据、指标、回测和审计基础设施。",
    r"\lead{主问题：同样的候选槽位，进化是否优于持续提出新想法？}",tex(summary(d)),
    "主比较预先固定为LightGBM、测试期平均RankIC、进化49槽相对新提案49槽。两个分支共享21个初代候选，各新增28槽，失败不补位。三个重复全部保留，未按测试成绩选择种子、模型或新参数。",
    table(["重复","新提案RankIC","进化RankIC","差值"],[[r["seed"],number(r["base"].rank_ic),number(r["new"].rank_ic),signed(r["rank_ic"])] for r in rows]),
    r"\lead{独立改进：只用训练期IC路径剔除冗余}",tex(summary(d,"llm_evolved_diverse","llm_evolved49")),
    r"\lead{结论边界}","本实验是不同数据与不同模型规模下的机制复现，不是原文120B模型、H100训练或Table 1/5数值的精确重现。股票池是静态快照；历史已经被研究者看过，模型知识也可能覆盖后续时期。代码层面的未来信息隔离不能把这段历史变成独立盲测。"]
    out += [r"\page{2\quad 原论文与本次系统：相邻看共同点和差异}",pair("source_figure1.png","local_workflow.png","77mm"),
    r"{\footnotesize 左：原文PDF第5页Figure 1；右：青序本次有限工作流。原图为署名文献摘录。}\par",
    table(["维度","原论文","本次实验"],[
      ["语言模型","本地gpt-oss-120b/H100",p["model"]["name"]+"官方API"],
      ["因子表示","生成Python/TA-Lib代码","受限OHLCV表达式；不执行任意代码"],
      ["搜索规模","初代80、父池32、子池96；多轮","21初代+2代；两臂各49槽，三个重复"],
      ["训练方式","126日滚动训练","训练期拟合一次；后续不重训"],
      ["数据","Qlib CSI300等多市场","Tushare静态1000股A股快照"],
      ["公开代码","提示词仓库；完整运行时未公开","协议、提案血缘、公式、测试与公开汇总"]],"p{24mm}XX"),
    "作者公开仓库仅提供提示词模板，未公开完整运行时、最终20个表达式、数据版本或全部随机状态。本次提示词依据七层角色组织，但数值质量检查、预算账本和安全DSL是青序的工程实现。经济解释是待检验的假设，不是已经观测到的订单流或投资者行为。"]
    dq=d["data_quality"]
    out += [r"\page{3\quad 数据、标签时点与时间隔离}",img("split_timeline.png","63mm"),
      table(["字段","实际记录"],[["数据日期",f"{dq['first']} 至 {dq['last']}"],["资产/记录/交易日",f"{dq['assets']} / {dq['rows']:,} / {dq['sessions']}"],
        ["可用训练日期",dq["training_dates"]],["最后训练标签退出日",dq["last_training_label_exit"]]],"p{43mm}X"),
      r"\[ y_{i,t}^{(10)}=\frac{O^{adj}_{i,t+11}}{O^{adj}_{i,t+1}}-1,\qquad t\text{日收盘形成信号。}\]",
      "训练样本不仅要求形成日在训练区间内，还要求标签退出日严格早于下一切分边界。验证/测试评价同样剔除尚未实现的退出标签；终点附近不凭空补未来收益。因子提案、修复、亲本选择、去冗余和模型拟合都只接收训练统计。",
      "价格按当时复权因子相对首个观测基准调整，成交量按相反比例调整，以保持价格乘成交量的金额口径。缺失OHLCV不向前填充；资格掩码不使用未来标签。1000股静态成员不是历史沪深300成分，因此有幸存者与静态选样偏差。"]
    out += [r"\page{4\quad 论文示例公式与手算：先确认程序算对了}",pair("source_listing1.png","source_listing3.png","67mm"),
      r"{\footnotesize 左：原文第8页Listing 1；右：第17页Listing 3。第17页Listing 2将第一式的分子改为$H-L$。}\par",
      r"\[ f_1=\frac{H-C}{V+\varepsilon},\qquad f_2=\frac{H-L}{V+\varepsilon},\qquad f_3=\tanh\!\left(\frac{|C-O|}{VC+\varepsilon}\right).\]",
      "以教学数值O=10、H=12、L=9、C=11、V=100000股代入；这些数值不是行情样本，也不用于实证。",
      table(["公式","手算值","DSL执行值","检查"],[[f"Listing {i+1}",f"{r['expected']:.10g}",f"{r['actual']:.10g}","通过"] for i,r in enumerate(checks["checks"])]),
      tex(f"原文epsilon=1e-9；本次实现epsilon={checks['implementation_epsilon']:g}。") + ("数值稳定项一致。" if checks["implementation_epsilon"]==1e-9 else "本次采用了不同稳定项，应视为注明差异的译写，不能声称逐字复现。"),
      "第一式的H-C是最高价至收盘的距离，不等于开盘到收盘的上涨；报告保留源公式，同时指出原经济叙述的歧义。三个示例按正向定义评价，不根据后验表现反转符号。"]
    out += [r"\page{5\quad 五项评价、资格门槛与受限解释}",
      r"\[ IC_t=\operatorname{corr}_i(f_{i,t},y_{i,t}),\quad RankIC_t=\operatorname{corr}_i(\operatorname{rank}f_{i,t},\operatorname{rank}y_{i,t}).\]",
      r"\[ ICIR=\overline{IC}/s(IC),\qquad RankICIR=\overline{RankIC}/s(RankIC),\quad ddof=1.\]",
      "预测信息比率不年化。每日横截面至少30个有效配对；均值按有效日期计算。MI采用每日秩五分位离散化的插件估计，以自然对数计量。原文给出互信息定义，但没有公开估计器，故MI是本次明确标注的替代实现，有限样本下可能向上偏。",
      table(["阶段","分位数","IC / RankIC底线","ICIR / RankICIR底线","MI底线"],[["qualified","65%","0.005","0.05","0.02"],["elite","80%","0.010","0.10","0.02"]],"lrrrr"),
      "每项实际门槛取池内分位数与绝对底线的较大者，五项全部满足才具资格；本地实现采用大于等于比较，不取绝对值。缺失或非有限值不能通过资格。本次在截至当前槽已评分的累计池上重算分位数，原文描述的是同代候选池；小规模累计池是明确的工程改造，不冒充作者的代际筛选实现。",
      r"\lead{资格、工程亲本池和最终建模池分别报告}",
      "进化优先使用qualified且训练RankIC为正的候选，并优先elite。若五项交集为空，允许使用质量通过、训练RankIC为正的工程回退亲本池；每次回退明确标记，绝不改称qualified。最终组合另按预先固定的正训练RankIC顺序选择最多20个因子；论文最终选择细节未公开，因此这也是本地规则。",
      r"\lead{安全与时序质量检查}",tex("表达式先通过AST白名单，再检查数值、覆盖率和因果性。最大单窗口120日；禁止任意代码、属性访问和未来位移。前缀重算与未来扰动检查都在训练窗口完成，覆盖率至少70%。通过这些检查说明计算边界满足要求，不等于因子有经济价值。")]
    attempt_rows=[]
    for seed in d["seeds"]:
        result=d["lineage_by_seed"][seed];cards=list(result["cards"].values());counts=Counter(c["status"] for c in cards)
        attempt_rows.append([seed,len(cards),counts.get("accepted",0),sum(v for k,v in counts.items() if k!="accepted"),sum(len(c["attempts"]) for c in cards),sum(bool(c["repair"]) for c in cards),sum(c["parent_policy"]=="fallback_engineering_positive_rankic" for c in cards)])
    out += [r"\page{6\quad 同预算对照、失败槽与真实调用}",img("matched_budget.png","66mm"),
      table(["重复","独立槽","通过","拒绝","请求尝试","修复槽","回退亲本"],attempt_rows,"lrrrrrr"),
      "同预算指相同的候选槽位上限：两臂都从共享21槽扩展到49槽。新提案对照获得同类训练统计，不获得父公式。重复表达式、无效公式和API失败均占用原槽且保留；每槽最多一次修复且计费，因此实际调用数及费用可能不同，不能声称token完全相同。",
      table(["预算项","实际账本"],[["全局费用上限",f"{budget['budget_cny']:.2f}元"],["全局调用/成功",f"{budget['calls']} / {budget['successful_calls']}"],
        ["成功请求计费",f"{budget['successful_cost_cny']:.6f}元"],["不确定请求保守预留",f"{budget['uncertain_reserved_cny']:.6f}元"],
        ["总已花或预留",f"{budget['reserved_or_spent_cny']:.6f}元"]],"p{60mm}X"),
      "预算为三个重复、连接检查及修复共同使用；失败请求按保守上界保留预算，预留不等于已确认账单。每次请求保存namespace、请求哈希和响应模型字段。父因子抽样使用本地种子，但API模型采样没有可控seed；缓存提供请求级复现，不能保证重新付费生成相同表达式。",
      "请求别名是deepseek-flash；官网2026年10月19日映射为DeepSeek-V4.1-Flash-1017。响应中的model字段只证明返回的名称，不独立证明不可变checkpoint；无法钉死服务端权重。原始响应与冻结公式保留后可离线重评。",
      r"{\footnotesize 模型身份来源：\url{https://api-docs.deepseek.com/quick_start/pricing}，当日首次调用页模型表脚注。}\par"]
    pools=d["pool_summary"]
    pool_rows=[]
    for r in pools.itertuples():
        candidates=d["qualification"]
        candidates=candidates[(candidates.replicate==r.replicate)&(candidates.arm==r.arm)]
        positive=int((pd.to_numeric(candidates.rank_ic,errors="coerce")>0).sum())
        pool_rows.append([r.replicate,ARM.get(r.arm,r.arm),r.proposal_slots,r.valid_candidates,r.qualified_candidates,r.elite_candidates,positive,r.selected_count])
    out += [r"\page{7\quad 候选池与最终选择：保留全部重复}",
      table(["重复","候选臂","槽位","质量通过","qualified","elite","正RankIC","入模"],pool_rows,"lp{29mm}rrrrrr","scriptsize"),
      "“正RankIC”指训练期平均RankIC大于零的质量通过候选数。质量通过只说明公式可计算且符合时序检查；qualified/elite要求五项共同达标；最终入模另按正训练RankIC排序、最多20个，去冗余臂再过滤相关路径。三种选择层级不能混称。论文三个字面公式作为独立基准原向执行，不因训练符号而删去。不可得值、拒绝理由、代数和父ID均在公开血缘记录中保留。",
      r"\lead{同预算对照解释的边界}","两个分支共享初代，重复检测在各臂内部执行，防止一条分支先提出某公式就额外惩罚另一条分支。相同槽位不消除随机模型输出差异；三个重复用于检查方向和离散程度，不足以声称统计显著或全市场稳定。"]
    out += [r"\page{8\quad 固定组合模型与组合交易}",
      table(["项目","本次固定口径"],[["因子/标签缩放","同日合资格横截面秩减0.5"],["rank组合","选定因子的等权平均秩"],["Ridge",f"alpha={p['combination']['ridge_alpha']}"],
       ["LightGBM",f"{p['combination']['lightgbm_trees']}树，{p['combination']['lightgbm_leaves']}叶，深度{p['combination']['lightgbm_max_depth']}，学习率{p['combination']['lightgbm_learning_rate']}"],
       ["拟合/选择","只用训练期；不在验证期再训练或改变选择"],["持仓与调仓","Top50/Drop5，保留继续持有股票的原份额"],
       ["成交与费用","收盘t评分，t+1开盘成交；买0.05%、卖0.15%，单笔最低5元"],
       ["账户/年化","初始100万元，现金收益0；252交易日，无风险年利率0"]],"p{30mm}X"),
      "论文也采用Top50/Drop5、开盘交易与上述费用；本次复用青序成交和现金账本，使用分数化复权研究份额，不等同A股整手订单。继续持有的股票不每天重置为等权，以免人为增加换手。",
      tex("撮合按下一开盘价执行，缺失开盘或涨跌停价则阻止该笔交易；依据Tushare涨跌停价，开盘触及涨停禁止买、触及跌停禁止卖。未成交卖单仍占持仓槽。缺失收盘使用此前收盘估值，连续缺失超过20个交易日按零记账，这是研究中的保守减记规则，不是实际清算结果。未启用成交参与率限制，也未模拟整手、订单排队、冲击或额外滑点。资格池另用涨跌停比率排除类似5%限制的股票，只是ST-like代理，并非完整历史ST识别。"),
      r"\[ Sharpe=\sqrt{252}\,\frac{\overline{r^{net}}}{s(r^{net})},\qquad AER=252\,\overline{(r^{net}-r^{bench})}.\]",
      "CAGR采用复利年化；AER是平均日超额收益的算术年化，两者不能混用。最大回撤包含初始资金峰值。沪深300是免交易费的价格指数代理，从同一评价首日开盘归一化；它不是可直接购买的ETF或含分红全收益基准。股票侧采用复权总收益研究单位，因此相对该价格指数的AER不等于相对沪深300全收益指数的alpha，分红口径可能贡献差异。",
      "验证区间的预测从首个验证交易日收盘开始，因此第一天账户持有现金、最早下一开盘交易；测试区间已有前期预测信号。没有为了补齐首日交易而使用当日收盘分数在当日开盘成交。"]
    out += [r"\page{9\quad 主实验：进化49槽对独立新提案49槽}",img("source_table3.png","51mm"),
      r"{\footnotesize 原文第16页Table 3：Agent到Agent\_E的RankIC由0.0318升到0.0420，但IC由0.0300降到0.0219。原表不是本次同槽位对照。}\par",
      img("primary_rankic.png","72mm"),tex(summary(d)),
      table(["重复","新提案RankIC","进化RankIC","差值","CAGR变化(pp)","Sharpe变化"],[[r["seed"],number(r["base"].rank_ic),number(r["new"].rank_ic),signed(r["rank_ic"]),points(r["cagr"]),signed(r["sharpe"],3)] for r in rows],"lrrrrr"),
      "这里的差值都取进化减新提案，保持数据、模型、交易规则和最终选择上限一致。RankIC检验横截面排序能力，CAGR与Sharpe检验交易后的账户表现；二者方向可能不同。收益下降的重复不能被删去，也不能改用表现最好的模型来替换主比较。",
      "本次两臂都保留七层组织及训练统计反馈，检验的差别是是否提供父公式并做变异/交叉，并未分别识别原文全部模块。三个种子不是三个独立市场样本，不凭这三个观察点作显著性或未来盈利保证。"]
    out += [r"\page{10\quad 原表与本次结果相邻：只作有条件的比较}",img("source_table1.png","104mm"),
      r"{\footnotesize 原文PDF第7页Table 1完整摘录；CSI300，10日标签，原论文训练/测试区间。}\par",
      table(["来源/配置","IC","RankIC","ICIR","RankICIR"],[["原文CogAlpha",*[number(PAPER[k]) for k in ("ic","rank_ic","ic_ir","rank_ic_ir")]]]+
            [[f"本次进化/{seed}",*[number(pick(d,seed,"llm_evolved49")[k]) for k in ("ic","rank_ic","ic_ir","rank_ic_ir")]] for seed in d["seeds"]],"Xrrrr"),
      "两组使用不同股票池、日期、模型、训练频率、搜索规模和指标估计细节，因此不计算直接复现误差或对原文提升百分比。本次可识别的比较是同一冻结数据内的进化与新提案对照；原表提供研究定位，不能作为可达到收益的承诺。"]
    out += [r"\page{11\quad 扣费净值：全部三个重复同样展示}",img("portfolio_all_seeds.png","194mm"),
      "所有曲线使用测试期和指定费用；各自账户均从初始资金开始。股票侧是复权总收益研究单位，沪深300侧是免交易费价格指数；两者分红口径不同。这里跑赢价格指数不能等同于跑赢沪深300全收益基准。净值好坏不改变此前冻结的因子或模型选择；主比较RankIC不依赖该指数基准。"]
    out += [r"\page{12\quad 独立改进：训练期IC路径去冗余}",img("diverse_rankic.png","80mm"),
      "在进化候选池不变的条件下，按正训练RankIC排序逐一加入因子；若与已选因子训练期每日RankIC序列的绝对相关超过0.8，则跳过。最多20个；相关不可得或有效共同日期不足也跳过。此规则与市场条件、测试收益无关，未在测试上寻找最优阈值。",
      table(["重复","原进化RankIC","去冗余RankIC","RankIC变化","CAGR变化(pp)","回撤变化(pp)"],[[r["seed"],number(r["base"].rank_ic),number(r["new"].rank_ic),signed(r["rank_ic"]),points(r["cagr"]),points(r["max_drawdown"])] for r in diverse],"lrrrrr"),
      tex(summary(d,"llm_evolved_diverse","llm_evolved49")),
      "回撤变化按新减旧，负值表示回撤幅度降低。训练IC路径不高度相关，并不保证因子值、预测误差或最终持仓完全独立；入模数量也可能同时减少，应连同候选池表解释。我们不把去冗余结果自动提升为平台默认策略。"]
    modelrows=[]
    for model in ("rank","ridge","lightgbm"):
        for r in delta_rows(d,model=model):modelrows.append([r["seed"],model,number(r["base"].rank_ic),number(r["new"].rank_ic),signed(r["rank_ic"]),points(r["cagr"])])
    costrows=[]
    for seed in d["seeds"]:
        for arm in ("llm_fresh49","llm_evolved49"):
            charged,free=pick(d,seed,arm),pick(d,seed,arm,cost="zero")
            costrows.append([seed,ARM[arm],percent(free.cagr),percent(charged.cagr),points(charged.cagr-free.cagr)])
    anchor_rows=[]
    for seed,arm in [(0,"fixed16")]+[(seed,"llm_initial21") for seed in d["seeds"]]:
        r=pick(d,seed,arm)
        anchor_rows.append([seed,ARM[arm],number(r.rank_ic),percent(r.cagr),number(r.sharpe,3),percent(r.max_drawdown)])
    out += [r"\page{13\quad 模型与费用敏感性：不替换主比较}",
      table(["重复","组合模型","新提案RankIC","进化RankIC","差值","CAGR变化(pp)"],modelrows,"llrrrr","scriptsize"),
      "rank、Ridge与LightGBM使用同一候选池和训练期规则。上表用于理解组合器的影响；主比较仍为LightGBM，不把其中最好的行改成事后主结论。pp表示百分点，报告中收益率与回撤差值均按百分点表示。",
      table(["重复","工程基线/初代","RankIC","CAGR","Sharpe","回撤"],anchor_rows,"lXrrrr","scriptsize"),
      "上表同为LightGBM、测试期与指定费用。固定16因子是本地工程基线，不等同论文的Alpha158；初代行显示增加提案前的结果，不能把与初代的差异全部归因于演化。",
      table(["重复","候选臂","零费CAGR","指定费CAGR","成本后变化(pp)"],costrows,"lXrrr"),
      "零费与指定费用是敏感性实验，扣费后路径会因可用现金和持仓演化而改变，不能把两条终值差简单等同于累计手续费。两种费用都不影响预测RankIC，因为预测分数与标签没有变。"]
    literal=[]
    for i in (1,2,3):
        r=pick(d,0,f"literal_{i}",model="rank")
        literal.append([f"Listing {i}",number(r.ic),number(r.rank_ic),percent(r.cagr),number(r.sharpe,3),percent(abs(r.max_drawdown))])
    out += [r"\page{14\quad 示例因子与单因子实验}",
      table(["原公式","本次测试IC","RankIC","CAGR","Sharpe","回撤"],literal,"Xrrrrr"),
      "论文报告Listing 1/2/3的示例IC分别为0.0090、0.0073、0.0141，RankIC为0.0061、0.0021、0.0087，但该示例的完整样本和筛选历史未公开。本次只比较同一数据上的原公式形状，不宣称还原作者的这几个数值。单因子未重新训练为多因子模型。",
      r"\lead{实际生成表达式：按训练期排序展示，不按测试选例}"]
    for seed in d["seeds"]:
        record=d["lineage_by_seed"][seed]
        candidates=[record["cards"][i] for i in record["arms"]["evolved"] if record["cards"][i]["status"]=="accepted" and record["cards"][i]["train_metrics"].get("rank_ic") is not None]
        candidates.sort(key=lambda c:(-c["train_metrics"]["rank_ic"],c["id"]))
        if candidates:
            c=candidates[0];expression=c["card"]["expression"]
            hypothesis=c["card"]["hypothesis"]
            excerpt=hypothesis if len(hypothesis)<=280 else hypothesis[:280].rsplit(" ",1)[0]+"…（节选）"
            out += [r"\lead{"+tex(f"重复{seed}：{c['card']['label']}")+"}",tex(f"训练RankIC={number(c['train_metrics']['rank_ic'])}；代数{c['generation']}；操作{c['kind']}。"),
                    r"{\footnotesize\ttfamily\raggedright "+tex(expression).replace(",",r",\allowbreak ").replace("(",r"(\allowbreak ")+r"\par}",tex(excerpt)]
    out += [r"\page{15\quad 平台接入、复现路径与验收范围}",
      "青序平台分别展示冻结协议、原文对照、三次重复、候选血缘和账本。因子挖掘作为辅助研究路线保留，未经独立后续验证不自动替换已有策略。阅读已保存的研究结果不需要重新付费调用模型。",
      table(["层次","对应实现/证据"],[["数据与因子","cogalpha.py：指标、质量、原公式和去冗余"],["真实提案","cogalpha_mining.py + research_api.py：槽位、血缘、缓存、共享预算"],
       ["冻结模型","cogalpha_models.py：rank/Ridge/LightGBM统一训练边界"],["研究编排","cogalpha_study.py；configs/project2_cogalpha.json"],
       ["P1复用","成交、持仓、现金、费用、绩效与账本核对"],["公开证据","performance/comparison/pool/qualification/lineage/manifest"],["材料","报告TeX/PDF、Beamer、可编辑PPTX及逐页讲稿"]],"p{28mm}X"),
      r"\lead{已验证范围}",tex(test_claim),tex(ui_claim),
      tex(f"正式研究账本核验：{d['validation']['portfolio_runs']}组；all_ledgers_passed={d['validation']['all_ledgers_passed']}。具体检查逐条见validation.json。"),
      r"\lead{重跑原则}","先核验协议和输入身份，再复用已有提案缓存执行evaluate阶段。mine/all阶段会在授权的共享预算内产生真实模型调用；阶段区分防止只读验收意外付费。原始Tushare数据、密钥和训练模型不放入公开仓库；公开聚合曲线及哈希支持核对，完整重算仍需要有权限的同版数据。",
      r"{\footnotesize\ttfamily python scripts/build\_project2\_cogalpha\_materials.py\par}"]
    out += [r"\page{16\quad 限制、后续验证和来源}",
      r"\lead{不能由当前实验推出的结论}","静态股票池存在幸存者偏差；调整价格和分数化份额不是可成交整手实盘。历史已经看过且LLM可能知道后续市场，时间切分不能完全消除研究者或模型知识污染。只有三次提案重复，不能支撑普遍显著性。MI估计器、最终选择、训练周期、数据和模型均与原文不同。",
      "后续应冻结候选后收集新的时间区间，以历史成分股票池和可交易约束再次验证，并加入分红口径一致的全收益基准、不同市场阶段、手续费与冲击。股票复权单位相对价格指数的超额不能直接解释为全收益基准alpha。若未来表现不佳，保留失败结果；不在同一已见测试期反复调参直到跑赢。",
      r"\lead{来源与对应位置}",
      r"\begin{enumerate}\item Fengyuan Liu等，\textit{Cognitive Alpha Mining via LLM-Driven Code-Based Evolution}，arXiv:2511.18850v4。Figure 1第5页；Table 1第7页；Listing 1第8页；Figure 2第12页；数据与指标第14--16页；Listing 2/3第17页；Table 5第19页。\url{https://arxiv.org/abs/2511.18850v4}",
      r"\item 作者提示词仓库：\url{https://github.com/uwFengyuan/CogAlpha_Prompt}。固定版本见下列身份记录。",
      r"\item 本项目冻结配置、输入哈希、公开汇总及完整槽位记录：evidence/project2\_cogalpha。原始图片摘录的页码、裁切范围与哈希见source\_reference.json。\end{enumerate}",
      r"\lead{精确身份}","论文文件SHA以独立来源勘误为准；冻结配置保留原字节及运行身份，勘误不改变实验。",
      r"{\scriptsize 论文PDF SHA256：\par\ttfamily "+tex(d["source_reference"]["pdf_sha256"])+r"\par}",
      r"{\scriptsize 提示词仓库commit：\par\ttfamily "+tex(p["official_prompts_commit"])+r"\par}",
      r"{\scriptsize 研究身份：\par\ttfamily "+tex(d["manifest"]["identity"])+r"\par}",
      "本报告保留限制和负面结果，目标是让公式、数据、实验、图表和结论彼此对应。评分由课程教师决定；这些材料不构成未来收益或课程成绩保证。",r"\end{document}"]
    return "\n\n".join(out)


def native_chart(d,arms):
    if not all(np.isfinite(pick(d,s,a).rank_ic) for a in arms for s in d["seeds"]):
        # The PPTX runtime coerces null to zero; use the annotated raster instead.
        return None
    return {"type":"bar","categories":[str(s) for s in d["seeds"]],
            "series":[{"name":ARM[a],"values":[float(pick(d,s,a).rank_ic) if np.isfinite(pick(d,s,a).rank_ic) else None for s in d["seeds"]]} for a in arms],"axisFormat":"0.000"}


def deck(d,checks,test_claim,ui_claim):
    slides=[]
    def add(title,claim,items,notes,kind="content",**extra):
        slides.append({"title":title,"claim":claim,"items":items,"notes":notes,"kind":kind,"seconds":SECONDS[len(slides)],
                       "source":"来源：Liu等，arXiv:2511.18850v4；本次冻结协议与真实实验汇总。",**extra})
    add("CogAlpha：让因子研发可检验","真实模型提案，固定对照，完整保留结果",
        ["Computational Finance · Project 2",MEMBERS,"独立于RSRS路线 · 三个重复 · 约15分钟"],
        "各位老师、同学好。我们这条独立研究路线复现CogAlpha的核心机制：让大模型提出可以解释的因子，再用明确公式、质量检查和固定实验检验它。我们关心的是进化机制有没有帮助，而不是展示一次碰巧赚钱的曲线。",kind="cover",cover_title="大模型辅助因子研发\n与受限机制复现")
    add("先固定问题，再看测试结果","主比较：相同49提案槽，进化是否胜过持续提新想法？",
        ["共享21个初代；两臂各新增28个槽", "LightGBM + 测试期RankIC为固定主比较", "三个重复全部报告；不改选最优种子或模型"],
        "如果大模型不断生成因子，我们很容易从很多尝试里挑出一个漂亮结果，却不知道收益来自新方法还是尝试次数。因此我们先固定研究问题：两个分支共享二十一个初代候选，之后一个分支做变异交叉，另一个继续提出新想法，各增加二十八个槽。主比较只看固定LightGBM在测试区间的RankIC。三个重复都保留，收益曲线也全部展示。这里相同的是提案槽位，修复次数、实际token和人民币花费可以不同，后面会给真实账本。")
    add("从原文工作流到青序实现","保留七层组织、质量检查、五指标与演化闭环",
        ["左：原文Figure 1；右：本次受限工作流", "不同模型、数据与搜索规模；属于机制复现"],
        "原文工作流从OHLCV出发，由七层、二十一个角色生成因子，经过质量与信息泄漏检查，再用五项指标筛选，并进行变异和交叉。右边是我们接入青序的流程。我们保留这条闭环，但只让模型输出受限表达式，不运行任意Python。原文使用本地一百二十B模型、H100和更大搜索池，公开仓库只有提示词，完整运行时和最终二十个因子并未公开。所以这次是小规模机制复现，不能说复现了原始训练过程或原表数值。",kind="comparison",original_image="source_figure1.png",figure="local_workflow.png")
    add("时间切分与标签退出边界","所有研发只看训练；模型先冻结，再评价后续区间",
        ["训练2020—2022；验证2023—2024；测试2025—2026.09", "标签退出必须在下一切分前；静态股票池与已看历史仍是限制"],
        "我们使用已有Tushare一千只股票快照，训练是二零二零到二零二二年，验证是二三到二四年，测试从二五年到二六年九月。这里最重要的是标签时间：今天收盘才知道信号，下一交易日开盘进入，第十一个后续交易日开盘退出，形成十个交易日的持有收益。只把形成日放在训练期还不够，还必须保证退出日早于验证边界，否则训练会看到验证期价格。提案、亲本、最终筛选和模型拟合都只用清理后的训练数据。不过这是一千股静态快照，我们也已看过这段历史，不能把代码隔离等同于独立盲测。",figure="split_timeline.png",formula=r"y_{i,t}=O^{adj}_{i,t+11}/O^{adj}_{i,t+1}-1")
    add("先做手算：原公式真的算对了吗？","原文含义与公式分开检查，不能只复制经济叙述",
        ["H−C是高点到收盘距离，不是C−O上涨", "教学输入O10/H12/L9/C11/V100000股；三式执行与手算一致"],
        "原文的第一个因子是最高价减收盘价，再除成交量。作者把它解释成价格上涨的流动性冲击，但最高到收盘的距离并不等于开盘到收盘的上涨，所以我们保留公式，同时标出解释歧义。第二个示例把分子改成全天振幅，第三个用收盘和开盘绝对差除以成交金额，再做双曲正切。我们用一组明确标为教学数据的数字进行手算，三式都与DSL执行一致。这只能证明实现一致，不能证明原文的预测能力在我们的数据上也存在。",formula=r"f_1=(H-C)/(V+\varepsilon),\quad f_3=\tanh\!\left(|C-O|/(VC+\varepsilon)\right)",formula_plain="f₁ = (H − C) / (V + ε)\nf₃ = tanh[ |C − O| / (VC + ε) ]")
    add("生成不等于执行，更不等于交易","模型只提公式和假设；程序决定能否进入研究池",
        ["OHLCV + 有限算子；单窗口≤120日", "AST、数值覆盖率、前缀重算、未来扰动", "失败记录保留；一次修复仍计费"],
        "每次模型只交一个JSON候选，包含名字、公式和可证伪的经济假设。表达式只能使用OHLCV与白名单算子，最大窗口一百二十日，没有导入、属性访问或任意代码执行。然后检查是否能计算、有效覆盖是否达到百分之七十，并通过前缀重算和改变未来数据的测试，检查历史因子有没有被未来影响。无效候选最多修复一次，修复也占用预算；重复公式不补位。通过这些检查只是得到可研究的候选，并不是得到有用的交易策略。")
    attempts=sum(len(c["attempts"]) for r in d["lineage"] for c in r["cards"].values())
    add("同提案预算的进化消融","每个重复77个独立槽；每个最终分支49槽",
        [f"实际提案请求尝试{attempts}次；失败与修复均保留", "新提案臂看自身训练摘要，但不接收父公式"],
        "这张图说明对照为什么公平。初代二十一个槽被两个分支共享，演化臂每代七次变异、七次交叉，一共两代；新提案臂每代生成十四个新候选。新提案也能看到自己的训练指标摘要，所以它不是完全无反馈的一次生成，关键差别是没有父公式，也不做公式变异交叉。每条臂最后都是四十九个槽。拒绝和重复都保留，不能为某一边偷偷补到四十九个成功因子。我们还记录真实费用，预算是上限而不是花费目标。本地种子只控制父因子抽样，API输出本身没有可控种子。",figure="matched_budget.png")
    add("五指标资格与工程回退分开","qualified必须五项同时达标；不能把回退池改名",
        ["65% / 80%分位与绝对底线共同约束", "MI为本地五分位插件估计；原估计器未公开", "无qualified时可用正训练RankIC回退，并明确标记"],
        "五项指标是IC、RankIC、两种信息比率和互信息。每个门槛取池内百分之六十五或百分之八十分位与绝对底线的较大值，五项全部通过才叫qualified或elite，负相关不能取绝对值蒙混过关。原文没有公开互信息估计器，我们用每日横截面秩五分位插件估计，因此阈值数值不保证与原文完全可比。当五项交集为空，我们允许从质量合格、训练RankIC为正的因子里选工程亲本，但每一次都标记为回退，不能把它重新叫合格。最终入模最多二十个，也是另行声明的固定规则。")
    primary_rows=delta_rows(d)
    add("主结果：三个重复完整展示",summary(d),
        ["差值定义：进化减新提案；主指标为测试RankIC", "不据此更换主模型、种子或超参数"],
        "现在看预先指定的主结果。"+summary(d)+"每组柱子使用相同数据切分、同一LightGBM设置和相同的组合交易规则。RankIC反映横截面排序关系，不是账户收益率，因此后面仍要查看费用后的净值与回撤。三个重复共享同一市场历史，只是提案随机性不同，不能把它们当作三个独立市场样本。无论差异是正是负，我们保留每一组；这些结果最多支持这次受限实验的判断，不能推出大模型因子研发在所有市场都优于传统方法。",figure="primary_rankic.png",chart=native_chart(d,["llm_fresh49","llm_evolved49"]))
    add("原表与本次结果：保留条件再比较","原文Table 1是文献基准；本次不计算直接复现误差",
        ["左：原Table 1表头与LLM/CogAlpha行摘录；中间行省略", "右：本次三个重复；股票池、日期、模型与训练方式均不同"],
        "左边保留原表表头和大模型相关行，完整表格在报告中。原文CogAlpha的RankIC是零点零八一四，右边是我们三个重复的结果。不能直接把右边减左边，说复现误差是多少，因为原文是不同的成分股、不同日期、更大的模型和搜索池，还采用一百二十六日滚动训练。我们是静态一千股、固定训练一次，而且互信息等细节也不同。相邻展示是为了让读者看到原始主张和本地结果各自成立的条件，而不是拿不同设定拼成提升百分比。",kind="comparison",original_image="source_table1_focus.png",figure="local_table_comparison.png",chart=native_chart(d,["llm_evolved49"]))
    first=d["seeds"][0];rows=[]
    for j,arm in enumerate(("llm_fresh49","llm_evolved49","llm_evolved_diverse")):
        frame=pd.read_csv(E/pick(d,first,arm).daily_file)
        indices=np.unique(np.r_[np.arange(0,len(frame),max(1,int(np.ceil(len(frame)/120)))),len(frame)-1])
        rows.append({"name":ARM[arm],"values":frame.iloc[indices].nav.astype(float).tolist()})
    rows.append({"name":"沪深300价格指数","values":frame.iloc[indices].benchmark_nav.astype(float).tolist()})
    add("交易结果：排序能力之后还要核账户","Top50/Drop5、下一开盘成交、真实费用规则",
        [f"预先固定展示第一重复{first}；报告同时列出三个重复", "股票复权总收益单位 vs 沪深300价格指数：分红口径不同"],
        "这页展示第一重复的净值，这个展示顺序预先固定，没有从三个重复中挑最漂亮的一条。完整报告把三个重复并排列出。组合持有排名靠前的五十只股票，每天最多替换五只，继续持有的部分保留原份额；今天收盘得到分数，下一天开盘成交。我们采用论文的买入万分之五、卖出千分之一点五以及单笔最低五元费用，同时检查现金、持仓与净值账本。使用的是分数化复权研究份额，包含开盘涨跌停拦截，但没有整手、成交排队或冲击模拟。灰色是沪深三百价格指数代理，它不含分红且免交易费。平台也展示年化夏普、最大回撤和成本，所以预测分数好看不能替代账户核验。",figure="portfolio_first_seed.png",chart={"type":"line","categories":frame.iloc[indices].date.astype(str).tolist(),"series":rows,"axisFormat":"0.00"})
    add("独立改进：训练IC路径去冗余",summary(d,"llm_evolved_diverse","llm_evolved49"),
        ["保持进化候选池；IC路径绝对相关阈值固定0.8", "去冗余后最多20因子；不利用测试选阈值"],
        "在进化机制之外，我们增加一个独立改进：先按训练RankIC排序，再剔除每日训练RankIC路径与已选因子过于相关的候选，绝对相关阈值固定为零点八。这样希望避免二十个入模因子其实重复表达同一种信号。"+summary(d,"llm_evolved_diverse","llm_evolved49")+"需要注意，IC路径不相关不等于最终持仓完全独立，而且去冗余也会减少入模数量。我们同时报告数量与收益指标，不把改进假设直接当成已经证实的优势。",figure="diverse_rankic.png",chart=native_chart(d,["llm_evolved49","llm_evolved_diverse"]))
    add("平台演示：从结论追到证据","每个结果都能追溯到协议、候选公式与调用记录",
        ["先选主比较：LightGBM / 测试 / 指定费用", "查看三个重复、失败槽、父因子与公式", "下载公开汇总；读取结果无需重新付费"],
        "接下来演示青序平台。先进入CogAlpha独立研究页，确认这里是冻结证据，不是点击一次就开始新的付费挖掘。默认查看LightGBM、测试区间和指定费用，再切换三个重复，核对刚才报告里的RankIC和净值。然后打开候选详情，看公式来自哪个层级、是哪一代、父因子是谁、是否经历修复、训练指标是多少。失败槽同样可以追踪，不会从历史里消失。最后展示协议和汇总导出。平台继续保留原来的策略与RSRS路线，研究候选不会因为一次历史表现好就自动变成默认交易策略。若现场网络或界面异常，可以用本地固定证据文件完成核对。",kind="demo")
    add("结论与限制一起保留","我们交付可检验的研究流程；不把历史结果当作承诺",
        ["静态股票池、历史已观察、LLM知识污染风险", "数据/模型/MI/训练方式不同，不能宣称精确原表重现", "下一步：冻结候选，用新的时间与成分历史验证"],
        "最后，我们把真实模型提案、受限公式、质量检查、固定预算对照和完整账本接成了一条可重跑的路线。主比较和独立改进是否有效，按三个重复的真实结果回答。限制同样明确：静态股票池有偏差，这段历史已经看过，模型知识可能覆盖后续市场，原文实现也未完全公开。接下来的有效工作是冻结候选，收集新的数据区间和历史成分再验证，而不是反复调这个测试集直到赚钱。谢谢大家。")
    add("备份：模型和收益口径","预测指标、账户收益和成本对照各有定义",
        ["Ridge alpha100；LightGBM 200树/15叶/深度5/学习率0.05", "ICIR不年化；Sharpe按252日年化，零无风险收益", "AER为平均日超额算术年化；CAGR为复利年化", "最大回撤包含初始资金；整数交易与冲击仍是限制"],
        "本页用于答辩。原文使用滚动训练和不同超参数，我们本次使用一次训练以清晰隔离后续数据。预测ICIR不年化，Sharpe和收益指标另按明确口径计算。零费敏感性只是解释成本影响，不改变模型与预测标签。")
    add("备份：来源与验收范围","论文摘录、实验身份、原始槽位和核账记录均保留",
        ["原文v4与官方提示词commit固定；裁图范围逐一记录", test_claim,ui_claim,"公开聚合证据；密钥、原始行情和模型不公开"],
        "本页用于核对交付证据。报告、演示和讲稿都从正式汇总生成；原文裁图保留来源、页码和哈希。软件测试范围单独说明，不把某一模块通过的数量当成全项目测试。公开仓库包含代码、配置、聚合结果和材料；完整重算仍需要合法访问同版本原始行情。")
    expansions={
      1:"这也解释了为什么需要研究平台：它要保存所有尝试和最初的规则，而不是只保存最后的赢家。候选名称、公式和执行设置都有记录，后续课程可以沿用这些接口继续研究。",
      2:"这里的七层是让模型从不同角度提出假设，不代表我们获得了七类新的真实观测。比如订单失衡和羊群行为只能从日频量价构造代理，不能说我们直接看到了真实委托或投资者身份。",
      3:"数据处理还要保持单位一致。价格复权后，成交量按相反比例调整，避免凭空改变成交金额；缺失行情不能用未来值填上。最后一段尚未实现的十日标签直接排除，评价日期数可以少于全部行情日期数。这些边界会影响指标的有效样本量。",
      4:"成交量在这里按股数计算，第三式分母是成交量乘价格，而不是只用成交量。三个公式都固定原来的正方向，不因为测试发现负相关就事后取反。双曲正切限制了极端值，但这个数学性质本身并不能证明收益预测能力。",
      5:"一个表达式即使运行成功，也可能是常数、覆盖率太低或大量除零。质量检查与预测能力是两道独立的门槛；我们把拒绝原因一起保存，方便回头判断是机制无效还是工程实现不合格。",
      6:"缓存是可复现的一部分。中断后已经完成的槽位直接复用，不再调用模型；预算、凭据或账本状态不明时停止。请求使用deepseek-flash别名，官网当天映射到具体版本，但响应并不独立证明不可变权重。因此保留原始回答与冻结公式，比承诺模型再次生成同一答案更准确。",
      7:"IC衡量线性关系，RankIC衡量排序关系，两种信息比率衡量这种关系跨日期的稳定程度。互信息可以捕捉非线性关系，但没有正负方向，不能单独告诉我们该买哪一边。因此五项门槛与最终入模规则需要完整说明。",
      8:"还要区分主比较与基准比较：这页回答进化是否优于同槽位的新提案，买入持有沪深三百回答的是组合相对市场的表现。两种问题都可以报告，但不能用其中一个的成功来替代另一个的失败。",
      9:"原文消融表里，加入演化后的RankIC上升，但IC并没有同步提高。这也提醒我们，评价一个模块不能只挑选对它有利的一个数字，更不能忽略指标定义。",
      10:"股票侧是复权总收益研究单位，基准却是价格指数，分红口径可能贡献一部分差异，所以跑赢这条线不等于跑赢沪深三百全收益指数。夏普使用每日扣费收益和二百五十二日年化，最大回撤包含初始资金峰值。验证首日因尚无前一日模型预测而持有现金，测试已有前期信号。零费与扣费路径差也不能简单视为手续费总额。",
      11:"这个改进不增加新提案，也不改变费用和模型，所以能比较清楚地观察筛选规则的影响。阈值固定后不根据测试曲线反复修改；如果收益下降，它同样是需要报告的结果。",
      12:"演示时还可以切换rank和Ridge，说明平台支持候选方法比较；但回到结论页时仍使用最初指定的LightGBM主比较，避免演示过程变成事后挑选。"
    }
    for index,extra in expansions.items():slides[index]["notes"]+=extra
    if len(slides)!=16 or sum(s["seconds"] for s in slides)!=900:raise AssertionError("Deck timing drift")
    return slides


def beamer(slides):
    out=[r"""\documentclass[aspectratio=169,10pt]{beamer}
\usepackage[UTF8]{ctex}\usepackage{booktabs,tabularx,graphicx,amsmath}
\definecolor{navy}{HTML}{142B43}\definecolor{teal}{HTML}{168579}
\setbeamercolor{frametitle}{bg=navy,fg=white}\setbeamercolor{structure}{fg=teal}
\setbeamercolor{block title}{bg=teal,fg=white}\setbeamercolor{block body}{bg=teal!6}
\setbeamertemplate{navigation symbols}{}\setbeamerfont{frametitle}{size=\large}
\setbeamersize{text margin left=8mm,text margin right=8mm}
\setbeamertemplate{footline}{\leavevmode\hbox{\begin{beamercolorbox}[wd=.82\paperwidth,ht=2.5ex,dp=1ex,leftskip=1em]{author in head/foot}青序量化研究平台\quad Project 2 / CogAlpha\end{beamercolorbox}\begin{beamercolorbox}[wd=.18\paperwidth,ht=2.5ex,dp=1ex,center]{date in head/foot}\insertframenumber/16\end{beamercolorbox}}}
\begin{document}
"""]
    for s in slides:
        out += [r"\begin{frame}{"+tex(s["title"])+"}",r"\begin{block}{}"+tex(s["claim"])+r"\end{block}"]
        if s["kind"]=="cover":out.append(r"\vspace{2mm}{\Huge\bfseries\color{navy}CogAlpha 因子研发}\par\vspace{4mm}")
        if s.get("original_image"):out.append(pair(s["original_image"],s["figure"],"43mm"))
        elif s.get("figure"):out.append(img(s["figure"],"42mm"))
        if s.get("formula") and not s.get("figure"):out.append(r"\["+s["formula"]+r"\]")
        items=s["items"][:2] if s.get("figure") else s["items"]
        out += [r"\small\begin{itemize}",*(r"\item "+tex(x) for x in items),r"\end{itemize}",r"\end{frame}"]
    out.append(r"\end{document}")
    return "\n".join(out)


def speaker(slides):
    out=[r"""\documentclass[UTF8,a4paper,11pt]{ctexart}
\usepackage[margin=22mm]{geometry}\usepackage{xcolor,fancyhdr}
\definecolor{ink}{HTML}{142B43}\definecolor{accent}{HTML}{168579}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small 青序 · Project 2 · CogAlpha逐页讲稿}\fancyhead[R]{\small 15分钟正文 + 2页备份}\fancyfoot[C]{\thepage}
\setlength{\headheight}{15pt}\setlength{\parindent}{2em}\setlength{\parskip}{7pt}\setlength{\emergencystretch}{2em}\linespread{1.12}
\begin{document}
{\Large\bfseries\color{ink}CogAlpha：课堂口述与操作提示}\par
""",tex(MEMBERS)+r"\par","正文14页合计900秒；第15--16页仅供答辩。按实际语速排练，操作提示不朗读。"]
    md=["# Project 2 CogAlpha：15分钟逐页讲稿","",MEMBERS,"","正文14页900秒；另2页答辩备份。操作提示不朗读。",""]
    elapsed=0
    for i,s in enumerate(slides):
        if i and i%2==0:out.append(r"\newpage")
        timing=f"建议{s['seconds']}秒；{elapsed//60:02d}:{elapsed%60:02d}开始" if s["seconds"] else "答辩备份，不计入正文15分钟"
        out += [r"\section*{\color{ink}"+tex(f"{i+1:02d} {s['title']}")+"}",r"{\small\color{accent}"+tex(timing)+r"}\par",tex(s["notes"])]
        if s["kind"]=="demo":out.append(r"\par{\small\color{accent}操作提示：CogAlpha研究页 → 固定主比较 → 三个重复 → 提案血缘 → 协议/汇总导出；不启动新挖掘。}")
        md += [f"## {i+1:02d} {s['title']}","",timing,"",s["notes"],""];elapsed+=s["seconds"]
    out.append(r"\end{document}")
    return "\n".join(out),"\n".join(md)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-sources",action="store_true")
    parser.add_argument("--source-pdf",type=Path,default=Path("D:/Desktop/2511.18850v4.pdf"))
    parser.add_argument("--test-claim",default="软件测试范围以交付验收记录为准。")
    parser.add_argument("--ui-claim",default="界面检查范围以交付验收记录为准。")
    args=parser.parse_args()
    if args.prepare_sources:
        prepare_sources(args.source_pdf);method_figures()
        dump(OUT/"preparation_status.json",{"sources_prepared":True,"method_figures_prepared":True,
            "final_results_filled":False,"final_pdfs_compiled":False,"message":"Waiting for complete real evidence; no performance values invented."})
        print("Source excerpts and method diagrams prepared; formal results still required.");return
    d=load();method_figures();experiment_figures(d);checks=manual_checks()
    slides=deck(d,checks,args.test_claim,args.ui_claim)
    write(OUT/"final_report.tex",report(d,checks,args.test_claim,args.ui_claim));write(OUT/"beamer.tex",beamer(slides))
    script,md=speaker(slides);write(OUT/"speaker_script.tex",script);write(ROOT/"docs/PROJECT2_COGALPHA_PRESENTATION_SCRIPT.md",md)
    dump(OUT/"deck_content.json",slides)
    dump(OUT/"authoring_checks.json",{"evidence_identity":d["manifest"]["identity"],"evidence_manifest_sha256":sha(E/"manifest.json"),"seed_count":len(d["seeds"]),
        "main_slides":14,"backup_slides":2,"seconds":sum(s["seconds"] for s in slides),
        "main_notes_chinese_characters":sum(sum('\u4e00'<=c<='\u9fff' for c in s["notes"]) for s in slides[:14]),
        "test_claim":args.test_claim,"ui_claim":args.ui_claim,"formula_handchecks_passed":True,
        "pdf_compiled":False,"visual_reviewed":False,"claims_from_real_evidence":True})
    dump(OUT/"preparation_status.json",{"stage":"formal_materials_authored_pending_pdf_review",
        "sources_prepared":True,"method_figures_prepared":True,"final_results_filled":True,
        "final_pdfs_compiled":False,"message":"Finalized real evidence loaded; compile and inspect the PDFs before release."})
    print(json.dumps({"out":str(OUT),"slides":len(slides),"seconds":900,"status":"authored; compile and visually inspect next"},ensure_ascii=False))


if __name__=="__main__":main()
