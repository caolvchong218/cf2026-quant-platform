"""Author the formal Project 2 report, Beamer deck and matched speaking script.

Numerical claims are read from the completed public study evidence. This script
does not run research, select a winner, compile PDFs, or assert UI acceptance.
PDF/PPTX rendering and visual acceptance remain separate release operations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "evidence/project2_rsrs"
OUT = ROOT / "reports/project2_rsrs"
FIG = OUT / "figures"
MEMBERS = "12412408 胡博文；12410819 谢明飞；12410436 王天希"
NAMES = {"buy_hold": "买入持有", "ma20": "单独MA20", "slope": "斜率",
         "standard": "标准分", "corrected": "修正标准分", "right": "右偏标准分",
         "standard_ma": "标准分+MA", "corrected_ma": "修正分+MA", "right_ma": "原右偏+MA",
         "right_ma_vol15": "15%波动目标", "right_ma_fixed75": "固定75%仓位",
         "right_ma_cash75": "固定75%仓位", "right_ma_vol10": "10%波动目标",
         "right_ma_vol20": "20%波动目标"}
INDEX = {"sh000300": "沪深300", "sh000016": "上证50", "sh000905": "中证500"}
PERIOD = {"historical_overlap": "共同区间", "post_publication": "发表后", "recent": "近年重叠区间",
          "source_compatible": "原样本兼容", "source_window": "原样本兼容", "source_period": "原样本兼容"}
SECONDS = [20,45,65,90,60,50,75,65,75,55,100,60,95,45,0,0]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(value, encoding="utf-8", newline="\n")


def tex(value):
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "_": r"\_",
               "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}",
               "β": r"\ensuremath{\beta}", "α": r"\ensuremath{\alpha}", "ε": r"\ensuremath{\varepsilon}",
               "Δ": r"\ensuremath{\Delta}", "σ": r"\ensuremath{\sigma}", "²": r"\textsuperscript{2}",
               "→": r"\ensuremath{\to}", "−": r"\ensuremath{-}", "√": r"\ensuremath{\surd}",
               "≠": r"\ensuremath{\ne}"}
    return "".join(mapping.get(c, c) for c in str(value))


def pct(value):
    return f"{float(value)*100:.2f}%"


def pp(value):
    return f"{float(value)*100:+.2f}"


def metric(row, key):
    alternatives = {"cagr": ("cagr", "annualized_return"), "annual_turnover": ("annual_turnover", "annualized_turnover")}
    for name in alternatives.get(key, (key,)):
        if name in row:
            return float(row[name])
    raise KeyError(f"Missing required metric {key}")


def pick(frame, strategy, period="post_publication", index="sh000300", cost=10):
    if period == "source_compatible":
        source = pd.read_csv(E / "source_compatible.csv")
        match = source[(source.index_id == index) & (source.strategy_id == strategy) &
                       (source.warmup == "partial_prior_M_min2") &
                       (source.end_assumption == "march_end") &
                       np.isclose(source.cost_bps, cost)]
        if len(match) != 1:
            raise ValueError(f"Expected one source-compatible row for {strategy}/{cost}")
        return match.iloc[0]
    match = frame[(frame.index_id == index) & (frame.period_id == period) &
                  (frame.strategy_id == strategy) & np.isclose(frame.cost_bps, cost)]
    if len(match) != 1:
        raise ValueError(f"Expected one result: {index}/{period}/{strategy}/{cost}, got {len(match)}")
    return match.iloc[0]


def table(headers, rows, widths=None, size="small"):
    spec = widths or ("l" + "r" * (len(headers)-1))
    lines = [rf"\begin{{center}}\{size}\begin{{tabularx}}{{\textwidth}}{{{spec}}}\toprule",
             " & ".join(map(tex, headers)) + r"\\\midrule"]
    lines.extend(" & ".join(map(tex, row)) + r"\\" for row in rows)
    lines.append(r"\bottomrule\end{tabularx}\end{center}")
    return "\n".join(lines)


def image(name, height="65mm", width=r".96\textwidth"):
    return rf"\begin{{center}}\includegraphics[width={width},height={height},keepaspectratio]{{figures/{name}}}\end{{center}}"


def side_images(left, right, height="62mm"):
    return (r"\begin{center}\begin{minipage}[t]{.49\textwidth}\centering" +
            rf"\includegraphics[width=\linewidth,height={height},keepaspectratio]{{figures/{left}}}" +
            r"\end{minipage}\hfill\begin{minipage}[t]{.49\textwidth}\centering" +
            rf"\includegraphics[width=\linewidth,height={height},keepaspectratio]{{figures/{right}}}" +
            r"\end{minipage}\end{center}")


def save_figure(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_inputs():
    required = ("performance.csv", "paired_comparison.csv", "diagnostics.csv", "reproduction_table.csv",
                "formula_checks.json", "bootstrap.json", "validation.json", "protocol.json", "source_compatible.csv", "data_quality.json")
    missing = [name for name in required if not (E / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Complete formal evidence first: {missing}")
    data = {Path(name).stem: pd.read_csv(E/name) if name.endswith(".csv") else read_json(E/name)
            for name in required}
    frame = data["performance"]
    if frame.duplicated(["index_id", "period_id", "strategy_id", "cost_bps"]).any():
        raise ValueError("Duplicate study identities")
    data["base"] = pick(frame, "right_ma")
    data["improved"] = pick(frame, "right_ma_vol15")
    controls = [name for name in ("right_ma_fixed75", "right_ma_cash75", "right_ma_75") if name in set(frame.strategy_id)]
    if len(controls) != 1:
        raise ValueError("Require exactly one declared fixed-75% sizing control")
    data["cash_id"] = controls[0]
    NAMES[controls[0]] = "固定75%仓位"
    data["cash"] = pick(frame, controls[0])
    data["hold"] = pick(frame, "buy_hold")
    if not data["validation"].get("all_ledgers_passed") or not data["validation"]["diagnostics"].get("passed"):
        raise ValueError("Ledger validation must pass before publishing materials")
    for _,row in data["reproduction_table"].iterrows():
        actual=pick(frame,row.strategy_id,"source_compatible",cost=0)
        for key in ("cagr","sharpe","max_drawdown"):
            if not np.isclose(metric(actual,key),row[f"source_compatible_{key}"],atol=1e-12):
                raise ValueError("Source-compatible chart/table endpoint mismatch")
    return data


def curves_frame():
    rows=[]
    performance=pd.read_csv(E/"performance.csv")
    for _,row in performance.iterrows():
        part=pd.read_csv(E/row.daily_file,parse_dates=["date"])
        for key in ("index_id","period_id","strategy_id","cost_bps"):
            part[key]=row[key]
        rows.append(part)
    source=pd.read_csv(E/"source_compatible.csv")
    source=source[(source.warmup=="partial_prior_M_min2") & (source.end_assumption=="march_end")]
    for _,row in source.iterrows():
        part=pd.read_csv(E/row.daily_file,parse_dates=["date"])
        for key in ("index_id","strategy_id","cost_bps"):
            part[key]=row[key]
        part["period_id"]="source_compatible"
        rows.append(part)
    # A source-compatible benchmark is supplied by each run's same-fee ledger.
    for fee in source.cost_bps.unique():
        row=source[np.isclose(source.cost_bps,fee)].iloc[0]
        part=pd.read_csv(E/row.daily_file,parse_dates=["date"])
        part["nav"]=part.benchmark_nav
        part["index_id"]=row.index_id;part["strategy_id"]="buy_hold"
        part["cost_bps"]=fee;part["period_id"]="source_compatible"
        rows.append(part)
    diagnostic=pd.read_csv(E/"diagnostics.csv")
    for _,row in diagnostic[(diagnostic.period_id=="historical_overlap") &
                            diagnostic.case.isin(["cost_0bps","prior_M_ddof0_lag2_next_open_10bps","cost_20bps","cost_30bps"])].iterrows():
        part=pd.read_csv(E/row.daily_file,parse_dates=["date"])
        part["index_id"]="sh000300";part["strategy_id"]="right_ma"
        part["cost_bps"]=row.cost_bps;part["period_id"]="diagnostic_cost_common"
        rows.append(part)
    return pd.concat(rows,ignore_index=True)


def draw_curves(curves, name, strategies, period, cost, title, costs=None):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = ["#142B43", "#168579", "#4265A6", "#BE8757", "#8162A8"]
    en = {"buy_hold": "Buy & hold", "standard_ma": "Standard + MA", "corrected_ma": "Corrected + MA",
          "right_ma": "Right-skew + MA", "right_ma_vol15": "15% volatility target", "right_ma_cash75":"Fixed 75% target"}
    for i, strategy in enumerate(strategies):
        fee = costs[i] if costs is not None else cost
        part = curves[(curves.index_id == "sh000300") & (curves.period_id == period) &
                      (curves.strategy_id == strategy) & np.isclose(curves.cost_bps, fee)].sort_values("date")
        if part.empty:
            raise ValueError(f"No curve for {strategy}/{period}/{fee}")
        label = f"{fee:g} bp / side" if costs is not None else en.get(strategy, strategy)
        ax.plot(part.date, part.nav, color=colors[i], label=label, lw=1.35)
    ax.grid(alpha=.2); ax.legend(fontsize=8, ncol=2); ax.set_ylabel("NAV / initial capital")
    ax.set_title(title, fontsize=10)
    save_figure(fig, name)


def figures(data):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    for name in ("source_figure25.png", "source_table5.png", "source_figure26.png"):
        source = E / "reference" / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, FIG / name)
    curves = curves_frame()
    data["source_period"] = "source_compatible"
    draw_curves(curves, "reproduction_curves", ["buy_hold", "standard_ma", "corrected_ma", "right_ma"],
                "source_compatible", 0, "2005-03-01 to 2017-03-31 | partial warmup | 0 bp / side")
    draw_curves(curves, "cost_curves", ["right_ma"]*4, "diagnostic_cost_common", 0,
                "Common window: 2007-08-01 to 2017-03-31 | fees only", costs=[0,10,20,30])
    draw_curves(curves, "main_curves", ["buy_hold", "right_ma", data["cash_id"], "right_ma_vol15"],
                "post_publication", 10, "Post-publication historical check | CSI 300 | 10 bp / side")
    rows = [data[k] for k in ("base", "cash", "improved", "hold")]
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
    labels = ["RSRS", "Fixed 75%", "Vol 15%", "Buy & hold"]
    for ax, key, title in zip(axs, ("cagr", "max_drawdown", "sharpe"), ("CAGR", "Maximum drawdown", "Sharpe")):
        ax.bar(np.arange(4), [metric(row,key) for row in rows], color=["#4265A6", "#BE8757", "#168579", "#8D97A2"])
        ax.set_xticks(np.arange(4), labels, rotation=20, ha="right", fontsize=8)
        ax.set_title(title); ax.grid(axis="y", alpha=.2)
        if key != "sharpe": ax.yaxis.set_major_formatter(PercentFormatter(1))
    save_figure(fig, "main_metrics")
    frame = data["performance"]
    pairs = []
    for index in INDEX:
        for period in ("historical_overlap", "post_publication", "recent"):
            b = pick(frame, "right_ma", period, index)
            v = pick(frame, "right_ma_vol15", period, index)
            pairs.append([index.replace("sh", ""), period.replace("historical_overlap","Overlap").replace("post_publication","Post").replace("recent","Recent"),
                          metric(v,"cagr")-metric(b,"cagr"), metric(b,"max_drawdown")-metric(v,"max_drawdown")])
    fig, axs = plt.subplots(1,2,figsize=(10,4))
    y = np.arange(len(pairs)); labels=[f"{p[0]} / {p[1]}" for p in pairs]
    for ax,j,title in zip(axs,(2,3),("CAGR difference: volatility - baseline", "Drawdown reduction: baseline - volatility")):
        vals=[p[j] for p in pairs]
        ax.barh(y, vals, color=["#168579" if value >= 0 else "#BE8757" for value in vals])
        ax.set_yticks(y, labels, fontsize=8); ax.invert_yaxis(); ax.axvline(0,color="#657384",lw=.7)
        ax.xaxis.set_major_formatter(PercentFormatter(1)); ax.set_title(title,fontsize=9); ax.grid(axis="x",alpha=.2)
    save_figure(fig,"robustness")
    return curves


def summary_rows(data, strategies=None, period="post_publication", cost=10):
    names = strategies or ["right_ma", data["cash_id"], "right_ma_vol15", "buy_hold"]
    return [[NAMES.get(name,name), pct(metric(row,"total_return")), pct(metric(row,"cagr")),
             f"{metric(row,'sharpe'):.3f}", pct(metric(row,"max_drawdown"))]
            for name in names for row in [pick(data["performance"],name,period,cost=cost)]]


def differences(data):
    b,v,c = data["base"],data["improved"],data["cash"]
    return {"cagr":metric(v,"cagr")-metric(b,"cagr"), "drawdown_reduction":metric(b,"max_drawdown")-metric(v,"max_drawdown"),
            "sharpe":metric(v,"sharpe")-metric(b,"sharpe"), "versus_cash_cagr":metric(v,"cagr")-metric(c,"cagr"),
            "versus_cash_drawdown_reduction":metric(c,"max_drawdown")-metric(v,"max_drawdown")}


def cost_row(data,fee):
    part=data["diagnostics"]
    name={0:"cost_0bps",10:"prior_M_ddof0_lag2_next_open_10bps",20:"cost_20bps",30:"cost_30bps"}[fee]
    rows=part[(part.period_id=="historical_overlap") & (part.case==name)]
    if len(rows)!=1: raise ValueError(f"Missing common-window fee diagnostic {fee}")
    return rows.iloc[0]


def warmup_rows(data):
    source=data["source_compatible"]
    rows=[]
    for strategy in ("standard_ma","corrected_ma","right_ma"):
        part=source[(source.strategy_id==strategy) & (source.end_assumption=="march_end") & np.isclose(source.cost_bps,0)]
        a=part[part.warmup=="partial_prior_M_min2"].iloc[0]
        b=part[part.warmup=="strict_prior_M"].iloc[0]
        rows.append([NAMES[strategy],pct(metric(b,"cagr")),pct(metric(a,"cagr")),pp(metric(a,"cagr")-metric(b,"cagr")),pp(metric(a,"max_drawdown")-metric(b,"max_drawdown"))])
    return rows


def json_lines(value, prefix=""):
    lines=[]
    for key,item in value.items():
        label=f"{prefix}.{key}" if prefix else key
        if isinstance(item,dict): lines.extend(json_lines(item,label))
        elif isinstance(item,(int,float,str,bool)) or item is None: lines.append((label,item))
    return lines


def report(data, test_text, ui_status):
    d=differences(data); b=data["base"]; v=data["improved"]
    content = [r"""\documentclass[UTF8,a4paper,10pt]{ctexart}
\usepackage[margin=19mm,top=20mm,bottom=20mm]{geometry}
\usepackage{amsmath,amssymb,booktabs,tabularx,graphicx,xcolor,fancyhdr,enumitem,tikz}
\usetikzlibrary{arrows.meta,positioning}
\usepackage[colorlinks=true,linkcolor=teal,urlcolor=teal]{hyperref}
\definecolor{ink}{HTML}{142B43}\definecolor{accent}{HTML}{168579}\definecolor{teal}{HTML}{168579}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small CF2026 Project 2 · 正式研究报告}\fancyhead[R]{\small 青序量化研究平台}
\fancyfoot[L]{\scriptsize 2026-10-19 · 原文复现 / 差异归因 / 固定规则改进}\fancyfoot[R]{\thepage}
\setlength{\headheight}{15pt}\setlength{\parskip}{5pt}\setlength{\emergencystretch}{2em}\setlength{\parindent}{2em}
\setlist{nosep,leftmargin=1.7em}\ctexset{section={format=\Large\bfseries\color{ink}},subsection={format=\normalsize\bfseries\color{accent}}}
\newcommand{\lead}[1]{\par\noindent{\bfseries\color{accent}#1}\par}
\newcommand{\page}[1]{\newpage\section*{#1}}
\begin{document}
\begin{center}{\small\color{accent}COMPUTATIONAL FINANCE / PROJECT 2}\\[8pt]
{\Huge\bfseries\color{ink}RSRS 择时研究}\\[6pt]
{\Large 原文复现、差异归因与波动率仓位控制}\\[8pt]
2026年10月19日\quad 正式课程报告\quad 截止日期：2026年11月1日
\end{center}
""",tex(MEMBERS),r"""
\section*{1\quad 摘要与主要结论}
本项目以光大证券2017年5月1日研报为研究对象，将RSRS斜率、标准分、修正分、右偏分和价格过滤规则接入青序P1平台。我们先复核公式及可交易的信息时点，再分别讨论原样本兼容实验、共同区间对照与发表后检验，最后用固定的15\%年化波动率目标调整仓位。
\lead{核心研究发现：暖机约定会改变早期复现，风险改进须与单纯减仓区分。}
原文第22页允许历史不足时使用已有斜率；准备版采用完整窗口暖机，二者不是同一规则。本次保留严格暖机主协议，并增加部分历史窗口兼容实验。原文没有公开精确交易日、完整费用计提和Wind快照，因此本报告不声称逐笔重建原始账户。
\subsection*{主检验：沪深300，2017-05-02至2026-10-15，买卖各10bp}
""",table(["策略","累计收益","CAGR","Sharpe","最大回撤"],summary_rows(data),"Xrrrr"),
    rf"主改进相对原右偏+MA：CAGR变化{tex(pp(d['cagr']))}个百分点，最大回撤变化{tex(pp(-d['drawdown_reduction']))}个百分点，Sharpe变化{d['sharpe']:+.3f}。固定75\%仓位对照用于判断效果是否只是少承担股票风险。",
    r"\lead{主端点存在有限改善，但没有确立收益优势。}15\%目标相对原策略改善了CAGR与回撤；固定75\%仓位的最大回撤仍更低。两组配对年化算术日收益差的95\%区间均跨零，不据此声称风险管理创造稳定alpha。",
    r"""\subsection*{研究结论的适用范围}
全部历史区间已在研究准备中观察。参数在本轮正式运行前固定，仍不构成未知数据上的盲测；近期区间包含在发表后区间中，不能当作独立重复证据。指数点位是研究代理，未加入ETF跟踪误差、分红、最小交易单位、冲击和真实券商成交。收益不保证未来表现。
\subsection*{交付与软件核验}
    """,tex(f"{test_text}；{ui_status}。"),
    tex(f"实际完成{data['validation']['main_runs']}组主实验；其中{data['validation']['legacy_oracle_runs']}组与独立账户对齐。另有16组口径诊断、24组源兼容实验与12组波动目标/费用稳健性检查，均保留结果。"),
    r"报告、16页Beamer、可编辑演示、逐页讲稿、配置、源码、数据说明与聚合证据一同交付。",
    r"""\page{2\quad 研究问题、来源与课程要求}
\lead{我们复现的是研报中RSRS价格优化策略的核心结论。}
原文：刘均伟、周萧潇，《基于阻力支撑相对强度（RSRS）的市场择时——技术择时系列报告之一》，光大证券，2017-05-01。目标为第19页图25、表5及第20页图26。图25比较三类标准分的价格优化策略；表5汇总收益、风险和交易统计；图26比较成本情景。
\subsection*{三个可回答的问题}
\begin{enumerate}\item 在明确的数据与成交规则下，核心公式、买卖状态及账户结果能否逐层重建？\item 与原文不同的结果，有多少可由暖机、标准化、均线索引、成交时点与费用解释？\item 保留原信号的15\%波动目标，能否改善回撤，代价是什么，是否优于固定减仓？\end{enumerate}
""",table(["课件要求","本项目的对应证据"],[
        ["接入P1（第4页）","cfquant.rsrs接入已有引擎、绩效与reconcile；界面和CLI共用核心。"],
        ["逐层核验（第4页）","手算OLS与标准分、阈值状态、未来扰动不影响过去、独立账本重建。"],
        ["并排对照（第5页）","原图25/表5/图26截图与本次同主题输出并排；三层区间分开。"],
        ["差异归因（第6页）","固定基准、单因素矩阵和实际差值；不能解释的差异明确保留。"],
        ["自主改进（第7页）","仅改仓位，增加固定75%仓位对照、费用和波动目标敏感性。"],
        ["可复现提交（第8页）","原文位置、数据哈希、代码版本、协议、seed、检查结果与材料。"]],"lX"),
    r"""\subsection*{来源优先级与复用}
研报排版正文是方法依据，第三方手写批注不作为作者陈述。开源QuantsPlaybook用来定位合法公开研究材料与实现差异，不直接运行或复用其账户逻辑。课程要求与原文方法分开审计，文档中写明每个补充假设。
\subsection*{原文内部存在的不一致}
摘要写2005年3月至2017年3月，正文部分段落写至2017年4月，图25横轴从2005-02-18起。摘要1971.05\%、正文1826.13\%与第20页无成本净值19.71也不能机械换算成同一个终值。我们不择取最方便的数字作为唯一原值。
""",
    r"""\page{3\quad 指标、状态机与可审计补充假设}
\subsection*{带截距滚动OLS}
对收盘日$t$及此前$N-1$日的高低价拟合
\[H_i=\alpha_t+\beta_tL_i+\varepsilon_i,\quad
\beta_t=\frac{\sum_i(L_i-\bar L)(H_i-\bar H)}{\sum_i(L_i-\bar L)^2},\quad
R_t^2=\frac{[\sum_i(L_i-\bar L)(H_i-\bar H)]^2}{\sum_i(L_i-\bar L)^2\sum_i(H_i-\bar H)^2}.\]
退化窗口保留缺失，不用未来或零值补齐。主规则以严格早于$t$的$M$个有效斜率计算
\[z_t=\frac{\beta_t-\mu_{t-1,M}}{\sigma_{t-1,M}},\qquad
z_t^{\mathrm{corr}}=z_tR_t^2,\qquad z_t^{\mathrm{right}}=z_tR_t^2\beta_t.\]
主标准差为总体口径ddof=0；原文明确早期可使用部分历史，但未明确是否含当日、ddof和最低有效样本数，故单独做控制实验。
""",table(["信号","N / M","入场 / 出场","价格过滤"],[
        ["原始斜率","18 / 无","β > 1.0 / β < 0.8","无"],
        ["标准分、修正分","18 / 600",">0.7 / <−0.7","另列MA版本"],
        ["右偏分","16 / 300",">0.7 / <−0.7","主候选增加MA"],
        ["MA入场条件","20日均线","MA[t] > MA[t−2]","不因MA下降强制卖出"]],"Xlll"),
    r"""\subsection*{持仓状态与执行}
空仓时只有信号超过上阈值且入场条件成立才开仓；持仓时信号低于下阈值才退出，其余时间延续状态。主候选right\_ma不是“每天挑最高分股票”，而是单一指数做多/现金的滞回状态机。
收盘$t$计算信号，$t+1$开盘成交，$t+1$收盘估值。单边成本按实际成交金额收取，现金利息和无风险收益率均为0，无杠杆，不强制末日清仓。次日开盘和费用是本项目补充约定，不是研报已完全规定的细节。
\subsection*{价格过滤的索引歧义}
原文同时使用“前一日/前三日”与“当日/3日前”的文字。主规则按执行日解释为MA[t]与MA[t−2]；lag=3只作独立敏感性。原文均线过滤已有，不列为本组创新。
""",
    r"""\page{4\quad 数据快照与三层比较范围}
\lead{先确定什么可以比较，再解释收益差异。}
本次复用研究准备版的三个指数公开日频快照：沪深300、上证50、中证500，来源为Tencent Finance公共行情。该指数研究与P1个股Tushare快照分开；不会把供应商更换本身写成已识别的收益原因。
""",table(["指数","记录数","首日","末日","缺失/重复"],
           [[INDEX[r["index_id"]],r["rows"],r["first"],r["last"],f"{r['missing_ohlc']}/{r['duplicate_dates']}"] for r in data["data_quality"]],"Xrlll","scriptsize"),
    table(["层次","区间与规则","能回答的问题"],[
        ["原样本兼容","2005-03-01至2017-03-31；另测4月末","部分历史暖机，贴近原图/表；首尾和Wind仍不完全同源。"],
        ["共同区间","2007-08-01至2017-03-31；严格暖机","同一输入、费用和账户下比较各信号及控制。"],
        ["发表后检验","2017-05-02至2026-10-15；严格暖机","方法发布后是否保持效果；本次已观察的历史。"],
        ["近期敏感性","2020-01-02至2026-10-15","市场阶段影响；与上一层重叠，不能视作独立样本。"]],"lXX"),
    r"""\subsection*{部分历史窗口兼容实验}
原文第22页明确举例：标准化历史不足600日时，用已有的斜率。兼容实现仍需至少两个有效斜率及非零标准差，沿用排除当日、ddof=0；这只还原已知的早期窗口原则，不能消除全部未公开细节。严格暖机与部分暖机的区别在同一期间、同一费用下单独比较。
\subsection*{数据质量与身份}
日期要求升序、唯一、无时区且对应日频记录；OHLC必须有限且为正，最高价不得低于最低价。不填补缺失交易日，不以最新值倒填历史。快照SHA-256、行数、日期范围和研究文件哈希随证据包保存，重跑先验证输入身份。
\subsection*{价格指数的经济含义}
研究按连续指数单位交易，输入无个股涨跌停或容量约束，价格指数没有直接持有工具和分红再投资。该账本可检验信号及资金会计，但不能称作可立即执行的ETF策略或沪深300全收益投资。可交易扩展应另加ETF实际OHLC、费率、分红和跟踪误差。
\subsection*{结论的独立性}
正式协议固定主端点、15\%目标、20日窗口、75\%降仓对照、有限敏感性及重采样seed；准备版已观察同段历史，因此“先写协议”提高本轮透明度，并不把既有历史转换成盲测。
""",
    r"""\page{5\quad 公式手算、因果性与账户核验}
\lead{净值曲线之前，先验证每一个会影响交易的中间量。}
\subsection*{可手算的OLS例子}
取$L=(1,2,3)$，$H=(2,3,5)$，则$\bar L=2$、$\bar H=10/3$，
\[\sum(L-\bar L)^2=2,\quad\sum(L-\bar L)(H-\bar H)=3,\quad
\sum(H-\bar H)^2=14/3.\]
因而$\beta=3/2$，$\alpha=1/3$，$R^2=27/28\approx0.964286$。如果此前斜率为$(1,2,3)$，当前$\beta=4$，则$z=(4-2)/\sqrt{2/3}=\sqrt6$。这些小例子只验证公式，不代表可投资样本。
""",real_formula_text(data),r"""
\subsection*{核验层次}
""",table(["层次","检查方法与通过含义"],[
        ["公式","向量化OLS对照逐窗独立计算；beta、R²、z及变换均保存。"],
        ["信号","阈值上下穿越、缺失窗口、MA仅限制入场；手工状态序列可对照。"],
        ["因果","扰动未来价格不应改变此前指标、目标和交易；信号日严格早于成交日。"],
        ["账户","逐笔成交重建份额、现金、费用和净值；无负现金或超过100%目标。"],
        ["独立参照","P1共享账本与准备版独立账户同输入比较；差异不得直接跳过。"],
        ["产物","结果、协议、输入身份与配置一同留存；软件测试与研究表现分开。"]],"lX"),
    r"""\subsection*{绩效口径}
设每日净值$V_t$、初始资金$V_0$、交易日数$T$，日收益$r_t=V_t/V_{t-1}-1$，则
\[\mathrm{CAGR}=(V_T/V_0)^{252/T}-1,\quad
\mathrm{Sharpe}=\sqrt{252}\,\bar r/s(r),\quad
\mathrm{MDD}=\max_t\left(1-\frac{V_t}{\max_{u\le t}V_u}\right).\]
Sharpe使用日收益样本标准差（ddof=1）、无风险利率0；最大回撤报告为正的损失幅度，原表用负号。初始资金纳入回撤峰值。换手按成交名义金额除以账户规模，不能把成交笔数等同于完整往返次数。
""",
    r"""\page{6\quad 图25：原文与本次兼容实验并排}
\lead{原文展示高净值；本次输出必须连同样本与假设一起阅读。}
""",side_images("source_figure25.png","reproduction_curves.png","73mm"),
    r"\noindent\small 左：原文第19页图25截图，保留标题、图例和WIND/光大证券研究所来源行。右：2005-03-01至2017-03-31，本次指数快照、部分历史暖机、零成本、次日开盘；相同主题不代表完全相同口径。\normalsize",
    table(["候选（兼容区间）","累计收益","CAGR","Sharpe","最大回撤"],summary_rows(data,["standard_ma","corrected_ma","right_ma"],data["source_period"],0),"Xrrrr"),
    r"""\subsection*{可检验的复现内容}
三类标准分、MA入场过滤与相同阈值都落实到代码中，指标和交易状态可下载逐日检查。右偏版本同时改变N/M，所以与标准分的差异不是单独乘beta的纯效应；这个限制在比较表中明确保留。
\subsection*{无法消除的可比性差异}
Wind原始行情、确切首末日、当日是否参与标准化及完整成交模型未公开。我们没有原始逐日净值去验证逐点误差。图形形状相似或累计收益接近都不是“完全复现”的充分条件；不同费用、暖机和交易时点的控制实验见后文。
""",
    r"""\page{7\quad 表5：原始统计与本次统计的边界}
\begin{center}\includegraphics[width=.83\textwidth,height=110mm,keepaspectratio]{figures/source_table5.png}\end{center}
\noindent\small 上：原文第19页表5完整截图，保留20个统计量及来源。全文抄录固定在来源审计，不用当前结果覆盖原表。\normalsize
\subsection*{同定义可解释的比较}
主要对照CAGR、Sharpe、最大回撤；回撤统一为正损失幅度。交易次数需区分成交笔数、入场次数和已结束往返，原文“持仓日”与开盘/收盘有仓日也可能不同。本次列取自reproduction\_table.csv，不把尚未对齐口径的两列差值称作估计误差。
""",
    table(["价格优化候选","原年化","本次年化","原Sharpe","本次Sharpe","原回撤","本次回撤"],
          [[NAMES[name], orig, pct(metric(row,"cagr")),sh,f"{metric(row,'sharpe'):.3f}",original_dd,pct(metric(row,"max_drawdown"))]
           for name,orig,sh,original_dd in (("standard_ma","27.37%","1.76","15.48%"),("corrected_ma","26.13%","1.64","15.76%"),("right_ma","28.19%","1.56","22.31%"))
           for row in [pick(data["performance"],name,data["source_period"],cost=0)]],"Xrrrrrr","scriptsize"),
    r"\small 本次列：2005-03-01至2017-03-31、部分历史窗口、零成本、次日开盘，与reproduction\_table.csv一致。2017-04-28截止仅列为端点敏感性。原文列的精确样本、费用和标准差细节未完全公开。\normalsize",
    r"""\page{8\quad 图26：费用敏感性与可交易假设}
""",side_images("source_figure26.png","cost_curves.png","78mm"),
    r"\small 左：原文第20页图26，原图例为零成本及单边0.2\%/0.3\%双边收取。右：本次共同区间2007-08-01至2017-03-31，0/10/20/30bp每边。样本不同，费用水平可映射；原文具体计提仍不完全公开。\normalsize",
    table(["每边成本","累计收益","CAGR","Sharpe","最大回撤"],
          [[f"{fee} bp",pct(metric(row,"total_return")),pct(metric(row,"cagr")),f"{metric(row,'sharpe'):.3f}",pct(metric(row,"max_drawdown"))]
           for fee in (0,10,20,30) for row in [cost_row(data,fee)]],"Xrrrr"),
    r"""\subsection*{怎样解释这张图}
每次只改变费率，信号、样本和执行时点保持一致。费用不仅直接减少现金，也改变此后可买份额，因此长期差值不能简单用“交易次数乘固定金额”复原。更高费率的结果不是另一套经过择优的策略。
\subsection*{次日开盘与次日收盘}
主结果只使用收盘形成、下一开盘成交。控制实验保持信号在$t$收盘形成，把成交推迟至$t+1$收盘，重新计算持仓、费用及买持基准。该对照仍有完整一日滞后，不使用同日收盘抢先交易。指数没有可直接交易账户，实际ETF扩展需独立验证。
""",
    r"""\page{9\quad 差异归因：有限单因素实验}
\lead{先排除公式错误，再量化口径差异；对照之间存在相互作用。}
全部对照以固定基准为参照，保留不改善和零差异结果。暖机在原样本兼容阶段尤为关键；共同区间已拥有充足历史时，严格与部分窗口可能趋同，不据此否认早期影响。
""",diagnostic_table(data),
    r"\subsection*{原样本区间的暖机对照：零成本，截止2017-03-31}",
    table(["候选","严格CAGR","部分CAGR","ΔCAGR(pp)","ΔMDD(pp)"],warmup_rows(data),"Xrrrr","scriptsize"),
    r"""\subsection*{哪些差异能够归因}
在相同数据下，改变标准化是否含当日、ddof、MA lag、费用或成交时点得到的差值，是该设定在指定样本和其他设定固定条件下的影响。它不证明原文采用其中某个未披露口径，也不能外推为所有市场阶段的固定效应。
\subsection*{哪些差异仍未解释}
由于未取得Wind原始快照，不能把剩余差距统称为数据供应商效应；也没有原作者代码证明其早期窗口最低样本数、波动率分母或日收益记账方式。对照作用有相互影响，禁止把各差值直接相加解释原文全部差距。
\subsection*{为何保留较差结果}
课程要求检验和解释，而非只展示收益最高曲线。固定主信号后，其余标准分、均线、暖机、费率和波动目标均保留在完整结果中，避免在看过最终区间后改写主候选或只报告冠军。
""",
    r"""\page{10\quad 自主改进：保留信号，调整风险仓位}
\lead{创新点是仓位映射；MA入场过滤来自原文。}
令原右偏+MA二值信号为$s_t\in\{0,1\}$，以截至$t$的20个日收益样本标准差估计年化波动率：
\[\widehat\sigma_t=\sqrt{252}\,s(r_{t-19:t}),\qquad
w_t=s_t\min\{1,0.15/\widehat\sigma_t\}.\]
信号与波动率均在收盘可得，次日开盘将账户调整至目标，允许现金、不加杠杆。缺少完整20日收益时目标为0；完整窗口若波动为0则比例上限为1，不除以零。每日调整可能增加换手。
""",image("main_metrics.png","57mm"),
    table(["发表后主端点","累计收益","CAGR","Sharpe","最大回撤"],summary_rows(data),"Xrrrr"),
    rf"波动目标相对原策略的回撤变化{tex(pp(-d['drawdown_reduction']))}个百分点，CAGR变化{tex(pp(d['cagr']))}个百分点。相对固定75\%仓位，回撤变化{tex(pp(-d['versus_cash_drawdown_reduction']))}个百分点，CAGR变化{tex(pp(d['versus_cash_cagr']))}个百分点；固定75\%的回撤更低。",
    rf"原策略、固定75\%与波动目标的年化换手分别为{metric(data['base'],'annual_turnover'):.2f}、{metric(data['cash'],'annual_turnover'):.2f}、{metric(data['improved'],'annual_turnover'):.2f}倍；平均形成目标分别为{tex(pct(data['base']['exposure']))}、{tex(pct(data['cash']['exposure']))}、{tex(pct(data['improved']['exposure']))}。",
    r"""\subsection*{控制实验解释}
固定75\%仓位使用相同二值信号并每日调仓，控制纯粹降低市场暴露的效果；其余25\%现金不计息。它仍未与波动目标精确匹配平均暴露或逐日换手，因此不能独立证明主动波动管理产生alpha。主评价是回撤，收益、Sharpe和换手必须同时报告。
""",
    r"""\page{11\quad 发表后净值与跨指数检查}
""",image("main_curves.png","73mm"),
    r"\small 沪深300，2017-05-02至2026-10-15，买卖各10bp；初始资金相同，现金不计息。固定75\%与波动目标均每日调整；原RSRS只在状态改变时交易。\normalsize",
    image("robustness.png","72mm"),
    r"""\small 上图每组对照共享数据、样本和10bp成本。CAGR差值=波动目标−基准，回撤改善=基准MDD−波动目标MDD，二者分别判断。共同区间、发表后与近期历史互有联系，三指数也高度相关，不能把九组结果当九次独立实验。\normalsize
\subsection*{稳健性而非重新选优}
10\%和20\%目标只检查15\%附近敏感性，不在此后挑最高收益目标替换主方案；零成本与双倍费率检验成本依赖。完整矩阵在performance.csv和paired\_comparison.csv中保留，本文突出研究问题而非隐藏失败案例。
""",
    r"""\page{12\quad 不确定性、重采样与失败保留}
\lead{历史差值不是已确立的未来优势。}
使用配对区块重采样：将基准与改进的同日收益一起抽取，保留两策略同期关联；区块长度20个交易日，1000次，随机seed为20261019。百分位区间只描述这一历史收益序列的重采样不确定性，不替代新的前向样本。
""",bootstrap_table(data),
    r"""\subsection*{重采样的局限}
市场结构变化、择时信号的长记忆、策略开发与多重比较均未被短区块完全建模。重采样后的收益路径不是重新按OHLC计算信号并成交的完整新市场，因此仅作描述性配对评价。区间跨零时尤其不能称为稳健改进；不跨零也不能证明可交易未来收益。
\subsection*{保留的负面证据}
主改进如果降低回撤但削弱收益，应陈述风险与收益交换；如果固定75\%已达到相近或更好效果，应把简单降仓列为有力替代。原研报的高年化数字与发表后较弱表现可以同时存在，不能用新时段表现否定公式实现，也不能用原文高收益忽略后续退化。
\subsection*{未进行的再优化}
没有依据最终区间逐个寻找最高收益N/M/阈值，没有用发表后冠军取代预先固定的right\_ma，也没有将RSRS研究替换P1默认LightGBM策略。P1原路线与P2研究在同一平台中可分别查看、复跑和解释。
""",
    r"""\page{13\quad P1接入、复现入口与交付验收}
\begin{center}\begin{tikzpicture}[node distance=3mm,every node/.style={draw=accent,fill=accent!5,rounded corners,align=center,text width=32mm,minimum height=16mm,font=\small}]
\node(a){已有数据接口\\OHLC/交易日};\node(b)[right=of a]{新增RSRS模块\\指标/状态/目标};\node(c)[right=of b]{已有回测引擎\\成交/现金/持仓};\node(d)[right=of c]{已有绩效分析\\Sharpe/回撤/核账};
\draw[-{Stealth},accent](a)--(b);\draw[-{Stealth},accent](b)--(c);\draw[-{Stealth},accent](c)--(d);\end{tikzpicture}\end{center}
\subsection*{新增与复用的边界}
新增cfquant.rsrs负责指数数据适配、因果指标、阈值状态及波动目标；已有engine负责订单与成交，analytics负责收益、风险与reconcile。统一界面提供结果浏览、配置与研究重跑。准备版独立实现保留为交叉检查，不复制成另一套正式主账户。
\subsection*{建议复现顺序}
\begin{enumerate}\item 阅读PROJECT2\_SOURCE\_AUDIT与固定protocol，核对原图表、样本与补充假设。\item 按数据说明准备三个快照；先检查SHA-256、字段、日期和行情质量。\item 运行公式/因果/账本检查，查看实际输出，不把“脚本退出成功”当全部验收。\item 运行正式矩阵；核对performance、paired\_comparison、diagnostics及重采样。\item 在青序中查看同一证据、手动切换样本与策略，再生成报告和演示。\end{enumerate}
""",table(["交付项","状态或定位"],[
        ["实验协议","configs/project2_rsrs.json及evidence/project2_rsrs/protocol.json"],
        ["实际结果","performance、daily、paired_comparison、diagnostics、reproduction_table"],
        ["数值检查","formula_checks.json、validation.json、bootstrap.json"],
        ["材料","报告、Beamer、可编辑PPTX、逐页讲稿及LaTeX源码"],
        ["软件验证",test_text], ["网页验证",ui_status]],"lX"),
    r"每一项检查的范围写入验收记录；本地文件准备与Git推送不等于已在Blackboard提交。提交前由组员核对课程入口、截止时刻、命名及上传回执。",
    r"""\page{14\quad 结论、局限与参考资料}
\lead{本项目的价值在于把研报陈述变成可逐层核验的研究过程。}
我们将RSRS方法接入P1平台，保留原样本兼容、共同区间与发表后三层结果；识别出原文部分历史暖机与准备版严格暖机的重要差别；通过单因素矩阵呈现可解释差异，保留尚未由原始数据解释的部分。
自主改进保留信号、按20日波动估计控制风险，并以固定75\%仓位检验“少冒风险”这一替代解释。收益、风险、换手和费用一同报告；改善与否都不改变预先固定候选或抹去失败结果。
\subsection*{主要限制与下一步}
\begin{enumerate}\item 原Wind快照、精确样本与作者账户未公开，尚未实现逐笔同源复制。\item 指数点位不是可交易证券；下一步用ETF实际历史、分红、交易单位、冲击和跟踪误差验证。\item 所有历史已经观察，缺少未知数据前向检验；应冻结规则后追加新交易日。\item 三指数及区间并不独立，有限重采样不消除开发偏差或结构变化。\item 固定75\%对照不精确匹配平均暴露；进一步可预先设计暴露/换手匹配控制，但本次不追加择优搜索。\end{enumerate}
\subsection*{参考资料与原始身份}
\begin{enumerate}
\item 课程课件：CF2026\_Project2.pdf，8页；SHA-256：\par{\scriptsize\texttt{346219417ef143471a8461102dfeef06e1560a1eecd08698f320ad2582fc1ad2}}。
\item 刘均伟、周萧潇，2017，《基于阻力支撑相对强度（RSRS）的市场择时——技术择时系列报告之一》，光大证券；方法见第7--10、13--20页，部分历史说明见第22页。
\item 原文PDF SHA-256：\par{\scriptsize\texttt{34597264ed6682d8edafb94c0c540bb60ce407ee00552310acf424a39031d9a}}。
\item 公开定位仓库：\url{https://github.com/hugo2046/QuantsPlaybook}；\par 固定提交：{\scriptsize\texttt{3f83a77f4c7996ed8185b8979a42c4372c1d811f}}。
\item 平台与研究代码：\url{https://github.com/caolvchong218/cf2026-quant-platform}；本次版本与实际推送回执以提交包内验收记录为准。
\end{enumerate}
\subsection*{学术与工具说明}
使用自动化工具辅助实现、资料核对、实验和写作；原文事实与本项目假设分列。组员须能够解释公式、账户、差异和限制。课程如另有AI使用披露模板，应按授课教师要求补充。
\end{document}
"""]
    return "\n".join(content)


def diagnostic_table(data):
    frame=data["diagnostics"]
    part=frame[(frame.index_id=="sh000300") & (frame.period_id=="historical_overlap")]
    labels={"prior_M_ddof0_lag2_next_open_10bps":"共同区间基准",
            "include_current_M":"标准化纳入当日", "sample_std_ddof1":"样本标准差ddof=1",
            "trend_lag3":"MA比较滞后3日", "next_close_execution":"次日收盘成交",
            "cost_0bps":"零成本", "cost_20bps":"20bp/边", "cost_30bps":"30bp/边"}
    rows=[[labels[row.case],pp(row.delta_cagr),f"{row.delta_sharpe:+.3f}",pp(row.delta_max_drawdown)] for _,row in part.iterrows()]
    return table(["唯一改变","ΔCAGR(pp)","ΔSharpe","ΔMDD(pp)"],rows,"Xrrr","scriptsize")


def real_formula_text(data):
    check=data["formula_checks"]
    values=check["manual"]
    if not check["passed"] or not check["signal_date_precedes_execution"]:
        raise ValueError("Actual-window formula check must pass before authoring")
    return (rf"\subsection*{{真实窗口的独立复算}}形成日{tex(check['formation_date'])}，执行日{tex(check['requested_execution_date'])}；"
            rf"独立最小二乘重复计算16日窗口和此前300个斜率。$\beta={values['beta']:.6f}$，$R^2={values['r2']:.6f}$，"
            rf"$z={values['z']:.6f}$，右偏分${values['right']:.6f}$；最大绝对误差为{max(check['absolute_errors'].values()):.2e}，"
            r"小于预定$10^{-8}$容差。16个观测点、残差和此前斜率日期均可核对。")


def bootstrap_table(data):
    rows=[]
    for item in data["bootstrap"]["comparisons"]:
        rows.append(["15%目标 − "+NAMES[item["baseline"]],pp(item["annualized_mean_daily_return_difference"]),
                     f"[{pp(item['ci_low'])}, {pp(item['ci_high'])}]"])
    return (r"\noindent 指标为252倍配对日收益差的算术均值，单位为百分点，\textbf{不是CAGR差值}。"+
            table(["比较","年化算术均值差(pp)","95%区间(pp)"],rows,"Xrr"))


def deck(data, test_text, ui_status):
    d=differences(data); b=data["base"]; v=data["improved"]
    slides=[]
    def add(title,claim,items,notes,**extra):
        slides.append({"title":title,"claim":claim,"items":items,"notes":notes,"kind":"content",
                       "seconds":SECONDS[len(slides)],"source":"来源：光大证券2017-05-01研报；本项目固定协议与公开研究证据。",**extra})
    add("RSRS：从研报到可核验实验","复现核心结论，解释差异，检验一项风险改进",
        ["Computational Finance · Project 2",MEMBERS,"2026年10月19日 · 约15分钟"],
        "各位老师、同学好。我们把RSRS择时研究接入青序平台。这次展示分为原文复现、差异归因和一项仓位改进。我们希望回答的不只是收益有多高，还包括结果是否可核验、差异是否有证据。",kind="cover")
    add("研究问题与正式要求","原文、实验、解释必须形成完整对应",
        ["目标：2017年光大RSRS研报，图25、表5、图26", "先核公式和账本，再比较结果", "改进不奏效也保留；复用P1平台"],
        "我们的研究对象是2017年光大证券的RSRS研报，重点复现价格优化策略的图二十五、表五和费用图二十六。课程要求把核心结论变成可核验实验，所以我们先说明公式和交易规则，再解释结果差异。新策略收益高并不能替代这条证据链；反过来，结果较弱但实现和解释完整，也有研究价值。我们复用P1的成交、资金和绩效模块，新增RSRS信号与风险仓位。")
    add("原文审计：暖机是关键发现","原文早期使用部分历史，准备版等待完整窗口",
        ["原文第22页：不足600个斜率时使用已有历史", "兼容实验与严格暖机实验分别展示", "样本首尾、成本和标准差细节未完全公开"],
        "审计原文时，我们发现一个会直接影响早期净值的细节。原文第二十二页说明，前六百天没有完整斜率历史时，可以使用已有数据；而准备版为了统一信息边界，等待完整窗口才发出信号。这两种处理不相同，不能把准备版称为完全按原文复现。因此正式实验增加部分历史窗口兼容版本，同时保留严格暖机主协议。原文还有截止三月还是四月、图形起点与正文不一致的问题，我们都保留，没有任选一个方便的说法。")
    add("RSRS指标与交易状态","滚动回归衡量高低价关系，阈值决定持仓状态",
        ["H = α + βL + ε；保留β及R²", "z → zR² → zR²β；右偏用N=16、M=300", "超过0.7入场，低于−0.7离场；MA只过滤入场"],
        "RSRS首先在最近N个交易日，用最低价解释最高价，做带截距的一元回归，得到斜率beta和拟合优度R平方。然后把斜率相对之前M个斜率标准化为z。修正分是z乘R平方，右偏分再乘beta。右偏版本的N是十六、M是三百，其他标准分版本用十八和六百，因此几种版本的差异不只有变换本身。策略是带记忆的状态机：空仓时超过零点七才考虑买，持仓后跌破负零点七才卖，中间区域继续原状态。均线只限制买入，不会因为均线下降单独触发卖出。",
        formula=r"z_t=\frac{\beta_t-\mu_{t-1,M}}{\sigma_{t-1,M}},\quad q_t=z_tR_t^2\beta_t",
        formula_plain="右偏分 = 斜率的历史标准分 × R² × 当前斜率")
    add("信息时点与P1接入","收盘形成目标，次日开盘成交，独立重建账户",
        ["数据 → RSRS指标/状态/权重 → P1成交引擎", "次日开盘、每边10bp、现金0利息、无杠杆", "手算、未来扰动、账本和独立实现交叉核验"],
        "所有当日最高价、最低价和收盘价在收盘后才完整可得，所以主结果按收盘形成目标，下一交易日开盘执行，再按收盘估值。次日开盘和每边十个基点的费用，是我们明确补充的执行约定，不冒充研报已有的全部规定。RSRS输出交给青序原回测引擎，费用按实际成交金额扣除。验证不仅看曲线，还包括OLS手算、阈值状态、扰动未来是否改变过去，以及从每笔成交重新计算现金、份额和净值。")
    add("三层样本分别回答不同问题","兼容原样本、共同区间、发表后检验分开呈现",
        ["原样本兼容：2005-03至2017-04，部分历史暖机", "共同区间：2007-08-01至2017-03-31，严格暖机", "发表后：2017-05-02至2026-10-15，已观察历史"],
        "我们把结果分成三层。第一层尽量接近原文的二零零五年三月到二零一七年四月，使用部分历史暖机，但无法消除原始Wind快照和精确日期缺失。第二层采用二零零七年八月开始的共同区间，保证严格暖机后的公平比较。第三层是研报发表后的历史检验。它可以检验方法后来表现怎样，却不能当作原图的复现。近期区间是第三层的子集，也不是独立样本。")
    add("图25：同主题，并排看差异","原文曲线与兼容实验连同假设一起比较",
        ["左：原文图25；右：本次部分窗口、零成本、次日开盘", "三类信号全部保留；不按最终收益选赢家"],
        "这里左边是原文图二十五，我们保留原图标题、图例和来源，右边是同主题的兼容实验。右图使用本次指数快照、部分历史窗口、零成本和下一开盘成交。我们没有把右图直接叫作原文的精确复制，因为仍缺Wind输入和作者完整交易规则。比较首先看三种方法是否都按定义实现，再看它们在明确设置下的相对表现。形状接近不等于完全复现，终值不同也不自动意味着公式错误，必须继续检查可以控制的因素。",
        kind="comparison",figure="reproduction_curves.png",original_image="source_figure25.png")
    add("表5：数字接近之前，先对齐定义","原文年化28.19%，不直接等于本次可比目标",
        ["原右偏+MA：年化28.19%、Sharpe 1.56、回撤22.31%", "本次年化/Sharpe/回撤统一定义，原表保留", "交易次数、持仓日与费用定义仍有差异"],
        "原表五的右偏价格优化策略报告年化百分之二十八点一九，夏普一点五六，最大回撤百分之二十二点三一。我们把原始二十行数据固定在审计记录中，同时给出本次统计。比较时先处理定义：原文回撤用负号，我们用正的损失幅度；成交笔数和已经完成的往返次数并不相同；原文也未完全写清持仓日统计方式。表格差值可以展示，但不能简单称为复现误差，更不能把发表后新时段数字填到原表里。",
        table={"headers":["候选","原年化","本次年化","本次Sharpe"],"rows":[[NAMES[n],o,pct(metric(r,"cagr")),f"{metric(r,'sharpe'):.3f}"] for n,o in (("standard_ma","27.37%"),("corrected_ma","26.13%"),("right_ma","28.19%")) for r in [pick(data["performance"],n,data["source_period"],cost=0)]]})
    add("图26与控制实验：解释差异","一次只改变一个主要因素，保留完整结果",
        ["共同区间成本0/10/20/30bp，样本与原图不同", "收盘t信号，比较次日开盘与次日收盘成交", "多个对照的收益差值不能机械相加"],
        "这里展示成本对照：本次共同区间内，信号和样本固定，仅改变每边费率。右图与原图二十六样本不同，只能对照成本作用，不能比较绝对终值。原图没有公开全部费用计提细节，所以仍不能宣称账户完全一致。其他对照分别改变暖机、标准化是否包含当日、标准差分母和均线索引。成交时点对照保持收盘t形成信号，比较下一开盘与下一收盘，因此都没有同日抢先交易。各个因素可能相互作用，不能把所有差值加起来，声称解释了原文全部差距。没有原Wind快照的剩余差异，也不能笼统归咎于数据源。",
        kind="comparison",figure="cost_curves.png",original_image="source_figure26.png")
    add("自主改进：15%波动率目标","保留原信号，以过去20日波动决定投入比例",
        ["w = 原信号 × min(1, 15% / 估计年化波动)", "次日开盘每日调整；现金允许、上限100%", "固定75%仓位对照：检查是否只是少冒风险"],
        "完成核心复现后，我们只改变一个主要因素：仓位大小。原信号买入时，过去二十日估计年化波动如果超过百分之十五，就降低投入比例；否则最多满仓。卖出信号仍是零仓位，现金不计利息，也不加杠杆。这是风险管理改进，不是新的择时预测。为了防止把单纯少买股票的效果说成创新收益，我们增加固定百分之七十五仓位对照，而且同样每日调整，费用照收。",
        formula=r"w_t=s_t\min\!\left(1,\frac{0.15}{\sqrt{252}\,s(r_{t-19:t})}\right)",formula_plain="保留RSRS买卖信号；高波动时降低仓位")
    add("主结果：收益、风险与简单减仓一起看",
        f"相对原策略：回撤{pp(-d['drawdown_reduction'])}pp；CAGR{pp(d['cagr'])}pp",
        [f"原策略：CAGR {pct(metric(b,'cagr'))}，回撤 {pct(metric(b,'max_drawdown'))}",
         f"15%目标：CAGR {pct(metric(v,'cagr'))}，回撤 {pct(metric(v,'max_drawdown'))}",
         f"相对固定75%：CAGR{pp(d['versus_cash_cagr'])}pp，回撤{pp(-d['versus_cash_drawdown_reduction'])}pp"],
        f"这是预先固定的主端点：沪深三百，发表后区间，每边十个基点。原右偏加均线策略年化为{pct(metric(b,'cagr'))}，最大回撤{pct(metric(b,'max_drawdown'))}；百分之十五波动目标的年化为{pct(metric(v,'cagr'))}，最大回撤{pct(metric(v,'max_drawdown'))}。所以我们用两个维度陈述：回撤下降{pct(d['drawdown_reduction']).replace('%','')}个百分点，年化增加{pct(d['cagr']).replace('%','')}个百分点。但与固定百分之七十五仓位相比，波动目标的收益更高，最大回撤也更大，固定减仓的风险效果更直接。这个对照很重要，因为我们关心波动控制是否超出简单降仓；平均暴露和实际换手仍未完全匹配，配对收益差区间也跨零，因此不把差值解释成确定的alpha。所有结果都保留，不因为不够漂亮换掉基准或挑另一个参数。",
        figure="main_metrics.png",table={"headers":["策略","CAGR","Sharpe","回撤"],"rows":[[NAMES[n],pct(metric(r,"cagr")),f"{metric(r,'sharpe'):.3f}",pct(metric(r,"max_drawdown"))] for n in ("right_ma",data["cash_id"],"right_ma_vol15","buy_hold") for r in [pick(data["performance"],n)]]})
    add("稳健性与不确定性","跨指数、费用与区块重采样检查依赖条件",
        ["三个指数 × 三个阶段；近期与发表后重叠", "10%/15%/20%只作敏感性，不再选赢家", "配对20日区块、1000次；历史已观察，非盲测"],
        "我们再看其他指数、其他阶段和费用水平，目的不是找出新的冠军，而是看结论是否依赖某一段市场。图中正负变化同时保留，收益改善与回撤改善分开呈现。三个指数相关，近期区间又包含在发表后区间中，所以不能把九组当作九次独立证据。我们还用二十日配对区块、一千次重采样描述差值的不确定性，但它不是新的市场，也没有重新生成完整OHLC交易。全部历史先前已经观察，所以整个研究仍是历史复核，而非未知样本上的成功证明。",figure="robustness.png")
    add("平台演示与可复现交付","同一界面查看原文、配置、结果和核验",
        ["打开P2研究页，切换三层样本与策略", "检查信号、账本、成本和对照结果", "导出配置/证据；报告、Beamer、PPTX与讲稿对应"],
        "下面用约一分钟演示平台。首先进入Project二研究页，展示研究协议和原文定位。然后选择沪深三百与发表后区间，先看原右偏加均线，再切换百分之十五波动目标和固定百分之七十五对照，确认曲线和指标同时变化。接着打开核验和差异对照，说明这里读取的是同一份可下载证据。若课堂网络或重跑耗时不适合现场，我们使用已保留的固定快照结果；现场重跑有独立配置和输出，不覆盖正式证据。代码复用P1资金账本，材料的每个数字来自同一实验输出，组员可以从公式一路追到最终指标。",kind="demo")
    add("结论：可解释的复现，比单条漂亮曲线更完整","实现可核验，差异有对照，改进保留代价",
        ["发现并量化原文部分历史暖机的影响", "固定信号加入风险仓位，并检验简单减仓替代", "下一步：真实ETF执行与冻结后的新数据"],
        "我们的结论有三点。第一，RSRS核心公式和账户可以接入平台并逐层核验，但原始快照和口径缺失限制了精确复制。第二，暖机等设定应通过单因素实验解释，不能泛泛地说数据不同。第三，仓位改进要同时报告收益代价，并与简单降仓比较。下一步应使用真实ETF和冻结后的新增数据检验。谢谢大家，欢迎提问。")
    add("备份：手算与指标口径","可从三个观测点验证回归，再核对日账本",
        ["L=(1,2,3)，H=(2,3,5)：β=1.5，α=1/3，R²=27/28", "此前β=(1,2,3)，当前β=4：z=√6", "Sharpe：252日年化、日收益样本标准差、rf=0"],
        "如果被问到如何确认公式正确，可以从三个观测点手算。低价一二三，高价二三五，带截距OLS斜率一点五，截距三分之一，拟合优度二十七分之二十八。此前斜率一二三、当前四，按总体标准差标准化得到根号六。再说明实际策略各指标都保存，账本核验与公式核验是不同层。")
    add("备份：限制与复现实务","没有把未知细节、历史复核或指数代理包装成确定性",
        ["Wind原始输入和作者精确样本未公开", "指数点位≠可直接交易ETF；分红与冲击未建模", test_text, ui_status],
        "如果被问到是否达到实盘要求，答案是本次研究还不能据此直接交易。价格指数不是证券，ETF有跟踪误差、交易单位、分红和冲击；原始Wind数据和作者账户也未公开。测试通过说明我们检查过的公式和软件行为符合约定，不证明策略未来有效。所有历史已观察，下一步需要冻结规则后的新数据。")
    supplements={
        1:"具体来说，图表并排只是第一步。还要交代使用什么样本、收益如何年化、什么时候成交、费用如何扣除，并且保留配置和输入身份，让另一个小组从同一数据出发能够重跑。如果某个差异无法通过控制实验解释，我们就明确列为未解释部分。",
        2:"这个发现也改变了我们的工作顺序：先对照研报中关于早期样本的文字，再增加兼容实验，而不是调参数把收益追到原文。主图统一选择三月底作为本次端点，四月底仅做敏感性，避免在原文歧义中挑最有利的日期。",
        3:"直观地看，斜率描述近期高点相对低点的变化关系，标准化把不同历史阶段放到相对位置，拟合优度则压低回归关系不稳定时的信号。但这种解释只是经济直觉，不能代替收益验证。我们把每个中间量都保留下来，能追查某一天为什么持仓或空仓。",
        4:"例如，真实窗口的独立复算是直接做最小二乘，再重复计算此前三百个回归；不是用主程序的输出再抄一遍公式。日账本另从成交反推份额和现金，因此可以区分指标算错、日期对齐错与账户记账错。",
        5:"数据来自三个指数的公开日线快照，而不是P1的一千只股票池。沪深三百和中证五百各有五千二百八十八条记录，上证五十有五千五百三十一条。我们保存了哈希和缺失检查，保证报告、演示和平台看到的是同一份输入。",
        6:"图中买入持有也按相同起点和成本重新建立账户，初始净值一致。右图的纵轴是账户净值除以初始资金，不是价格点数，也没有把原文终值作为归一化目标。对照时可以关注相对排序、回撤阶段和长期变化，避免只盯住最后一个数字。",
        7:"我们自己的夏普统一用每日扣费收益、二百五十二日年化和零无风险收益率。最大回撤包含初始资金这个峰值。原文没有全部披露这些设置，所以即使年化收益接近，夏普或交易统计仍可能不同；需要比较定义才能解释。",
        8:"共同区间里，纳入当前斜率和改用样本标准差恰好没有改变最终交易结果，这个零差异也保留。它说明小的数值变动未必跨过交易阈值，不能推断两种指标永远等价。均线索引和成交延后一日则产生可见变化。",
        9:"二十日窗口和百分之十五目标在正式运行前固定，百分之十和二十只检查附近敏感性。我们不会看到哪个目标最高就改口说那是主策略。波动管理在剧烈行情中降低仓位，也可能错失反弹，并且每天调整要支付额外费用。",
        10:"换手也支持这个谨慎解释。原策略年化换手约四点六一倍，波动目标提高到五点四六倍，固定七成五仓位约三点六五倍。波动控制不是免费获得风险改善，账户需要实际调仓；更复杂的操作只有带来清楚的增益才有意义。",
        11:"两组配对年化算术日收益差的百分之九十五区间都跨过零，所以这里没有给出稳健的收益优势结论。注意这个统计量是日收益差均值乘二百五十二，不是年化复合收益率的差。区分这两种口径也是报告核验的一部分。",
        12:"现场还可以查看真实窗口的公式复算记录和日期边界，把公式与账户连接起来。提交材料保留原文定位、固定协议、代码、依赖、快照说明及数值输出；另一位同学可以先复核聚合证据，再用本地数据重跑。平台只是研究入口，最终证据来自保存的实际实验。",
    }
    for index,text in supplements.items():
        slides[index]["notes"] += text
    if sum(s["seconds"] for s in slides) != 900:
        raise AssertionError("Main presentation must total 900 seconds")
    return slides


def beamer(slides):
    out=[r"""\documentclass[aspectratio=169,10pt]{beamer}
\usepackage[UTF8]{ctex}\usepackage{booktabs,tabularx,graphicx,amsmath}
\definecolor{navy}{HTML}{142B43}\definecolor{teal}{HTML}{168579}
\setbeamercolor{frametitle}{bg=navy,fg=white}\setbeamercolor{structure}{fg=teal}
\setbeamercolor{block title}{bg=teal,fg=white}\setbeamercolor{block body}{bg=teal!6}
\setbeamertemplate{navigation symbols}{}\setbeamerfont{frametitle}{size=\large}
\setbeamersize{text margin left=8mm,text margin right=8mm}
\setbeamertemplate{footline}{\leavevmode\hbox{\begin{beamercolorbox}[wd=.82\paperwidth,ht=2.5ex,dp=1ex,leftskip=1em]{author in head/foot}青序量化研究平台\quad Project 2 / RSRS\end{beamercolorbox}\begin{beamercolorbox}[wd=.18\paperwidth,ht=2.5ex,dp=1ex,center]{date in head/foot}\insertframenumber/16\end{beamercolorbox}}}
\begin{document}
"""]
    for i,s in enumerate(slides):
        out.append(rf"\begin{{frame}}{{{tex(s['title'])}}}")
        out.append(rf"\begin{{block}}{{}}{tex(s['claim'])}\end{{block}}")
        if s["kind"]=="cover":
            out.append(r"\vspace{3mm}{\Huge\bfseries\color{navy}RSRS 择时研究}\par\vspace{3mm}")
        if s.get("original_image"):
            out.append(side_images(s["original_image"],s["figure"],"44mm"))
        elif s.get("table"):
            t=s["table"];out.append(table(t["headers"],t["rows"],"X"+"r"*(len(t["headers"])-1),"scriptsize"))
        elif s.get("figure"):
            out.append(image(s["figure"],"43mm"))
        if s.get("formula"):
            out.append(r"\["+s["formula"]+r"\]")
        short=s["items"][:2] if (s.get("figure") or s.get("table")) else s["items"]
        out.append(r"\small\begin{itemize}")
        out.extend(r"\item "+tex(item) for item in short)
        out.append(r"\end{itemize}")
        # Matched notes are published separately and embedded in the editable
        # presentation. Keeping them out of Beamer avoids fragile note parsing.
        out.append(r"\end{frame}")
    out.append(r"\end{document}")
    return "\n".join(out)


def speaker(slides):
    out=[r"""\documentclass[UTF8,a4paper,11pt]{ctexart}
\usepackage[margin=22mm]{geometry}\usepackage{xcolor,fancyhdr}
\definecolor{ink}{HTML}{142B43}\definecolor{accent}{HTML}{168579}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small 青序 · Project 2 · 逐页演讲稿}\fancyhead[R]{\small 正文15分钟 + 2页答辩备份}\fancyfoot[C]{\thepage}
\setlength{\headheight}{15pt}\setlength{\parindent}{2em}\setlength{\parskip}{7pt}\setlength{\emergencystretch}{2em}\linespread{1.13}
\begin{document}
{\Large\bfseries\color{ink}RSRS研究：课堂口述与操作提示}\par
""",tex(MEMBERS)+r"\par",r"正文14页合计900秒；第15--16页仅供答辩。按实际语速排练，操作提示不朗读。"]
    md=["# Project 2：15分钟逐页讲稿", "", MEMBERS, "", "正文14页约900秒，另2页答辩备份。操作提示不朗读。", ""]
    elapsed=0
    for i,s in enumerate(slides):
        if i>0 and i%2==0:out.append(r"\newpage")
        timing=f"建议{s['seconds']}秒；{elapsed//60:02d}:{elapsed%60:02d}开始" if s["seconds"] else "答辩备份，不计入正文15分钟"
        out.extend([rf"\section*{{\color{{ink}}{i+1:02d}\quad {tex(s['title'])}}}",
                    r"{\small\color{accent}"+tex(timing)+r"}\par",tex(s["notes"])])
        if s["kind"]=="demo":
            prompt="操作提示：P2研究页 → 沪深300/发表后/10bp → 原策略、固定75%、15%目标 → 核验/导出。必要时用固定证据截图。"
            out.append(r"\par\noindent{\small\color{accent}"+tex(prompt)+"}")
        md.extend([f"## {i+1:02d} {s['title']}","",timing,"",s["notes"],""])
        elapsed+=s["seconds"]
    out.append(r"\end{document}")
    return "\n".join(out),"\n".join(md)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--tests-passed",type=int)
    parser.add_argument("--tests-skipped",type=int,default=0)
    parser.add_argument("--tests-scope",default="RSRS相关自动测试")
    parser.add_argument("--ui-status",default="界面验收状态以交付检查记录为准")
    args=parser.parse_args()
    test_text=(f"{args.tests_scope}{args.tests_passed}项通过、{args.tests_skipped}项跳过" if args.tests_passed is not None
               else "软件测试统计以交付检查记录为准")
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    data=load_inputs();figures(data)
    slides=deck(data,test_text,args.ui_status)
    write(OUT/"final_report.tex",report(data,test_text,args.ui_status))
    write(OUT/"beamer.tex",beamer(slides))
    script,md=speaker(slides)
    write(OUT/"speaker_script.tex",script)
    write(ROOT/"docs/PROJECT2_PRESENTATION_SCRIPT.md",md)
    write(OUT/"deck_content.json",json.dumps(slides,ensure_ascii=False,indent=2)+"\n")
    checks={"schema":"qingxu.project2.material_authoring.v1","slides":len(slides),"main_slides":14,
            "main_seconds":sum(s["seconds"] for s in slides),"notes_chinese_characters":sum(sum('\u4e00'<=c<='\u9fff' for c in s["notes"]) for s in slides[:14]),
            "test_claim":test_text,"ui_claim":args.ui_status,"pdf_compiled":False,"pdf_visual_reviewed":False,
            "evidence_hashes":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in E.iterdir() if p.is_file()},
            "source_figures":"Everbright 2017-05-01 pp19-20, captions and source retained",
            "numeric_claims":"formal evidence only; no inherited draft metrics"}
    write(OUT/"authoring_checks.json",json.dumps(checks,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"output":str(OUT),"slides":len(slides),"main_seconds":checks["main_seconds"],
                      "notes_chinese_characters":checks["notes_chinese_characters"]},ensure_ascii=False))


if __name__=="__main__":main()
