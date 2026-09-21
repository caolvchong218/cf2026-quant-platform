"""Create the ten-page report and Beamer sources from the published evidence."""
from pathlib import Path
import json,os,shutil,subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parents[1]; E=ROOT/'evidence/research_v2'
OUT=ROOT/'reports/final'; FIG=OUT/'figures'; FIG.mkdir(parents=True,exist_ok=True)
load=lambda name:json.loads((E/name).read_text(encoding='utf-8'))
val=pd.read_csv(E/'validation.csv'); test=pd.read_csv(E/'test.csv'); chosen=load('selection.json')['selected']
r=test[test.model==chosen].iloc[0]; controls=pd.read_csv(E/'controls.csv')
factors=pd.read_csv(E/'factor_summary.csv'); audit=load('data_audit.json'); cpi=load('cpi.json')
bench=load('benchmark_metrics.json'); uncertainty=load('uncertainty.json')
legacy=pd.read_csv(E/'legacy_controls.csv'); years=pd.read_csv(E/'legacy_years.csv')
names={'multifactor_raw':'标准化多因子','multifactor_neutral':'中性化多因子','ridge':'Ridge','lightgbm':'LightGBM','mlp':'小型 MLP'}
english=['Equal z-score','Neutralized','Ridge','LightGBM','Small MLP']
colors=['#168579','#4265a6','#bc7940','#7d6395','#8897a6']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
def save(fig,name):
    fig.tight_layout();fig.savefig(FIG/(name+'.pdf'));fig.savefig(FIG/(name+'.png'),dpi=170);plt.close(fig)
curves=pd.read_csv(E/'curves.csv',parse_dates=['date'])
fig,ax=plt.subplots(figsize=(9.4,3.2))
for label,en,color in [('提交策略 · 扣费','Selected strategy (net)',colors[0]),('沪深300 · 价格指数','CSI 300 price index',colors[1]),('现金 · 零利息','Cash (0% interest)',colors[4])]:
    d=curves[curves.series==label];ax.plot(d.date,d.nav,label=en,color=color,lw=1.6)
ax.set_ylabel('NAV (initial = 1)');ax.legend(loc='upper left',fontsize=8);save(fig,'nav')
fig,ax=plt.subplots(figsize=(9.4,2.6))
for label,en,color in [('提交策略 · 扣费','Selected strategy',colors[0]),('沪深300 · 价格指数','CSI 300 price index',colors[1])]:
    d=curves[curves.series==label];a=d.nav.to_numpy();ax.plot(d.date,1-a/np.maximum.accumulate(np.r_[1,a])[1:],label=en,color=color)
ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_ylabel('Drawdown');ax.legend(fontsize=8);save(fig,'drawdown')
for table,name,title in [(val,'validation','Validation: 2023-2024'),(test,'test_models','Final period: 2025-2026-09-18')]:
    fig,ax=plt.subplots(figsize=(9.4,3.2));x=np.arange(5)
    ax.bar(x-.18,table.annualized_return,width=.36,color=colors[0],label='Net CAGR')
    ax.bar(x+.18,table.max_drawdown,width=.36,color=colors[1],label='Max drawdown')
    ax.set_xticks(x,english);ax.axhline(0,color='#687788',lw=.7);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend();ax.set_title(title);save(fig,name)
fig,ax=plt.subplots(figsize=(9.4,2.8));x=np.arange(4)
ax.bar(x-.18,controls.annualized_return,width=.36,color=colors[0],label='Net CAGR');ax.bar(x+.18,controls.max_drawdown,width=.36,color=colors[1],label='Max drawdown')
ax.set_xticks(x,['Selected + risk','Equal weight, full','Double cost','Zero cost rerun']);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend();save(fig,'controls')
fig,ax=plt.subplots(figsize=(9.4,2.7));ax.bar(years.year.astype(str),years.net_return,color=[colors[0] if y>0 else '#b5615b' for y in years.net_return]);ax.axhline(0,color='#687788');ax.set_ylabel('Legacy annual net return');ax.yaxis.set_major_formatter(PercentFormatter(1));save(fig,'legacy')
fig,ax=plt.subplots(figsize=(9.4,3.3));short=['Mom60-20','Rev5','Low vol','Low range','Trend20','Volume','Liquidity','Low turnover','E/P','B/P','Dividend','ROE']
ax.bar(short,factors.rank_ic_mean,color=[colors[0] if x>0 else '#b5615b' for x in factors.rank_ic_mean]);ax.axhline(0,color='#687788');ax.set_ylabel('Mean daily Rank IC');ax.tick_params(axis='x',labelsize=8);save(fig,'factor_ic')

def pct(x):return f'{100*x:.2f}'+r'\%'
def rows(table):
    return '\n'.join(f"{names[x.model]} & {pct(x.annualized_return)} & {pct(x.max_drawdown)} & {x.sharpe:.3f} & {pct(x.mean_cash_weight)} \\\\" for x in table.itertuples())
rev=load('provenance.json')['git_revision'][:8]
sub={'REV':rev,'VAL_ROWS':rows(val),'TEST_ROWS':rows(test),'RETURN':pct(r.total_return),'CAGR':pct(r.annualized_return),'DD':pct(r.max_drawdown),
     'SHARPE':f'{r.sharpe:.3f}','CASH':pct(r.mean_cash_weight),'COST':f'{r.total_cost:,.2f}',
     'CPI_STOCK':pct(cpi['strategy_return']),'CPI':pct(cpi['inflation']),'REAL':pct(cpi['real_return']),
     'LOW':pct(uncertainty['cagr_95_percentile_interval'][0]),'HIGH':pct(uncertainty['cagr_95_percentile_interval'][1]),
     'TRAIN':f"{load('selection.json')['training_rows']:,}",'HASH':load('provenance.json')['data_sha256'],
     'IC_ROWS':'\n'.join(f"{load('factor_cards.json')[x.factor][0]} & {x.ic_mean:.4f} & {x.rank_ic_mean:.4f} & {pct(x.mean_factor_coverage)} \\\\" for x in factors.itertuples())}
report=r'''\documentclass[UTF8,a4paper,10pt]{ctexart}
\usepackage[margin=19mm,top=20mm,bottom=19mm]{geometry}
\usepackage{amsmath,amssymb,booktabs,tabularx,graphicx,xcolor,fancyhdr,enumitem,tikz}
\usetikzlibrary{arrows.meta,positioning}
\usepackage[colorlinks=true,linkcolor=teal,urlcolor=teal]{hyperref}
\definecolor{ink}{HTML}{142B43}\definecolor{accent}{HTML}{168579}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small CF2026 Project 1}\fancyhead[R]{\small 青序量化研究平台}
\fancyfoot[L]{\scriptsize 研究代码 @REV@ · 固定历史快照 · 描述性结果}\fancyfoot[R]{\thepage/10}
\setlength{\headheight}{14pt}\setlength{\parskip}{4pt}\setlength{\parindent}{2em}
\setlist{nosep,leftmargin=1.7em}\ctexset{section={format=\Large\bfseries\color{ink}},subsection={format=\normalsize\bfseries\color{accent}}}
\newcommand{\lead}[1]{\noindent{\bfseries\color{accent}#1}\par}
\newcommand{\page}[1]{\newpage\section*{#1}}
\begin{document}
\begin{center}
{\small\color{accent}COMPUTATIONAL FINANCE / PROJECT 1}\[8pt]
{\Huge\bfseries\color{ink}青序量化研究平台}\[8pt]
{\Large 可复现的多因子研究与风险控制}\[8pt]
2026年9月21日\quad 最终提交报告\quad 平台 v2.0（Python）
\end{center}
\section*{1\quad 研究问题与主要结论}
\lead{在可核验的账本上，比较简单因子、预测模型与风险约束。}
Project 1 的主体是正确、完整、可复现且可扩展的研究平台。按课程课件，基础项占80分、扩展项占20分，没有强制收益或Sharpe门槛。本报告以数据质量、信息时点、实际成交与实验控制为主线，同时检验扣费收益是否超过同期中国CPI。

沿用2019年末选定的1,000只主板股票，使用2019-10-08至2026-09-18的1,656,599条日线。新增每日估值、历史财报与行业区间，构建12个有经济解释的因子。候选包括标准化等权多因子、中性化多因子、Ridge、LightGBM和小型MLP。模型训练截至2022年，2023--2024年用于选择，2025年之后仅用于评估。

\begin{center}\begin{tabular}{llll}\toprule
最终预选策略 & 累计净收益 & 年化净收益 & 最大回撤\\\midrule
标准化多因子＋风险控制 & @RETURN@ & @CAGR@ & @DD@\\\bottomrule
\end{tabular}\end{center}
最终评估为2025-01-02至2026-09-18，共417个交易日。Sharpe为@SHARPE@，平均现金比例@CASH@。在2025年2月至2026年8月的19个共同完整月份内，策略累计@CPI_STOCK@，CPI累计@CPI@，实际购买力收益@REAL@。\textbf{跑赢同期CPI且回撤低于20\%的研究目标达成；年化5\%--8\%的较高期望未达成。}

同期沪深300价格指数年化8.55\%、回撤12.05\%。提交策略降低了风险，也牺牲了上涨参与度，不能据此宣称战胜市场或显著alpha。LightGBM后续表现较好，但验证期为负收益，未用最终期排名替换预选策略。
\subsection*{交付与证据导航}
平台包含策略研究、交互回测、因子诊断、数据复现及扩展指南。仓库提供源码、测试、固定依赖、合成样本与真实研究汇总。配套本报告、20页Beamer演示、可编辑PowerPoint、讲稿和提交清单。第2--6页说明实现，第7--9页说明实验，第10页逐项核验与披露边界。

\page{2\quad 架构、接口与可维护性}
\lead{数据、分数、目标权重、成交和统计分别承担独立职责。}
选择Python，复用NumPy、pandas、SciPy、scikit-learn及LightGBM。Streamlit和Plotly负责交互；界面不实现第二套收益计算。参考Qlib的因子表达与训练/验证工作流，保留透明的小型会计引擎，以便逐项核对课程的缺失、费用和标签口径。Backtrader仅作为独立参考测试依赖。
\begin{center}\begin{tikzpicture}[node distance=5mm and 5mm, every node/.style={draw=ink,rounded corners,align=center,font=\small,minimum height=9mm},>=Stealth]
\node(a){原始快照\\质量与哈希};\node[right=of a](b){因子/模型\\日期×资产分数};\node[right=of b](c){组合构建\\目标权重＋现金};\node[right=of c](d){成交账本\\成交/持仓/净值};
\draw[->](a)--(b);\draw[->](b)--(c);\draw[->](c)--(d);
\node[below=of c](e){诊断、实验记录与中文界面};\draw[->](b)|-(e);\draw[->](d)|-(e);
\end{tikzpicture}\end{center}
\noindent\begin{tabularx}{\textwidth}{lX}\toprule
模块 & 输入、输出与职责\\\midrule
data / incremental & 厂商响应转统一行情；原始响应与标准数据分开，重复键报错，增量合并另写新目录。\\
factors / features & 三个注册式基准因子及12个扩展因子；完整日历对齐，输出长表或日期×资产面板。\\
models & 只对训练区间拟合，输出后续分数与固定参数、随机种子。\\
risk / portfolio & 分数变为目标权重；保留缓冲、波动率预算及行业/单股上限。\\
engine & 下一开盘执行、现金约束、涨跌停、部分成交、缺失估值及逐日账本。\\
analytics & IC、分组、收益与回撤；独立从成交重建现金和单位余额。\\
experiment / research & 配置、数据和源码哈希，固定对照、时间分割、选择记录、历史运行归档。\\
app / ui\_research & 调用核心或读取已落盘研究结果，导出CSV/PDF，不在刷新时偷偷重训。\\\bottomrule
\end{tabularx}
\subsection*{核心数据契约}
行情键为\texttt{date, asset}；因子长表为\texttt{date, asset, factor, value}，高分方向在实验前固定。目标权重保留同样日期和资产轴，要求有限、非负且和不超过1。成交结果分为daily、trades、positions、orders，拒单原因和部分成交均留痕。CLI与界面使用同一配置与函数。
\subsection*{扩展路径与版本}
新增技术因子注册FactorSpec；基本面因子显式声明公告滞后；预测模型只输出相同面板。新组合只替换权重构建，成交与分析无需改写。完整研究重跑前归档旧目录；特征缓存核对输入文件及源码指纹，避免把旧特征冒充新版本。公开仓库不包含Token或厂商逐股原始数据。

\page{3\quad 数据来源、质量与偏差}
\lead{固定选池日先于所有收益研究，历史缺失保留并计数。}
2019-12-31按当日成交额取符合主板代码、价格至少3元且有成交的前1,000只，并列按代码。没有使用今日仍上市名单，后来停止报价的原成员也保留。该池不包括2020年以后IPO，单日流动性选择偏向当时热点，结论不代表全A股或沪深300。
\begin{center}\begin{tabular}{lr}\toprule
核验项目 & 数量或结果\\\midrule
统一日线 / 日历交易日 & 1,656,599 / 1,690\\
完整股票×日历网格缺行 & 33,401\\
重复行情键 / 异常剔除 & 0 / 0\\
涨跌停价格缺失 / 最新有报价资产 & 48 / 956\\
每日估值记录 / 重复键 & 1,656,599 / 0\\
财务指标记录 / 单次返回最高行数 & 57,389 / 67（接口上限100）\\
行业历史区间 / 特征网格 & 1,432 / 1,690,000\\
原行情清单哈希 / 新研究清单哈希 & 3,006 / 2,073个文件通过\\\bottomrule
\end{tabular}\end{center}
\noindent\begin{tabularx}{\textwidth}{lX}\toprule
来源接口 & 字段、单位与用途\\\midrule
daily / adj\_factor & OHLC原价为元/股，vol源单位手转股；统一乘法复权供收益计算。\\
stk\_limit / trade\_cal & 原始涨跌停价和完整交易日历；缺限制价时不执行对应方向。\\
daily\_basic & total\_mv为万元，PE/PB为倍数，dv\_ttm与换手率为百分数；每日横截面。\\
fina\_indicator & ann\_date为公告日，end\_date为报告期；ROE为百分数，严格滞后使用。\\
index\_member\_all & 申万一级行业进出区间；未知或冲突标UNKNOWN，不将当前行业反填历史。\\
index\_daily / cn\_cpi & 沪深300价格指数；全国CPI月环比，按共同完整月份比较。\\\bottomrule
\end{tabularx}
\subsection*{复权、可得性与存储}
所有OHLC用\(P^{adj}_{i,t}=P^{raw}_{i,t}A_{i,t}/A_{i,ref}\)，每资产固定首个有效参考因子。行情及研究数据合计约1.63GB，包括原始缓存、标准数据、特征和模型分数。原始请求记录字段、日期与UTC下载时间，保存SHA-256；同请求可断点续拉。

历史行业未知比例约0.52\%，年度ROE平均覆盖约98.90\%（按每日合资格样本计算）。采用公告时点仍不能消除厂商事后修订，当前数据不是完整的逐版本财报库。保留此局限，不用“无未来函数”概括所有数据风险。

\page{4\quad 因子设计与标准接口}
\lead{12个因子覆盖不同假设，经济方向固定，负结果同样保留。}
记\(C,H,L,V\)为同尺度价格与股数，\(r_t=C_t/C_{t-1}-1\)。下表所有分数均为“越大越优先”的事前方向，缺失不补未来值。完整字段、窗口、失败情形和实现位置见\texttt{docs/FACTOR\_CARDS\_V2.md}。
\begin{center}\small\begin{tabular}{lll}\toprule
因子 & 公式与窗口 & 主要假设或风险\\\midrule
中期动量 & \(C_{t-20}/C_{t-60}-1\) & 跳过短期反转；趋势突变失效\\
短期反转 & \(-(C_t/C_{t-5}-1)\) & 超调修复；持续下跌失效\\
低波动 & \(-s(r,20)\) & 防御性；牛市可能落后\\
低振幅 & \(-\overline{(H-L)/C}_{20}\) & 日内风险；与低波动冗余\\
均线趋势 & \(C_t/\overline C_{20}-1\) & 趋势延续；震荡易反复\\
量能变化 & \(\log(\overline V_5/\overline V_{20})\) & 量能确认；方向可能不稳定\\
流动性 & \(-\overline{|r|/(C^{raw}V/10^6)}_{20}\) & 冲击较低；可能偏大盘\\
低换手 & \(-turnover\_rate\) & 降低拥挤；也可能缺乏交易\\
盈利收益率 & \(1/PE_{TTM},\ PE>0\) & 价值；亏损公司无定义\\
账面市值比 & \(1/PB,\ PB>0\) & 价值；资产质量有差异\\
股息率 & \(dv_{TTM}/100\) & 分红；历史分红不可保证\\
盈利质量 & 最新已公告年度\(ROE/100\) & 盈利能力；财报修订与滞后\\\bottomrule
\end{tabular}\end{center}
动量和反转要求整个价格窗口完整，波动等滚动统计要求完整观测；不将停牌前后压缩为相邻日。PE/PB非正为缺失，财报同公告、同报告期的冲突副本剔除，年度报表超过550日不再使用。建模时至少10个有效因子；剩余标准化缺值填0表示当日横截面中性值，不将缺失收益标签填0。
\subsection*{基础项与扩展项的关系}
原始20日动量、5日反转、20日低波动保留注册入口，可独立调窗口、诊断和回测。12因子研究采用固定配方，标准化等权平均为一个透明基线。中性化改变行业/市值暴露，不保证提高收益；本次没有因为某因子最终IC为负而翻转符号。
\subsection*{与Qlib和参考论文的关系}
参考Qlib Alpha158中的滞后、滚动量价表达以及训练/验证分离，复用LightGBM和scikit-learn模型实现。此处12因子是明确列出的自定义子集，\textbf{未声称完整复制Alpha158，也未将Qlib引擎集成进主账本}。CogAlpha论文采用代码形式的因子与LLM演化搜索，本项目借鉴可检查的因子表示，未开展大规模自动搜索。

\page{5\quad 中性化、标签与模型比较}
\subsection*{每日横截面处理}
合资格条件为原价至少3元、过去20日平均成交额至少2,000万元、市值有效、至少10个因子有效，并以涨跌停比例排除近似5\%限制样本。最后一项只是ST类风险代理，不等于完整历史ST名单。
每个形成日单独对因子取1\%/99\%分位数截尾，使用当日均值和样本标准差得到\(z_{i,t}\)。中性化回归为
\[
z_{i,t}=\sum_g\alpha_{g,t}\mathbf1\{industry_{i,t}=g\}+\beta_t\log(MV_{i,t})+\epsilon_{i,t},\qquad
n_{i,t}=\epsilon_{i,t}/s_t(\epsilon).
\]
实现用分组去均值后回归市值，等价于包含行业虚拟变量的OLS残差。测试核验行业内残差均值为0、残差与市值正交。UNKNOWN作为单独组，不能凭当前行业补历史。
\subsection*{信息时间轴}
\begin{center}\begin{tabular}{ll}\toprule
区间 & 用途\\\midrule
2019年10月起 & 因子预热，选池日在2019-12-31\\
2020--2022 & 模型训练，有效训练样本@TRAIN@条\\
2023--2024 & 五个候选的验证与唯一选择\\
2025-01-02--2026-09-18 & 417日最终评估，禁止据此替换预选模型\\\bottomrule
\end{tabular}\end{center}
收盘\(t\)标签为\(y_{i,t}=O_{i,t+21}/O_{i,t+1}-1\)。训练目标使用当日横截面标签百分位秩减0.5，保留相对收益排序信息；跨入2023年的训练标签剔除。财务数据在公告日之后才可用；任何全样本标准化都被禁止。
\subsection*{固定模型与选择规则}
Ridge的\(\alpha=100\)；LightGBM使用250棵树、学习率0.03、15叶、最大深度5、叶最少500样本、L2=10；MLP为12--32--16--1，ReLU、Adam、8个固定训练轮次，种子20260921。MLP在预算内未收敛，结果只代表该固定配置。表格特征没有天然相邻顺序，本次选择小型全连接网络而非在任意因子顺序上卷积。

优先选择验证期年化为正且回撤不超过20\%的候选，再按年化收益/最大回撤排序。先保存selection.json，再进入最终期。原动量全区间表现已被看过，因此本次最终段不是完全未知的盲测；这一程序控制不能消除所有研究者选择风险。

\page{6\quad 组合、成交与账本核验}
\lead{每笔费用来自实际成交金额，模型只决定分数。}
每20个交易日调仓，目标50只。原持仓仍在前70名则优先保留，其余按分数补足，减少边界换手。先按波动率倒数构建权重，单股目标不超过4\%、行业目标不超过25\%；上限造成的余量留现金，不强制重新分配。

过去60日的候选组合收益估计年化波动率\(\hat\sigma\)。股票总预算为
\[
e_t=0.90\min\{1,0.12/\hat\sigma_t\}\times
\begin{cases}1,&CSI300_t\ge MA_{120,t},\\0.25,&\text{其他情况}.\end{cases}
\]
再应用个股与行业上限。所有估计截至形成日，次日开盘执行。风险估计中的少量历史缺价按0收益处理，是可低估风险的近似；实际缺价成交仍受阻。权重、波动和20\%回撤均为目标，不能保证市场变化后的实际风险。
\subsection*{成交与估值规则}
开盘按目标名义金额计算连续可分的复权单位，先卖后买，保留费用缓冲，现金不足同比缩减买单。涨停不买、跌停不卖、缺开盘或限制价不成交。每资产成交预算不超过此前20日平均成交额的1\%，超过时部分成交；这是事前容量代理，不是当日成交量的真实参与率保证。
\[
fee_t=0.001\,BuyNotional_t+0.0015\,SellNotional_t,\quad
Turnover_t=\frac{BuyNotional_t+SellNotional_t}{OpeningNAV_t}.
\]
买10bp、卖15bp是统一研究假设，不冒充历史法定税费。停牌持仓沿用最后已知收盘价并计数，连续缺失超过20个交易日保守计零；保留单位，恢复报价时重新估值，不虚构卖出。复权单位近似分红自动再投资，尚未模拟整手、完整分红现金或结算规则。
\subsection*{独立核验与扩展测试}
从成交逐笔重建现金与单位余额，再核对每日NAV、费用、换手与收益连乘。13个主研究运行全部通过，每次检查417或484个交易日。数值容差现金绝对\(10^{-6}\)元、相对\(10^{-10}\)。测试另覆盖手算、拆股连续性、独立Backtrader参考、未来扰动、公告滞后、中性化残差、无杠杆、部分成交及增量幂等。最终测试数量与日志保存在验收清单，不能用通过数代替对假设的审查。

\page{7\quad 原始动量为何亏损}
\lead{原账本能够复核，亏损不能只归因于费用。}
原始策略为20日动量、前30只等权、每5个交易日调仓。2020-01-02至2026-09-18累计净收益为\(-83.57\%\)，最大回撤89.83\%，费用254,496.22元。2022--2024连续大幅亏损，说明该固定样本和持有方式下的趋势假设没有稳定兑现。
\noindent\includegraphics[width=\textwidth]{figures/legacy.pdf}
\begin{center}\begin{tabular}{lrr}\toprule
口径 & 全区间累计收益 & 含义\\\midrule
原始扣费账本 & -83.57\% & 实际模拟路径\\
同成交路径加回费用 & -58.12\% & 费用退回闲置现金\\
零费用独立重跑 & -75.16\% & 仓位规模随无费用净值变化\\\bottomrule
\end{tabular}\end{center}
费用确实侵蚀收益，但零费用重跑仍亏损75.16\%。两个“毛收益”口径差异来自再投资与路径依赖，不能把累计费用简单加回当作可执行零费策略。原实验日均双边换手约20\%，长期高频再平衡增加了成本拖累。
\subsection*{同区间重新比较}
为避免把六年旧策略与不到两年新策略直接比较，原动量在2025-01-02重新从100万元启动。该期累计净收益\(-6.02\%\)、最大回撤33.46\%、费用101,397.93元。新预选策略为@RETURN@、@DD@和@COST@元。两者同时改变了因子与组合，所以这是\textbf{整体系统比较}，不能把差额全部归因于某一个因子。
\subsection*{能下的结论与不能下的结论}
账本、拒单和缺失估值均有证据，不支持把亏损直接判成程序记账错误。新策略减少换手与市场暴露，并引入防御/价值等因子；究竟哪项贡献最大需要单变量对照。固定池、行业暴露、停牌估值和成本假设仍会影响结果，未完成逐笔市场微观结构归因。

\page{8\quad 模型选择与最终区间表现}
\lead{验证期选出的简单基线保留为提交策略。}
所有候选使用相同股票池、期间、费用与风险规则。原始标准化等权基线使用z分数，其余学习模型使用中性化特征，因此模型间差异不能脱离输入处理解释。标准化基线与中性化等权基线构成预处理对照。
\begin{center}\small\begin{tabular}{lrrrr}\toprule
验证期2023--2024 & 年化净收益 & 最大回撤 & Sharpe & 平均现金\\\midrule
@VAL_ROWS@
\bottomrule\end{tabular}\end{center}
验证期标准化多因子年化5.84\%、回撤7.20\%，风险收益比最高。中性化和复杂模型未稳定提高验证收益，故没有把复杂度当作选择理由。
\begin{center}\small\begin{tabular}{lrrrr}\toprule
最终期2025--2026/9/18 & 年化净收益 & 最大回撤 & Sharpe & 平均现金\\\midrule
@TEST_ROWS@
\bottomrule\end{tabular}\end{center}
LightGBM最终期年化13.09\%，但验证期为\(-2.21\%\)，不能后验替换模型。提交策略年化@CAGR@、波动6.64\%，CPI目标达成但未达到更高的年化期望。
\noindent\includegraphics[width=\textwidth]{figures/nav.pdf}
\noindent\small 沪深300为价格指数且未扣复制成本，未含现金分红，策略则采用复权研究单位；两者不是完全对等的可投资基准。最终评估始终从独立100万元现金启动，不与验证期拼接冒充连续实盘。

\page{9\quad 因子诊断、压力测试与不确定性}
每日横截面先计算Pearson IC和Spearman Rank IC（并列平均秩），再汇总时间均值和样本标准差；至少30个有效配对，常数或不足样本为NaN。下表为2023-01-03至2026-08-18的878个形成日、20日未来标签，展示中性化因子。标准化前后对照另存preprocessing.csv。
\begin{center}\small\begin{tabular}{lrrr}\toprule
因子 & 平均IC & 平均Rank IC & 平均覆盖率\\\midrule
@IC_ROWS@
\bottomrule\end{tabular}\end{center}
低振幅与低波动Rank IC较高，但信息有重叠；流动性和趋势方向在此样本为负，未事后翻转。分组先用当日分数固定五组，再匹配未来收益，分别记录形成数与有效数。重叠20日标签的分组均值及高低组差是诊断量，不能连乘成可交易净值，不能据此声称独立样本显著性。
\subsection*{相同分数下的组合与费用对照}
\noindent\includegraphics[width=\textwidth]{figures/controls.pdf}
取消整套风险构建、改为满仓等权后，年化4.81\%、回撤13.12\%；风险组合为3.65\%、7.01\%。这是组合模块整体对照，不能单独识别趋势或波动控制的贡献。双倍费用下年化3.18\%、回撤7.24\%，说明低换手改善了费用耐受性。

共同完整月份2025/02--2026/08中，按CPI月环比连乘：策略@CPI_STOCK@，通胀@CPI@，实际收益@REAL@。不把同比连乘，也不把部分9月与整月CPI比较。对已实现净收益作20日循环分块、2,000次bootstrap，年化收益95\%分位区间为[@LOW@, @HIGH@]。该区间仅是固定策略的条件性描述，未涵盖模型选择及未来市场结构变化，不能保证未来跑赢通胀。

\page{10\quad 课程核验、复现与研究边界}
\noindent\begin{tabularx}{\textwidth}{lX}\toprule
课程要求 & 已交付的实现与证据\\\midrule
数据与质量 & 原始/标准分离、字段单位、复权尺度、缺失和异常报告、源文件哈希。\\
至少3因子 & 原3个可配置因子＋12因子卡，统一长表，经济方向与缺失规则明确。\\
回测与输出 & 日期/调仓/持仓/成本可配置，实际成交费用，净值、持仓、成交和订单导出。\\
IC与分组 & 日横截面IC/Rank IC、样本量/覆盖/分布；先分组再匹配标签。\\
绩效与实验 & 初始NAV参与回撤，252日年化，固定控制变量，保留失败与选择协议。\\
正确性与复现 & 手算和独立参考、时点扰动、账本核对，源码/数据/参数指纹、测试及CI。\\
扩展项 & 历史中性化、风险权重与现金、事前流动性部分成交、增量合并幂等和修订审计。\\
提交材料 & 可运行代码/依赖/README/样本，10页PDF，本地完整数据获取指南，Beamer及PPTX。\\\bottomrule
\end{tabularx}
\subsection*{复现入口}
安装固定依赖与\texttt{.[dev,research]}，运行\texttt{python -m pytest -q}和\texttt{python -m cfquant.cli run --config configs/demo.yaml}。真实实验先按EXPANDED\_DATA.md下载固定1,000股行情，再运行补充数据脚本及\texttt{python -m cfquant.cli research}，最后发布证据并生成报告。无Token仍能演示合成样本和已发布真实研究摘要，不能重拉厂商数据。完整步骤见REPRODUCE\_V2.md。

\noindent 行情SHA-256：\par{\ttfamily\scriptsize @HASH@}\par
原始60股、扩展旧动量与升级研究分别保留身份，不混用区间。本地运行归档与公开Git提交提供版本历史；公开仓库只含可分享合成数据和派生汇总。
\subsection*{限制与后续方向}
固定历史池、厂商修订、行业重构、ST代理、长期缺失计零、连续复权单位以及固定费用均限制外推。风险目标不是硬保证，MLP预算不足以代表神经网络上限。最终区间曾被原始动量研究观察，全部结果为描述性。后续在新的未观察区间采用滚动训练、动态历史股票池、真实分红现金与整手成交，并以严格验证检验更多Qlib因子，不能反复挑选当前最终期冠军。
\subsection*{主要依据与工具来源}
\begin{enumerate}\scriptsize
\item 课程原件CF2026\_Project1.pdf（19页）：评分结构、交付、会计与因子诊断口径。
\item Microsoft Qlib：\url{https://github.com/microsoft/qlib}，Alpha158/LightGBM公开基准配置与量价表达。
\item Liu et al., Cognitive Alpha Mining via LLM-Driven Code-Based Evolution, v4：\url{https://arxiv.org/abs/2511.18850v4}。
\item Tushare Pro官方文档：\url{https://tushare.pro/document/2?doc_id=32}（估值）；doc\_id=79（财务，单次100行）；335（历史行业）；228（CPI）；27/28/183/26（日线/复权/限制/日历）。
\item scikit-learn：\url{https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html}；LightGBM：\url{https://github.com/microsoft/LightGBM}；Backtrader：\url{https://github.com/mementum/backtrader}。
\end{enumerate}
\end{document}
'''
report=report.replace(r'}\[8pt]',r'}\\[8pt]')
report=report.replace('最终测试数量与日志保存在验收清单，不能用通过数代替对假设的审查。','另以大面板逐项除法复现并修复本机可选NumExpr加速的错误常数结果，统一禁用该路径并重新生成全部研究；独立账本不能替代上游数值验证。最终测试与复算日志见验收清单。')
report=report.replace(r'\noindent\includegraphics',r'\par\noindent\includegraphics')
report=report.replace(']{figures/legacy.pdf}',']{figures/legacy.pdf}\\par')
report=report.replace(']{figures/nav.pdf}',']{figures/nav.pdf}\\par')
report=report.replace(r'\noindent\small 沪深300',r'\noindent 沪深300')
report=report.replace(r'\texttt{python -m cfquant.cli run --config configs/demo.yaml}',r'\path{python -m cfquant.cli run --config configs/demo.yaml}')
report=report.replace(r'和\path{python -m cfquant.cli run --config configs/demo.yaml}',r'并运行演示命令：\par\noindent\texttt{python -m cfquant.cli run --config configs/demo.yaml}\par')
report=report.replace(r'\setlength{\parskip}{4pt}',r'\setlength{\parskip}{4pt}\setlength{\emergencystretch}{2em}')
for k,v in sub.items():report=report.replace('@'+k+'@',v)
(OUT/'final_report.tex').write_text(report,encoding='utf-8')

def compile_tex(name,output):
    build=ROOT/'tmp/final-latex'/name;build.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();rt=Path('D:/Downloads/MM-LaTeX/.miktex-runtime')
    if rt.exists():env.update(MIKTEX_USERCONFIG=str(rt/'config'),MIKTEX_USERDATA=str(rt/'data'),MIKTEX_USERINSTALL='D:/MiTex')
    exe=shutil.which('xelatex') or 'D:/MiTex/miktex/bin/x64/xelatex.exe'
    for _ in range(2):
        proc=subprocess.run([exe,'-interaction=nonstopmode','-halt-on-error',f'-output-directory={build}',name+'.tex'],cwd=OUT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (build/'compile.txt').write_bytes(proc.stdout)
        if proc.returncode:raise RuntimeError(proc.stdout.decode('utf-8',errors='replace')[-5000:])
    shutil.copy2(build/(name+'.pdf'),OUT/output)
compile_tex('final_report','CF2026_Final_Report.pdf')
print('Final report compiled',flush=True)
