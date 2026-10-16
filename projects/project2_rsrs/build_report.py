"""Build a self-contained Chinese LaTeX report and offline result browser."""
from pathlib import Path
import argparse
import json
import html
import pandas as pd

LABELS={'buy_hold':'持有指数','ma20':'20日均线对照','slope':'斜率','standard':'标准分','corrected':'修正标准分',
        'right':'右偏标准分','standard_ma':'标准分＋均线','corrected_ma':'修正分＋均线','right_ma':'右偏分＋均线'}


def table(rows, columns, headers):
    lines=[r'\begin{center}\small\begin{tabular}{l'+ 'r'*(len(columns)-1)+r'}\toprule',
           ' & '.join(headers)+r'\\\midrule']
    for _,row in rows.iterrows():
        cells=[]
        for col in columns:
            x=row[col]
            if col=='strategy':cells.append(LABELS[x])
            elif col=='symbol':cells.append({'sh000300':'沪深300','sh000016':'上证50','sh000905':'中证500'}[x])
            elif col in ['total_return','cagr','max_drawdown','exposure']:cells.append(f'{x*100:.2f}'+r'\%')
            elif col in ['buys','sells','year']:cells.append(str(int(x)))
            else:cells.append(f'{x:.3f}')
        lines.append(' & '.join(cells)+r'\\')
    return '\n'.join(lines+[r'\bottomrule\end{tabular}\end{center}'])


def build(root: Path, team: Path):
    results=root/'results'; out=root/'report';out.mkdir(exist_ok=True)
    t=pd.read_csv(results/'performance.csv');cost=t[t.cost_bps==10]
    hs=cost[cost.symbol=='sh000300'];post=hs[hs.period=='post_publication'];historical=hs[hs.period=='historical_overlap']
    sup=json.loads((results/'supplementary.json').read_text()); boot=json.loads((results/'bootstrap.json').read_text())
    sens=pd.read_csv(results/'sensitivity.csv'); diag=pd.read_csv(results/'predictive_diagnostics.csv')
    members=json.loads(team.read_text(encoding='utf-8'))
    team_line=r'\quad '.join(f'{m["student_id"]} {m["name"]}' for m in members)
    overview=table(post,['strategy','total_return','cagr','sharpe','max_drawdown','buys'],['策略','累计收益','年化收益','Sharpe','最大回撤','买入次数'])
    hist=table(historical[historical.strategy.isin(['buy_hold','standard','right','standard_ma','right_ma'])],
      ['strategy','total_return','cagr','sharpe','max_drawdown'],['策略','累计收益','年化收益','Sharpe','最大回撤'])
    cross=table(cost[(cost.period=='post_publication')&(cost.strategy=='right_ma')],
      ['symbol','total_return','cagr','sharpe','max_drawdown'],['指数','累计收益','年化收益','Sharpe','最大回撤'])
    recent=table(hs[(hs.period=='recent')&hs.strategy.isin(['buy_hold','standard_ma','right_ma'])],
      ['strategy','total_return','cagr','sharpe','max_drawdown'],['策略','累计收益','年化收益','Sharpe','最大回撤'])
    tex=r'''\documentclass[UTF8,a4paper,11pt]{ctexart}
\usepackage[margin=21mm,top=22mm,bottom=22mm]{geometry}
\usepackage{amsmath,amssymb,booktabs,tabularx,graphicx,xcolor,fancyhdr,enumitem}
\usepackage[colorlinks=true,linkcolor=teal,urlcolor=teal]{hyperref}
\definecolor{ink}{HTML}{16334B}\definecolor{accent}{HTML}{148779}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small 计算金融 · 研报复现}\fancyhead[R]{\small RSRS：历史效果与发表后检验}
\fancyfoot[L]{\scriptsize 研究稿 v0.1 · 课程 Project 2 要求待核实}\fancyfoot[R]{\thepage}
\setlength{\headheight}{15pt}\setlength{\parskip}{4pt}\setlist{nosep,leftmargin=1.8em}
\ctexset{section={format=\Large\bfseries\color{ink}},subsection={format=\normalsize\bfseries\color{accent}}}
\newcommand{\lead}[1]{\noindent\textbf{\color{accent}#1}\par}
\graphicspath{{../results/figures/}}
\begin{document}
\begin{center}
{\small\color{accent}COMPUTATIONAL FINANCE · REPRODUCTION STUDY}\\[10pt]
{\Huge\bfseries\color{ink}RSRS 择时策略复现}\\[8pt]
{\Large 原报告历史重叠样本与发表后检验}\\[12pt]
{\small @@TEAM@@}\\[7pt]
2026年10月16日\quad 数据截止：2026年10月15日
\end{center}
\section*{研究结论}
\lead{原报告时期的良好效果没有在后续市场中稳定延续。}
本研究复现光大证券2017年5月1日研报《基于阻力支撑相对强度（RSRS）的市场择时》。从免费公开的 QuantsPlaybook 仓库取得研报，按原文实现斜率、标准分、修正标准分和右偏标准分，并加入原报告的均线入场过滤。研究使用公开指数行情，按收盘形成信号、下一交易日开盘成交，逐日核算资金与交易成本。

主要检查对象在跑回测前固定为\textbf{右偏标准分＋均线过滤}。在沪深300的2007年8月至2017年3月历史重叠区间，其扣费累计收益428.76\%，最大回撤17.95\%。2017年5月至2026年10月，其累计收益降为14.55\%，同期持有指数为21.41\%，最大回撤分别为35.01\%和45.60\%。它降低了回撤，但没有在该时期跑赢基准。标准分＋均线版本小幅领先，但不能据此认定存在稳定的超额收益。

\subsection*{本稿完成范围}
本稿交付可重跑代码、三指数数据快照、162组主回测、全部账本、27组参数邻域实验、实现口径敏感性、ETF代理检验、公式与时序测试，以及本报告。所有结果均来自本次运行，未删除负收益版本。

\subsection*{课程要求核实状态}
2026年10月16日，通过学校已登录账号同时核查 Blackboard 的 Project 页面、成绩栏、Lecture及课程通知。Project 栏仅出现 CF2026\_Project1，成绩栏没有 Project 2，Lecture00没有第二项目的具体要求。因此\textbf{本稿是已完成的研报复现研究稿，尚不能声称满足 Project 2 的全部提交要求}。截止日期、页数、展示形式、评分细则仍待课程文件确认。本次未向 Blackboard 提交任何材料。

\subsection*{贡献与使用边界}
研报思想与原始结果归原作者。代码由 Codex 辅助编写与测试，组员应理解并核对研究内容后按课程规则披露工具使用。本报告不虚构个人分工，也不将回测结果作为实际投资建议。
\newpage
\section*{1\quad 原始研报与复现设计}
研报作者为光大证券分析师刘均伟、联系人周萧潇，发布日期2017年5月1日。公开副本包含第三方手写批注，本研究仅依据排版正文和表格，不把批注当成原作者结论。原文使用 Wind 数据，在2005年至2017年的宽基指数上测试做多择时策略。

\begin{center}\small\begin{tabularx}{\linewidth}{lXX}\toprule
项目 & 原文内容 & 本次处理\\\midrule
数据 & Wind指数日频行情 & 腾讯公开指数OHLC，Tushare重叠核验\\
标准分 & $N=18,M=600,S=0.7$（第10页） & 按原参数固定\\
右偏分 & $N=16,M=300,S=0.7$（第16页） & 按原参数固定\\
价格过滤 & 前一日MA20与前三日MA20比较（第18页） & 入场过滤，不强制提前平仓\\
收益样本 & 2005年3月至2017年3月/4月 & 历史重叠区间另设，并检验发表后\\
成本 & 主表无成本，附成本情景 & 同报0与单边10bp\\
执行细节 & 完整数据、成交时点及预热处理未披露 & 明确次日开盘、完整历史预热\\\bottomrule
\end{tabularx}\end{center}

\subsection*{为何不逐项追求原表数值相等}
标准分需要600个历史斜率，公开沪深300数据从2005年1月起，无法在2005年3月自然形成完整预热。原文未说明早期如何补足历史、是否使用扩展窗口。把缺失值补零、使用全样本均值或回填未来数据，都可能改变早期信号。本研究保留缺失，\textbf{不声称逐数复刻原报告的2005年起始净值}。

原文也存在表述差异：摘要和第19页对价格过滤版本给出不同累计收益值，测试终点在不同段落写作3月或4月；第17页“净值15.74”与“累计收益1573.60\%”并不完全对应。因此本研究以公式和交易逻辑为复现目标，披露差异，不调整实现去凑原文收益。

\subsection*{固定实验协议}
历史重叠区间为2007-08-01至2017-03-31，确保三指数均有600个有效历史斜率。发表后区间为2017-05-02至2026-10-15，近年区间为2020-01-02至2026-10-15。日期与参数写入配置文件后才运行。所有数据都是当下回看的历史数据，\textbf{“发表后”不等于本团队从未查看过的盲测样本}。近期窗口只是分段诊断，不是新的独立验证集。

我们先检查了公开复现代码。其Notebook依赖聚宽、TA-Lib及本地文件，且部分过滤实现把入场条件作用于全部持仓。为保持原文含义，本项目复用成熟数值库和公开数据接口，独立实现小型策略账本，并以线性代数直接解核对OLS。
\newpage
\section*{2\quad 数据与质量核验}
\begin{center}\begin{tabular}{llrr}\toprule
指数 & 代码 & 起始日 & 记录数\\\midrule
沪深300 & sh000300 & 2005-01-04 & 5,288\\
上证50 & sh000016 & 2004-01-02 & 5,531\\
中证500 & sh000905 & 2005-01-04 & 5,288\\\bottomrule
\end{tabular}\end{center}
三指数共16,107条日频记录，统一截止2026-10-15。包含 date、open、close、high、low、volume。通过腾讯公开指数接口按两年分段获取，接口与字段说明参考AKShare官方文档。保存每段原始响应、整理后的CSV及SHA-256，不需任何账号Token即可重跑已保存快照。

质量检查要求日期唯一、按时间递增、所有必要价格非空且为正，且最低价不高于开收盘价、最高价不低于开收盘价。所有检查通过。没有使用补零或未来填充。指数供应商的起始日期不代表指数正式发布日，部分早期记录是回溯历史序列。

\subsection*{独立数据源交叉核对}
将腾讯沪深300开盘、收盘与原平台保存的Tushare快照按日期连接，2019-10-08至2026-09-18共匹配1,690个交易日。开盘价最大绝对差0.0052点，收盘价最大绝对差0.0051点，与腾讯保留两位小数的精度相符。此核对支持\textbf{重叠区间开收盘价}一致性，不等于验证了全部历史高低价。

\subsection*{与原平台一千股数据的关系}
本题研究指数择时，无需将固定一千只股票拼接成“沪深300”。这避免了固定股票池的幸存者偏差和指数复制误差。原平台的Tushare行情仅用于独立核对和ETF补充实验；未将未来收益标签、原模型预测或新因子搜索结果带入本次信号。

\subsection*{价格指数与可交易资产}
主实验使用\textbf{价格指数}作为学术研究代理。指数本身不能直接买卖，指数收益也不含红利再投资。成本使用统一的研究假设，未模拟指数成分股逐只成交、融资或冲击。

补充实验用510330沪深300ETF复权开收盘价承接指数信号，样本止于已有快照的2026-09-18。ETF序列按复权因子统一调整，近似包含分红调整效果，但不逐笔处理基金现金分配、整手约束及跟踪误差来源。因此主实验和ETF结果分别陈列，不混用为同一个基准。
\newpage
\section*{3\quad 指标公式、状态机与成交}
在$t$日收盘，取最近$N$个交易日高低价，带截距拟合
\[
 H_i=\alpha_t+\beta_t L_i+\varepsilon_i,\qquad i=t-N+1,\ldots,t.
\]
\[
 \beta_t=\frac{\sum_i(L_i-\bar L)(H_i-\bar H)}{\sum_i(L_i-\bar L)^2},\qquad
 R_t^2=\frac{[\sum_i(L_i-\bar L)(H_i-\bar H)]^2}{\sum_i(L_i-\bar L)^2\sum_i(H_i-\bar H)^2}.
\]
定义过去$M$个\textbf{不含当日}斜率的均值$\mu_{t-1,M}$和总体标准差$\sigma_{t-1,M}$，则
\[
 z_t=\frac{\beta_t-\mu_{t-1,M}}{\sigma_{t-1,M}},\qquad
 z_t^{\mathrm{corr}}=z_tR_t^2,\qquad
 z_t^{\mathrm{right}}=z_tR_t^2\beta_t.
\]
退化回归或零标准差标记为不可用，完整预热前不产生信号。主实验采用过去$M$项口径，并另测试含当日、样本标准差等解释。

\subsection*{做多/现金状态机}
斜率策略使用上阈值1.0、下阈值0.8。其余版本使用$S=0.7$及$-S$。空仓且分数严格大于上阈值时买入，持仓且严格小于下阈值时平仓，位于阈值之间时维持原状态。价格过滤仅增加入场条件
\[
 \mathrm{MA}_{20,t}>\mathrm{MA}_{20,t-2}.
\]
这对应$t+1$开盘下单时的“前一日”与“前三日”均线比较。原文另有“与3日前比较”的措辞，本研究将$t$与$t-3$的比较作为敏感性实验。\textbf{均线变差本身不强制平仓}，避免擅自改变研报退出逻辑。

\subsection*{现金账本与时序}
信号只使用截至$t$收盘的信息，首次可执行时点为$t+1$开盘；持仓按当日收盘估值。以现金$C$、开盘价$O$、单边成本$c$为例，买入单位数为$q=C/[O(1+c)]$，卖出净回款为$qO(1-c)$。净值为$V_t=C_t+q_tP_t^{\mathrm{close}}$。每段初始净值1，继承预热后的信号状态，以新资金在首日开盘执行。不强制期末卖出，期末持仓按收盘估值。

现金利率与Sharpe无风险利率均设为0，不允许杠杆、做空或负现金。年化收益$V_T^{252/T}-1$，Sharpe为$\sqrt{252}\bar r/s_r$，最大回撤包含初始净值1。收益差使用同日期同成本基准，累计收益差以百分点计。
\newpage
\section*{4\quad 原报告历史重叠区间}
\lead{历史重叠样本支持价格过滤的效果，但不能据此验证原文全部数字。}
以下均为2007-08-01至2017-03-31、单边10bp成本、次日开盘成交。原文主表使用不同起始时点与无成本口径，因此这里只比较机制与方向，不计算虚假的“复现误差”。
@@HIST@@
\begin{center}\includegraphics[width=.98\linewidth]{nav_historical_overlap.png}\end{center}
右偏原始版本最大回撤52.61\%，加入均线入场过滤后降至17.95\%，与原报告“减少熊市过早抄底”的解释一致。标准分＋均线的Sharpe更高，也与不同风险指标可能给出不同排序相符。

这个窗口从2007年较高点附近开始，持有指数累计亏损22.92\%。起点由完整预热需求确定，但它使结果与2005年起始的原报告不可直接比较。图中的高超额收益不能外推为所有起止时间都成立。
\newpage
\section*{5\quad 发表后检验}
\lead{固定右偏＋均线版本收益落后，标准分＋均线只小幅领先。}
2017-05-02至2026-10-15，单边10bp，全部版本按相同资金、时点和估值方法比较。
@@OVERVIEW@@
\begin{center}\includegraphics[width=.98\linewidth]{nav_post_publication.png}\end{center}
主要检查对象的累计收益14.55\%，比持有指数低6.86个百分点，年化收益1.50\%。标准分＋均线累计23.95\%，比基准高2.54个百分点，最大回撤26.52\%。它可作为后续候选，但本次已同时观察多个版本，不能事后将表现最好的版本称为事前选定的胜者。
\newpage
\section*{6\quad 跨指数与近年检验}
右偏＋均线版本在发表后同样不具有跨指数稳定优势。下表是固定相同参数的结果。
@@CROSS@@
\begin{center}\includegraphics[width=.98\linewidth]{cross_index.png}\end{center}
在上证50上它超过持有指数，但中证500累计收益为负。完整主结果表还保留其他版本：标准分＋均线在中证500表现相对较好，说明过滤效果和参数选择具有市场依赖性。

\subsection*{2020年以来的沪深300}
@@RECENT@@
近年窗口中，右偏＋均线仍未跑赢持有指数，且存在35.01\%回撤。本次没有验证实际CPI购买力目标，不能由正收益推断跑赢通胀。

\subsection*{ETF代理的实用性检查}
2020-01-02至2026-09-18，采用510330复权价格并承接相同指数信号，右偏＋均线累计收益6.07\%、年化0.92\%、Sharpe 0.135、最大回撤33.61\%；同期ETF持有累计25.60\%、年化3.59\%、最大回撤42.08\%。ETF实验保留了降低部分风险、同时明显放弃上涨收益的特征。
\newpage
\section*{7\quad 稳健性、相关性与局限}
\subsection*{参数与实现口径}
对右偏＋均线预先列出$N\in\{14,16,18\}$、$M\in\{250,300,600\}$、$S\in\{0.5,0.7,0.9\}$共27组邻域参数，仅用于揭示敏感性，不据此修改主参数。发表后年化收益范围为$-1.06\%$至$4.07\%$，中位数1.85\%。
\begin{center}\includegraphics[width=.96\linewidth]{sensitivity.png}\end{center}
把均线比较改成$t$对$t-3$，主策略发表后累计收益从14.55\%变为17.84\%，仍低于基准。含当日标准化及样本标准差两种变体在本次阈值规则下没有改变最终交易路径；这不表示两种指标数值相同。

\subsection*{聚合相关性不等于每日预测准确率}
原文的约0.75相关性来自指标区间对应的未来预期收益图，不能理解为每日上涨预测准确率75\%。本研究同时报告逐日相关性与分箱均值相关性，分箱固定宽度0.2，仅保留至少10个观测的箱。历史重叠区间逐日Pearson约0.115、分箱均值相关性约0.540；发表后分别约$-0.002$和$-0.324$。未来10日收益只进入诊断，不用于策略输入，且标签终点不得超过所评估窗口。

\subsection*{统计不确定性}
对主策略相对基准的逐日收益差进行20日循环分块、2,000次重抽样，固定随机种子。沪深300年化算术日均收益差估计$-1.55\%$，95\%描述性区间$[-9.65\%,6.54\%]$，跨过0。这是依赖块长与平稳性近似的探索性区间，并非CAGR区间，也未校正多策略选择。重叠10日标签亦会带来相关性，故不报告误导性的独立样本显著性。
\newpage
\section*{8\quad 复现入口与结论}
\lead{复现支持了“历史上有效”的局部事实，未支持“长期稳定有效”的强结论。}
本研究完成公式、参数、状态机和成交账本的可审计实现。历史重叠窗口的均线过滤改善明显；在发表后的长窗口中，固定右偏＋均线版本未跑赢沪深300，且跨指数、ETF检验不稳定。课程研究可以保留这些负结果，用以说明回测过拟合、模型衰减和研究口径的重要性。\textbf{本次不替换原平台默认策略。}

\subsection*{如何重新运行}
解压后用Python 3.11或更高版本，在项目目录运行：
\begin{quote}\ttfamily\small
python -m pip install -r code/requirements.txt\\
python -m pytest code/test\_rsrs.py -q\\
python code/run\_study.py --data data --output rerun\_results\\
\hspace*{1em}--supplemental data/supplemental
\end{quote}
命令最后两行应合并为一行。已有快照可全程离线分析。只在主动刷新行情时运行fetch\_data.py，需联网。配置文件限定数据截止日和全部样本窗口。验证SHA-256后才能使用主数据。

\subsection*{核验与交付物}
七项测试覆盖OLS独立解、严格预热、未来数据扰动不影响过去、过滤只作用入场、次日开盘与隔夜跳空、成本和首日回撤、日期错位及退化数据。162份主回测账本逐一由成交重建现金和持仓，校验费用及复利净值一致性。另保留ETF账本核验。报告数字由结果CSV支撑，输出清单记录文件哈希。

离线浏览页可筛选指数、时期、策略和成本，查看完整指标与对应账本。data保存公开行情及补充快照，results保存全部结果，code保存源码和测试，report保存本稿及可编辑LaTeX。原研报与课程导论仅在本机sources中用于阅读，不将第三方Notebook作为本项目原创代码发布。

\subsection*{仍待确认的课程事项}
Project 2正式文件、截止时间、是否必须使用Project 1平台界面、是否要求PPT/课堂展示、报告语言与提交文件命名尚未取得。当前版本不声称已经通过这些要求的逐项验收，也没有上传BB。取得要求后，应在本稿基础上对齐内容与格式。

\subsection*{参考资料}
\begin{enumerate}[label={[\arabic*]},leftmargin=2em]\small
\item 刘均伟、周萧潇. 基于阻力支撑相对强度（RSRS）的市场择时：技术择时系列报告之一. 光大证券，2017-05-01. 公式与策略见第8--10、13--20页。
\item \href{https://github.com/hugo2046/QuantsPlaybook}{QuantsPlaybook公开研报与复现资料}，固定提交3f83a77。只将其中研报作为方法来源，Notebook作为实现差异参考。
\item \href{https://akshare.akfamily.xyz/data/index/index.html}{AKShare官方指数接口文档}，腾讯日频历史行情字段及接口约定。访问日期2026-10-16。
\item \href{https://gu.qq.com/sh000300/zs}{腾讯财经沪深300指数行情}。本研究快照截至2026-10-15。
\end{enumerate}
\end{document}
'''
    for key,value in {'TEAM':team_line,'OVERVIEW':overview,'HIST':hist,'CROSS':cross,'RECENT':recent}.items():tex=tex.replace('@@'+key+'@@',value)
    (out/'RSRS_Report.tex').write_text(tex,encoding='utf-8')
    # Plain HTML can be opened offline without a server or third-party scripts.
    records=t.round(8).to_dict(orient='records')
    browser='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RSRS 研报复现</title><style>body{font:16px/1.65 system-ui,"Microsoft YaHei";max-width:1180px;margin:40px auto;padding:0 22px;color:#18334a;background:#f5f8fa}h1{font-size:34px}select{font:inherit;padding:8px;margin:4px 16px 4px 0}table{border-collapse:collapse;background:white;width:100%;font-size:14px}th,td{padding:9px;border-bottom:1px solid #dce5eb;text-align:right}th:first-child,td:first-child{text-align:left}img{max-width:100%;background:white}a{color:#087c70}.note{border-left:4px solid #148779;padding:12px 20px;background:#e8f3ef}section{margin-top:36px}footer{margin-top:50px;color:#607386}</style>
<h1>RSRS 研报复现与发表后检验</h1><p>光大证券 2017-05-01 研报 · 数据截止 2026-10-15 · 研究稿 v0.1</p>
<p class="note">课程状态：BB 暂未见 Project 2 要求。本研究已经运行，但尚未完成课程要求逐项验收，也未提交 BB。</p>
<p>主检查对象“右偏＋均线”在发表后沪深300累计收益 14.55%，持有指数 21.41%。本页展示全部结果，支持查看亏损版本。成本含单边 10 bp 和零成本两档，现金利率及 Sharpe 无风险利率均为 0。</p>
<p><a href="report/RSRS_Report.pdf">阅读中文报告</a>　<a href="results/performance.csv">下载完整指标 CSV</a>　<a href="使用说明.txt">复现说明</a></p>
<section><h2>结果筛选</h2><label>指数 <select id="symbol"><option value="sh000300">沪深300</option><option value="sh000016">上证50</option><option value="sh000905">中证500</option></select></label>
<label>时期 <select id="period"><option value="post_publication">发表后 2017–2026</option><option value="historical_overlap">历史重叠 2007–2017</option><option value="recent">近年 2020–2026</option></select></label>
<label>单边成本 <select id="cost"><option value="10">10 bp</option><option value="0">0 bp</option></select></label>
<label>策略 <select id="strategy"><option value="all">全部版本</option></select></label><div style="overflow-x:auto"><table><thead><tr><th>策略</th><th>累计收益</th><th>年化收益</th><th>Sharpe</th><th>最大回撤</th><th>超额百分点</th><th>持仓比例</th><th>账本</th></tr></thead><tbody id="rows"></tbody></table></div></section>
<section><h2>沪深300净值与回撤</h2><p>此图固定为沪深300、单边10 bp，时期跟随上方选择。完整其他指数结果见表格及账本。</p><img id="chart" src="results/figures/nav_post_publication.png"></section>
<section><h2>参数敏感性</h2><p>27组预定邻域全部保留，没有使用最佳格子替换主参数。</p><img src="results/figures/sensitivity.png"></section>
<section><h2>复现口径</h2><p>日终数据形成信号，次日开盘成交。均线条件只控制入场。完整600/300个历史斜率预热，缺失不补零。主实验是价格指数研究代理，ETF补充见报告。发表后样本也是回顾性历史研究，并非团队从未看过的盲测。</p></section>
<footer>原文：刘均伟、周萧潇，光大证券，2017。公开研报平台 <a href="https://github.com/hugo2046/QuantsPlaybook">QuantsPlaybook</a>。代码、数据与结果均在本目录。</footer>
<script>const data=@@DATA@@,labels=@@LABELS@@; const el=id=>document.getElementById(id); for(const [k,v] of Object.entries(labels)){let o=document.createElement('option');o.value=k;o.textContent=v;el('strategy').appendChild(o)}
function render(){let d=data.filter(r=>r.symbol===el('symbol').value&&r.period===el('period').value&&r.cost_bps==el('cost').value&&(el('strategy').value==='all'||r.strategy===el('strategy').value));el('rows').replaceChildren();for(let r of d){let tr=document.createElement('tr');let vals=[labels[r.strategy],(r.total_return*100).toFixed(2)+'%',(r.cagr*100).toFixed(2)+'%',r.sharpe.toFixed(3),(r.max_drawdown*100).toFixed(2)+'%',r.total_return_difference_pp.toFixed(2),(r.exposure*100).toFixed(1)+'%'];for(let v of vals){let td=document.createElement('td');td.textContent=v;tr.appendChild(td)}let td=document.createElement('td'),a=document.createElement('a');a.href=`results/ledgers/${r.symbol}_${r.period}_${r.strategy}_${r.cost_bps}bps.csv`;a.textContent='CSV';td.appendChild(a);tr.appendChild(td);el('rows').appendChild(tr)}el('chart').src=`results/figures/nav_${el('period').value}.png`};for(let id of ['symbol','period','cost','strategy'])el(id).addEventListener('change',render);render();</script></html>'''
    browser=browser.replace('@@DATA@@',json.dumps(records,ensure_ascii=False)).replace('@@LABELS@@',json.dumps(LABELS,ensure_ascii=False))
    (root/'打开研究结果.html').write_text(browser,encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--team',type=Path,required=True)
    a=p.parse_args();build(a.root,a.team)
