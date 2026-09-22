"""Updated ten-page report and 20-slide Beamer narrative; preserves V2 artifacts."""
from pathlib import Path
import json, os, re, shutil, subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parents[1];E=ROOT/'evidence/research_v3';OUT=ROOT/'reports/benchmark_v3'
FIG=OUT/'figures';FIG.mkdir(parents=True,exist_ok=True)
load=lambda n:json.loads((E/n).read_text(encoding='utf-8'))
m=load('metrics.json');decision=load('decision.json');val=pd.read_csv(E/'validation.csv');test=pd.read_csv(E/'test.csv')
stress=pd.read_csv(E/'stress.csv');curves=pd.read_csv(E/'curves.csv',parse_dates=['date'])
short=['Theme S','Theme M','Ridge S','Ridge M','LGBM S','LGBM M','Blend S','Blend M']
labels=['主题/稳定','主题/预算','Ridge/稳定','Ridge/预算','LGBM/稳定','LGBM/预算','融合/稳定','融合/预算']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})

def save(fig,name):
    fig.tight_layout();fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');fig.savefig(FIG/(name+'.png'),dpi=140,bbox_inches='tight');plt.close(fig)

fig,ax=plt.subplots(figsize=(10,3.8))
mapping={'V3最新候选 · 扣费':'V3 rolling LightGBM (net)','沪深300 · 价格指数':'CSI300 price index',
         '沪深300 · 全收益指数':'CSI300 total-return index','沪深300ETF · 复权未扣费':'CSI300 ETF adjusted (uncosted)','V2原预选 · 扣费':'V2 previous selection (net)'}
for (name,g),color in zip(curves.groupby('series',sort=False),['#168579','#4265a6','#955db1','#b8863f','#89949f']):
    ax.plot(g.date,g.nav,label=mapping[name],color=color,lw=1.5)
ax.set_ylabel('NAV (initial = 1)');ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2);save(fig,'nav')
for table,name,title in [(val,'validation','2023–2024 validation'),(test,'test_models','2025–2026 historical review')]:
    fig,ax=plt.subplots(figsize=(10,3.4));x=np.arange(8)
    ax.bar(x-.18,table.excess_return_pp,.36,label='Net return minus price index',color='#168579')
    ax.bar(x+.18,table.max_drawdown,.36,label='Maximum drawdown',color='#4265a6')
    ax.set_xticks(x,short,fontsize=8);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.axhline(0,color='gray',lw=.7)
    ax.set_title(title);ax.legend(fontsize=8);save(fig,name)
fig,ax=plt.subplots(figsize=(9,3.3));x=np.arange(3)
ax.bar(x-.18,stress.iloc[:3].total_return,.36,color='#168579',label='Strategy net return')
ax.bar(x+.18,stress.iloc[:3].max_drawdown,.36,color='#4265a6',label='Maximum drawdown')
ax.axhline(m['total_return_benchmark'],color='#955db1',ls='--',label='CSI300 total return')
ax.set_xticks(x,['Selected V3','Double costs','One-day signal delay']);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(fontsize=8);save(fig,'controls')
for name in ['legacy','factor_ic','drawdown']:
    for suffix in ['pdf','png']:shutil.copy2(ROOT/'reports/final/figures'/f'{name}.{suffix}',FIG/f'{name}.{suffix}')

original=(ROOT/'reports/final/final_report.tex').read_text(encoding='utf-8')
prefix=original.split(r'\begin{document}')[0].replace('研究代码 17026b81','研究版本 V3 / 2026-09-22')
parts=re.split(r'(?=\\page\{)',original)
esc=lambda s:str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
pct=lambda x:f'{x*100:.2f}'+r'\%'
page1=r'''\begin{document}
\begin{center}{\small\color{accent}COMPUTATIONAL FINANCE / PROJECT 1}\\[8pt]
{\Huge\bfseries\color{ink}青序量化研究平台}\\[8pt]
{\Large 指数目标、滚动预测与策略版本管理}\\[8pt]
2026年9月22日\quad V3研究更新\quad 最终提交报告
\end{center}
\section*{1\quad 目标与主要结果}
\lead{从购买力目标升级为扣费后超过沪深300，同时保留选择失败与全部候选。}
平台覆盖数据质量、因子、横截面诊断、含费用回测、独立账本和可复现交付。用户将策略目标改为战胜沪深300。本轮固定1,000股历史池及2025-01-02至2026-09-18区间，预先限定8个候选，比较经济主题、年度滚动Ridge、年度滚动LightGBM及固定融合。

当前默认策略为\textbf{年度滚动LightGBM＋波动预算}，用24个标准化/中性化输入预测股票相对排序，50只目标持仓，每20个交易日调仓，允许现金、不用杠杆。
\begin{center}\begin{tabular}{lr}\toprule
2025--2026固定417日历史区间 & 结果\\\midrule
累计净收益 / 年化净收益 & 25.13\% / 14.51\%\\
最大回撤 / 年化Sharpe & 9.88\% / 1.255\\
沪深300价格指数 / 全收益指数 & 14.55\% / 19.94\%\\
超过价格指数 / 全收益指数 & 10.58 / 5.19个百分点\\
交易费用 / 平均现金 & 24,506.76元 / 17.87\%\\\bottomrule
\end{tabular}\end{center}
\textbf{本段历史同时超过价格指数和含红利再投资的全收益指数。}双倍交易费用仍收益22.48\%，信号延后一日收益27.31\%。2023起连续运行累计44.09\%、最大回撤19.70\%，但2024年度落后价格指数7.28个百分点，不能声称每年都战胜市场。

\subsection*{选择边界必须与结果一起阅读}
V3验证期预选的价值质量组合在2025--2026仅收益4.30\%，没有超过指数。当前LightGBM是在查看本轮历史结果后，从通过验证净收益、超额收益和回撤门槛的候选中选出。V2已经看过同一最终区间，本次\textbf{不是独立盲测成功}。保留失败预选和8个候选，下一步应冻结规则做新数据前向验证。
\subsection*{交付与操作}
“回测实验”现在可切换V1单因子、V2各模型和V3候选，默认最新历史验收达标版本。账本正确与收益/风险验收分开；亏损或回撤超门槛显示未达标。报告、20页Beamer、可编辑PPTX和讲稿与平台使用同一派生证据。
'''
page2=parts[1].replace('app / ui\\_research','app / strategies / UI')
page2+=r'版本目录独立记录信号、组合规则、分数来源和状态；切换版本即清除旧结果。参数重跑沿用同一成交核心，每次另存运行目录。仅公开聚合日账本，逐股财务和模型缓存保留在本地。'+'\n'
page3=parts[2].replace('沪深300价格指数；全国CPI月环比','沪深300价格/全收益指数；全国CPI月环比')
page3=page3.replace('行情及研究数据合计约1.63GB','原行情及V2研究缓存合计约1.63GB')
page3+=r'新增基准由Tushare指数目录确认H00300.CSI，再取全收益日线；另取510330.SH基金日线和复权因子，各1,690日。价格指数、全收益指数、ETF代理分开命名，未猜测或替代指数代码。'+'\n'
page4=parts[3]
page5=r'''\page{5\quad 中性化、年度训练与8个候选}
\lead{因子处理按日进行；新模型只能使用当时已实现的标签。}
当日合资格因子按1\%/99\%截尾并用样本标准差标准化。对当日行业虚拟变量和对数市值回归取残差，再缩放为单位样本标准差。LightGBM同时输入12个标准化值和12个残差值，Ridge只输入12个残差值。
\[
z_{i,t}=\sum_g\alpha_{g,t}\mathbf1\{g_{i,t}=g\}+\beta_t\log MV_{i,t}+\epsilon_{i,t},\qquad n_{i,t}=\epsilon_{i,t}/s_t(\epsilon).
\]
每年第一次交易前一交易日形成新模型，训练此前3年。标签为次日开盘至20交易日后开盘收益的日横截面百分位减0.5，训练标签退出日必须严格早于模型形成日。整个下一年度使用该模型，到下一年才重训。
\begin{center}\small\begin{tabular}{lrr}\toprule
预测年度 & 训练观测数 & 最大标签退出日早于形成日\\\midrule
2023 & 654,563 & 已核验\\2024 & 645,816 & 已核验\\2025 & 636,902 & 已核验\\2026 & 634,716 & 已核验\\\bottomrule
\end{tabular}\end{center}
\begin{itemize}
\item 经济主题：价值（盈利/账面/股息）占50\%，ROE占25\%，低波动/低振幅占25\%。
\item Ridge：alpha=100，年度滚动重训，无最终期参数扫描。
\item LightGBM：250树，学习率0.03，15叶，最大深度5，最小叶500，L2=10，列采样0.8，种子20260921。
\item 固定融合：LightGBM百分位50\%、Ridge百分位25\%、经济主题百分位25\%。
\end{itemize}
四种信号各配稳定仓位及波动预算两种组合，共8个候选。稳定版本取消波动缩放和趋势折减；其他持仓、缓冲、上限及成交规则相同。2023--2024先记录验证结果及预选，然后再运行固定2025--2026区间。全部尝试保留在同一表。
\subsection*{与原建议的关系}
CNN需要有意义的局部邻接结构，简单排列的因子列没有天然邻接。V2已经保留小型MLP对照；本轮使用适合表格特征的树模型及线性基线，并未假定模型越复杂越好。年度训练减少长期冻结模型的陈旧性，但不能消除分布变化或过拟合。
'''
page6=parts[5].replace('0.90\\min\\{1,0.12','0.95\\min\\{1,0.20').replace('\\\\0.25,&','\\\\0.75,&')
page6=page6.replace('和20\\%回撤','和25\\%回撤').replace('13个主研究运行','21个V3研究及压力运行')
page6=page6.replace('每次检查417或484个交易日','分别检查417、484或901个交易日')
page7=r'''\page{7\quad 失败、预选与开发后的候选}
\lead{更改研究目标不等于抹去原始失败。}
V1的20日动量、5日调仓长区间净收益为$-83.57\%$，最大回撤89.83\%；零费用独立重跑仍为$-75.16\%$。费用加回同成交路径得到$-58.12\%$，并非零费用独立策略，因为再投资路径不同。V2预选多因子在固定最终区间收益6.12\%，实现CPI目标但落后指数。
\subsection*{V3选择过程}
协议先写定8个候选与验证规则：2023--2024净收益、累计超额为正，最大回撤不超过25\%；先比较最差年度超额，再比较累计超额。价值质量主题稳定仓位排名第一，验证收益24.27\%、回撤9.16\%；但随后收益4.30\%，落后价格指数10.25个百分点。\textbf{这个预选失败不能被后续冠军替换。}
\begin{center}\includegraphics[width=.98\textwidth]{figures/validation.pdf}\end{center}
按用户要求进一步寻找能超过指数的历史候选，筛掉验证回撤超25\%的版本，再查看2025--2026超额。LightGBM稳定仓位虽然最终收益27.39\%，但验证回撤25.73\%，未作为默认。采用LightGBM波动预算版本，验证收益15.05\%、回撤19.70\%，最终收益25.13\%、回撤9.88\%。

以上是\textbf{开发后历史验收}，并非只有训练样本上的拟合收益，但也不属于全新盲测。决策文件保存preselected和selected两个字段，明确选择先后及8次候选规模。
'''
rows='\n'.join(esc(label)+' & '+pct(r.total_return)+' & '+pct(r.excess_return_pp)+' & '+pct(r.max_drawdown)+r'\\' for label,(_,r) in zip(labels,test.iterrows()))
page8=r'''\page{8\quad 固定区间的全候选与基准}
\lead{同一417日区间，策略费用已扣；基准名称及计息规则明确。}
\begin{center}\small\begin{tabular}{lrrr}\toprule
候选（稳定/波动预算） & 净收益 & 超价格指数 & 最大回撤\\\midrule
'''+rows+r'''\bottomrule\end{tabular}\end{center}
\begin{center}\includegraphics[width=.98\textwidth]{figures/nav.pdf}\end{center}
价格指数从前一交易日收盘归一化，累计14.55\%；全收益指数计红利再投资，累计19.94\%。策略同区间净收益25.13\%，相对全收益指数财富增长为$(1+25.13\%)/(1+19.94\%)-1\approx4.33\%$，与相差5.19个百分点是不同口径。

首次开盘起算价格指数收益14.64\%，结论不变。510330.SH复权未扣费代理收益19.82\%，另保留同开盘建仓扣买费的可交易近似；ETF不能改名为全收益指数。策略beta约0.30，年化跟踪误差15.60\%，信息比率0.292；落后阶段及主动净值回撤约18.41\%仍存在。
'''
page9=r'''\page{9\quad 压力测试、诊断与长期区间}
\begin{center}\includegraphics[width=.98\textwidth]{figures/controls.pdf}\end{center}
双倍费用重跑净收益22.48\%、回撤10.19\%；信号延后一交易日重跑净收益27.31\%、回撤10.67\%。两者仍超过同区间全收益指数19.94\%。延迟改善不是可再选择的默认参数，只是固定压力检查；没有据此改成更高收益的延迟版本。
\subsection*{从2023年连续运行，避免年度重置资金}
累计收益44.09\%，同期全收益指数28.86\%，策略最大回撤19.70\%。2023、2024、2025、2026年截至9月的净收益分别为7.12\%、7.41\%、22.44\%、2.28\%。2024落后价格指数7.28个百分点；长期优势不能解释为每年都赢。该连续账本与2025从现金起步的复核账本不同，年度收益不能互换。
\subsection*{课程因子诊断仍独立保留}
逐日计算Pearson IC和平均并列秩Spearman Rank IC，常数或不足有效样本返回缺失；不把股票日混为一次相关。先依据形成日分数分五组，再匹配未来标签，保留形成/有效人数；缺失标签不重新排名。878个形成日的12因子诊断保留在V2证据目录，低波动/低振幅平均Rank IC约0.072/0.079，流动性约$-0.051$，未因结果为负翻转符号。

重叠20日标签均值不连乘为净值，不声称独立显著性。V2的bootstrap不能直接转用于V3；本轮没有提供消除全部模型选择偏差的置信区间。风险目标是约束研究选择和形成时权重，不能承诺未来不超过25\%回撤。
'''
page10=r'''\page{10\quad 课程验收、复现与提交}
\noindent\begin{tabularx}{\textwidth}{lX}\toprule
要求 & 完成位置与验收证据\\\midrule
数据/字段/质量/版本 & data、incremental、质量报告与SHA清单；原始与处理数据分离。\\
至少三因子及配置 & 原三注册因子＋12因子卡、长表接口和窗口参数。\\
回测、费用、指标 & 下一开盘、实际成交收费、现金/限制价/缺失估值；期初NAV参与回撤。\\
IC/Rank IC和分组 & 逐日横截面、有效人数、形成后标签匹配、负结果保留。\\
实验与完整性 & 8候选、失败预选、21个V3账本、独立环境4张表复算一致。\\
扩展 & 历史行业/市值中性化、风险预算、部分成交、增量幂等。\\
代码与演示 & 版本选择/参数重跑、固定依赖、合成样本、10页报告、20页Beamer。\\\bottomrule
\end{tabularx}
\subsection*{可运行入口}
本机双击launch.cmd。新环境按README安装固定依赖和项目，运行pytest及configs/demo.yaml。无凭据可看公开真实聚合证据和所有版本日账本；原始数据缺失时不能重跑真实模型，不将合成样本冒充实证。完整研究入口为scripts/run\_benchmark\_research.py，资格与压力检查为qualify\_benchmark\_strategy.py；已有运行不会覆盖，重复实验须另行归档。

回测页默认V3最新合格历史候选，选择器可切换V2全部模型和V1失败实验。独立账本通过使用中性提示；策略亏损、未超过指数或回撤超门槛使用未达标提示。切换版本清除旧结果，并展示实际日期和资金，防止把旧图误认成新策略。
\subsection*{研究边界}
固定2019池、财报事后修订、行业重构和缺失估值限制外推；复权分数单位尚未覆盖整手、结算、完整分红现金和盘口执行。当前候选在历史复核后选择，未来表现未验证。所有失败、参数和源文件身份保留；公开不代表厂商数据再分发授权，Token和原始缓存均留本地。
\subsection*{来源与提交}
课程依据CF2026\_Project1.pdf（19页）；参考Qlib工作流与增强指数概念，未声称复制Alpha158或Qlib引擎。来源说明见BENCHMARK\_METHOD\_SOURCES\_V3.md，链接中证编制方案、Tushare目录/指数日线/基金复权、sklearn时间序列验证。CogAlpha仅借鉴代码式因子表示。

完整源代码、证据、LaTeX、PDF、PowerPoint和讲稿纳入Git。提交前填写组员姓名、学号并按教师要求命名；最终评分由教师决定。本报告已涵盖课程基础项和有实际证据的扩展，收益结果不能替代工程正确性，也不能作为未来收益保证。
\end{document}
'''
(OUT/'final_report.tex').write_text(prefix+page1+page2+page3+page4+page5+page6+page7+page8+page9+page10,encoding='utf-8')

# Reuse stable assignment/data/factor slides, replace every strategy claim.
slides=json.loads((ROOT/'reports/final/deck_content.json').read_text(encoding='utf-8'))
def update(n,title,claim,items,notes,**extra):
    old=slides[n-1];slides[n-1]={'title':title,'claim':claim,'items':items,'notes':notes,'seconds':old['seconds'],**extra}
update(1,'青序量化研究平台','从失败账本到超过指数的历史候选',
       ['Computational Finance · Project 1','1,000只股票，12因子，8个V3候选','固定复核：2025-01-02 至 2026-09-18'],
       '本次展示从平台正确性开始，解释原始失败、用户升级的指数目标，以及最后找到的滚动LightGBM候选。必须同时说明：这个候选经过历史结果筛选，不是独立盲测成功。',kind='cover',date='2026年9月22日')
update(6,'年度更新模型，标签先实现','每年只训练形成日前三年的可见观测',
       ['2023/2024先验证，清除跨形成日标签','2025/2026逐年重训，同一年度固定模型','原V2已看过最终区间，不能恢复盲测资格'],
       '形成日是每年第一次交易前一个交易日。标签退出日必须严格早于形成日，因此不能把还没实现的20日收益放进训练。每年滚动最近三年训练，随后一年使用新模型。此次最终区间已被V2观察，滚动训练不能改变这个事实。',kind='timeline')
update(8,'中性化与标准化同时输入','LightGBM使用12个标准化值与12个残差值',
       ['每日1%/99%截尾、样本标准差缩放','当日行业与对数市值回归残差','缺财报不补未来值；个别标准化缺值取0','中性化减少暴露，但不保证提高收益'],
       '保留两套表示让模型区分原始风格和剔除行业市值后的相对位置。全部横截面处理只使用当天数据。基本面严格晚于公告日才可见，但仍披露厂商历史修订。')
update(9,'候选有明确边界','四种信号 × 两种组合，共8个候选',
       ['经济主题：价值50%、质量25%、低风险25%','Ridge：12个中性化输入，alpha=100','LightGBM：24输入、250树、15叶、深度5','融合：树模型50%、线性25%、主题25%'],
       '候选和容量在本轮开始前写进协议。每种信号配稳定股票仓位和波动预算组合，避免只展示胜出的模型。神经网络仍在V2版本对照中，本轮不要求复杂模型必须胜出。')
update(10,'仓位目标与指数目标相适应','最高95%股票敞口；趋势偏弱时预算乘0.75',
       ['20日调仓、50股、20名缓冲','波动目标20%，单股4%、行业25%','其余持现金，禁止杠杆与卖空','实际权重会漂移，回撤目标不是保证'],
       '上一版平均一半现金更适合防御。新方案提高股票参与度，同时保留波动、行业和单股限制。最新候选最终期平均现金约17.87%。权重和波动是形成时目标，无法承诺未来回撤上限。')
update(11,'账本正确与策略达标分开','版本切换后清除旧结果，亏损显示未达标',
       ['新旧策略共用成交与独立核对核心','下一开盘，实际成交收费，限制价/缺价阻断','最新默认V3，V1失败与V2对照仍可选择','固定结果可查看；本机可改参数重跑'],
       '绿色账本提示曾容易让用户误以为策略合格。现在会分别显示资金核对与收益风险判断。回测页直接提供V1、V2、V3版本，不再只给旧动量入口。',kind='demo')
def modelchart(n,table,fig,title,claim,notes):
    update(n,title,claim,[],notes,figure=fig,chart={'type':'bar','categories':short,
      'series':[{'name':'超价格指数（百分点）','values':(table.excess_return_pp*100).tolist()},
                {'name':'最大回撤（%）','values':(table.max_drawdown*100).tolist()}],'percent_points':True})
modelchart(12,val,'validation','验证选择没有被事后改写','价值主题稳定仓位排名第一，但随后失败',
           '先看2023到2024，规则要求净收益、超额收益为正且回撤不超过25%，按最差年度超额优先。价值主题稳定仓位被预选。LightGBM稳定仓位回撤25.73%，没有通过风险门槛。')
modelchart(13,test,'test_models','历史复核保留全部8个候选','当前候选为滚动LightGBM波动预算版',
           '预选价值主题最后只赚4.30%，没有超过指数。用户要求继续研究后，我们在通过验证门槛的候选中根据这张历史结果表选择LightGBM波动预算。这是开发后的候选选择，不能说事前选中了赢家。稳定版树模型收益更高，但验证回撤超门槛，未作为默认。')
wide=curves.pivot(index='date',columns='series',values='nav');sample=wide.iloc[::5]
if sample.index[-1]!=wide.index[-1]:sample=pd.concat([sample,wide.iloc[[-1]]])
names=['V3最新候选 · 扣费','沪深300 · 价格指数','沪深300 · 全收益指数']
update(14,'策略超过价格与全收益指数','净收益25.13%，最大回撤9.88%',
       ['价格指数14.55%，全收益指数19.94%','分别超过10.58和5.19个百分点'],
       '该结果覆盖固定417日，费用已扣。全收益指数包含红利再投资，是比价格指数更严格的基准。策略收益25.13%、年化14.51%、回撤9.88%。我们也检查首次开盘基准和ETF代理，结论没有改变。',figure='nav',chart={'type':'line','categories':[d.strftime('%Y-%m-%d') for d in sample.index],
       'series':[{'name':name,'values':sample[name].tolist()} for name in names]},footnote='PPT曲线每5日取点；完整417日见Beamer、平台与报告。')
update(15,'加倍费用与延迟信号仍胜出','压力检查不再用于选择更好参数',
       ['双倍费用：22.48%；延后一日：27.31%','2023起连续收益44.09%，最大回撤19.70%'],
       '双倍费用和延迟一天都重新运行完整账本，并超过同区间全收益指数。延迟一天更高不能据此改默认参数。另从2023起连续运行，避免每年重新起步；2024落后指数7.28个百分点，仍原样呈现。',figure='controls',chart={'type':'bar','categories':['标准费用','双倍费用','延迟一天'],
       'series':[{'name':'净收益','values':(stress.iloc[:3].total_return*100).tolist()},{'name':'最大回撤','values':(stress.iloc[:3].max_drawdown*100).tolist()}],'percent_points':True})
update(17,'现场演示：版本、目标与账本','两分钟切换最新、旧版与失败案例',
       ['策略研究：查看价格/全收益指数和选择边界','回测实验：默认最新V3，切换V2原预选','V1自由实验：查看明确的历史失败标记','展开参数重跑：看版本、日期、费用及明细'],
       '先打开策略研究说明25.13与两个指数。然后进入回测实验，确认默认滚动LightGBM；切换V2，收益应变为6.12并显示未超过指数；再切V1展示失败警示。最后返回V3，说明改参数会生成新的实验，旧结果不会混入。现场可用预热缓存演示。',kind='demo')
update(18,'平台满足研究与演示目标','历史上跑赢指数；未来仍须前向验证',
       ['达成：扣费后超过价格及全收益指数','验证：52项测试、21个账本、独立复算','保留：失败预选、8个候选、落后年份','下一步：冻结V3，使用未来新数据检验'],
       '最终交付包括可维护平台、完整策略版本、报告和演示。结果达到这段历史的目标，但不能保证未来赚钱或满分。最重要的研究边界是当前候选在看过最终历史结果后被选中，需要新的前向数据来验证。')
update(20,'备份：老师可能追问什么','把选型、基准和失败讲清楚',
       ['为什么不选最高收益？稳定树模型验证回撤超25%','为何不用CNN？因子列没有天然局部邻接','是不是独立盲测？不是，明确披露事后选择','来源：中证、Tushare、Qlib、sklearn、CogAlpha'],
       '应如实回答：价值主题预选失败，滚动LightGBM是在本轮历史候选中进一步筛选的。价格和全收益指数分别给结论，ETF只是代理。我们保留52项测试、源码身份和独立复算，不能用工程验证来替代市场泛化证明。')
(OUT/'deck_content.json').write_text(json.dumps(slides,ensure_ascii=False,indent=2),encoding='utf-8')
beamer_builder=(ROOT/'scripts/build_deck_content.py').read_text(encoding='utf-8').split('def tex(s):',1)[1]
beamer_builder='def tex(s):'+beamer_builder
beamer_builder=beamer_builder.replace("docs/PRESENTATION_SCRIPT.md","docs/PRESENTATION_SCRIPT_V3.md").replace("tmp/final-latex/beamer","tmp/v3-latex/beamer").replace("CF2026_Beamer.pdf","CF2026_V3_Beamer.pdf")
exec(compile(beamer_builder,'beamer_builder','exec'),globals())
build=ROOT/'tmp/v3-latex/report';build.mkdir(parents=True,exist_ok=True)
env=os.environ.copy();rt=Path('D:/Downloads/MM-LaTeX/.miktex-runtime')
env.update(MIKTEX_USERCONFIG=str(rt/'config'),MIKTEX_USERDATA=str(rt/'data'),MIKTEX_USERINSTALL='D:/MiTex')
for _ in range(2):
    p=subprocess.run(['D:/MiTex/miktex/bin/x64/xelatex.exe','-interaction=nonstopmode','-halt-on-error',f'-output-directory={build}','final_report.tex'],cwd=OUT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (build/'compile.txt').write_bytes(p.stdout)
    if p.returncode:raise RuntimeError(p.stdout.decode('utf-8',errors='replace')[-3000:])
shutil.copy2(build/'final_report.pdf',OUT/'CF2026_V3_Final_Report.pdf')
print('V3 report, Beamer, source and script ready.',flush=True)
