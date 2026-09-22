"""Build V4 platform report, Beamer and the shared editable-slide narrative.

Run without --compile to author source files while integration evidence is pending.
PDF compilation requires the local XeLaTeX installation. V3 artifacts are inputs
and are never modified. Screenshots remain optional until the final UI is verified.
"""
from pathlib import Path
import argparse
import json
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from cfquant.analytics import compare_strategies

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/platform_v4'
FIG = OUT / 'figures'
E = ROOT / 'evidence/research_v3'


def tex(value):
    return str(value).replace('\\', r'\textbackslash{}').replace('&', r'\&').replace('%', r'\%').replace('_', r'\_').replace('#', r'\#')


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(FIG / (name + '.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)


def test_counts(path):
    if not path.exists():
        return None
    suites = ET.parse(path).getroot().findall('.//testsuite')
    totals = {key: sum(int(s.get(key, 0)) for s in suites) for key in ['tests', 'failures', 'errors', 'skipped']}
    if totals['failures'] or totals['errors']:
        raise ValueError(f'Cannot publish a passing-test statement from {path}')
    return {'passed': totals['tests'] - totals['skipped'], 'skipped': totals['skipped']}


def figures():
    for name in ['nav', 'validation', 'test_models', 'controls', 'legacy', 'factor_ic']:
        for extension in ['png', 'pdf']:
            shutil.copy2(ROOT / f'reports/benchmark_v3/figures/{name}.{extension}', FIG / f'{name}.{extension}')
    daily = pd.read_csv(E / 'daily/rolling_lightgbm__managed.csv')
    curves = pd.read_csv(E / 'curves.csv', parse_dates=['date'])
    benchmarks = {name: part.set_index('date').nav for name, part in curves.groupby('series')
                  if name in ['沪深300 · 价格指数', '沪深300 · 全收益指数']}
    result = compare_strategies({'V3': daily}, benchmarks)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(10, 2.8))
    dd = result['drawdowns']['V3']
    ax.fill_between(dd.index, -dd, 0, color='#BD645E', alpha=.22)
    ax.plot(dd.index, -dd, color='#BD645E', lw=1.3)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel('Drawdown from running peak')
    ax.set_title('V3 rolling LightGBM + risk budget | 417 sessions, first-day loss included')
    ax.grid(axis='y', alpha=.2)
    save(fig, 'v3_drawdown')
    monthly = result['monthly']['V3'].unstack('month').reindex(columns=range(1, 13))
    fig, ax = plt.subplots(figsize=(10, 2.4))
    limit = np.nanmax(np.abs(monthly.to_numpy())) * 100
    image = ax.imshow(monthly * 100, cmap='RdYlGn', vmin=-limit, vmax=limit, aspect='auto')
    ax.set_xticks(range(12), [str(i) for i in range(1, 13)])
    ax.set_yticks(range(len(monthly)), monthly.index.astype(str))
    for row in range(len(monthly)):
        for col in range(12):
            value = monthly.iloc[row, col]
            if np.isfinite(value):
                ax.text(col, row, f'{value*100:.1f}', ha='center', va='center', fontsize=10)
    ax.set_title('V3 compounded monthly return (%) | partial first/last months retained')
    fig.colorbar(image, ax=ax, shrink=.8)
    save(fig, 'v3_monthly')
    return result


def integration_status(args):
    default_status = ROOT / 'evidence/qlib_bridge/result.json'
    if args.qlib_status or default_status.exists():
        path = Path(args.qlib_status) if args.qlib_status else default_status
        if not path.is_absolute():
            path = ROOT / path
        q = json.loads(path.read_text(encoding='utf-8'))
    else:
        q = {}
    passed = q.get('passed') is True
    if passed:
        count = q['assets']
        if isinstance(count, list):
            count = len(count)
        expressions = q['factor_count']
        if isinstance(expressions, (list, dict)):
            expressions = len(expressions)
        alpha = q.get('alpha158', {})
        qshort = f'Qlib原生{expressions}表达式与Alpha158配置已实测'
        qlong = (f'Qlib {q["packages"]["pyqlib"]}在独立环境读取本地转换的数据；'
                 f'{count}股共{q["source_rows"]:,}条日线，{expressions}个基础表达式与独立pandas计算逐项核对。')
        compared = sum(check['compared_values'] for check in q['checks'].values())
        qlong += f'共核对{compared:,}个数值，有限值与缺失位置一致。'
        if alpha.get('configured'):
            qlong += (f'另实际运行官方Alpha158的{alpha["configured"]}项配置，'
                      f'{alpha["with_finite_values"]}项至少有一个有限值，'
                      f'VWAP输入覆盖率{alpha["vwap_coverage"]:.1%}。')
        qlong += '这属于因子计算和数据接口验证，未训练Qlib交易策略，也未替换现有成交账本。'
    else:
        qshort = 'Qlib原生表达式接入尚待运行证据核验'
        qlong = ('Qlib接入独立安装环境与本地数据适配，当前材料尚未读取到通过的运行证据，'
                 '因此不能声称已经完成原生计算、复刻Alpha158或连接Qlib回测引擎。')
    bridge = args.bridge_status
    if not bridge:
        connection = ROOT / 'evidence/connections/status.json'
        state = json.loads(connection.read_text(encoding='utf-8')) if connection.exists() else {}
        if state.get('file_bridge_available') is True and state.get('account_connected') is False:
            bridge = ('同花顺/SuperMind文件桥已完成：可导出历史权重与代码、导入模拟成交CSV并校验金额、保存本地回执。'
                      '账号/API未连接，未收到真实账户文件，未提交任何订单；现金流核对不等于盈利。')
        else:
            bridge = '同花顺文件桥与账户连接状态待核验；不表示交易已经接通。'
    return qshort, qlong, bridge, q


def slides(qshort, qlong, bridge, q=None):
    result = []

    def add(title, claim, items, notes, seconds=60, **extra):
        result.append({'title': title, 'claim': claim, 'items': items, 'notes': notes,
                       'seconds': seconds, 'source': '来源：本项目 evidence/research_v3、platform_v4；数据与假设见最终报告。', **extra})

    add('青序量化研究平台', '可操作的研究工具、完整策略版本与可核验的结果',
        ['Computational Finance · Project 1', '平台增强版 V4 / release v2.2.0', '1,000只股票 · 12因子 · 历史策略V3'],
        '各位老师、同学好。我们展示的是青序量化研究平台。这次项目有两个相互联系的成果：一是可以实际使用的研究工作台，二是通过这个工作台得到并保留完整证据的策略。大家等会儿会看到如何比较策略、解释回撤、检查一笔订单，以及保存一个实验。当前策略在固定历史区间内扣费后超过沪深三百，但它经过了历史结果筛选，所以我们会把结果和局限一起讲清楚。今天的顺序是先介绍数据和研究方法，再解释策略结果，最后直接操作平台。演示结束后，报告、源码和每页讲稿都能对应到同一份证据。',
        40, kind='cover', date='2026年9月22日')
    add('课程要求与交付范围', '数据、因子、回测、诊断与复现形成完整工作流程',
        ['至少三因子 → 原三因子 + 12因子研究集', '含费用回测 → 订单、成交、现金、持仓与独立核对', '诊断与扩展 → IC、分组、中性化、风险与Qlib', '可交付 → 源码、报告、演示与逐页讲稿'],
        '我们先把作业要求理解为一条完整研究流程。数据要知道从哪里来，因子要有定义和信息时点，回测要把费用与成交约束落实到每一笔交易，最后还要解释因子是否有效。课程要求至少三个因子，我们保留原来的基础因子，又加入十二个有经济含义的研究因子。扩展部分选择中性化、风险控制和外部研究工具接入。界面增强也围绕具体动作展开，例如比较一个新假设，或者追踪一次没有成交的订单。评分应当根据这些实际实现与证据判断，不能因为净值图上涨，就忽略数据泄漏和工程正确性。接下来先看这些功能共用的架构。', 50)
    add('平台架构', '模型更新时，复用同一套组合与成交账本',
        ['数据快照\n字段与质量检查', '因子和模型\n日期 × 资产分数', '组合与风险\n目标权重和现金', '成交与分析\n订单、净值和证据'],
        '平台在架构上把四件事分开。数据层处理行情、财报与交易日，因子和模型层输出股票分数，组合层把分数转成目标权重，成交层才决定实际买卖了多少。这种分离让我们可以把动量换成树模型，却不必重新实现一套现金会计。命令行、回测界面和报告都使用同一个核心。新增的策略对比与风险透镜直接读取公开日账本，因此没有原始数据的同学，也能离线复核结果。需要重新训练或按股票检查行情时，才使用本机私有快照。这样既减少了维护成本，也能明确哪些按钮是在查看证据，哪些按钮真的产生了新的实验。', 60, kind='flow')
    add('数据来源与可见范围', '固定历史股票池，保留缺失与数据身份',
        ['价格：2019-10-08 至 2026-09-18', '固定2019年末池，不代表全部A股', '原始数据与Token留本地，公开聚合结果'],
        '目前行情快照包含一千只股票，一百六十五万多条日线，覆盖一千六百九十个交易日。股票池在二零一九年末固定，因此不是把今天还上市的股票直接套回过去；但它也没有包括后来的新股，仍有明确的选择偏差。财报只有在公告之后才进入特征，历史行业也按有效区间匹配。我们记录缺失网格，不凭空补出可以交易的价格。原始快照和研究缓存总量约一点六三GB，新增小规模接入证据单独保存。公开仓库提供来源、哈希、聚合账本与复现方法，账号凭据和厂商原始数据不随源码发布。数据量只是基础，时点正确才决定研究能否成立。', 55,
        table=[['项目', '真实快照'], ['资产 / 交易日', '1,000 / 1,690'], ['日频行情 / 每日估值', '各1,656,599条'], ['财务 / 历史行业区间', '57,389 / 1,432条'], ['缺失网格', '33,401条；不伪造成交']])
    add('十二个研究因子', '先解释经济含义，再检查实际预测表现',
        ['因子卡提供字段、窗口、方向和缺失处理', '例：跳月动量Rank IC −0.016；低波动 +0.072', '负方向结果保留，不按最终期翻转符号'],
        '这十二个因子覆盖趋势、短期反转、低风险、量价、价值、分红和盈利质量。例如盈利收益率把估值转成收益尺度，低波动因子描述过去价格风险，跳月动量避开最近一个月的短期反转。这些因子不是彼此独立的十二条信息，低波动和低振幅就可能高度相关。我们用每日横截面IC和分组结果检查它们，而不是只看最后哪条曲线最好。界面里可以选一个因子，查看定义、指定区间的覆盖率和有效样本，也能对照十二个因子的Rank IC。因子诊断所用的远期标签不能回流成当天的交易依据，这条时间边界在后面的模型训练中同样适用。', 65,
        table=[['类别', '因子'], ['趋势 / 反转', '跳月动量、5日反转、均线趋势'], ['风险 / 量价', '低波动、低振幅、量能、流动性、低换手'], ['价值 / 分红', '盈利收益率、账面市值比、股息率'], ['质量', '最新已公告年度ROE']])
    add('年度模型与信息时点', '训练标签退出日严格早于模型形成日',
        ['每年滚动此前三年，同一年度固定模型', '标签：下一开盘至20交易日后开盘收益秩', '按日截尾与标准化，行业/市值回归取残差', 'LightGBM输入12个标准化值 + 12个残差值'],
        '模型每年更新一次，只使用形成日前三年的可见数据。训练目标不是简单预测涨跌，而是预测股票未来二十个交易日收益在当日横截面中的相对排名。最容易出错的地方是标签边界：即使特征日期早于新年，二十日收益也可能还没有实现，所以必须要求标签退出日严格早于模型形成日。特征处理也按日进行，先截尾和标准化，再回归当时的行业与对数市值得到中性化残差。树模型同时看到原始风格和相对风格。我们选择受限深度的LightGBM，并保留Ridge和小型神经网络对照，因为更复杂的模型不自动代表更好的策略，更不自动解决信息泄漏。', 75, kind='timeline')
    add('组合、现金与成交规则', '分数领先不等于立刻全部买入',
        ['50股目标，20日调仓，20名保留缓冲', '股票预算最高95%，单股4%，行业25%', '60日风险估计；20%目标波动；弱趋势预算×0.75', '下一开盘执行，实际买10bp / 卖15bp'],
        '有了分数以后，我们选择五十只股票，并通过排名缓冲降低不必要的换手。组合允许现金，最高股票预算是百分之九十五；单股和行业目标分别不超过百分之四和百分之二十五。目标年化波动为百分之二十，风险估计较高时缩减仓位，沪深三百低于一百二十日均线时，再把预算乘零点七五。交易采用收盘形成信号、最早下一开盘执行，买卖费用分别为十和十五个基点。涨跌停、缺价、现金不足与过去成交额决定能否完成交易，未成交的部分不收费。这些约束都是形成时和模拟执行时的规则，之后价格会变化，所以目标权重和回撤门槛都不是未来风险的硬保证。', 70,
        formula=r'fee_t=0.001B_t+0.0015S_t', formula_plain='费用 = 0.001 × 实际买额 + 0.0015 × 实际卖额')
    add('失败记录与候选选择', '保留验证预选失败，明确后续历史重选',
        ['V1动量：扣费−83.57%；零费重跑仍−75.16%', 'V2原预选：固定最终区间+6.12%，落后指数', 'V3先预选价值主题，随后仅+4.30%', '历史复核后采用LightGBM预算版；不是独立盲测'],
        '这页是策略研究最需要如实说明的部分。原始动量累计亏损百分之八十三点五七，真正把费用设成零重新运行，仍然亏损百分之七十五点一六，因此不能把失败全部归因于手续费。第二版改善了回撤，但收益没有超过指数。第三版限定八个候选，先按二零二三到二零二四年的验证结果预选价值主题，随后发现它在最终区间只赚百分之四点三。当前滚动树模型是在进一步查看历史结果后选出的。稳定仓位树模型收益更高，但验证回撤超过百分之二十五，所以没有作为默认。我们保留这些选择过程，承认最终历史区间已经参与开发，需要未来新数据验证泛化。', 80)
    wide = pd.read_csv(E / 'curves.csv', parse_dates=['date']).pivot(index='date', columns='series', values='nav')
    sample = wide.iloc[::5]
    if sample.index[-1] != wide.index[-1]:
        sample = pd.concat([sample, wide.iloc[[-1]]])
    names = ['V3最新候选 · 扣费', '沪深300 · 价格指数', '沪深300 · 全收益指数']
    add('固定区间的收益与基准', '净收益25.13%，最大回撤9.88%',
        ['价格指数14.55%，全收益指数19.94%', '分别超过10.58和5.19个百分点'],
        '这张图使用二零二五年一月二日至二零二六年九月十八日相同的四百一十七个交易日。策略已经扣除实际费用，累计收益百分之二十五点一三，最大回撤百分之九点八八。沪深三百价格指数同期收益百分之十四点五五，计入红利再投资的全收益指数是百分之十九点九四。因此，策略分别超过十点五八和五点一九个百分点。这里不能把ETF代理改称全收益指数，也不能混淆百分点差和相对财富增长。为了避免首日亏损被抹掉，策略从期初资金起算，指数从前一交易日收盘起算。平台中的日期切片和策略比较采用完全相同的边界定义。', 65,
        figure='nav', chart={'type': 'line', 'categories': [d.strftime('%Y-%m-%d') for d in sample.index],
                            'series': [{'name': name, 'values': sample[name].tolist()} for name in names]},
        footnote='PowerPoint每5日取点；完整417日曲线见报告与平台。')
    stress = pd.read_csv(E / 'stress.csv').iloc[:3]
    add('压力检查与落后年份', '提高费用后仍超过同期全收益指数',
        ['双倍费用：22.48%；延迟一日：27.31%', '2023起连续44.09%，2024年落后价格指数'],
        '收益结果还需要承受一些简单但实际的扰动。我们把买卖费用同时加倍，完整重跑账本，收益仍为百分之二十二点四八；把信号延迟一日，收益为百分之二十七点三一。延迟结果更好，不代表可以再把它挑成默认参数，否则压力测试又变成一次找冠军。另一个检查从二零二三年开始连续运行，不在每年年初重置资金，累计收益百分之四十四点零九，最大回撤百分之十九点七。但二零二四年仍然落后价格指数七点二八个百分点。这个年份提醒我们，长期累计领先不能被解释为每年都赢。接下来我们用平台工具把这种差异具体拆开。', 55,
        figure='controls', chart={'type': 'bar', 'categories': ['标准费用', '双倍费用', '延迟一天'], 'percent_points': True,
                                 'series': [{'name': '净收益', 'values': (stress.total_return*100).tolist()},
                                            {'name': '最大回撤', 'values': (stress.max_drawdown*100).tolist()}]})
    add('工具一：同区间策略对比', '选择版本与日期，导出同一计算口径的结果',
        ['最多五版策略；价格或全收益基准', '累计/相对净值、回撤、相关性与指标表', '区间从前收盘起算，首日损益保留'],
        '策略对比页解决的是公平比较问题。我们可以同时选择最多五个版本，限定相同日期，再选择价格指数或全收益指数。图表既可以看普通净值，也可以看策略净值除以基准净值后的相对表现。下面的表把收益、回撤、费用和现金比例放在一起，避免只比较一个收益数字。日期改变以后，系统使用所选首日前的净值作为起点，因此不会把首日损失归一掉。需要注意，这里是在切片一条已经运行的持仓路径，并不是从现金重新开仓；如果要改变实际交易参数，应该回到回测实验重新执行。所有图表、统计和计算假设可以一次下载为比较包，方便写报告或与组员核对。', 70,
        screenshot='evidence/ui-v4-compare.png')
    add('工具二：风险透镜', '定位回撤、月份与现金配置的影响',
        ['月收益日历、日收益分布、回撤恢复表', '60日滚动收益、波动与下行偏差', '现金/费用/换手；静态资金分配情景'],
        '风险透镜把一个最大回撤数字拆成可以解释的过程。月收益日历能看出收益集中在哪些月份，回撤表则列出峰值、谷底、恢复日期和持续交易日。尚未恢复的回撤保留为空，不伪装成已经恢复。滚动曲线用最近六十个交易日计算收益与风险，不足窗口时不补值。现金和换手图让我们看到防御仓位以及费用发生在哪些时段。最后还有一个可以动手操作的现金配置实验：把部分初始资金投入策略，其余保持零利息现金，并输入通胀假设查看购买力。它是一种静态资金分配的算术情景，没有重新模拟资金规模对成交容量的影响，也不是新增的交易策略。', 70,
        screenshot='evidence/ui-v4-risk.png')
    add('工具三：因子与行情探索', '把模型输入连接到具体日期和股票',
        ['十二因子按日期诊断；行情按股票筛选', '因子卡、覆盖率、IC/Rank IC与分组诊断', '行情K线与成交量，真实数据缺失明确说明'],
        '因子与行情入口可以把抽象模型拉回具体数据。我们可以先选择一个研究因子，阅读它的窗口、字段和方向，再查看形成日的样本覆盖、IC和分组结果。如果一个因子突然没有有效值，就能沿着日期检查价格、成交量或财报缺失，而不是直接把空值当成模型信号。行情页提供K线和成交量查询，方便检查异常价格和某次信号对应的市场背景。公开离线模式下，界面只展示已有的聚合证据；没有本机原始行情时不会伪造K线。这个入口也适合课堂提问，例如老师指定一只股票或一个日期，我们可以现场说明它的数据是否可见、因子是否有效，再讨论预测是否合理。', 60)
    add('工具四：实验档案与研究笔记', '从一条净值曲线追到订单与配置',
        ['按实验、日期、股票、成交状态检索', '查看资金、持仓、费用与来源身份', '保存本地笔记，导出研究记录'],
        '实验档案让每次尝试都有可追溯的上下文。参数重跑会生成新的运行目录，档案页可以打开这次实验，检查实际使用的策略、日期、资金与费用。进一步按股票或订单状态筛选，就能解释一次涨停买不到、缺价估值或者部分成交对账本产生了什么影响。我们还加入本地研究笔记，用来记录这次改变了哪个假设、观察到什么现象、下一步需要验证什么。笔记与实验关联，但不改变历史账本。这样的设计比只保留最终冠军更有用，因为小组后续项目可以沿着旧实验继续工作。课堂演示时，我们会打开一个已完成实验并查看一笔订单，不让大家等待重新训练。', 60,
        screenshot='evidence/ui-v4-experiments.png')
    qtable = {}
    if q and q.get('passed') is True:
        compared = sum(check['compared_values'] for check in q['checks'].values())
        qtable = {'table': [['接入层', '实际结果与范围'],
                            ['Qlib数据桥', f'{q["packages"]["pyqlib"]} / {q["assets"]}股 / {q["source_rows"]:,}条日线'],
                            ['独立核验', f'{q["factor_count"]}项表达式，{compared:,}个数值核对'],
                            ['Alpha158', f'官方{q["alpha158"]["configured"]}项表达式实际执行'],
                            ['同花顺文件桥', 'CSV导入与金额核对；账号/API未连接']]}
    add('Qlib与外部工具连接', qshort,
        ['Qlib尚未训练新策略；V3历史结果保留', '同花顺尚未登录，未提交订单'],
        '老师推荐Qlib，因此我们把它作为可复用的研究组件，而不是只在报告中放一个名字。'+qlong+
        '一个外部因子接进来以后，仍然要经过我们自己的信息时点检查、诊断和同区间比较，不能因为来自知名库就直接上线。另一个连接是同花顺。'+bridge+
        '课堂上应当清楚区分三种状态：能导出文件，能调用已授权的数据接口，以及能向交易系统提交订单。这三者不是同一件事。本项目的实证结果仍由本地研究账本产生。', 80,
        source='https://github.com/microsoft/qlib ; 本项目Qlib与连接状态证据；不声称交易接通。', **qtable)
    add('现场演示顺序', '两分钟完成一次可复核的研究操作',
        ['总览确认V3 → 对比V2与全收益指数', '风险透镜改日期 → 查看回撤恢复与月历', '因子或行情查询 → 实验档案检索订单', '下载比较包 → 保存一条研究笔记'],
        '下面用约两分钟实际操作一次。先看研究总览，确认当前默认版本、数据区间和历史选择边界。进入策略对比，保留V3，再加入V2原预选，并将基准设为全收益指数，大家可以看到两版策略的差异。接着进入风险透镜，修改分析日期，指向月收益和最深回撤，说明这是一段持仓路径的切片。然后打开因子或行情页面，展示一个具体日期的样本。最后进入实验档案，筛选一笔订单，查看成交状态和费用，保存一条带有实验背景的笔记。如果教室网络或现场页面不稳定，我们切换到预先保存的真实截图继续说明；演示不现场下载数据，也不现场等待模型重训。', 100, kind='demo')
    add('验证与复现', '公开聚合证据可离线检查，真实重跑要求本机数据',
        ['21组V3研究/压力账本，独立环境复算', '首日损益、缺日拒绝、回撤恢复等边界测试', '公开数据分析与私有原始数据重跑分层', 'Git版本、配置、SHA与导出包共同定位结果'],
        '我们用两类检查支持这些功能。第一类验证研究核心，包括订单、费用、现金、持仓和净值的独立重建，第三版共有二十一组研究与压力账本，并在独立环境中复算。第二类验证新增工具容易出错的边界，例如首日损失是否保留，不同策略是否比较相同日期，缺失日是否被偷偷插值，以及未恢复回撤是否被误记成恢复。公开仓库提供真实聚合日账本，所以没有账号也能打开对比与风险页。真正重新训练或者查看逐股行情，则需要本机合法取得的数据。我们提供源码、固定依赖、操作说明和版本号，让下一位同学能够判断自己是在复现哪一个结果。', 50)
    add('结论与下一步', '保留有效工具，也保留尚未解决的问题',
        ['平台：可比较、可解释、可追踪、可扩展', '策略：该历史区间扣费后超过两类沪深300基准', '限制：历史重选、固定池与执行近似', '下一步：冻结规则，积累真正的新数据验证'],
        '最后回到本项目的目标。我们交付的不只是一个收益数字，而是能够继续研究的平台：可以输入新信号，按统一规则回测，公平比较版本，解释风险，再把证据留给下一次实验。当前策略在这段历史上超过价格指数和全收益指数，但我们没有删除原始亏损、预选失败和落后年份。它仍然存在固定股票池、厂商历史修订以及成交近似的限制，历史重选也意味着还不能宣称独立盲测成功。下一步最有价值的工作是冻结当前规则，持续积累未见过的新数据，然后按相同标准检查。以上就是我们的展示，谢谢各位老师和同学，欢迎针对因子、账本和平台操作提问。', 40)
    add('备份：统计口径', '初始资金参与首日收益与最大回撤',
        ['收益：NAV终值 / 区间前收盘 − 1', '回撤：1 − NAV / 包含初始值的历史高点', '下行偏差：全部观察日作分母，目标为零', '月度收益复利；重叠20日标签不能连乘为净值'],
        '如果老师追问统计公式，可以用这一页说明。区间第一天收益不是零，而是第一天收盘净值与前一日净值或初始资金的比值。最大回撤的高水位也包含初始资金，因此一开始就亏损不会被遗漏。年化波动采用样本标准差乘根号二百五十二，下行偏差把正收益日记为零，但分母仍然是全部观察日。月度收益由日收益复利得到，首尾不足一个月会明确显示。因子二十日远期标签存在重叠，它适合诊断排序关系，不能直接连乘当作真实投资净值。', 0,
        formula=r'r_t=\frac{NAV_t}{NAV_{t-1}}-1,\qquad DD_t=1-\frac{NAV_t}{\max_{0\leq s\leq t}NAV_s}',
        formula_plain='日收益 = 今日净值 / 昨日净值 − 1；回撤包含初始资金')
    add('备份：答辩问题', '如实解释模型选择、现金与连接状态',
        ['为什么不用CNN？因子列没有天然局部邻接', '为什么不是最高收益版本？验证回撤门槛', '现金情景能当策略吗？只是静态资金分配', 'Qlib接入范围：因子计算，尚未训练新策略'],
        '如果问为什么没有按最初建议使用CNN，我们的回答是，表格因子列没有天然空间邻接，先用线性模型、受限树模型和小型MLP比较更合理。如果问为什么没有选收益最高的稳定树模型，是因为它的验证回撤超过了事先保留的风险门槛。如果问现金实验是否就是一个新策略，需要说明它只是资金缩放情景，改变交易规模仍然需要重新回测。最后，Qlib数据表达式接入、同花顺文件桥和真正的交易授权必须分别回答，以最终运行证据为准，不能把准备好的接口说成已经接通的交易账户。', 0)
    return result


def write_beamer(deck):
    preamble = r'''\documentclass[aspectratio=169,10pt]{beamer}
\usepackage[UTF8]{ctex}\usepackage{booktabs,tabularx,graphicx,amsmath,tikz}
\usetikzlibrary{arrows.meta,positioning}
\definecolor{navy}{HTML}{142B43}\definecolor{teal}{HTML}{168579}
\setbeamercolor{frametitle}{bg=navy,fg=white}\setbeamercolor{structure}{fg=teal}
\setbeamercolor{block title}{bg=teal,fg=white}\setbeamercolor{block body}{bg=teal!6}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{footline}{\leavevmode\hbox{\begin{beamercolorbox}[wd=.80\paperwidth,ht=2.5ex,dp=1ex,leftskip=1em]{author in head/foot}青序量化研究平台\quad Project 1 / V4\end{beamercolorbox}\begin{beamercolorbox}[wd=.20\paperwidth,ht=2.5ex,dp=1ex,center]{date in head/foot}\insertframenumber/20\end{beamercolorbox}}}
\setbeamerfont{frametitle}{size=\large}\setbeamersize{text margin left=8mm,text margin right=8mm}
\begin{document}
'''
    frames = []
    for slide in deck:
        body = r'\begin{frame}{' + tex(slide['title']) + '}\n'
        body += r'\begin{block}{}' + tex(slide['claim']) + r'\end{block}' + '\n'
        if slide.get('kind') == 'cover':
            body += r'\vspace{7mm}{\Huge\bfseries\color{navy}量化研究与风险分析}\par\vspace{5mm}' + '\n'
            body += r'{\large\color{teal}可维护平台\quad 可复核策略\quad 可继续研究}\par\vspace{6mm}' + '\n'
        if slide.get('kind') == 'flow':
            body += r'''\begin{center}\begin{tikzpicture}[node distance=4mm, every node/.style={draw=teal,fill=teal!5,rounded corners=1mm,align=center,text width=27mm,minimum height=18mm,font=\small}]
\node(a){数据快照\\字段与质量};\node(b)[right=of a]{因子与模型\\统一分数};\node(c)[right=of b]{组合与风险\\权重与现金};\node(d)[right=of c]{成交与分析\\账本与证据};
\draw[-{Stealth},teal](a)--(b);\draw[-{Stealth},teal](b)--(c);\draw[-{Stealth},teal](c)--(d);
\end{tikzpicture}\end{center}
'''
        visual = False
        screenshot = ROOT / slide.get('screenshot', '__missing__')
        if screenshot.is_file():
            dest = FIG / screenshot.name
            shutil.copy2(screenshot, dest)
            body += r'\centering\includegraphics[width=.95\textwidth,height=.55\textheight,keepaspectratio]{figures/' + dest.name + r'}\par' + '\n'
            visual = True
        elif slide.get('figure'):
            body += r'\centering\includegraphics[width=.96\textwidth,height=.51\textheight,keepaspectratio]{figures/' + slide['figure'] + r'.pdf}\par' + '\n'
            visual = True
        if slide.get('table'):
            body += r'\begin{center}\small\begin{tabularx}{.97\textwidth}{lX}\toprule' + '\n'
            for i, row in enumerate(slide['table']):
                body += ' & '.join(tex(x) for x in row) + r'\\' + (r'\midrule' if i == 0 else '') + '\n'
            body += r'\bottomrule\end{tabularx}\end{center}' + '\n'
        if slide.get('formula'):
            body += r'\[' + slide['formula'] + r'\]' + '\n'
        if slide['items'] and slide.get('kind') != 'flow':
            body += (r'\scriptsize' if visual else r'\small') + r'\begin{itemize}' + '\n'
            body += ''.join(r'\item ' + tex(item).replace('\n', r'\quad ') + '\n' for item in slide['items'])
            body += r'\end{itemize}' + '\n'
        if slide.get('footnote'):
            body += r'{\tiny ' + tex(slide['footnote']) + '}\n'
        body += r'\note{' + tex(slide['notes']) + '}\n' + r'\end{frame}' + '\n'
        frames.append(body)
    (OUT / 'beamer.tex').write_text(preamble + '\n'.join(frames) + r'\end{document}', encoding='utf-8', newline='\n')


def write_script(deck):
    seconds = sum(s['seconds'] for s in deck[:18])
    chinese = sum(len(re.findall(r'[\u4e00-\u9fff]', s['notes'])) for s in deck)
    lines = ['# 青序量化研究平台 V4：逐页演讲稿', '',
             f'对应20页Beamer / PPTX。正文1–18页建议{seconds//60}分{seconds%60:02d}秒；19–20页仅答辩时使用。',
             f'口述正文及备份共{chinese:,}个汉字，不含标题与操作卡。语速因人而异，请按实际排练微调停顿。', '',
             '演示前：运行 launch.cmd，预热研究总览、策略对比、风险透镜和实验档案；准备同版本PDF离线备份。',
             '本稿正文可直接口述；方括号内是演示动作，不朗读。第16页预留操作时间，避免讲解重复。', '']
    elapsed = 0
    actions = {9: '指向曲线图，分别指出价格指数和全收益指数。',
               11: '展示真实策略对比截图；现场操作留到第16页。',
               12: '指出月历、最深回撤与未恢复状态。',
               14: '指出实验身份与订单筛选入口。',
               15: '只展示当前已经核验的接入状态；遇到账号未验证状态直接说明。',
               16: '切换浏览器：总览 → 对比V3/V2 → 风险透镜 → 因子/行情 → 实验档案 → 保存笔记；完成后返回第17页。'}
    for i, slide in enumerate(deck, 1):
        lines += [f'## {i}. {slide["title"]}', '',
                  f'建议{slide["seconds"]}秒；从{elapsed//60:02d}:{elapsed%60:02d}开始。' if i <= 18 else '答辩备份，不计入正文时长。', '']
        if i in actions:
            lines += ['[演示动作：' + actions[i] + ']', '']
        lines += [slide['notes'], '']
        elapsed += slide['seconds']
    (ROOT / 'docs/PRESENTATION_SCRIPT_V4.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    preamble = r'''\documentclass[UTF8,a4paper,12pt]{ctexart}
\usepackage[margin=22mm]{geometry}\usepackage{xcolor,fancyhdr}
\definecolor{ink}{HTML}{142B43}\definecolor{accent}{HTML}{168579}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small 青序量化研究平台 V4 · 逐页演讲稿}
\fancyhead[R]{\small 正文19分05秒 + 答辩备份}\fancyfoot[C]{\thepage/10}
\setlength{\headheight}{16pt}\setlength{\parindent}{2em}\setlength{\parskip}{7pt}
\setlength{\emergencystretch}{2em}\linespread{1.20}
\begin{document}
'''
    parts = [preamble]
    elapsed = 0
    for i, slide in enumerate(deck, 1):
        if i > 1 and i % 2 == 1:
            parts.append(r'\newpage')
        if i == 1:
            parts.append(r'{\Large\bfseries\color{ink}课堂演示口述稿}\par' + '\n')
            parts.append('每两页幻灯片对应一页讲稿；留白供手写批注。正文可直接朗读，方括号内为操作指令。演示前预热平台并准备同版本PDF。\n')
        parts.append(r'\section*{\color{ink}' + tex(f'{i:02d}  {slide["title"]}') + '}\n')
        timing = f'建议{slide["seconds"]}秒；从{elapsed//60:02d}:{elapsed%60:02d}开始。' if i <= 18 else '答辩备份，不计入正文时长。'
        parts.append(r'{\small\color{accent}' + tex(timing) + r'}\par' + '\n')
        if i in actions:
            parts.append(r'{\small\color{accent}[演示动作：' + tex(actions[i]) + r']}\par' + '\n')
        parts.append(tex(slide['notes']) + '\n\n')
        if i % 2 == 1:
            parts.append(r'\vspace{6mm}\noindent\textcolor{accent!40}{\rule{\textwidth}{0.4pt}}\par' + '\n')
        elapsed += slide['seconds']
    parts.append(r'\end{document}')
    (OUT / 'speaker_script.tex').write_text('\n'.join(parts), encoding='utf-8', newline='\n')
    return chinese, seconds


def report(qlong, bridge, args):
    original = (ROOT / 'reports/benchmark_v3/final_report.tex').read_text(encoding='utf-8')
    prefix = original.split(r'\begin{document}')[0].replace('研究版本 V3 / 2026-09-22', '平台 V4 / v2.2.0 / 2026-09-22')
    sections = re.split(r'(?=\\page\{)', original)
    first = sections[0].split(r'\begin{document}', 1)[1]
    first = first.replace('指数目标、滚动预测与策略版本管理', '策略研究、风险分析与可扩展工作台')
    first = first.replace('V3研究更新', 'V4平台增强').replace('本轮固定', '策略研究固定')
    first = first.split(r'\subsection*{交付与操作}')[0]
    first += r'''\subsection*{平台增强版的实际能力}
研究总览汇集结果与入口；同区间策略对比支持版本、日期和基准选择；风险透镜提供月历、回撤恢复、60日滚动风险和静态现金情景；因子、行情与实验档案支持追查样本和订单。本版增强工具和可复现性，保留V3策略参数及历史选择边界，未把界面升级包装成新的独立验证。
'''
    page2 = r'''\page{2\quad 软件结构与研究工作台}
\lead{数据、信号、组合、成交和分析分别维护，界面与命令行调用同一核心。}
\begin{center}\begin{tikzpicture}[node distance=4mm, every node/.style={draw=accent,fill=accent!5,rounded corners=1mm,align=center,text width=32mm,minimum height=13mm,font=\small}]
\node(a){数据快照\\字段/质量};\node(b)[right=of a]{因子/模型\\日期$\times$资产分数};\node(c)[right=of b]{组合/风险\\权重与现金};\node(d)[right=of c]{成交/分析\\账本与证据};
\draw[-{Stealth},accent](a)--(b);\draw[-{Stealth},accent](b)--(c);\draw[-{Stealth},accent](c)--(d);
\end{tikzpicture}\end{center}
\begin{center}\small\begin{tabularx}{\textwidth}{lX}\toprule
模块 & 职责与稳定接口\\\midrule
data / incremental & 行情及交易日、字段校验、幂等更新与数据身份。\\
factors / features / models & 日期$\times$资产分数；因子卡、可见财报与年度模型。\\
risk / engine & 目标权重、现金和成交约束；订单、成交、持仓及日账本。\\
analytics / strategies & 指标、独立核对、同区间分析与不可变策略版本。\\
界面 / CLI / 导出 & 共用核心；筛选、图表、参数实验和研究记录。\\\bottomrule
\end{tabularx}\end{center}
\subsection*{四项可直接使用的增强工具}
\begin{enumerate}
\item \textbf{策略对比：}最多五版策略，统一日期与价格/全收益基准，展示相对净值、风险、费用和相关性，导出CSV及计算假设。
\item \textbf{风险透镜：}月收益、下行风险、最深回撤及恢复日期；查看现金、换手和费用，交互分析静态现金配置与通胀假设。
\item \textbf{因子与行情：}12因子独立入口、日期诊断与样本查询；K线及成交量追查异常，缺原始数据时明确可用边界。
\item \textbf{实验档案：}按运行、日期、资产及订单状态筛选，查看参数、成交与账本，保存关联实验的本地研究笔记。
\end{enumerate}
\subsection*{公开证据与本地数据各有用途}
聚合日账本足以运行策略对比、风险诊断和结果导出，无Token也能离线展示真实历史结果。逐股行情、训练缓存和交易明细依赖本机快照；缺少时说明状态。每次参数回放保存新的运行目录，策略切换清除旧图，不混用旧结果。新增模型只需输出统一分数，新增分析以DataFrame作为输入。
'''
    shot = ROOT / 'evidence/ui-v4-overview.png'
    if shot.exists():
        shutil.copy2(shot, FIG / shot.name)
        page2 += r'\begin{center}\includegraphics[width=.86\textwidth,height=57mm,keepaspectratio]{figures/ui-v4-overview.png}\end{center}' + '\n'
    page3, page4, page5, page6, page7, page8 = sections[2:8]
    page4 = page4.replace('此处12因子是明确列出的自定义子集，\\textbf{未声称完整复制Alpha158，也未将Qlib引擎集成进主账本}',
                          '主策略12因子是明确列出的自定义研究集；独立Qlib表达式及Alpha158接入状态见第10页，\\textbf{未将Qlib引擎替换进主成交账本}')
    page4 += r'''\subsection*{实际横截面诊断}
已保存878个形成日的研究诊断；先按当日分数形成五组，再匹配下一开盘至20日后开盘标签。下表给出三个代表因子，完整12因子与有效人数见公开证据和独立因子页。
\begin{center}\small\begin{tabular}{lrrr}\toprule
因子 & 平均IC & 平均Rank IC & Q5--Q1平均20日收益\\\midrule
跳月动量 & $-0.0035$ & $-0.0161$ & $-0.0831\%$\\
5日反转 & $0.0176$ & $0.0187$ & $0.3264\%$\\
20日低波动 & $0.0378$ & $0.0722$ & $0.6377\%$\\\bottomrule
\end{tabular}\end{center}
负值不翻转；重叠20日标签不视为独立样本，也不连乘为可交易净值。
'''
    page9 = r'''\page{9\quad 风险透镜与交互分析口径}
\lead{新增分析读取真实V3日账本，月收益与回撤恢复帮助解释风险。}
\begin{center}\includegraphics[width=.98\textwidth]{figures/v3_monthly.pdf}\end{center}
\begin{center}\includegraphics[width=.98\textwidth]{figures/v3_drawdown.pdf}\end{center}
\subsection*{日期、首日损益与回撤恢复}
同区间比较取交易日严格交集；内部缺日拒绝计算，不插值。区间第一日以此前收盘净值或明确初始资本作分母，保留首日损益；高水位包含初始资本。日期筛选是既有持仓路径的切片，不是从现金重新回测。未恢复回撤保留缺失恢复日期；无观测初始日期时不臆造峰值日期。
\subsection*{滚动风险与现金实验}
60日完整窗口计算复利收益、样本波动率和下行偏差，按252交易日年化；下行偏差以全部观察日作分母，目标为零。月历包括首尾不完整月份。静态资金分配采用$N_t^{mix}=wN_t+(1-w)$，现金零利息、不再平衡；输入通胀仅是购买力假设。该情景未重算规模对费用与成交容量的影响。费用占比为区间费用/期初净值，不等于零费用重跑的收益差。

\subsection*{既有压力检查保持不变}
双倍费用完整重跑净收益22.48\%，延迟一日27.31\%；2023起连续收益44.09\%、最大回撤19.70\%，2024仍落后价格指数7.28个百分点。延迟更好没有被用于修改默认参数；全部21组V3运行仍保留。
'''
    test_text = (f'主环境{args.test_count}项测试通过，{args.main_skipped}项可选Qlib集成测试因无依赖跳过；'
                 f'独立Qlib环境{args.qlib_passed}项通过，包含被跳过的真实引擎集成项') if args.test_count else '新增分析边界测试及交互检查随版本封存'
    page10 = r'''\page{10\quad 外部接入、验收与复现}
\subsection*{Qlib与同花顺连接状态}
''' + tex(qlong) + '\n\n' + tex(bridge) + r'''

\subsection*{课程要求与可检查的完成位置}
\begin{center}\small\begin{tabularx}{\textwidth}{lX}\toprule
要求 & 实现与证据\\\midrule
数据与因子 & 字段/复权/缺失/版本；原三因子、12因子卡及可配置窗口。\\
回测与指标 & 下一开盘、实际成交收费、现金/限制价/缺价、独立资金核对。\\
IC与分组 & 逐日横截面、有效样本数、先形成分组再匹配标签；负结果保留。\\
扩展与实践 & 历史行业市值中性化、风险预算、部分成交、增量审计、Qlib状态证据。\\
可用功能 & 版本回放、同区间对比、风险透镜、因子/行情、实验档案与笔记。\\
材料与复现 & 10页报告、20页Beamer/PowerPoint、逐页讲稿、源码与哈希。\\\bottomrule
\end{tabularx}\end{center}
''' + tex(test_text) + r'''；V3原研究21个账本核对和独立环境复算保留。工程检查验证计算实现，不能替代未来市场泛化证明。
\subsection*{运行与来源}
本机双击launch.cmd。新环境按REPRODUCE\_V4安装固定依赖，运行自动化测试和合成示例；无凭据先查看公开真实证据，补足合法原始数据后再运行训练。原V3研究目录防覆盖；改变假设必须保存新实验。材料、指标、Git标签v2.2.0和数据身份共同定位交付。

课程依据CF2026\_Project1.pdf（19页）；金融基准采用中证与Tushare官方定义。Qlib官方代码/文档及接入版本用于表达式核查；来源与连接状态详见仓库文档。CogAlpha仅借鉴代码式因子表示，未声称复刻整篇方法。
\subsection*{提交边界}
当前候选经历史复核后选择，不是独立盲测。固定2019池、财报修订、历史行业重构和缺失估值限制外推；复权连续单位尚不覆盖整手、完整分红现金、结算与真实盘口。原始缓存和Token不公开；文件桥不代表账户接通或下单。提交前填写组员姓名、学号并按教师要求命名；材料已准备，评分与课程提交状态由实际验收决定。
\end{document}
'''
    (OUT / 'final_report.tex').write_text(prefix + r'\begin{document}' + first + page2 + page3 + page4 + page5 + page6 + page7 + page8 + page9 + page10, encoding='utf-8', newline='\n')


def documents(qlong, bridge, chinese, seconds):
    text = f'''# V4 平台增强版验收映射

平台发行版：v2.2.0；材料版本：V4；策略仍为 V3 年度滚动 LightGBM + 波动预算。

| 课程 / 交付项 | 可检查位置 | 边界 |
|---|---|---|
| 数据质量与身份 | data、incremental、数据质量及SHA清单 | 原始厂商缓存不公开 |
| 因子与配置 | 原三因子、12因子卡、独立诊断页面 | 公告后可见；不使用未来财报 |
| 含费用回测 | engine、daily/trades/positions/orders | 连续复权单位，不等于实盘整手交易 |
| IC/Rank IC/分组 | 因子诊断、逐日有效人数 | 重叠标签不连乘为净值 |
| 策略比较 | analytics.compare_strategies、策略对比页面 | 前收盘起算；日期切片不是现金重启 |
| 风险分析 | 风险透镜、回撤恢复、月历、60日滚动 | 缺日拒绝；不保证未来风险上限 |
| 资金配置情景 | 风险透镜的现金页 | 静态缩放、零利息、未重算容量 |
| 研究追踪 | 实验档案、订单筛选、本地笔记 | 查看、回放及编辑笔记各自分开 |
| 外部接入 | Qlib/连接状态证据 | 以下状态以实际运行记录为准 |
| 提交材料 | reports/platform_v4 | 10页报告、20页幻灯片、源码、讲稿 |

Qlib：{qlong}

同花顺：{bridge}

讲稿：正文18页建议{seconds//60}分{seconds%60:02d}秒，2页备份；口述内容共{chinese:,}个汉字，标题逐页与 deck_content.json 一致。

研究结果仍为固定417日净收益25.13%、最大回撤9.88%；价格指数14.55%、全收益指数19.94%。当前候选经过历史结果筛选，不是独立盲测。

本清单是验收映射，不是评分证明。自动测试、交互截图、Qlib输出和提交包哈希由发布时的独立记录提供；不以作者生成材料的成功代替功能核验。
'''
    (ROOT / 'docs/ACCEPTANCE_V4.md').write_text(text, encoding='utf-8', newline='\n')
    (ROOT / 'docs/REPRODUCE_V4.md').write_text('''# V4 平台与材料复现

版本 v2.2.0 增强了工作台，保留 V3 策略及历史研究文件。先阅读 README 中的环境安装步骤。

1. 本机执行 `launch.cmd`，或在已安装环境运行 `streamlit run app.py`。
2. 无原始数据时，研究总览、策略对比和风险透镜读取 `evidence/research_v3/daily` 与 V2 日账本，可离线查看真实聚合结果。
3. 从策略对比选择版本、日期和基准，下载 CSV 与 methodology.json；筛选日期是已有路径切片。真实重新开仓应在回测实验中提交参数。
4. 风险透镜提供月历、回撤恢复、滚动风险及静态现金情景。查看计算假设，缺失值不等于零。
5. 独立12因子页面读取公开 evidence CSV，可离线查看定义、IC与日期诊断。真实行情及逐股成交明细需要对应本机私有数据/运行记录；缺少时按页面说明补足合法快照，不用合成数据冒充实证。
6. 运行 `python -m pytest -q -p no:cacheprovider`；合成示例入口为 `python -m cfquant.cli run --config configs/demo.yaml`。
7. Qlib在隔离环境安装，按接入文档转换本地数据并运行原生表达式。先确认状态证据，不将未验证的账户接口说成已接通。
8. 材料作者入口为 `python scripts/build_platform_materials.py --compile`；最终作者应传入真实Qlib状态、同花顺连接说明和已通过测试数量。无状态证据时材料只记录待验证。

材料位于 `reports/platform_v4`。`deck_content.json`同时提供Beamer、原生可编辑PowerPoint和逐页讲稿内容；PPTX由 `scripts/build_platform_presentation.mjs` 单独构建。使用报告和讲稿时保持同一V4版本。

原始私有数据、凭据和训练缓存不在公开仓库；真实训练需要本机合法快照。V3既有研究目录禁止覆盖，改变假设必须新建运行身份。平台不连接证券交易账户。
''', encoding='utf-8', newline='\n')


def compile_pdfs(only=None):
    env = os.environ.copy()
    rt = Path('D:/Downloads/MM-LaTeX/.miktex-runtime')
    env.update(MIKTEX_USERCONFIG=str(rt / 'config'), MIKTEX_USERDATA=str(rt / 'data'), MIKTEX_USERINSTALL='D:/MiTex')
    executable = shutil.which('xelatex') or 'D:/MiTex/miktex/bin/x64/xelatex.exe'
    for stem, final in [('final_report', 'CF2026_V4_Final_Report.pdf'), ('beamer', 'CF2026_V4_Beamer.pdf'),
                        ('speaker_script', 'CF2026_V4_Speaker_Script.pdf')]:
        if only is not None and stem not in only:
            continue
        build = ROOT / f'tmp/v4-latex/{stem}'
        build.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            process = subprocess.run([executable, '-interaction=nonstopmode', '-halt-on-error', f'-output-directory={build}', stem + '.tex'],
                                     cwd=OUT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (build / 'compile.txt').write_bytes(process.stdout)
            if process.returncode:
                raise RuntimeError(process.stdout.decode('utf-8', errors='replace')[-4000:])
        shutil.copy2(build / (stem + '.pdf'), OUT / final)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--qlib-status')
    parser.add_argument('--bridge-status')
    parser.add_argument('--test-count', type=int)
    args = parser.parse_args()
    main_tests = test_counts(ROOT / 'evidence/platform_v4/tests-final.xml')
    qlib_tests = test_counts(ROOT / 'evidence/qlib_bridge/test-results.xml')
    if main_tests:
        if args.test_count is not None and args.test_count != main_tests['passed']:
            raise ValueError('--test-count disagrees with the published JUnit evidence')
        args.test_count = main_tests['passed']
    args.main_skipped = main_tests['skipped'] if main_tests else 0
    args.qlib_passed = qlib_tests['passed'] if qlib_tests else 0
    FIG.mkdir(parents=True, exist_ok=True)
    analysis = figures()
    qshort, qlong, bridge, q = integration_status(args)
    deck = slides(qshort, qlong, bridge, q)
    if args.test_count:
        deck[16]['items'][1] = f'主环境{args.test_count}通过、{args.main_skipped}跳过；Qlib隔离{args.qlib_passed}通过'
    (OUT / 'deck_content.json').write_text(json.dumps(deck, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
    chinese, seconds = write_script(deck)
    write_beamer(deck)
    report(qlong, bridge, args)
    documents(qlong, bridge, chinese, seconds)
    receipt = {'release': 'v2.2.0', 'slides': len(deck), 'main_talk_seconds': seconds,
               'spoken_chinese_characters': chinese, 'qlib_evidence_passed': q.get('passed') is True,
               'main_tests': main_tests, 'qlib_tests': qlib_tests,
               'source_strategy': 'V3 rolling_lightgbm__managed',
               'first_day_return_included': True, 'net_return': float(analysis['summary'].loc['V3', 'total_return'])}
    (OUT / 'authoring_checks.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
    if args.compile:
        compile_pdfs()
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
