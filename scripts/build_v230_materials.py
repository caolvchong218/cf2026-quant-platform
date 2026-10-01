"""Author V230 report, twenty-slide narrative and matched speaking script.

This reuses the existing platform_v4 visual style and scientific figures.
It only writes sources/figures; PDF and PPTX compilation and visual acceptance
are separate operations. New numerical claims come from public V4 evidence.
Final test counts and UI acceptance are injected explicitly, never inherited.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/platform_v230"
FIG = OUT / "figures"
OLD = ROOT / "reports/platform_v4"
E = ROOT / "evidence/research_v4"
MEMBERS = "12412408 胡博文；12410819 谢明飞；12410436 王天希"
NAMES = {"fixed_whitebox": "固定经济白箱", "mined_rank": "筛选排名组合",
         "mined_ridge": "年度白箱Ridge", "blend": "主线/排名各50%", "mainline": "V3主线LightGBM"}
ORDER = ["fixed_whitebox", "mined_rank", "mined_ridge", "blend", "mainline"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(value, encoding="utf-8", newline="\n")


def tex(value):
    translations = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "_": r"\_",
                    "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}"}
    return "".join(translations.get(char, char) for char in str(value))


def pct(value):
    return f"{float(value)*100:.2f}%"


def inputs():
    table = pd.read_csv(E / "test.csv").set_index("candidate")
    validation = pd.read_csv(E / "validation.csv").set_index("candidate")
    stress = pd.read_csv(E / "stress.csv")
    protocol = read_json(E / "protocol.json")
    decision = read_json(E / "decision.json")
    if set(table.index) != set(ORDER) or table.index.has_duplicates:
        raise ValueError("Require exactly the five declared V4 candidates")
    if protocol["candidate_budget"] != 28 or protocol["llm_called"] or decision["llm_called"]:
        raise ValueError("This narrative requires the bounded 28-factor, offline research run")
    cards = read_json(E / "factor_cards.json")
    if len(cards) != 28 or len({card["id"] for card in cards}) != 28:
        raise ValueError("Duplicate or missing factor identities: regenerate evidence first")
    return table, validation, stress, protocol, decision


def figure_save(fig, name):
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{extension}", bbox_inches="tight", dpi=160)
    plt.close(fig)


def figures(table, stress):
    for name in ("nav", "legacy", "factor_ic", "v3_monthly", "v3_drawdown", "controls"):
        for extension in ("pdf", "png"):
            shutil.copy2(OLD / "figures" / f"{name}.{extension}", FIG / f"{name}.{extension}")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    labels = ["Fixed whitebox", "Mined rank", "Mined ridge", "50/50 targets", "V3 mainline"]
    x = np.arange(5)
    fig, ax = plt.subplots(figsize=(10, 3.1))
    ax.bar(x-.18, table.loc[ORDER, "total_return"], width=.36, label="Net total return", color="#168579")
    ax.bar(x+.18, table.loc[ORDER, "max_drawdown"], width=.36, label="Maximum drawdown", color="#BB7957")
    ax.axhline(table.loc["mainline", "benchmark_total_return"], color="#4265A6", ls="--", lw=1,
               label="CSI 300 total-return benchmark")
    ax.set_xticks(x, labels); ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(ncol=3, fontsize=8); ax.grid(axis="y", alpha=.2)
    ax.set_title("Five retained candidates | 2025-01-02 to 2026-09-18 | historical development")
    figure_save(fig, "v4_candidates")
    fig, ax = plt.subplots(figsize=(10, 3.4))
    palette = ["#8C99A6", "#BB7957", "#4265A6", "#8878A5", "#168579"]
    for name, label, color in zip(ORDER, labels, palette):
        daily = pd.read_csv(E / "daily" / (name + ".csv"), parse_dates=["date"])
        ax.plot(daily.date, daily.nav / 1e6, label=label, color=color, lw=1.6 if name == "mainline" else 1.1)
    curves = pd.read_csv(ROOT / "evidence/research_v3/curves.csv", parse_dates=["date"])
    for title, label, color in (("沪深300 · 价格指数", "CSI 300 price", "#49505C"),
                                 ("沪深300 · 全收益指数", "CSI 300 total return", "#4265A6")):
        part = curves[curves.series == title]
        ax.plot(part.date, part.nav, ls="--", lw=1, label=label, color=color)
    ax.legend(ncol=4, fontsize=8); ax.grid(axis="y", alpha=.2); ax.set_ylabel("Wealth / initial capital")
    ax.set_title("Same observed sessions and initial capital; actual costs included")
    figure_save(fig, "v4_curves")
    strict = stress[stress.scenario == "lot_min_fee_tax_slippage"].set_index("candidate")
    names = ["mainline", "mined_ridge", "blend"]
    fig, ax = plt.subplots(figsize=(10, 2.9)); x = np.arange(3)
    ax.bar(x-.18, table.loc[names, "total_return"], width=.36, color="#168579", label="Original reference fee policy")
    ax.bar(x+.18, strict.loc[names, "total_return"], width=.36, color="#4265A6", label="Lot / minimum fee / tax / slippage scenario")
    ax.set_xticks(x, ["V3 mainline", "Mined ridge", "50/50 targets"])
    ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.2)
    ax.set_title("Execution stress changes fills and fee assumptions; differences are not solely lot effects")
    figure_save(fig, "strict_stress")


def deck_content(table, validation, stress, protocol, decision, test_text, ui_status):
    deck = deepcopy(read_json(OLD / "deck_content.json"))
    for slide in deck:
        slide.pop("screenshot", None)
        slide["kind"] = slide.get("kind", "content")
        slide["source"] = "来源：本项目 evidence/research_v3、evidence/research_v4；固定历史快照，完整口径见提交报告。"
        for series in slide.get("chart", {}).get("series", []):
            if series.get("name") == "V3最新候选 · 扣费":
                series["name"] = "V3主线LightGBM · 扣费"
    main = table.loc["mainline"]
    ridge = table.loc["mined_ridge"]
    blend = table.loc["blend"]
    strict = stress[stress.scenario == "lot_min_fee_tax_slippage"].set_index("candidate")
    duration = [40,50,60,55,65,75,70,80,65,55,75,85,60,65,80,105,60,55,0,0]
    def replace_slide(index, title, claim, items, notes, **extra):
        deck[index] = {"title": title, "claim": claim, "items": items, "notes": notes,
                       "seconds": duration[index], "source": deck[index]["source"], "kind": "content", **extra}
    replace_slide(0, "青序量化研究平台", "课程主线保留，辅助研究有独立来源与边界",
        ["Computational Finance · Project 1 · V230 / v2.3.0", "1,000只股票 · 原12经济因子 + 16个DSL公式", MEMBERS],
        "各位老师、同学好。我们展示青序量化研究平台。本项目把数据、因子、含费用回测和诊断连成一个可继续维护的研究工作台。本次更新保留年度LightGBM主线，同时加入白箱因子研究、成交压力、资金归因和冻结后的观察。我们会先解释课程核心，再展示同区间实际结果，最后介绍哪些新功能已经产生证据、哪些只是接口准备。历史收益不是未来保证，语言模型建议也不是计算结果。今天所有数字来自保留的真实聚合账本，合成示例只用于软件流程。", kind="cover", date="2026年10月1日")
    replace_slide(1, "课程要求与交付范围", "调整价格、三因子、费用账本、IC与分组仍是核心",
        ["原动量/反转/低波动入口，保留12经济因子", "含费用回测：订单、成交、现金、持仓与独立重建", "IC/Rank IC/五组诊断与可配置窗口", "扩展：28候选公式、白箱模型、压力与冻结观察"],
        "我们先把课程要求理解为完整研究流程。数据必须有来源和复权规则，因子必须说明定义、方向和信息时点；回测必须把费用落实到真实模拟成交，最后用IC、分组和风险指标检查。课程至少要求三个因子，我们保留动量、反转、低波动，再维护十二个经济因子。本次新增十六个受约束公式，与原因子合成二十八个独立候选身份。报告并不只介绍新按钮，还保留原始失败、主线训练、现金账本和完整结果。材料包括报告、二十页演示、逐页讲稿及可维护源码。")
    deck[2]["notes"] += "本次白箱路线输出同样的分数和目标权重，因此比较时共用原成交账本；严格成交压力使用另一个明确政策，原账本保留。语言模型只能提出待验证公式，不能跳过评价器直接生成可信收益。"
    deck[2]["seconds"] = duration[2]
    deck[3]["seconds"] = duration[3]
    replace_slide(4, "经济因子与28个公式候选", "公式先解释，计算和筛选后才产生数值证据",
        ["12经济因子 + 16个dsl_命名的量价公式", "白名单：历史滞后、滚动统计、横截面秩与安全除法", "不执行任意Python；所有公式用相同评价流程", "例：低波动Rank IC +0.072；跳月动量 −0.016"],
        "原十二因子覆盖趋势、反转、风险、量价、价值、分红和盈利质量。诊断仍保留原方向：低波动的平均Rank IC约零点零七二，跳月动量约负零点零一六，不因为最后区间表现就改符号。新公式语言只允许历史滞后、滚动均值与波动、相关性、横截面秩和安全除法。例如二十日动量除以二十日波动率，是一个风险调整趋势假设。十六个新公式统一使用dsl前缀，与原经济因子区分。由此得到二十八个候选身份，避免重名覆盖原定义。IC标签属于诊断输入，不能倒流成形成日特征；重叠二十日标签也不是独立交易净值。",
        formula=r"f_{i,t}=\frac{C_{i,t}/C_{i,t-20}-1}{s_{20}(r_i)}",
        formula_plain="示例：过去20日动量 ÷ 过去20日收益波动")
    replace_slide(5, "年度模型与严格标签边界", "此前三年训练，label_exit严格早于形成日",
        ["下一开盘进入，20交易日后开盘退出的收益秩", "每年更新一次：特征日与标签退出日均早于形成日", "主线：12标准化因子 + 12行业/市值残差", "辅助：训练期先选6因子，年度排名/Ridge重估"],
        "模型每年更新一次，只使用形成日前三年已经可见的样本。标签从信号日的下一开盘进入，再过二十个交易日在开盘退出，因此特征日期在新年之前还不够，标签的退出日期也必须严格早于模型形成日。每日截尾、标准化和行业市值中性化都只读取当天横截面。主线LightGBM保留原参数和十二个标准化值、十二个残差值。白箱路线在二零二三之前筛定最多六个因子，此后只按年度重新估计方向或Ridge系数，alpha固定一百。二零二三到二零二四用于预选辅助策略，之后的已观察区间只能叫历史开发对照。", kind="timeline")
    replace_slide(6, "组合、现金与执行政策", "先形成目标，再由约束决定实际成交",
        ["原主线：50股、20日调仓、20名缓冲、允许现金", "最高95%预算；单股4%、行业25%为形成日目标", "基础账本：买10bp/卖15bp；前20日成交额1%", "严格压力：原股等价整手、T+1、最低佣金、税与滑点"],
        "有了分数以后，先按原规则构建五十股目标，每二十日调仓，保留二十名排名缓冲。最高股票预算百分之九十五，单股目标百分之四、行业目标百分之二十五；波动预算和趋势折减都截至形成日。下一开盘成交时，涨跌停、缺价、现金不足与此前成交额上限会改变实际买卖。原账本仍按买十、卖十五个基点收费。新压力路线保持复权单位会计，用当日复权开盘价与原始开盘价的比值换算原股等价股数，整手买入，完整退出可处理零股，并限制当天新买单位出售。它确实改变成交，但没有完整模拟企业行动现金、清算或盘口。",
        formula=r"NAV_t=Cash_t+\sum_i u_{i,t}P^{mark}_{i,t}",
        formula_plain="净值 = 实际现金 + 持仓单位 × 当日可用估值")
    deck[7]["seconds"] = duration[7]
    deck[7]["notes"] += "本轮辅助路线还保留验证期预选文件；它没有因为新历史冠军而替代主线。"
    deck[8]["seconds"] = duration[8]
    replace_slide(9, "费用压力与实际成交约束", "严格情景改变费率和成交，须分别解释",
        [f"主线严格情景收益 {pct(strict.loc['mainline','total_return'])}",
         f"白箱Ridge {pct(strict.loc['mined_ridge','total_return'])}；组合 {pct(strict.loc['blend','total_return'])}",
         "与基础费率不同，差额不能全归因于整手", "双倍基线费用、原延迟与连续运行结果保留"],
        f"压力测试要完整重跑成交，不能简单把成本加回净值。原主线双倍费用收益约百分之二十二点四八，延迟一天约百分之二十七点三一，这些原结果保留。新严格情景下，主线收益{pct(strict.loc['mainline','total_return'])}，白箱Ridge为{pct(strict.loc['mined_ridge','total_return'])}，固定权重组合为{pct(strict.loc['blend','total_return'])}。严格路线另设最低佣金、买卖佣金、日期税率和滑点现金成本，所以相对原账本的收益差并不单独代表整手效应。它仍然是复权收益研究，原股等价整手和T加一约束不能被称为完整实盘结算。",
        figure="strict_stress", chart={"type":"bar","categories":["主线","白箱Ridge","50/50组合"],"percent_points":True,
        "series":[{"name":"原基线","values":[float(table.loc[n,"total_return"]*100) for n in ["mainline","mined_ridge","blend"]]},
                  {"name":"严格情景","values":[float(strict.loc[n,"total_return"]*100) for n in ["mainline","mined_ridge","blend"]]}]})
    replace_slide(10, "白箱因子研发与模型辅助", "建议、DSL校验、实测评价各自留下来源",
        ["训练期覆盖/Rank IC/冗余筛选，预算最多6因子", "固定白箱、排名组合、年度Ridge分别留结果", "OpenAI兼容/Ollama接口；离线解释明确标识", "本次真实研究 llm_called=false，未发生模型请求"],
        "因子实验室把研究分成三个步骤。先提出可以解释的公式，经过白名单校验后计算，再按训练期覆盖、有效日期、Rank IC强度和相关性筛选，最多保留六个因子。固定经济白箱、筛选后的排名组合与年度Ridge分别回测，不能只保留一个最好结果。模型辅助层支持兼容Chat Completions的服务或本机Ollama，但本次真实研究没有发起语言模型请求，协议明确记录llm_called为false。离线解释是确定性文本，服务返回也标为未核验建议。只有评价器重新计算得到的IC、净值和成本才进入研究证据，不能把模型说的预期收益放进成绩表。")
    replace_slide(11, "五个候选的完整历史对照", "白箱预选保留为辅助，主线继续保留",
        [f"2023–2024按夏普预选白箱Ridge：{validation.loc['mined_ridge','sharpe']:.3f}",
         f"2025起：Ridge {pct(ridge.total_return)}；50/50组合 {pct(blend.total_return)}",
         f"主线 {pct(main.total_return)}；全收益指数 {pct(main.benchmark_total_return)}",
         "全部五候选保留；历史开发对照不能改称未知盲测"],
        f"这页列出本轮全部五个候选，而不是只展示赢家。相同四百一十七日内，固定经济白箱收益{pct(table.loc['fixed_whitebox','total_return'])}，筛选排名收益{pct(table.loc['mined_rank','total_return'])}，年度白箱Ridge收益{pct(ridge.total_return)}，主线与排名目标各占一半的组合收益{pct(blend.total_return)}，主线仍为{pct(main.total_return)}。白箱Ridge是在二零二三到二零二四的验证夏普上预选的，但之后没有超过百分之十九点九四的全收益指数。五十对五十组合略超过全收益基准，也仍弱于主线。结论是这些方法值得继续观察，尚无证据支持替换默认策略。所有已观察结果都属于历史开发比较。",
        figure="v4_candidates", chart={"type":"bar","categories":["固定白箱","排名","Ridge","50/50","主线"],"percent_points":True,
        "series":[{"name":"扣费收益","values":[float(table.loc[n,"total_return"]*100) for n in ORDER]},
                  {"name":"最大回撤","values":[float(table.loc[n,"max_drawdown"]*100) for n in ORDER]}]})
    replace_slide(12, "风险分析与资金账本归因", "逐股现金流加末日持仓，贡献核对组合净损益",
        ["卖出现金流 − 买入现金流 − 实际成本 + 期末价值", "完全退出股票仍保留损益；成本分项有据才显示", "行业聚合需要明确标签；不能称因果因子归因", "月历、回撤恢复、滚动风险与同区间比较继续保留"],
        "收益归因首先解决账本问题。每只股票的损益等于卖出现金流，减去买入现金流和实际成本，再加最后一天的持仓价值；完全退出的股票也不能被丢掉。各股损益之和应等于末日组合净值减初始资金。新账本记录佣金、税与滑点时可以分别汇总，原账本没有记录的分项保持未知。行业归因只按明确提供的标签汇总，不证明某行业或因子造成收益。已有风险透镜继续展示月收益、回撤谷底、恢复和六十日滚动统计。日期切片使用此前净值保留首日损益，未恢复回撤保持开放状态，不能把缺失日期补成已经恢复。",
        formula=r"\Pi_i=S_i-B_i-C_i+V_{i,T},\qquad \sum_i\Pi_i=NAV_T-NAV_0",
        formula_plain="逐股损益相加 = 期末组合净值 − 初始资金")
    replace_slide(13, "冻结计划与前向观察", "冻结后记录可以复查，未来表现仍待积累",
        ["冻结策略/公式/系数/配置/代码/数据身份与截止日", "只追加截止日之后的日期；signal_asof严格早于成交", "同日重试幂等；冲突、改写及链身份错误拒绝", "历史补录与前向纸面记录分别标识，非实盘证明"],
        "当前最有价值的下一步，是先冻结规则再积累新数据。冻结文件应保留策略、公式、年度系数、训练和执行配置、代码与数据身份，以及最后已知交易日。观察只允许追加截止日之后的日期，信号形成日必须早于成交日；同样内容的重复登记返回已有记录，冲突改写被拒绝，既有记录有哈希链。冻结之前已经发生的日期，即使后来才录入，也只属于历史补录。冻结以后登记的内容首先是纸面观察；本地时钟、人工净值和哈希不能证明真实信息获取时点或账户成交。我们不把历史净值填入前向成绩，更不提前声称前向验证成功。")
    replace_slide(14, "外部接入与尚未实测的扩展", "接口状态、原生计算和交易授权分别报告",
        ["Qlib 0.9.7：8表达式核对；Alpha158配置实测", "同花顺/SuperMind：文件桥，账号/API未连接", "指数增强：缺历史成分权重，尚无真实增强实测", "1.4.1的主要历史改动是启动修复，不能包装为策略提升"],
        "外部工具接入需要区分状态。此前Qlib零点九点七在独立环境读取二十股数据，八个表达式与pandas逐项核对，还运行Alpha158的一百五十八项配置；这验证因子接口，没有训练Qlib策略，也没有替换主账本。同花顺与SuperMind提供文件桥，支持观察权重导出和模拟成交CSV核验，但账号和API未连接，没有提交订单。本次指数增强组件也保留明确门槛：没有真实历史成分权重时，不发布增强收益，不把手工样例当指数实测。历史发行一点四点一的主要用途是修复启动；启动正常、增加页面和策略优越性是不同证据，需要分别说明。")
    replace_slide(15, "课堂演示：沿证据走一遍", "公开账本可离线查看，真实重跑依赖合法快照",
        ["总览 → 主线与五候选同区间比较", "因子实验室 → 公式/来源/训练筛选/离线模型状态", "成交压力 → 资金归因 → 冻结观察的待验证状态", "实验档案查看订单与参数；不现场等待年度重训"],
        "下面预留一分四十五秒演示。先在总览确认主线与快照截止日，打开同区间对比，说明候选使用相同日期和费用账本。在因子实验室查看公式、训练筛选来源和离线模型状态。接着查看严格政策，说明原股等价整手与复权单位仍是研究近似；打开资金归因核对损益，展示冻结计划和等待新增数据的状态。实验档案保留订单、参数和失败结果。现场用预先准备的数据；环境不可用时展示同版本PDF及证据CSV，不使用未核验截图，也不等待下载或重训。", kind="demo")
    replace_slide(16, "验证、来源与复现边界", "软件核验与未来市场表现分别成立",
        [test_text, "界面核验：" + ui_status, "主线及V4各自保存协议、输入/源码身份和核对", "无凭据读公开聚合结果；合法快照用于训练与逐股查询"],
        "复现时先区分四件事：源码已写好，研究已产生真实输出，软件测试通过，以及界面实际完成交互。它们不能互相替代。这里列出的最终测试与界面状态都对应封存证据，检查范围以实际记录为准。独立账本从实际成交成本重建现金、单位、每日净值和收益；上游还有标签退出边界、公式前缀因果性、缺价和幂等冲突检查。公开证据支持离线查看聚合结果；真实训练及逐股行情需要合法快照。改变假设就保存新的实验身份，修正前旧结果也保留归档。工程检查不会证明未来市场一定有效。")
    replace_slide(17, "结论：保留主线，继续积累证据", "辅助路线更可解释，仍需新数据验证",
        [f"主线扣费 {pct(main.total_return)}；历史最大回撤 {pct(main.max_drawdown)}", "白箱/组合全部结果公开，尚不足以替代主线", "真实LLM服务未调用；指数增强缺权重未实测", "下一步：冻结规则、追加新日、同口径核验"],
        "最后回到课程目标。我们保留了调整价格、明确因子、含费用成交和独立诊断，也把这些环节放进可以继续维护的工作台。主线在固定历史区间取得百分之二十五点一三净收益，但经过历史复核选择，不能说成未知盲测成功。新的白箱与组合给出了更透明的解释和另一条研究路线，完整对照尚不足以取代主线。语言模型接口没有在本轮真实调用，指数增强也没有缺权重时的假实测。下一步应冻结参数，积累真正新增的数据，并继续用相同账本、成本和基准检查。谢谢各位，欢迎针对公式、训练边界和成交记录提问。")
    deck[18]["seconds"] = 0
    replace_slide(19, "备份：答辩问题与证据边界", "可解释、工程可用与策略有效性要分别回答",
        ["为何不选新白箱？验证预选后仍未胜全收益基准", "模型是否参与实测？没有；离线确定性解释已标识", "整手/T+1是否实盘？原股等价约束，未完整企业行动结算", "指数增强是否已跑？缺历史真实权重，门槛尚未满足"],
        "如果问新白箱是不是更好，我们回答验证预选的Ridge在后续历史只有百分之十八点三七，低于全收益基准和主线，所以继续保留辅助身份。如果问语言模型是否生成了这次成绩，回答没有，本轮协议记录未调用，所有分数和指标来自代码评价。若追问整手与T加一，说明它们限制实际模拟成交，但会计仍为复权单位，未完整处理分红现金与清算。若问指数增强，真实历史权重缺失时不会给出收益。原有负收益、预选失败和落后年份仍在档案中，冻结与新数据观察是下一步，而不是用接口准备替代验证。")
    for index, slide in enumerate(deck):
        slide["seconds"] = duration[index]
    assert len(deck) == 20 and sum(slide["seconds"] for slide in deck[:18]) == 1200
    return deck


def write_beamer(deck):
    preamble = (OLD / "beamer.tex").read_text(encoding="utf-8").split(r"\begin{document}")[0]
    preamble = preamble.replace("Project 1 / V4", "Project 1 / V230")
    frames = []
    for slide in deck:
        body = r"\begin{frame}{" + tex(slide["title"]) + "}\n"
        body += r"\begin{block}{}" + tex(slide["claim"]) + r"\end{block}" + "\n"
        if slide.get("kind") == "cover":
            body += r"\vspace{4mm}{\Huge\bfseries\color{navy}量化研究与证据验证}\par\vspace{4mm}" + "\n"
        if slide.get("kind") == "flow":
            body += r"\begin{center}\begin{tikzpicture}[node distance=3mm,every node/.style={draw=teal,fill=teal!5,rounded corners,align=center,text width=27mm,minimum height=20mm,font=\small}]" + "\n"
            body += r"\node(a){数据快照\\字段与身份};\node(b)[right=of a]{因子与模型\\统一分数};\node(c)[right=of b]{组合与风险\\目标和现金};\node(d)[right=of c]{成交与分析\\账本与证据};" + "\n"
            body += r"\draw[-{Stealth},teal](a)--(b);\draw[-{Stealth},teal](b)--(c);\draw[-{Stealth},teal](c)--(d);\end{tikzpicture}\end{center}" + "\n"
        visual = bool(slide.get("figure"))
        if visual:
            body += r"\centering\includegraphics[width=.96\textwidth,height=.47\textheight,keepaspectratio]{figures/" + slide["figure"] + r".pdf}\\[1mm]" + "\n"
        if slide.get("table"):
            body += r"\begin{center}\small\begin{tabularx}{.97\textwidth}{lX}\toprule" + "\n"
            for index, row in enumerate(slide["table"]):
                body += " & ".join(tex(cell) for cell in row) + r"\\" + (r"\midrule" if index == 0 else "") + "\n"
            body += r"\bottomrule\end{tabularx}\end{center}" + "\n"
        if slide.get("formula"):
            body += r"\[" + slide["formula"] + r"\]" + "\n"
        if slide["items"] and slide.get("kind") != "flow":
            body += (r"\scriptsize" if visual else r"\small") + r"\begin{itemize}" + "\n"
            body += "".join(r"\item " + tex(item).replace("\n", r"\quad ") + "\n" for item in slide["items"])
            body += r"\end{itemize}" + "\n"
        body += r"\note{" + tex(slide["notes"]) + "}\n" + r"\end{frame}" + "\n"
        frames.append(body)
    source = preamble + r"\begin{document}" + "\n" + "\n".join(frames) + r"\end{document}" + "\n"
    if len(re.findall(r"(?<!\\)\\\[", source)) != len(re.findall(r"(?<!\\)\\\]", source)):
        raise AssertionError("Unbalanced display-math delimiters in generated Beamer")
    write(OUT / "beamer.tex", source)


def write_script(deck):
    main_seconds = sum(slide["seconds"] for slide in deck[:18])
    characters = sum(len(re.findall(r"[\u4e00-\u9fff]", slide["notes"])) for slide in deck)
    preamble = (OLD / "speaker_script.tex").read_text(encoding="utf-8").split(r"\begin{document}")[0]
    preamble = preamble.replace("平台 V4", "平台 V230").replace("正文19分05秒", "正文20分钟")
    md = ["# 青序量化研究平台 V230：逐页演讲稿", "", MEMBERS, "",
          "对应20页Beamer / 可编辑PPTX。正文1–18页建议20分钟；19–20页为答辩备份。",
          f"口述正文及备份共{characters:,}个汉字。方括号内容是操作提示，不朗读。", "",
          "本稿不证明界面已完成验收。演示前按最终记录核对页面，准备相同版本PDF和证据CSV离线备份。", ""]
    parts = [preamble, r"\begin{document}", r"{\Large\bfseries\color{ink}课堂演示口述稿}\par", tex(MEMBERS),
             "每两页幻灯片对应一页讲稿；正文20分钟，另有两页答辩备份。操作提示不朗读。"]
    elapsed = 0
    for index, slide in enumerate(deck, 1):
        if index > 1 and index % 2:
            parts.append(r"\newpage")
        timing = f"建议{slide['seconds']}秒；从{elapsed//60:02d}:{elapsed%60:02d}开始。" if index <= 18 else "答辩备份，不计入正文时长。"
        md += [f"## {index}. {slide['title']}", "", timing, ""]
        parts += [r"\section*{\color{ink}" + tex(f"{index:02d}  {slide['title']}") + "}",
                  r"{\small\color{accent}" + tex(timing) + r"}\par"]
        if index == 16:
            action = "按最终验收状态演示：总览→候选对比→公式与离线状态→压力/归因→冻结观察；环境不可用则看证据CSV。"
            md += ["[演示动作：" + action + "]", ""]
            parts += [r"{\small\color{accent}[演示动作：" + tex(action) + r"]}\par"]
        md += [slide["notes"], ""]
        parts.append(tex(slide["notes"]) + "\n")
        if index % 2:
            parts.append(r"\vspace{4mm}\noindent\textcolor{accent!40}{\rule{\textwidth}{0.4pt}}\par")
        elapsed += slide["seconds"]
    parts.append(r"\end{document}")
    write(OUT / "speaker_script.tex", "\n\n".join(parts))
    write(ROOT / "docs/PRESENTATION_SCRIPT_V230.md", "\n".join(md))
    return characters, main_seconds


def latex_table(headers, rows, specification):
    result = r"\begin{center}\small\begin{tabular}{" + specification + r"}\toprule" + "\n"
    result += " & ".join(tex(cell) for cell in headers) + r"\\\midrule" + "\n"
    result += "\n".join(" & ".join(tex(cell) for cell in row) + r"\\" for row in rows)
    return result + "\n" + r"\bottomrule\end{tabular}\end{center}" + "\n"


def report(table, validation, stress, protocol, decision, test_text, ui_status):
    original = (OLD / "final_report.tex").read_text(encoding="utf-8")
    preamble = original.split(r"\begin{document}")[0].replace("平台 V4 / v2.2.0 / 2026-09-22", "平台 V230 / v2.3.0 / 2026-10-01")
    preamble = preamble.replace(r"\definecolor{accent}{HTML}{168579}", r"\definecolor{accent}{HTML}{168579}\definecolor{teal}{HTML}{168579}")
    main = table.loc["mainline"]
    sections = []
    def page(title, body):
        sections.append((r"\section*{" if not sections else r"\page{") + title + "}\n" + body)
    page("1\\quad 目标、成员与主要结论", r"""
\begin{center}{\small\color{accent}COMPUTATIONAL FINANCE / PROJECT 1}\\[6pt]
{\Huge\bfseries\color{ink}青序量化研究平台}\\[6pt]
{\Large 课程主线、白箱研究与证据验证}\\[6pt]
2026年10月1日\quad V230 / v2.3.0\quad 完整提交报告
\end{center}
""" + tex(MEMBERS) + r"""
\subsection*{研究目标与可交付范围}
以可维护的Python平台完成调整价格获取、可配置因子、含实际成交成本的回测、IC/Rank IC与分组诊断、风险分析和复现。保留原动量、反转、低波动入口和12个经济因子；本次增加16个受限DSL公式，形成28个独立候选身份，以及白箱模型、成交压力、资金归因和冻结观察。
\subsection*{默认主线的固定历史结果}
""" + latex_table(["指标", "结果"], [
        ["区间 / 交易日", "2025-01-02—2026-09-18 / 417日"],
        ["扣费净收益 / 年化", pct(main.total_return) + " / " + pct(main.annualized_return)],
        ["夏普 / 最大回撤", f"{main.sharpe:.3f} / " + pct(main.max_drawdown)],
        ["沪深300价格 / 全收益指数", pct(main.benchmark_return) + " / " + pct(main.benchmark_total_return)],
        ["超过价格 / 全收益指数", f"{main.excess_return_pp*100:.2f} / {main.excess_vs_total_return*100:.2f}个百分点"],
        ["实际基础费用", f"{main.total_cost:,.2f}元"]], "ll") + r"""
\lead{本次辅助候选不足以替代年度LightGBM主线。}
白箱Ridge按2023--2024验证夏普预选；随后历史收益低于全收益指数和主线。50/50目标组合略超过全收益指数，但仍弱于主线。所有五候选保留，不因最终区间冠军重写预选。当前区间已在前期研究中观察，全部属于历史开发复核，\textbf{不是独立未知盲测，也不保证未来收益}。
\subsection*{状态与责任}
本次真实研究未调用语言模型；指数增强缺真实历史权重，尚未实测。软件源码、真实研究输出、最终测试与界面交互分别留证；软件检查与历史研究成绩都不能替代新增数据上的前向验证。
""")
    page("2\\quad 软件结构、课程要求与发行边界", r"""
\lead{复用同一套分数、组合和资金账本，新增模块保留独立政策。}
\begin{center}\begin{tikzpicture}[node distance=4mm,every node/.style={draw=accent,fill=accent!5,rounded corners,align=center,text width=32mm,minimum height=15mm,font=\small}]
\node(a){数据快照\\质量与身份};\node(b)[right=of a]{因子/模型\\日期$\times$资产分数};\node(c)[right=of b]{组合/风险\\目标权重和现金};\node(d)[right=of c]{成交/分析\\账本与证据};
\draw[-{Stealth},accent](a)--(b);\draw[-{Stealth},accent](b)--(c);\draw[-{Stealth},accent](c)--(d);\end{tikzpicture}\end{center}
\begin{center}\small\begin{tabularx}{\textwidth}{lX}\toprule
课程项 & 实现与可检查位置\\\midrule
调整价格与质量 & data/incremental；原价、复权、完整交易日、缺失与快照哈希。\\
至少三因子 & 原三个注册因子及12经济因子；定义、窗口、方向和缺失处理。\\
含费用回测 & engine与四张账本；限价、现金、缺价与滞后容量，成交才收费。\\
IC/Rank IC与分组 & 逐日横截面、有效人数、先形成五组再匹配未来标签。\\
收益/风险/成本 & 年化、样本夏普、最大回撤、双边换手、实际费用。\\
扩展及可复现 & 中性化、风险预算、Qlib接口、DSL与LLM建议边界、冻结观察。\\
报告与源码 & 10页报告结构、20页演示、逐页讲稿、固定依赖与源码。\\\bottomrule
\end{tabularx}\end{center}
\subsection*{本次新增模块}
\texttt{factor\_lab}校验有限表达式，不执行任意Python；\texttt{research\_lab}记录训练筛选、年度白箱与完整候选；\texttt{llm\_research}区分离线说明和未核验模型文本；\texttt{execution\_constraints}独立运行严格政策；\texttt{return\_attribution}重建资金损益；\texttt{forward\_observer}保留冻结身份与追加记录。界面与CLI复用核心，每次重跑保存独立实验，不改旧账本。
\subsection*{公开查看与本地重跑}
公开日账本支持同区间比较、风险分析和证据导出；真实训练、逐股行情与完整成交需要合法本地快照。目录册过滤解析到项目根外的junction，用公开聚合回退；本地V4重跑按显式数据来源定位输入。纠正前和失败目录归档保留，默认不混入当前结果。
\subsection*{发行号不代表策略收益提升}
材料V230及软件v2.3.0承载本次辅助研究。历史1.4.1的主要工作是启动修复：正常启动、提供研究工具与策略战胜基准是不同证据，不能把启动修复描述为实证收益提升。
""")
    # Reuse the verified source/units page, without transferring old test claims.
    old_pages = re.split(r"(?=\\page\{)", original)
    data_page = old_pages[2]
    data_page = data_page.replace(
        "2019-12-31按当日成交额取符合主板代码、价格至少3元且有成交的前1,000只，并列按代码。没有使用今日仍上市名单，后来停止报价的原成员也保留。该池不包括2020年以后IPO，单日流动性选择偏向当时热点，结论不代表全A股或沪深300。",
        "2019-12-31按成交额选主板、价格至少3元且有成交的前1,000只，并列按代码；停报原成员保留，不按今日上市名单重选。该池缺2020年后IPO且偏向当时热点，结论不代表全A股或沪深300。")
    data_page = data_page.replace(
        r"所有OHLC用\(P^{adj}_{i,t}=P^{raw}_{i,t}A_{i,t}/A_{i,ref}\)，每资产固定首个有效参考因子。原行情及V2研究缓存合计约1.63GB，包括原始缓存、标准数据、特征和模型分数。原始请求记录字段、日期与UTC下载时间，保存SHA-256；同请求可断点续拉。",
        r"OHLC统一用\(P^{adj}_{i,t}=P^{raw}_{i,t}A_{i,t}/A_{i,ref}\)，参考因子固定为各股首个有效值。原行情及V2缓存约1.63GB，含原始/标准数据、特征和模型分数。请求留字段、日期、UTC下载时间与SHA-256，可断点续拉。")
    data_page = data_page.replace(
        r"历史行业未知比例约0.52\%，年度ROE平均覆盖约98.90\%（按每日合资格样本计算）。采用公告时点仍不能消除厂商事后修订，当前数据不是完整的逐版本财报库。保留此局限，不用“无未来函数”概括所有数据风险。",
        r"历史行业未知约0.52\%，年度ROE覆盖约98.90\%（每日合资格样本）。公告时点不消除厂商事后修订；数据非完整逐版本财报库，不能以“无未来函数”概括风险。")
    data_page = data_page.replace(
        "新增基准由Tushare指数目录确认H00300.CSI，再取全收益日线；另取510330.SH基金日线和复权因子，各1,690日。价格指数、全收益指数、ETF代理分开命名，未猜测或替代指数代码。",
        "Tushare指数目录确认H00300.CSI全收益日线；另取510330.SH基金日线及复权因子，各1,690日。价格、全收益与ETF代理分开命名，不猜测或替代指数代码。")
    sections.append(data_page)
    page("4\\quad 因子定义、DSL与实际诊断", r"""
\lead{经济方向与公式身份固定；未来收益只在诊断和已实现训练标签中使用。}
""" + latex_table(["类别", "12经济因子"], [
        ["趋势与反转", "跳月动量、5日反转、均线趋势"],
        ["风险与量价", "低波动、低振幅、量能变化、流动性、低换手"],
        ["价值与分红", "盈利收益率、账面市值比、股息率"], ["质量", "最新已公告年度ROE"]], "ll") + r"""
记$C,H,L,V$为统一尺度价格与股数，$r_t=C_t/C_{t-1}-1$。代表式为跳月动量$C_{t-20}/C_{t-60}-1$、反转$-(C_t/C_{t-5}-1)$、低波动$-s_{20}(r)$、低振幅$-\overline{(H-L)/C}_{20}$，价值为正PE/PB的倒数，财报按公告后严格滞后使用。窗口不完整、估值非正或公告冲突时缺失，不后向补值。
\subsection*{横截面评价保留原方向}
先按当日分数形成五组，再匹配下一开盘至20日后开盘收益。原研究代表结果如下，完整逐日样本人数及12因子结果见公开证据。
""" + latex_table(["因子", "平均IC", "Rank IC", "Q5−Q1平均20日收益"], [
        ["跳月动量", "−0.0035", "−0.0161", "−0.0831%"],
        ["5日反转", "+0.0176", "+0.0187", "+0.3264%"],
        ["20日低波动", "+0.0378", "+0.0722", "+0.6377%"]], "lrrr") + r"""
负结果保留；重叠20日收益标签不可当成相互独立样本，也不可连乘成可交易净值。诊断不使用未来有效标签反向决定形成分组。
\subsection*{28个有唯一身份的公式候选}
12原经济因子加16个量价DSL种子：动量、均价反转、波动变化、相对量能、K线压力、价量相关及风险调整趋势等。新公式统一使用\texttt{dsl\_}前缀，避免覆盖低波动、低振幅等原经济定义。候选清单、表达式和来源均保存，程序对重复ID拒绝，不把同名替换隐匿在结果中。
\[
f_{i,t}=\frac{C_{i,t}/C_{i,t-20}-1}{s_{20}(r_i)}\qquad\text{（风险调整趋势示例）}.
\]
白名单允许正向历史滞后、滚动均值/标准差/相关性、横截面秩和安全除法，禁止未来移位及任意代码执行。参考CogAlpha的可检查量价公式表示，\textbf{本次未调用LLM、未运行其完整演化搜索方法}。所有公式进入统一计算和筛选器后才产生IC等证据。
""")
    folds = read_json(E / "folds.json")
    page("5\\quad 中性化、年度训练与有限候选选择", r"""
\lead{此前三年训练；标签退出日严格早于模型形成日。}
按日合资格横截面1\%/99\%截尾并样本标准化，再回归当时行业和对数市值，取残差缩放。财报使用公告后数据，历史行业按有效区间；厂商修订仍是数据局限。
\[
z_{i,t}=\sum_g\alpha_{g,t}\mathbf1\{g_{i,t}=g\}+\beta_t\log MV_{i,t}+\epsilon_{i,t},\quad n_{i,t}=\epsilon_{i,t}/s_t(\epsilon).
\]
标签$y_t=O_{t+21}/O_{t+1}-1$为下一开盘至20交易日后开盘收益秩；年末形成新模型，训练此前3年且要求特征日、\texttt{label\_exit}均严格早于形成日。随后年度固定，到下一年才更新。
\subsection*{主线保持原模型与参数}
年度LightGBM输入12个标准化值加12个中性化残差；250树、学习率0.03、15叶、最大深度5、最小叶500、L2=10、列采样0.8、种子20260921。原8个候选、Ridge基线和模型融合全部保留，不按本次辅助成绩反写主线。
\subsection*{白箱路线的训练与筛选}
28因子仅在2023之前已实现的训练标签上筛选；覆盖至少80\%、至少60个有效日期，按绝对平均Rank IC及ID排序，相关阈值0.75排除冗余，最多6因子。选定身份此后冻结，方向和Ridge系数按年度此前3年重估，Ridge的alpha=100。下表为本次白箱实际训练边界。
""" + latex_table(["年度", "形成日", "训练行数", "最大标签退出日"], [
        [f["year"], f["cutoff"], f"{f['train_rows']:,}", f["max_label_exit"]] for f in folds], "lrrr") + r"""
固定经济白箱、筛选排名组合和年度Ridge先在2023--2024按夏普预选；ID用于并列规则。之后才输出已观察2025起历史比较，不用最终区间重选辅助冠军。与主线各50\%的目标组合是固定研究对照，经相同成交引擎重新记账，不平均净值曲线冒充组合。长期固定池与已观察区间使本次仍属于历史开发，非独立盲测。
""")
    page("6\\quad 组合规则、严格执行与独立账本", r"""
\lead{费用只来自实际成交；约束作用在成交数量上。}
主线与辅助基础比较每20交易日调仓、50股、20名缓冲、逆波动权重；单股目标4\%、行业25\%，总预算最高95\%。过去60日估计组合波动，20\%波动预算；沪深300低于120日均线时乘0.75，截断与余量留现金。
\[
e_t=0.95\min(1,0.20/\hat\sigma_t)\,[\mathbf1_{CSI300_t\ge MA_{120,t}}+0.75\mathbf1_{CSI300_t<MA_{120,t}}].
\]
形成日目标会因受阻成交和价格漂移偏离，不能保证未来回撤。下一开盘先卖后买，基础买10bp/卖15bp；缺开盘/限制价或涨跌停阻挡相关方向。容量为此前20日平均原价成交额的1\%。缺估值沿用已知收盘，超20日保守计零并保留单位，不虚构卖出。
\subsection*{严格成交研究情景}
""" + latex_table(["参数", "明确研究设定"], [
        ["整手及T+1", "买100股等价整手；完整卖出允许零股；新买单位当日不可卖"],
        ["佣金", "买卖均0.03%；每笔最低5元"],
        ["印花税", "仅卖出；可配置日期表，2023-08-28起0.05%"],
        ["滑点与容量", "开盘参考名义金额0.05%独立现金成本；滞后20日成交额1%"]], "ll") + r"""
\[
q^{raw\,equiv}_{i,t}=u_{i,t}\frac{O^{adj}_{i,t}}{O^{raw}_{i,t}},\qquad
Cost=Commission+StampTax+SlippageCost.
\]
原/复权价格比仅换算股数，不推断拆股/分红。\textbf{严格路线仍为复权收益单位研究，未完整模拟企业行动现金、真实股权余额和结算。}开盘名义金额加独立滑点成本避免重扣；严格费率不同，收益差不能全归因整手。
\subsection*{独立重建}
\[
Cash_t=Cash_{t-1}-Buy_t+Sell_t-ActualCost_t,\quad NAV_t=Cash_t+\sum_i u_{i,t}P^{mark}_{i,t}.
\]
逐笔重建实际成本、现金、单位、净值、换手和收益，另核验原股换算、整手、T+1及容量；现金容差为绝对$10^{-6}$、相对$10^{-10}$。基础路线按平费率核验，严格路线按实际\texttt{trade.cost}和成本分项核验。
""")
    candidate_rows = [[NAMES[n], pct(table.loc[n,"total_return"]), f"{table.loc[n,'sharpe']:.3f}",
                       pct(table.loc[n,"max_drawdown"]), f"{table.loc[n,'excess_vs_total_return']*100:+.2f}"] for n in ORDER]
    page("7\\quad 失败记录与本次全部候选", r"""
\lead{旧失败保留；本次辅助预选也不能被后续结果替换。}
原20日动量长区间扣费亏损83.57\%，零费独立重跑仍亏损75.16\%；同成交路径加回费用不等于零费重新投资。V2原预选固定区间收益6.12\%，落后指数。V3验证先预选价值主题，后续仅4.30\%；当前主线经历史复核后采用，明确不是独立盲测。
\subsection*{先记录2023--2024辅助预选}
""" + latex_table(["辅助候选", "验证净收益", "验证夏普", "验证回撤"], [
        [NAMES[n], pct(validation.loc[n,"total_return"]), f"{validation.loc[n,'sharpe']:.3f}", pct(validation.loc[n,"max_drawdown"])]
        for n in ORDER[:3]], "lrrr") + r"""
按验证夏普预选年度白箱Ridge，然后原样保留选择文件。以下为相同417交易日的全部候选，超全收益以\textbf{百分点}计。
""" + latex_table(["候选", "扣费收益", "夏普", "回撤", "超全收益(pp)"], candidate_rows, "lrrrr") + r"""
\begin{center}\includegraphics[width=.97\textwidth,height=52mm,keepaspectratio]{figures/v4_candidates.pdf}\end{center}
全收益基准19.94\%；三个白箱均未超过，固定50/50目标组合略超过但弱于主线。组合可能改善某些风险表现，却不能用较低回撤证明全面优越。全部候选、失败和修正前输出归档；这轮结论是\textbf{继续保留主线，辅助路线待前向观察}。
""")
    strict = stress[stress.scenario == "lot_min_fee_tax_slippage"]
    page("8\\quad 主线曲线、基准与执行压力", r"""
\lead{相同起点和日期；价格、全收益与ETF代理明确区分。}
\begin{center}\includegraphics[width=.98\textwidth,height=66mm,keepaspectratio]{figures/v4_curves.pdf}\end{center}
策略从初始资金起算，指数用前一交易日收盘，保留首日损益；全收益指数计红利再投资，不能用ETF代理改名替代。主线收益25.13\%，分别超过价格14.55\%与全收益19.94\%；与全收益差5.19个百分点，相对财富增幅约4.33\%，两者不是同一口径。首次开盘起算价格基准结论不变，ETF复权代理另保留。
\subsection*{严格实际成交情景}
""" + latex_table(["候选", "严格净收益", "夏普", "严格回撤", "实际成本(元)"], [
        [NAMES[row.candidate], pct(row.total_return), f"{row.sharpe:.3f}", pct(row.max_drawdown), f"{row.total_cost:,.2f}"]
        for row in strict.itertuples()], "lrrrr") + r"""
独立最低佣金/税/滑点政策与基础买10bp、卖15bp不同，因此整手、现金与费用同时改变实际路径。严格情景并非单独增加整手约束的因果对照，亦不代表完整实盘可执行收益。
\subsection*{原压力和落后年份}
双倍基础费用主线22.48\%，信号延迟一日27.31\%；延迟更优未被据此改为默认。2023起连续运行44.09\%、最大回撤19.70\%，2024仍落后价格指数7.28个百分点。不能把累计领先描述为每年或每个市场状态都胜出。
\subsection*{基准排序修正}
厂商日线先按日期升序排列，再选首日前最近收盘。修正前指标归档，修正没有改变交易或预选；现用表与曲线采用一致边界。基准缺日拒绝，不插值，不从最早日期错误选取区间起点。
""")
    page("9\\quad 风险、资金归因与冻结观察", r"""
\lead{分析已有路径，不虚构未发生的前向成绩。}
\begin{center}\includegraphics[width=.98\textwidth,height=40mm,keepaspectratio]{figures/v3_drawdown.pdf}\end{center}
同区间对比取严格交易日交集；内部缺日拒绝。切片第一日以此前净值起算，保留首日收益，高水位包含初始资金；它不是从现金重新开仓。月历包含首尾部分月份，60日滚动统计不补不足窗口；未恢复回撤不伪造恢复日期。静态现金情景$N^{mix}_t=wN_t+(1-w)$不再平衡、现金零利息，未重算规模与容量。
\subsection*{描述性资金账本归因}
\[
\Pi_i=SellCash_i-BuyCash_i-Cost_i+V_{i,T},\qquad \sum_i\Pi_i=NAV_T-NAV_0.
\]
已退出股票仍保留损益，各股贡献除以初始资金后相加等于组合收益。实际成本逐笔使用，未记录的佣金/税/滑点分项保持未知；按明确行业标签汇总时保留未知组。该功能是描述性现金流与期末估值归因，不冒充行业或因子的因果解释。
\subsection*{冻结身份与追加观察}
冻结策略ID、完整公式、年度参数、训练/执行配置、数据/代码哈希、UTC时点及最后已知交易日。只接受\texttt{observation\_date}>截止日；\texttt{signal\_asof}<成交日。独占新建文件，不覆盖旧计划；同内容重试幂等，冲突或前日插入拒绝，记录链经读回检查。
冻结前已经发生的新增历史属于历史补录；冻结后首先是纸面记录。\textbf{本地时钟、人工记录和哈希不能认证真实信息获取或账户成交}，不把旧净值贴成前向胜出。未来绩效仍待积累新数据。
\subsection*{指数增强的实测门槛}
组件需要当时真实可得的成分、权重及风险数据。当前缺完整历史指数权重，未运行真实增强策略；手工/合成权重只验证约束与流程，不发布其投资收益。指数方法准备、真实实测与交易授权分别报告。
""")
    page("10\\quad 外部接口、验收注入与复现", r"""
\subsection*{LLM辅助的实际状态}
支持OpenAI兼容Chat Completions或本机Ollama；密钥临时传入，不写配置、结果或材料。发送明确研究问题和选定公式/证据；建议需DSL校验，再经评价器实测。\textbf{本次真实研究\texttt{llm\_called=false}，未发生真实模型请求}；离线说明标为确定性，模型文本标为未核验，不能制造IC或收益数字。
\subsection*{保留的原生接入与账户边界}
此前Qlib 0.9.7独立环境读取20股33,754条日线，8表达式对比269,554个数值；实际运行158项Alpha158配置，VWAP输入覆盖100\%。属于因子接口验证，未训练Qlib策略，未替换主成交账本。原接入证据保留，不由本次源码编写重新证明。

同花顺/SuperMind为文件桥：历史观察权重与代码导出、模拟成交CSV金额/重复核验、保存本地回执。账号/API未连接，未收到真实账户成交文件，未提交订单；历史权重不是今日信号，现金核对不等于盈利。
\subsection*{最终软件与界面验收}
""" + tex(test_text) + "\n\n" + "界面核验：" + tex(ui_status) + r"""
。测试数量和界面状态对应最终封存记录，记录范围之外的账户连接、真实指数增强与实盘成交没有据此验证。研究账本检查、数值前缀检查、UI交互和新增市场数据分别提供不同证据；公开聚合证据支持结果核对，真实训练与逐股查询仍依赖合法数据快照。
\subsection*{复现入口及可编辑材料}
\begin{itemize}
\item 双击\texttt{launch.cmd}，或按README安装固定依赖并启动Streamlit；缺私有快照时查看公开真实聚合证据。
\item 真实研究使用\texttt{scripts/run\_factor\_research.py}，新建输出身份；合法原始数据及训练缓存不随公开包发布。
\item 材料作者入口\texttt{scripts/build\_v230\_materials.py}只写源码和科学图；Beamer、报告、讲稿由独立XeLaTeX步骤编译。
\item \texttt{deck\_content.json}保持20条，正文18条20分钟、备份2条；同源生成可编辑PPTX与讲稿，图数据来自封存CSV。
\end{itemize}
\subsection*{来源与交付声明}
课程依据CF2026\_Project1.pdf；主线及历史诊断来自\texttt{evidence/research\_v2,v3}，新完整候选与压力来自\texttt{evidence/research\_v4}，原生Qlib及文件桥有独立状态证据。中证/Tushare用于数据与基准，CogAlpha只借鉴可检查表达式。源码、指标、数据身份和发布验收共同定位本版；材料准备、Git发布及Blackboard提交是各自状态，不以本报告生成替代。
""")
    if len(sections) != 10:
        raise AssertionError("Report must retain ten explicit sections/pages")
    write(OUT / "final_report.tex", preamble + r"\begin{document}" + "\n" + "\n".join(sections) + r"\end{document}" + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-passed", "--test-count", type=int, dest="tests_passed")
    parser.add_argument("--tests-skipped", type=int, default=0)
    parser.add_argument("--ui-status", default="本次交互验收待最终发布记录")
    args = parser.parse_args()
    if args.tests_passed is not None and args.tests_passed < 0 or args.tests_skipped < 0:
        parser.error("Test counts must be nonnegative")
    test_text = (f"最终测试：{args.tests_passed}项通过，{args.tests_skipped}项跳过；以封存日志为准"
                 if args.tests_passed is not None else "最终测试数量：待封存验收记录注入，未继承旧版数字")
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(exist_ok=True)
    table, validation, stress, protocol, decision = inputs()
    figures(table, stress)
    deck = deck_content(table, validation, stress, protocol, decision, test_text, args.ui_status)
    write(OUT / "deck_content.json", json.dumps(deck, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    write_beamer(deck)
    characters, seconds = write_script(deck)
    report(table, validation, stress, protocol, decision, test_text, args.ui_status)
    metadata = {"release": "v2.3.0", "materials": "V230", "members": MEMBERS,
                "source_only": True, "pdf_compiled": False, "visual_acceptance": "pending",
                "report_sections": 10, "slides": 20, "main_slides": 18, "backup_slides": 2,
                "main_talk_seconds": seconds, "spoken_chinese_characters": characters,
                "tests_passed_injected": args.tests_passed, "tests_skipped_injected": args.tests_skipped,
                "ui_status_injected": args.ui_status, "llm_called": False,
                "whitebox_preselected": decision["whitebox_preselected"],
                "source_hashes": {name: hashlib.sha256((E / name).read_bytes()).hexdigest()
                                  for name in ("test.csv", "validation.csv", "stress.csv", "protocol.json", "decision.json", "factor_cards.json")},
                "expected_pdf_names": {"final_report.tex": "CF2026_V230_Final_Report.pdf",
                                       "beamer.tex": "CF2026_V230_Beamer.pdf",
                                       "speaker_script.tex": "CF2026_V230_Speaker_Script.pdf"}}
    write(OUT / "authoring_checks.json", json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(OUT), "slides": 20, "main_talk_seconds": seconds,
                      "spoken_chinese_characters": characters, "sources_only": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
