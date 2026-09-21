from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; E=ROOT/'evidence/research_v2'
table=pd.read_csv(E/'test.csv');selected=json.loads((E/'selection.json').read_text())['selected'];r=table[table.model==selected].iloc[0]
readme=f'''# 青序 · CF2026量化研究平台

计算金融Project 1最终版：可复现的真实数据研究、12因子、模型比较、含费用成交账本及中文交互平台。

**提交策略：验证期选出的标准化多因子＋风险约束。** 2025-01-02至2026-09-18累计净收益{r.total_return:.2%}，年化{r.annualized_return:.2%}，最大回撤{r.max_drawdown:.2%}。共同完整月份内跑赢CPI，收益仍低于同期沪深300价格指数；这段历史不保证未来收益。LightGBM最终期更强，但没有据此替换验证期预选策略。

![策略研究页面](evidence/ui-research-final.png)

## 先运行，再看材料

本机双击`launch.cmd`，浏览器打开 http://127.0.0.1:8501 。新电脑用Python3.12/3.13：

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-research-lock.txt
.venv/Scripts/python.exe -m pip install -e ".[dev,research]"
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m cfquant.cli run --config configs/demo.yaml
.venv/Scripts/python.exe -m streamlit run app.py
```

无Token可以查看已发布真实研究证据、运行合成样本并演示完整界面。原始Tushare数据与凭据仅在本地，重新下载需自己的权限。新旧股票池及样本身份明确分开。

## 最终交付

- [10页最终报告](reports/final/CF2026_Final_Report.pdf)及[LaTeX源码](reports/final/final_report.tex)。
- [20页Beamer演示PDF](reports/final/CF2026_Beamer.pdf)、[Beamer源码](reports/final/beamer.tex)和[可编辑PowerPoint](reports/final/CF2026_Presentation_Final.pptx)。
- [约20分钟讲稿与演示步骤](docs/PRESENTATION_SCRIPT.md)。正文18页，2页答辩备份。
- [完整复现指南](docs/REPRODUCE_V2.md)、[课程逐项验收](docs/ACCEPTANCE_V2.md)、[12因子卡](docs/FACTOR_CARDS_V2.md)。
- [事前实验协议](docs/RESEARCH_PROTOCOL_V2.md)、[数值重复性修复说明](docs/NUMERICAL_AUDIT.md)、[派生证据](evidence/research_v2)。
- 原始60股项目与失败案例仍保留在[历史说明](docs/README_V1_ARCHIVE.md)，不冒充最终报告。

## 数据与研究设计

固定2019-12-31的1,000只主板股票，行情2019-10-08至2026-09-18，共1,656,599条日线、1,690个交易日。追加同规模每日估值、57,389条财务指标和1,432个历史行业区间。行情和研究缓存约1.63GB。

训练2020–2022、验证2023–2024、最终评估2025–2026-09-18。对照包括标准化多因子、中性化多因子、Ridge、LightGBM、小型MLP。严格按公告日滞后，清除跨训练边界的未来标签。原动量全期已被观察，最终区间不称完全未知的盲测。

| 最终期候选 | 年化净收益 | 最大回撤 | 选择依据 |
|---|---:|---:|---|
'''
names={'multifactor_raw':'标准化多因子','multifactor_neutral':'中性化多因子','ridge':'Ridge','lightgbm':'LightGBM','mlp':'小型MLP'}
for x in table.itertuples():readme+=f'| {names[x.model]} | {x.annualized_return:.2%} | {x.max_drawdown:.2%} | '+('验证期预选' if x.model==selected else '保留对照')+' |\n'
readme+='''
## 平台能力与边界

数据、因子/模型、目标组合、成交账本、统计及界面相互独立。组合可持现金；支持持仓缓冲、单股/行业目标上限、波动预算、趋势仓位、事前流动性预算与部分成交。增量合并输出新的标准行情及修订审计，测试覆盖分段等于全量和重复幂等。

本机发现可选NumExpr加速在大面板除法中偶发错误，已统一禁用并加入实际失败复现的回归测试。重新生成全部研究，独立环境复算结果见repeatability.json。

研究用连续可分的复权单位，未实现完整整手、分红现金、税费历史、结算和盘口；固定历史池、厂商财报修订、行业重构和长期缺失估值仍限制外推。风险权重是形成时目标，实际权重会漂移。原始动量累计−83.57%、零费用重跑−75.16%均保留。

参考Qlib工作流与因子表达，复用sklearn/LightGBM；未声称完整复制Alpha158或集成Qlib成交引擎。参考CogAlpha的代码式因子表示，未运行LLM演化挖掘。

## Git与许可证

公开仓库只包含代码、合成样本、报告和派生汇总，真实行情、财报、Token及虚拟环境不进入Git。提交历史保留原始平台、数据扩展、研究协议、模型与风险模块、数值修复及最终材料。当前未附加项目开源许可证；公开可见不等于授予任意再分发许可，第三方依赖保留各自许可证。
'''
(ROOT/'README.md').write_text(readme,encoding='utf-8')
accept='''# Project 1最终验收清单

按课程CF2026_Project1.pdf的19页要求逐项核验。课程基础80分、扩展20分，无强制收益/Sharpe门槛；最终评分由教师决定，本文不自授满分。

| 要求 | 实现 | 验收入口/证据 |
|---|---|---|
| 原始/标准数据分离、版本身份 | 原始JSON请求与UTC、标准market、SHA清单 | data.py、数据页、expanded_data.json、data_audit.json |
| 日期资产唯一、异常/缺失/单位 | 键检查、OHLC约束、股/手转换、固定复权 | CONVENTIONS、EXPANDED_DATA、质量报告 |
| 至少3个独立逻辑因子 | 原3个注册因子和12个扩展因子卡 | factors.py、features.py、FACTOR_CARDS_V2.md |
| 高分方向、可配置接口 | FactorSpec与Config，date/asset/factor/value | 交互回测与CLI、tests/test_research.py |
| 含费用回测与明细 | 下一开盘、现金约束、限制价、缺失估值 | daily/trades/positions/orders、reconcile |
| IC/RankIC与有效样本 | 每日横截面、并列平均秩、常数NaN | 因子诊断页、factors/*/daily.csv |
| 分组及缺失 | 先形成后匹配标签、形成/有效人数 | factors/*/groups.csv、相关测试 |
| 年化/回撤/换手/费用 | 初始NAV参与回撤、252日年化、双边换手 | analytics.py、手算及独立参考测试 |
| 有控制的对照 | 标准化/中性化、固定模型、组合模块、费用 | 协议、validation/test/controls/preprocessing |
| 完整性与复现 | 源码/数据/参数身份、失败归档、缓存指纹 | provenance、repeatability、测试与CI |
| 扩展：中性化 | 历史行业+对数市值OLS残差 | 正交/行业均值/时点测试 |
| 扩展：组合与现金 | 缓冲、权重上限、风险/趋势敞口 | risk.py、risk.csv与组合对照 |
| 扩展：部分成交 | 前20日均额预算，按实填收费 | liquidity_limited订单及未来量扰动测试 |
| 扩展：增量 | 重算统一参考尺度、修订审计、新目录 | incremental.py与幂等/分段测试 |
| 报告与代码材料 | 10页报告、运行入口、固定依赖、样本 | reports/final、README、configs/demo.yaml |
| 演示材料 | 真正Beamer PDF/源码、原生图表PPTX、讲稿 | PRESENTATION_SCRIPT.md及最终提交目录 |

## 必须保留的解释

- 预选策略依据验证期选择，LightGBM后续收益较高不能用来改写选择。
- CPI比较只涵盖2025年2月至2026年8月的共同完整月份，与417日累计收益不同。
- 原始动量亏损结果保留；6年旧策略和不到2年新策略不直接作单变量改善归因。
- 最终区间曾被原始动量研究观察，非完全未知盲测；固定池和厂商历史修订限制泛化。
- MLP固定8轮没有声称收敛。中性化和CNN均不是保证收益的步骤。
- 风险控制为目标，不保证实际漂移后的权重、波动或回撤；研究单位未模拟完整交易所细节。
- 增量merge只是标准行情合并和审计，未声称自动更新完整GUI数据集或动态全市场池。

## 提交前仅需人工补充

在课程提交系统填写小组姓名、学号等身份信息，核对教师要求的文件命名。报告正文未伪造成员名单。按课上规则提交最终报告和源码包，演示使用Beamer PDF或PowerPoint；无需上传Tushare Token和厂商原始数据。
'''
(ROOT/'docs/ACCEPTANCE_V2.md').write_text(accept,encoding='utf-8')
