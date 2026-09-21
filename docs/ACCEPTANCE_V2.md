# Project 1最终验收清单

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
