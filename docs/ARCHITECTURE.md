# 架构与扩展

## 语言与复用取舍

团队熟悉 Python/Java；本项目主要处理表格数据、统计诊断、可配置研究和可视化。Python 直接复用 NumPy、pandas、SciPy、Plotly 和 Streamlit。核心是独立包，界面不是业务逻辑的载体。

已检查 Backtrader、Alphalens、Qlib 官方项目。Backtrader 在测试中作为独立参考，验证次日开盘、现金、费用和逐日净值。主账本保持小型且显式，便于核对课件口径。Alphalens 的因子诊断概念被借鉴，缺失标签不重分组、并列与最小样本规则显式实现。Qlib 适合后续模型研究，此次不引入其完整训练和存储体系。没有复制第三方源代码。

~~~mermaid
flowchart LR
    A[Tushare / CSV] --> B[data: 快照与质量]
    B --> C[factors: 注册与分数]
    C --> D[portfolio: 目标权重]
    D --> E[engine: 成交与现金]
    B --> E
    C --> F[analytics: 因子诊断]
    E --> G[analytics: 绩效与复核]
    F --> H[experiment: 配置和证据]
    G --> H
    H --> I[CLI / Streamlit / 报告]
~~~

| 模块 | 输入 | 输出与职责 |
|---|---|---|
| data | API / CSV | 缓存、唯一键面板、日历、质量、SHA-256 |
| factors | 完整日历上的价格面板 | 日期×资产分数；注册元信息；统一长表 |
| portfolio | 单日分数、持仓数 | 非负目标权重；并列按代码 |
| engine | 行情、日历、分数、配置 | 净值、现金、持仓、成交、订单；不依赖因子公式 |
| analytics | 分数与标签，或账本 | 诊断、风险收益、独立现金和单位复核 |
| experiment | 配置和路径 | 保存数据/源码哈希、Git版本、耗时与结果 |
| app / cli | 用户参数 | 调用同一计算包，不另写金融公式 |

## 新增因子

函数签名 fn(close: DataFrame, window: int) -> DataFrame，保持日期和资产轴，不接收未来标签。通过 register(FactorSpec(...)) 注册，在运行入口导入注册模块。tests/test_research.py 演示区间振幅插件，无需修改 engine.py。

先写因子卡：公式、假设、字段、窗口、方向、缺失政策、失效条件。再做 causal_check（前缀重算与未来扰动），独立核对样本公式，最后批量诊断。未来可扩展为多字段 MarketView；执行核心仍只接收统一分数。

## 后续课程

- P2：固定论文的资产池、时间划分和成本；不把本次结果称为论文复现。
- P3：注册新因子；标准化仅使用当天横截面或训练集拟合的统计量。
- P4：添加 ModelSignal，按时间分训练/验证/测试；清除跨分界标签，预处理只拟合训练集，输出相同的日期×资产分数。
- P5：替换目标权重模块，升级真实股数、事件分红、整手、最低佣金、容量和排队模型。

## 与 CogAlpha 的关系

参考可解释因子表示、质量检查和泄漏测试；没有实现 LLM 生成、交叉变异、21角色或大规模搜索。作者仓库是提示词模板，不是完整平台。神经网络可成为预测组件，不能替代时点对齐、复权、现金与成本核算，也不是 Project 1 的必要条件。

参考：
https://github.com/mementum/backtrader
https://www.backtrader.com/docu/order-creation-execution/order-creation-execution/
https://github.com/quantopian/alphalens
https://github.com/microsoft/qlib
https://github.com/uwFengyuan/CogAlpha_Prompt
https://arxiv.org/abs/2511.18850

