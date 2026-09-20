# Project 1 验收映射

| 课件要求 | 实现 / 证据 |
|---|---|
| 数据获取与可复现快照 | data.py、download命令、manifest.json与校验值 |
| 日频字段、唯一键、质量报告 | load_market、quality.json、CONVENTIONS.md |
| 复权口径一致 | 同一因子缩放OHLC、原始字段保留、拆股连续性测试 |
| 三个不同逻辑因子 | momentum/reversal/low_volatility及因子卡 |
| 统一接口与方向 | FactorSpec注册、date/asset/factor/value长表 |
| 参数化回测与真实成交成本 | Config、engine、daily/trades/orders/positions |
| 账本与重复核验 | reconcile、两次基准一致、手算和Backtrader对照 |
| IC / Rank IC | 逐日横截面、平均秩、缺失和最低样本规则 |
| 分组收益 | 形成日分组、有效人数、缺失不重分组、与净值分开 |
| 绩效与风险 | 累计/年化收益、波动、Sharpe、回撤、换手、费用、持仓集中度 |
| 小型控制变量实验 | EXERIMENT_PLAN、5日与20日动量比较 |
| 复现包和研究报告 | CLI、数据清单、版本哈希、8页PDF与LaTeX |
| 自主拓展 | 可注册因子+批量诊断+时间泄漏检查；新增插件不改成交核心 |

## 验收边界

真实数据结果来自本地Tushare快照；仓库附带的合成样本仅验证安装和流程。账号凭据和原始课件不入Git。研究模型不等同于完整实盘交易系统，限制见CONVENTIONS。

## 版本管理

Git保留初始化、核心引擎、界面与演示、研究口径、验证与报告等阶段。数值运行ID包括配置、数据、日历和源码哈希；当前研究用run_index.json定位，不凭目录字典序猜测版本。正式验收代码使用v1.0.0标签。
