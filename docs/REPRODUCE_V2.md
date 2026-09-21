# 最终研究复现指南

## 1. 无凭据验收（推荐先做）

Python 3.12 或 3.13。在仓库根目录运行：

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-research-lock.txt
.venv/Scripts/python.exe -m pip install -e ".[dev,research]"
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m cfquant.cli run --config configs/demo.yaml
.venv/Scripts/python.exe -m streamlit run app.py
```

默认策略研究页读取已发布的真实研究汇总，不需要凭据。交互回测在真实数据缺席时使用明确标识的合成样本，不能把合成样本回测当作报告实证。`launch.cmd` 为本机一键启动入口。公开仓库不需要安装Qlib或神经网络GPU环境。

所有课程指标使用252交易日年化、样本标准差ddof=1。包导入会把pandas的全局`compute.use_numexpr`设为False：本机NumExpr 2.14.1可选加速曾在大面板除法中偶发返回错误常数，失败复现和修复测试见`NUMERICAL_AUDIT.md`。在自己的notebook中也应保留此策略，不要在运行中重新打开该选项。

## 2. 获取真实研究输入

需要自己的Tushare接口权限。Token仅放本地文件或环境变量，不能放YAML、Git或截图。按`EXPANDED_DATA.md`下载原1,000股快照：

```powershell
.venv/Scripts/python.exe -m cfquant.cli --root data/private/mainboard1000_20260918 download --token-file YOUR_TOKEN_FILE --assets 1000 --start 20191001 --end 20260918 --selection-date 20191231 --study-start 20200102 --delay 0.4 --workers 4 --max-gib 3
.venv/Scripts/python.exe scripts/download_research_data.py --token-file YOUR_TOKEN_FILE
```

补充脚本固定这次提交的数据身份，追加每日估值、财务公告、历史行业、沪深300价格指数和CPI。每资产接口若达到返回上限则停止，不能默默接受截断。财务上限为100条，本次最大67条；升级更长时期时应分报告期请求后去重。缓存保留UTC下载时间和不含Token的请求参数。

时间/权限变化后厂商可能修订历史值，重新下载不保证逐字节等于当前快照。比较manifest与质量报告，不能只比较行数。当前完整价格及研究数据约1.63GB，不含虚拟环境和多次回测输出。

## 3. 完整计算

```powershell
.venv/Scripts/python.exe -m cfquant.cli research
.venv/Scripts/python.exe scripts/publish_research_evidence.py
.venv/Scripts/python.exe scripts/audit_strategy_failure.py
.venv/Scripts/python.exe scripts/supplement_research_evidence.py
```

训练2020–2022，验证2023–2024，最终评估2025–2026-09-18。所有参数在`RESEARCH_PROTOCOL_V2.md`和`models.py`、`RiskPolicy`中固定。不要为复现改动参数或根据最终期重新选模型。运行先归档旧`runs/research_v2`，然后重建；特征缓存核对输入、核心源码和输出哈希，身份不匹配会重新计算。

`runs/research_v2`保存13个主实验完整日账本、持仓、交易、订单、费用、配置和检查。模型信息、随机种子、输入特征和选择规则也落盘。`evidence/research_v2`只发布派生结果、因子诊断及QA，原始财报和逐股数据仍仅在本地。原始动量复算另外保存在`runs/research_v2_audit`。

研究选择是验证期正收益且MDD≤20%的候选中，按CAGR/MDD排序；如没有合资格者按相同比值记录候选并披露未达标。最终期有更好的模型不能据此改写提交名称。MLP固定8轮，未收敛的限制应保留。

## 4. 重建材料

安装XeLaTeX、ctex和Beamer。先完成上面的实证与证据发布，再运行：

```powershell
python scripts/build_final_materials.py
python scripts/build_deck_content.py
```

输出在`reports/final`：10页报告、20页Beamer PDF、可编辑`.tex`和矢量图。讲稿在`docs/PRESENTATION_SCRIPT.md`，正文18页约19.1分钟，另2页备份。PPTX由同一`deck_content.json`用Artifact Tool生成，图表含嵌入工作簿；该构建器依赖本机提供的Artifact Tool运行时，普通电脑可以直接使用交付PPTX或重建Beamer PDF。

## 5. 扩展的可验收边界

- `features.neutralize`：按日行业/对数市值残差；UNKNOWN不会反填，测试检查正交性。
- `risk.build_targets`：目标权重、现金和波动预算；市场变化后实际暴露可能漂移。
- `engine.run_backtest(..., participation_limit=.01)`：以之前20日均额作事前成交预算，记录部分成交；不声称真实当日参与率或盘口成交。
- `cfquant merge --existing OLD.csv --incoming NEW.csv --destination NEW_FOLDER`：输出新的market.csv和修订审计，重算统一复权尺度；不自动注册成完整GUI数据集，也不生成缺失的日历/股票池文件。原文件不变。测试证明分段等于全量和重复导入幂等。

尚未实现动态全市场股票池、完整ST历史、交易所整手/分红现金/结算和市场冲击模型。它们属于后续改进，不能列作已完成功能。
