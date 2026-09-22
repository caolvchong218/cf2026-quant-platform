# V3复现、版本管理与验收

## 新目标与结果

当前候选 `rolling_lightgbm__managed`，年度滚动LightGBM＋波动预算。固定2025-01-02至2026-09-18，累计净收益25.13%、年化14.51%、最大回撤9.88%；沪深300价格指数14.55%，全收益指数19.94%。双倍费用22.48%、信号延后一日27.31%，都高于全收益指数。2023起连续运行44.09%、最大回撤19.70%；2024年度落后价格指数7.28个百分点。

V3验证期预选 `economic__steady` 后续失败，当前版本属于看过最终历史区间后选出的开发候选，不是独立盲测赢家。`selection.json`保留事前选择，`decision.json`记录事后筛选、8个候选和前向验证需要。切勿改写这两个文件的含义。

## 完整计算

先按REPRODUCE_V2.md取得原1000股快照和研究特征；安装固定research依赖。V3沿用V2特征文件身份，旧的NumExpr加速禁用策略保持生效。

```powershell
python scripts/download_benchmark_checks.py
python scripts/run_benchmark_research.py
python scripts/qualify_benchmark_strategy.py
python -m pytest -q -p no:cacheprovider
```

基准下载默认读取本机约定的Token文件；换电脑请调整`--token-file`参数。原始响应留`data/private/benchmark_v3/raw`，不能提交Token或厂商原始行情。指数目录可能达到返回行数上限，本次已找到并核实H00300.CSI准确匹配，不声称获得全部指数目录。

V3分数和模型形成边界在`runs/research_v3`。已存在该运行时研究入口拒绝覆盖；重复前使用另一个明确命名的归档目录保留旧运行，再重算。不要直接删除原实验。候选参数、阶段边界和默认选择规则见RESEARCH_PROTOCOL_V3.md及benchmark_research.py。

模型每年形成于第一次交易的前一交易日；训练日期在此前3年，标签退出严格早于形成日。预测区间衔接不重叠，日期不是随机拆分。数据具有后续修订的局限，滚动训练不恢复已观察区间的盲测资格。

## 平台入口

`strategies.py`目录注册每个版本的信号、组合、分数路径、状态。`ui_strategy_backtest.py`加载固定结果或调用同一成交核心重新运行。默认优先最新历史合格候选；旧策略仍可选择，V1动量明确标为失败对照。

参数重跑写入`runs/interactive_versions/时间戳`，保留daily/trades/positions/orders、费用、参数、风险权重及检查。更换版本会清除旧结果，面板标明实际使用的版本、日期和初始资金。模型版本只允许2023起的可见预测，不能把未来训练的模型套回训练期。

公开仓库包含13个模型版本的聚合逐日账本；无本地特征/分数时仍可查看，但重跑按钮禁用。侧栏股票池用于单因子自由实验，模型版本固定其1000股历史池，并在主区域明确显示。

## 验收口径

- 资金核对通过只说明计算一致；收益/回撤/指数目标单独判断。
- 当前固定区间历史门槛：扣费净收益正、超过同期沪深300价格指数、最大回撤不高于25%。全收益指数和压力结果进一步附列。
- 52项测试覆盖原会计、因果、中性化、缺失、数值修复，以及新增年度训练边界、日期覆盖、基准手算和选择门槛。
- 21个V3实验独立账本核对通过；另一Python环境从版本重跑入口复算最新策略，4张表按rtol=1e-9、atol=5e-6一致。
- 浏览器核验最新默认、V2切换、V1警示、返回V3真实重跑；报告10页、Beamer20页，PowerPoint原生6图表2表格并经最终重导入渲染。

## 材料

```powershell
python scripts/build_benchmark_materials.py
```

需要XeLaTeX、ctex、Beamer及绘图依赖，生成`reports/benchmark_v3`和`docs/PRESENTATION_SCRIPT_V3.md`。PowerPoint生成器使用`MATERIALS_DIR=reports/benchmark_v3`和`PPTX_NAME`，依赖本机Artifact Tool；普通电脑可直接使用PPTX或重建真正Beamer PDF。

V2原报告仍在`reports/final`供历史追溯；最新材料以README指向的`reports/benchmark_v3`为准。桌面提交包将V2存入明确历史子目录，根目录只放最新报告、演示和源码。
