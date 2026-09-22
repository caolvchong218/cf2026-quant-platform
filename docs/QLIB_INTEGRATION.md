# 真正接入 Microsoft Qlib

本平台的 Qlib 实验室调用正式安装包 **pyqlib 0.9.7**。桥接层只转换数据格式、固定表达式请求、独立核对结果和提供界面；表达式解析、滚动运算和 Alpha158 配置来自 Qlib。现有 V3 投资策略不因这项接入而更换或重训。

## 接入链路

```text
既有真实 Tushare 快照
    → 固定20股工程样本，SH/SZ各10股，按代码排序选取
    → Qlib calendars / instruments / features/*.day.bin
    → qlib.init + qlib.data.D.features
    → 8个基础表达式 + 官方Alpha158DL完整配置
    → pandas独立核对、IC/Rank IC/分组诊断
    → 平台“Qlib 实验室”与可下载公开聚合证据
```

样本选择不根据未来收益或因子表现；它是验证接口与计算的一组小样本，不能代表全市场。输入行情提供2019-10-08至2026-09-18的历史，默认诊断自2020-01-02开始，前段保留因子预热。

## 为什么独立环境

Qlib带有机器学习、优化与工作流依赖。为保持原平台的已验证环境与策略结果稳定，在 `.venv-qlib` 中单独安装和执行，Streamlit主环境不需要导入Qlib。页面使用固定脚本、参数白名单和独立进程重跑，不执行用户自定义命令或任意表达式。

Windows Python 3.12：

```powershell
python -m venv .venv-qlib
.venv-qlib\Scripts\python -m pip install --cache-dir tmp/qlib-pip-cache pyqlib==0.9.7 pandas==2.3.3 numpy==2.3.5 scipy==1.16.3 pyarrow==21.0.0
.venv-qlib\Scripts\python scripts/run_qlib_bridge.py --assets 20 --start 2020-01-02 --end 2026-09-18
```

如需运行隔离环境内的集成测试，再安装pytest，执行：

```powershell
.venv-qlib\Scripts\python -m pip install pytest
.venv-qlib\Scripts\python -m pytest tests/test_qlib_bridge.py -q -p no:cacheprovider
```

主环境仍可执行格式、缺失日期、独立参考和参数校验测试；真实Qlib测试在无Qlib的环境明确跳过。`evidence/qlib_bridge/result.json` 保留真实环境版本与实际核对结果。首次安装需要网络，后续本地数据接入不访问Tushare、不读取Token。

## 数据格式与单位

采用Qlib官方本地数据布局：`calendars/day.txt`保存交易日，`instruments/all.txt`保存代码与区间，`features/<symbol>/<field>.day.bin`先写一个小端float32日历起点索引，再写每日float32值。每次建立新目录，不覆盖旧provider；缺少行情的交易日保留NaN，禁止压缩掉日期或向前填充。

本平台已有OHLC是固定参考日复权价格。桥中使用同一价格与复权比例 `factor = adj_factor / adjustment_reference`；Qlib成交量按 `raw_volume_shares / factor` 转为调整后的单位，使价格与量的乘积保持金额含义。原始Tushare `vol` 为手（100股），`amount` 为千元；加载原始日线后独立核验 `vol * 100` 与标准快照股数一致。

VWAP按 `amount * 1000 / raw_volume_shares * factor` 得到真实当日日均成交价的复权值。这里不以收盘价替代缺失VWAP。若原始amount文件缺失，VWAP相关特征会缺失并反映到覆盖率，不能声称158项全部可用。

## 独立核对范围

8项：20日动量、5日反转、20日低波动、20日均线趋势、20日低振幅、5/20日成交量趋势、日内收益、收盘区间位置。

`pandas_reference` 从同一float32输入使用独立pandas公式计算；对照不仅比数值，也比较有限值/缺失位置，避免只在恰好同时有值的少数样本上通过。Qlib滚动算子原生采用 `min_periods=1`，Std采用 `ddof=1`。该语义已依据官方源码核对；不能直接将Qlib结果与要求完整窗口的另一公式比较后误报差异。

数值核对遵循Qlib原生语义，诊断展示另要求完整历史窗口和有效报价。绝对容差2e-6、相对容差2e-5写入每项证据；float32存储不能要求float64逐位相等。

Alpha158直接调用 `qlib.contrib.data.loader.Alpha158DL.get_feature_config` 的官方配置，并交给 `D.features` 实际执行。公开公式、有效观测数量和覆盖率；158项经济有效性与全部公式的独立验证尚未完成。**生成158个特征，不等于得到158个独立有效因子，更不等于已训练出更好的投资策略。**

## 诊断和保存

标签为形成日后的下一交易日开盘入场、20个交易日后开盘退出。逐日按横截面计算Pearson IC、Spearman Rank IC；有效股票至少10只，常数截面保持NaN；分组先按当日分数形成，再匹配未来标签。末端未实现标签不伪造，重叠20日均值不连乘成可交易净值。

- 私有：`data/private/qlib_bridge/<run_id>/` 包含provider、逐股表达式、Alpha158与来源子集。
- 公开：`evidence/qlib_bridge/` 包含聚合IC、分组、覆盖率、配置、包版本和校验值；不包含厂商原始数据。
- 每次运行保留 `history/<run_id>/`；只有8个独立对照全部通过才更新当前展示证据。
- 缺少本地Qlib环境或数据的电脑可以查看保存的真实聚合证据；页面不会把该状态描述为已经在此机重跑。

## 一手来源

已依据以下官方材料核对API和格式，并以安装包实际执行确认：

- [Qlib官方项目与安装说明](https://github.com/microsoft/qlib/tree/v0.9.7)
- [Qlib数据层与自有CSV转换说明](https://qlib.readthedocs.io/en/latest/component/data.html)
- [v0.9.7官方dump_bin数据布局](https://github.com/microsoft/qlib/blob/v0.9.7/scripts/dump_bin.py)
- [v0.9.7 Rolling/Std/Ref算子源码](https://github.com/microsoft/qlib/blob/v0.9.7/qlib/data/ops.py)
- [v0.9.7 Alpha158DL公式生成源码](https://github.com/microsoft/qlib/blob/v0.9.7/qlib/contrib/data/loader.py)
- [Tushare日线字段：vol手、amount千元](https://tushare.pro/document/2?doc_id=27)

Qlib为MIT许可。项目没有复制其运算引擎或将自写pandas计算包装成Qlib；本地bin适配层参考公开格式，表达式运算由正式安装包执行。
