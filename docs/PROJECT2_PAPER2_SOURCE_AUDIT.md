# Project 2 Paper 2 来源、复现边界与平台接入审计

> 审计对象：`2511.18850v4.pdf`（第二条独立 Project 2 复现路线）
>
> 审计日期：2026-10-19。本文只记录来源核验、论文数值、可复现边界和接入建议；不把论文附件中的实验设置自动当作课程硬性要求。

## 1. 结论先行

这篇论文是 **Cognitive Alpha Mining via LLM-Driven Code-Based Evolution**，提出 CogAlpha：由 OHLCV 数据出发，让多层 LLM agent 生成可执行 alpha 代码，用质量检查和信息泄漏检查筛选，再用 mutation/crossover 迭代演化，最后用 Ridge 或 LightGBM 组合多个 alpha。论文的主实验以 CSI300、10 日预测期为主，报告了 CogAlpha + LightGBM 的 IC=0.0591、RankIC=0.0814、ICIR=0.3410、RankICIR=0.4350（PDF 第 7 页 Table 1；跨数据集的完整结果见第 19 页 Table 5）。

**可以借鉴，但不能声称按原论文完整复现。** 作者公开仓库只提供 prompt 模板，明确不包含运行时、数据、实验输出、私有端点或 API key。原论文使用本地 `gpt-oss-120b`、单张 H100、21 个 agent、较大的进化池和多轮迭代；普通课程电脑上不适合直接照搬。青序平台已有安全 factor DSL、因子候选实验、滚动训练和可切换 LLM 研究接口，因此适合做一条**离线、可复核、无密钥的 CogAlpha-inspired 路线**：把七层 agent 的思想转成因子卡片和受限 DSL 候选，通过统一的质量/泄漏检查、因子指标和小规模 mutation/crossover 产生候选；可选的 LLM 只负责提出表达式和解释，不能直接执行任意 Python 或决定持仓。

建议第二条路线的正式表述是“基于 CogAlpha 机制的受限可复现改造实验”，并与 RSRS 主线保持独立报告、独立回测和独立 PPT；二者共享青序的数据、指标、审计和界面基础设施。论文中的 Table 1/5 数值只能作为文献基准，不能直接当作本地 Tushare 快照的结果。

## 2. 来源和身份核验

### 2.1 本地原始 PDF

| 项目 | 核验结果 |
|---|---|
| 文件 | `D:/Desktop/2511.18850v4.pdf` |
| SHA-256 | `20C6C76A0E9EA5FDF869EEA0A20A3594AB61181FEE9913B1EC9361B3669ECEA45` |
| PDF 页数 | 35 |
| PDF 标题元数据 | Cognitive Alpha Mining via LLM-Driven Code-Based Evolution |
| 作者元数据 | Fengyuan Liu; Yi Huang; Sichun Luo; Yuqi Wang; Yazheng Yang; Xinye Li; Zefa Hu; Junlan Feng; Qi Liu |
| arXiv | [arXiv:2511.18850v4](https://arxiv.org/abs/2511.18850v4) |
| DOI（预印本元数据） | [10.48550/arXiv.2511.18850](https://doi.org/10.48550/arXiv.2511.18850) |

使用 bundled Python `pdfplumber` 逐页抽取文本，并用 Poppler `pdftoppm -r 150` 渲染关键页。第 5、7、12、16、17、18、19、20 页渲染图已在临时审计目录中视觉检查；表格和图形清晰可读。

### 2.2 正式出版来源和代码来源

- ACL Anthology 记录：[2026.acl-long.538](https://aclanthology.org/2026.acl-long.538/)，正式标题和九位作者与本地 PDF 一致；正式 PDF：[aclanthology.org/2026.acl-long.538.pdf](https://aclanthology.org/2026.acl-long.538.pdf)。
- 作者 Fengyuan Liu 的公开 GitHub 账号：[github.com/uwFengyuan](https://github.com/uwFengyuan)。
- 作者公开仓库：[CogAlpha_Prompt](https://github.com/uwFengyuan/CogAlpha_Prompt)。核验的 `main` 分支 commit：`e6f422ddd8250f03b6c887bcf3a82ce8507c46e4`；README 原文快照：[raw README](https://raw.githubusercontent.com/uwFengyuan/CogAlpha_Prompt/e6f422ddd8250f03b6c887bcf3a82ce8507c46e4/README.md)。

作者仓库 README 明确说明：公开内容是 prompt templates，涵盖 Seven-Level Agent Hierarchy、Multi-Agent Quality Checker 和 Thinking Evolution；**不公开 runtime code、datasets、experiment outputs、private endpoints、API keys 或 local paths**。因此该仓库可用于核对提示词层级和方法命名，不能作为原论文运行时的完整复现包。

搜索到的 `MapleCity1314/Alpha-Agent`、`thesimbl9/CogAlpha-Repro`、`GK0421/cogalpha-research-mvp` 均为第三方项目；它们可以作为工程思路参考，但没有作者官方背书，不能用来宣称复现原论文，当前审计也没有复制其代码。

## 3. 论文方法的可核验内容

### 3.1 数据、表示和七层 agent（PDF 第 3--5 页）

论文把每日 OHLCV 作为原始输入，生成的 alpha 是以 `(date, ticker)` 为 MultiIndex、返回 `pd.Series` 的 Python 函数。七层 agent 共 21 个：

1. **Market Structure & Cycle**：`AgentMarketCycle`、`AgentVolatilityRegime`；
2. **Extreme Risk & Fragility**：`AgentTailRisk`、`AgentCrashPredictor`；
3. **Price–Volume Dynamics**：`AgentLiquidity`、`AgentOrderImbalance`、`AgentPriceVolumeCoherence`、`AgentVolumeStructure`；
4. **Price–Volatility Behavior**：`AgentDailyTrend`、`AgentReversal`、`AgentRangeVol`、`AgentLagResponse`、`AgentVolAsymmetry`；
5. **Multi-Scale Complexity**：`AgentDrawdown`、`AgentFractal`；
6. **Stability & Regime-Gating**：`AgentRegimeGating`、`AgentStability`；
7. **Geometric & Fusion**：`AgentBarShape`、`AgentCreative`、`AgentComposite`、`AgentHerding`。

### 3.2 检查、筛选和进化（PDF 第 4--6 页）

Quality Checker 由 Code Quality、Code Repair、Judge、Logic Improvement 四类角色组成。它执行生成代码，检查运行错误、NaN、溢出/下溢、非法 log、不稳定归一化，并进行信息泄漏单元测试。失败或不安全的候选不能进入候选池。

每个因子用 IC、RankIC、ICIR、RankICIR、MI 五项指标评价。同一代中五项都超过 65 分位的因子为 qualified，超过 80 分位的为 elite；论文还给出绝对下界：普通因子 `IC/RankIC >= 0.005`、`ICIR/RankICIR >= 0.05`、`MI >= 0.02`，elite 因子对应 `0.01, 0.01, 0.1, 0.1, 0.02`。这些是论文实验阈值，并非青序必须固定使用的参数。

Adaptive Generation 每代抽取两个有效因子和两个最差无效因子，向下一轮 prompt 提供原因摘要。Thinking Evolution 包含 Mutation Agent 和 Crossover Agent，支持 mutation、crossover、crossover-then-mutation；每代保留前两个 elite。

论文第 5 页 Figure 1 的工作流是：`OHLCV -> 7-L Hierarchy -> Quality Checker -> 5-M Evaluation -> Thinking Evolution -> Quality Checker -> 5-M Evaluation -> Candidates Pool`，并将候选池反馈到下一轮评价。

### 3.3 论文示例因子

第 7--8 页 Listing 1 给出的简单示例为

\[
  \mathrm{Alpha} = \frac{day\_high-day\_close}{day\_volume+\varepsilon}.
\]

初始示例报告 IC=0.0090、RankIC=0.0061。Listing 2 的一个演化版本报告 IC=0.0073、RankIC=0.0021，并被丢弃。第 8 页 Listing 3 的演化示例用绝对的 `close-open` 变化除以 dollar volume（`volume*close+eps`），再用 `tanh` 限制极端值，报告 IC=0.0141、RankIC=0.0087。论文声称完整进化后有不少单因子达到 `|IC|>0.05`、`|RankIC|>0.07`，这是论文自己的汇总性描述，不能当成本地实验保证。

## 4. 原论文实验设定和原表数值

### 4.1 数据集、切分和训练设置（PDF 第 6 页、第 14 页）

主实验以 CSI300 300 只大盘 A 股成分股、10-day forward return 为目标；因子在当日信息上计算，论文写明在开盘价买卖。CSI300/CSI500/S&P500 从 Qlib 下载，HSI/HSCI 从 Yahoo Finance 下载。CSI 数据集的切分为 train 2011-01-01--2019-12-31、validation 2020-01-01--2020-12-31、test 2021-01-01--2024-12-01；S&P500 为 2007--2014 / 2015 / 2016--2020-12-01；HSI/HSCI 为 2011--2019 / 2020 / 2021--2025-12-01。论文附录 B.1 对市场规模和来源有说明，但未提供可直接校验的 constituent snapshot、数据版本、复权方案和完整下载清单。

默认 LLM 是本地 `gpt-oss-120b`；任务 agent 和 evolution agent 的 temperature 从 `{0.7, 0.8, 0.9, 1.0, 1.1, 1.2}` 随机选择，quality checker 固定 0.8，max tokens 4096。LightGBM 是生成 alpha 的默认训练器。初始池 80，parent pool 32，children pool 96（3×32），每个任务 agent 24 代、每代 3 个 inner sub-cycles，每 2 代注入新因子；失败或 NaN 比例超过 30% 的候选丢弃，前两个 elite 保留。实验使用 NVIDIA H100；论文称单代在 CSI300 上约 1 小时。

### 4.2 Table 1：CSI300、10-day（PDF 第 7 页）

下表逐项抄录原表，指标顺序为 `IC, RankIC, ICIR, RankICIR, AER, IR`。论文说明生成 alpha 的结果是由 20 个生成 alpha 组合而成。

| 模型 | IC | RankIC | ICIR | RankICIR | AER | IR |
|---|---:|---:|---:|---:|---:|---:|
| Linear | 0.0165 | 0.0211 | 0.1612 | 0.1655 | -0.0076 | -0.0756 |
| MLP | 0.0227 | 0.0327 | 0.2227 | 0.3037 | 0.0678 | 0.9351 |
| RandomForest | 0.0240 | 0.0410 | 0.2932 | 0.4385 | 0.0784 | 0.8381 |
| LightGBM | 0.0269 | 0.0412 | 0.2811 | 0.3327 | 0.0878 | 1.0980 |
| XGBoost | 0.0257 | 0.0376 | 0.2783 | 0.4093 | 0.1081 | 1.3166 |
| CatBoost | 0.0197 | 0.0239 | 0.2196 | 0.3043 | 0.0462 | 0.5373 |
| Adaboost | 0.0187 | 0.0284 | 0.2709 | 0.3369 | 0.1138 | 1.2633 |
| Transformer | -0.0090 | -0.0022 | -0.0800 | -0.0181 | 0.0492 | 0.6361 |
| GRU | 0.0074 | 0.0176 | 0.0747 | 0.1370 | 0.0335 | 0.3386 |
| LSTM | 0.0096 | 0.0216 | 0.0886 | 0.1619 | 0.0593 | 0.6030 |
| CNN | 0.0268 | 0.0392 | 0.2432 | 0.3117 | 0.0763 | 0.9642 |
| Alpha158 | 0.0358 | 0.0402 | 0.2737 | 0.2866 | 0.0946 | 0.8556 |
| Alpha360 | 0.0200 | 0.0136 | 0.1674 | 0.1067 | 0.1198 | 1.0762 |
| AutoAlpha | 0.0211 | 0.0177 | 0.2030 | 0.1588 | 0.0658 | 0.6307 |
| AlphaAgent | 0.0246 | 0.0289 | 0.2407 | 0.2721 | 0.1072 | 1.2310 |
| Llama3-8B | 0.0121 | -0.0074 | 0.0972 | -0.0540 | 0.0520 | 0.5077 |
| Llama3-70B | 0.0205 | 0.0229 | 0.1786 | 0.1915 | 0.0681 | 0.6312 |
| gpt-oss-20B | 0.0061 | 0.0075 | 0.0613 | 0.0680 | 0.0464 | 0.4885 |
| gpt-oss-120B | 0.0300 | 0.0318 | 0.2501 | 0.2595 | 0.0789 | 0.8015 |
| GPT-4.1 | 0.0118 | 0.0114 | 0.1069 | 0.1037 | 0.0360 | 0.3628 |
| o3 | 0.0019 | -0.0050 | 0.0203 | -0.0475 | 0.0218 | 0.2278 |
| CogAlpha + gpt-oss-120B | **0.0591** | **0.0814** | **0.3410** | 0.4350 | **0.1639** | **1.8999** |

### 4.3 Table 3：消融实验（PDF 第 16 页，Appendix B.7）

`A`=Adaptive Generation，`G`=Diversified Guidance，`H`=Seven-Level Hierarchy，`E`=Thinking Evolution。列顺序为 `IC, RankIC, ICIR, RankICIR`。

| 配置 | IC | RankIC | ICIR | RankICIR |
|---|---:|---:|---:|---:|
| Agent | 0.0300 | 0.0318 | 0.2501 | 0.2595 |
| Agent_E | 0.0219 | 0.0420 | 0.1932 | 0.3322 |
| Agent_EA | 0.0315 | 0.0491 | 0.2568 | 0.3583 |
| Agent_EAG | 0.0414 | 0.0501 | 0.3239 | 0.3599 |
| Agent_EAGH（CogAlpha） | **0.0591** | **0.0814** | **0.3410** | **0.4350** |

### 4.4 Table 4：超参数分析（PDF 第 17 页，Appendix B.10）

配置记为 `P{parent pool}_G{generations per cycle}_H{inner subcycle length}`；列顺序为 `IC, RankIC, ICIR, RankICIR`。

| 配置 | IC | RankIC | ICIR | RankICIR |
|---|---:|---:|---:|---:|
| P16_G24_H8 | 0.0362 | 0.0508 | 0.3168 | 0.3805 |
| P32_G24_H8 | 0.0315 | 0.0475 | 0.2364 | 0.3379 |
| P48_G24_H8 | 0.0199 | 0.0285 | 0.1835 | 0.2480 |
| P32_G24_H2 | 0.0394 | 0.0625 | 0.3128 | 0.4364 |
| P32_G24_H4 | 0.0281 | 0.0477 | 0.2309 | 0.3507 |
| P32_G24_H12 | 0.0340 | 0.0524 | 0.2928 | 0.3759 |
| P32_G8_H8 | 0.0283 | 0.0447 | 0.2413 | 0.3542 |
| P32_G16_H8 | 0.0326 | 0.0433 | 0.2505 | 0.3458 |

论文认为 65--80 分位阈值整体优于 80--90 和 85--95，原因是候选池较宽且较少过早收敛；这属于作者设定下的结果。

### 4.5 Table 5：跨数据集、预测期和训练器（PDF 第 19 页，Appendix B.11）

下表完整保留原表数值，列顺序为 `IC, RankIC, ICIR, RankICIR`。原表的最佳值以粗体显示。

| Dataset | Horizon | Models | Training Method | IC | RankIC | ICIR | RankICIR |
|---|---|---|---|---:|---:|---:|---:|
| CSI300 | 10 days | CNN | CNN | 0.0268 | 0.0392 | 0.2432 | 0.3117 |
| CSI300 | 10 days | Linear | Linear | 0.0165 | 0.0211 | 0.1612 | 0.1655 |
| CSI300 | 10 days | XGBoost | XGBoost | 0.0257 | 0.0376 | 0.2783 | 0.4093 |
| CSI300 | 10 days | CogAlpha | Ridge | 0.0539 | 0.0714 | **0.3471** | **0.4380** |
| CSI300 | 10 days | CogAlpha | LightGBM | **0.0591** | **0.0814** | 0.3410 | 0.4350 |
| CSI300 | 30 days | CNN | CNN | 0.0445 | 0.0644 | 0.4118 | 0.6023 |
| CSI300 | 30 days | Linear | Linear | 0.0222 | 0.0482 | 0.2174 | 0.4218 |
| CSI300 | 30 days | XGBoost | XGBoost | 0.0402 | 0.0507 | 0.4181 | 0.5896 |
| CSI300 | 30 days | CogAlpha | Ridge | 0.0815 | 0.1069 | 0.4408 | 0.5403 |
| CSI300 | 30 days | CogAlpha | LightGBM | **0.0886** | **0.1243** | **0.4933** | **0.6740** |
| CSI500 | 10 days | CNN | CNN | 0.0353 | 0.0490 | 0.2615 | 0.3959 |
| CSI500 | 10 days | Linear | Linear | 0.0273 | 0.0391 | 0.2787 | 0.3671 |
| CSI500 | 10 days | XGBoost | XGBoost | 0.0282 | 0.0290 | 0.2458 | 0.3046 |
| CSI500 | 10 days | CogAlpha | Ridge | **0.0455** | **0.0738** | **0.2903** | **0.4752** |
| S&P500 | 10 days | CNN | CNN | -0.0040 | -0.0028 | -0.0508 | -0.0325 |
| S&P500 | 10 days | Linear | Linear | 0.0031 | 0.0004 | 0.0408 | 0.0045 |
| S&P500 | 10 days | XGBoost | XGBoost | -0.0020 | 0.0045 | -0.0434 | 0.0670 |
| S&P500 | 10 days | CogAlpha | Ridge | **0.0217** | **0.0226** | **0.2189** | **0.1726** |
| HSI | 10 days | CNN | CNN | 0.0218 | 0.0177 | 0.1603 | 0.1248 |
| HSI | 10 days | Linear | Linear | 0.0158 | 0.0142 | 0.1647 | 0.0967 |
| HSI | 10 days | XGBoost | XGBoost | -0.0031 | 0.0003 | -0.0272 | 0.0026 |
| HSI | 10 days | CogAlpha | Ridge | **0.0327** | **0.0400** | **0.1903** | **0.2330** |
| HSCI | 10 days | CNN | CNN | 0.0385 | 0.0242 | 0.4006 | 0.2304 |
| HSCI | 10 days | Linear | Linear | 0.0171 | 0.0142 | 0.2258 | 0.1401 |
| HSCI | 10 days | XGBoost | XGBoost | 0.0295 | 0.0144 | 0.3153 | 0.1690 |
| HSCI | 10 days | CogAlpha | Ridge | **0.0562** | **0.0495** | **0.5396** | **0.4394** |

### 4.6 论文回测定义（PDF 第 15 页，Appendix B.2--B.3）

论文的组合回测是按预测收益排名，选前 50 只股票；每日最多替换 5 个持仓（drop-5），在开盘价交易。买入成本 0.05%，卖出成本 0.15%，每笔交易最低费用 5 元。指标定义为：

- `IC`/`RankIC`：横截面上 alpha 与后续收益的 Pearson/Spearman 相关，再按日期平均；
- `ICIR = mean(IC_t) / std(IC_t)`，`RankICIR` 同理，论文说明采用 Qlib 实现；
- 每日超额收益写作 `r_t = r_port - r_bench - cost_t`；`AER = mean(daily excess) × 252`；`IR = mean / sample std × sqrt(252)`。

“开盘交易”与 10 日标签的精确起止价格在正文中没有给出足够的伪代码；复现时必须明确写出“因子观测日、可交易日、前视窗口和开盘价是否含端点”，并在报告中固定这一选择。

## 5. 对青序平台的可复用部分

当前青序代码已经形成了与论文目标相近但更安全的接入面：

- `src/cfquant/factor_lab.py` 有受限 AST DSL，禁止 `eval/exec/import/attribute/subscript`，只允许受控的 OHLCV 运算、delay、returns、rolling、corr、rank、zscore、clip 和 safe_divide，并对表达式长度、节点数、深度和窗口设上限；`builtin_candidates()` 已包含 16 个从 CogAlpha 思路改写的确定性 OHLCV 种子。
- `summarize_factor()` 能按日期计算 IC、RankIC、IR 和 coverage；标签对齐和截点由调用层负责，适合继续接入论文的 forward-10d / forward-30d 约定。
- `src/cfquant/llm_research.py` 已有 offline、OpenAI-compatible、Ollama 三种可切换接口；请求/响应不应保存密钥，模型文本只能作为未验证的解释或候选提案，不能当作证据。
- `src/cfquant/ui_factor_lab.py` 已有公式实验、候选比较、贡献度、保存模型提案和 LLM 解释入口；这里是展示“CogAlpha-inspired factor evolution”的最小改动接入点。
- `research_lab.py` / `ui_research_validation.py` 已有滚动 Ridge/LightGBM、训练/OOS 分离、因子卡片、泄漏防护和回测基础；第二条路线应复用这些执行和审计模块，不应复制一套未经核验的 backtest。

**安全边界：不能直接执行 LLM 生成的 Python。** 原论文允许生成 Python/TA-Lib，但青序的可维护性和安全性要求把候选先翻译成 DSL，或者只允许人工审阅后转换；不允许把论文 Listing 代码直接接入 `exec`。代码级表示、七层分类和 evolution 思想可以保留，运行时应采用受限表达式和显式算子。

## 6. 最小完整复现范围

为了让第二条 Project 2 在有限计算资源上可审计，推荐把下列内容定义为“最小完整范围”：

1. 固定一个带日期快照的 A 股数据集，优先使用青序已有 Tushare 数据；若不能证明它是论文同版本 CSI300 成分快照，就在报告中称为“CSI300-like / A-share reproduction-adapted”，不要称为 exact CSI300 reproduction。
2. 固定 10-day forward label，因子在 `t` 日收盘后计算，最早在下一可交易日开盘执行；明确 `t+1` 到 `t+10` 的收益定义，避免把当天收盘信息用于当天开盘。
3. 固定 train/validation/test 时间边界，先只报告 OOS test；所有标准化、候选选择、阈值选择和训练器选择只能使用 train/validation。
4. 实现 7 个层级的 deterministic seed cards，每层若干安全 DSL 候选；对候选做运行、NaN/Inf、零分母、窗口预热和历史信息泄漏检查。
5. 使用固定的候选组合协议（例如训练期横截面 IC/RankIC 过滤后等权或 Ridge/LightGBM），在报告中固定阈值、持仓数、调仓、成本和成交时点；不能看完 test 后再挑 winner。
6. 至少报告 IC、RankIC、ICIR、RankICIR、coverage、年化收益、年化波动、Sharpe、最大回撤、换手率和成本，并与 Linear/原始 16 seed/单因子最佳/组合模型及基准比较。
7. 若接入 LLM，第一版只开启“候选提案 + 解释”模式；记录 provider、model、temperature、max tokens、prompt hash 和时间。没有密钥或无法复现时，离线模式仍必须得到完整结果。

论文的 21 agent、80 初始池、32 parent、96 children、24 代×3 sub-cycles、H100 和本地 120B 模型属于论文规模，不是课程演示的必要条件。可将小规模 7 层候选进化称为受限改造，并把缩减原因写进局限性。

## 7. 控制实验矩阵（一次只改变一个主要因素）

不要在 test 上反复挑选最好版本。以下矩阵可作为报告的预注册实验计划；参数可由主 Agent 根据已有数据规模落地，但每格要保留配置文件和结果。

| 编号 | 主要变化 | 固定内容 | 目的 |
|---|---|---|---|
| C0 | 论文启发的 7 层 deterministic seeds | 同一数据、标签、切分、组合模型、执行规则 | 得到无 LLM 的可复核基线 |
| C1 | C0 + bounded mutation/crossover | 只改变候选产生方式，候选数量和筛选阈值固定 | 测量“进化”是否改善 OOS |
| C2 | C0 + LLM proposal（可选） | 使用 C0 的同一 DSL、同一 checker、同一候选预算 | 测量 LLM 提案是否增加有效候选 |
| C3 | 交易成本：0 / 5bp / 论文买卖成本 | 因子与模型均固定 | 分离信号收益与成本敏感性 |
| C4 | 组合模型：等权 / Ridge / LightGBM | 候选池、label、切分固定 | 对照训练器影响，不把训练器收益误归因于因子 |
| C5 | 预热/标准化：仅历史窗口 / 错误全样本（仅作为泄漏演示，不用于结论） | 其余固定 | 验证预热和标准化实现没有未来数据 |
| C6 | 执行时点：下一日开盘 / 当日收盘模拟 | 明确成本和标签，不能混用 | 解决论文开盘成交与平台默认时点差异 |

C5 的全样本标准化只能作为负面泄漏单元测试或教学演示，不能进入最终策略竞赛。主结果至少保留 C0、C1、C3、C4、C6；每次实验只改变表中一个主要因素。

## 8. 不可比差值和主要陷阱

以下差异必须在两套报告中明确，否则容易把不同实验误写成超越论文：

- 论文使用 Qlib CSI300/CSI500 数据或 Yahoo 数据；青序目前主要是 Tushare 日频快照。成分历史、停牌、复权、缺失值、行业和市值字段可能不同。
- 论文的 10-day/30-day 标签、开盘成交、top-50/drop-5 组合和买卖成本与青序原有策略执行器不同；IC 结果和策略收益不能直接横向混合。
- 论文用 20 个 generated alphas 组合，但没有在正文给出每个最终因子的完整、固定源码和随机种子；LLM 输出具有随机性，论文也承认精确复现困难。
- 论文没有公开完整运行时和实验输出；无法获得相同的 `gpt-oss-120b` checkpoint、Qlib 数据版本、成分快照和随机状态时，不能声称再现 Table 1/5。
- 论文的 Figure 3 显示 65--80、80--90、85--95 三类阈值的累计回报，但这不是青序当前主策略的回测；应把它作为原论文图示和阈值动机，不当成本地结果。
- 论文评价的 AER/IR 是超额组合指标，青序的 Sharpe、最大回撤和总收益可能采用不同无风险、频率、成本或成交时点；报告中需同时给公式和配置。
- 原论文生成任意 Python/TA-Lib 代码；青序受限 DSL 不支持任意函数、属性访问或隐式跨股票操作。安全改造后可复现研究结论的结构，但不是原代码执行的逐行复现。

## 9. 原文截图建议

以下坐标以 150 dpi 渲染 PNG 的左上角为原点，图像尺寸为 `1241×1754`；坐标是可用于报告插图的裁切框 `(left, top, right, bottom)`，保留图题/表题和正文署名，禁止裁掉作者、表头或说明文字。裁切前应在最终 PDF 中再次目视核对。

| 目的 | PDF 页 | 建议坐标 | 用途 |
|---|---:|---|---|
| Figure 1：整体工作流 | 5 | `(175, 150, 1085, 785)` | 说明七层 agent、质量检查、五项评价和进化闭环；保留 Figure 1 caption |
| Table 1：CSI300 10-day 基准 | 7 | `(145, 165, 1100, 725)` | 给出论文基准数值；保留所有模型、列头和 caption |
| Figure 3：三组阈值累计收益 | 18 | `(220, 140, 1090, 680)` | 说明阈值选择背景；不要把图中曲线当成本地回测 |
| Table 5：跨数据集结果 | 19 | `(145, 165, 1100, 940)` | 论文主要跨市场数值；保留表头、表题和粗体标记 |

第 12 页 Figure 2 如需要展示进化过程，可使用同一页中图形和 caption 的整栏框 `(150, 145, 1095, 950)`，最终插图以视觉复核结果为准。建议正文只放 Figure 1、Table 1 或 Table 5 的局部，详细数值放附录，以免 15 分钟 pre 过载。

## 10. 报告可使用的严谨表述

可以写：

> 我们复用了 CogAlpha 的七层因子组织、质量检查、信息泄漏检查和候选进化思想，并将生成表达式限制在青序的安全 DSL 中；在本地 Tushare 快照上按预先固定的时间切分和执行规则进行改造实验。

不可以写：

> 我们完整复现了 CogAlpha，并达到论文 Table 1/5 的数值。

除非后续取得相同 Qlib 数据、成分历史、模型 checkpoint、完整运行时、随机状态和标签/交易实现，否则第二种说法没有证据支持。原论文也有伦理说明：数据公开，ChatGPT-5.2 只用于语法检查和建议，AI 生成内容不能直接作为最终提交；课程报告应保留真实实验记录和人工解释。

## 11. 与 RSRS 轨道的关系

第二篇论文应作为独立实验轨道，拥有自己的数据切分、配置、结果、报告和 Beamer/PPT 讲稿；它不应覆盖或替换 RSRS 复现。两条轨道可以共享青序的平台层：数据快照、字段对齐、因子 DSL、质量审计、回测执行、Sharpe/回撤/成本指标、结果保存和 UI 导航。最终总平台可从候选策略中切换 RSRS 与 CogAlpha-inspired 组合，但每条论文结果必须在页面和报告中标注来源、版本和不可比限制。

## 12. 核验命令记录

本审计使用的关键本地核验：

```text
C:/Users/14108/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe
  + pdfplumber 0.11.9：逐页抽取 35 页文本
C:/Users/14108/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe
  + -r 150：渲染第 5、7、12、16、17、18、19、20 页
Get-FileHash -Algorithm SHA256 D:/Desktop/2511.18850v4.pdf
  -> 20C6C76A0E9EA5FDF869EEA0A20A3594AB61181FEE9913B1EC9361B3669ECEA45
git ls-remote https://github.com/uwFengyuan/CogAlpha_Prompt.git refs/heads/main
  -> e6f422ddd8250f03b6c887bcf3a82ce8507c46e4
```

本文件是研究审计，不包含 Token、账号、生成式 API 响应或付费服务调用；临时抽取文本和渲染图片位于 `tmp/paper2_audit/`，不应打包进公开仓库或课程提交包。

## 13. 追加精确核对：Listing 1--3、MI 和阈值方向

本节根据 PDF 第 8 页和 Appendix B.9（PDF 第 17 页）对前文的摘要作代码级补充。下面代码中的 `’` 是 PDF 排版后的引号，实际实现时应替换成普通 Python 引号；代码正文属于论文原文示例，不代表青序允许直接执行。

### 13.1 Listing 1：初始因子（PDF 第 8 页）

```python
def factor_upward_impact_per_vol(df):
    """Liquidity-impact: price rise (high-close) per unit of
    traded volume. A large positive value means the
    stock moved up strongly while volume stayed low,
    indicating thin liquidity and higher expected short-
    term return.
    Formula: (high - close) / (volume + ε)."""
    df_copy = df.copy()
    eps = 1e-9
    df_copy['price_up'] = df_copy['high'] - df_copy['close']
    df_copy['factor_upward_impact_per_vol'] = df_copy['price_up'] / (
        df_copy['volume'] + eps
    )
    return df_copy['factor_upward_impact_per_vol']
```

论文报告该初始版本 IC=0.0090、RankIC=0.0061。它使用的是 `high-close`，不是 `close-open`。

### 13.2 Listing 2：mutation 版本（PDF 第 17 页 Appendix B.9）

```python
def factor_dayhigh_impact_per_vol(df):
    """Price-impact proxy: (high-low) per unit of volume.
    Larger values indicate that price moves a lot while
    little volume trades, signalling thin liquidity."""
    df_copy = df.copy()
    df_copy['price_range'] = df_copy['high'] - df_copy['low']
    df_copy['factor_dayhigh_impact_per_vol'] = (
        df_copy['price_range'] / (df_copy['volume'] + 1e-9)
    )
    return df_copy['factor_dayhigh_impact_per_vol']
```

Listing 2 只把 `high-close` 换成完整日内区间 `high-low`，其余逻辑没有增加复杂变换。论文报告 IC=0.0073、RankIC=0.0021，低于 Listing 1，因此按论文叙述在后续轮次丢弃。这个 literal mutation 是最适合青序首先复现的实例，因为它不依赖 LLM，也不依赖 TA-Lib。

### 13.3 Listing 3：多轮演化示例（PDF 第 17 页 Appendix B.9）

```python
def factor_price_impact_per_vol_tanh_1d(df):
    """Impact proxy: absolute daily price move per dollar
    volume.
    Steps:
    1) Compute absolute price move (|Close-Open|).
    2) Compute dollar volume (Volume*Close).
    3) Form raw impact = absolute move / (dollar volume + ε).
    4) Apply tanh to bound the factor within (-1, 1)."""
    df_copy = df.copy()
    eps = 1e-9
    df_copy.loc[:, "abs_move"] = (
        df_copy["close"] - df_copy["open"]
    ).abs()
    df_copy.loc[:, "dollar_vol"] = df_copy["volume"] * df_copy["close"]
    df_copy.loc[:, "raw_impact"] = df_copy["abs_move"] / (
        df_copy["dollar_vol"] + eps
    )
    df_copy.loc[:, "factor_price_impact_per_vol_tanh_1d"] = np.tanh(
        df_copy["raw_impact"]
    )
    return df_copy["factor_price_impact_per_vol_tanh_1d"]
```

论文报告 Listing 3 IC=0.0141、RankIC=0.0087。这里的“dollar volume”是 `volume * close`，不是单纯的 `volume`；代码使用了当日 `open` 和 `close` 的绝对变化，再用 `tanh` 做有界化。这个示例可以作为第二个 literal baseline，但不要把它误称为论文最终候选池中的唯一因子；论文只把它作为演化过程中的一个可解释示例。

### 13.4 MI 的定义、估计器和正负方向

论文第 15 页公式 (6) 定义

\[
  MI(F,R)=\iint p(f,r)\log\frac{p(f,r)}{p(f)p(r)}\,df\,dr,
\]

其中 `F` 是因子值，`R` 是后续总收益，`p(f,r)` 是联合密度，`p(f)` 和 `p(r)` 是边缘密度。论文只给出这个理论定义和“MI captures nonlinear dependence”的解释，**没有公开具体密度估计器、离散化箱数、kNN 参数、随机种子或 Qlib 代码调用**。因此不能把 sklearn 的某一个 `mutual_info_regression` 配置说成论文的 exact estimator。

筛选方向是“越大越好”，不是按绝对值筛选：论文写的是五个指标都超过同代经验分布的 percentile，且对指标施加下界。对 CSI300 等默认任务，下界为普通候选 `IC >= 0.005`、`RankIC >= 0.005`、`ICIR >= 0.05`、`RankICIR >= 0.05`、`MI >= 0.02`；elite 候选为 `IC/RankIC >= 0.01`、`ICIR/RankICIR >= 0.1`、`MI >= 0.02`。因此负 IC 或负 RankIC 不会因“绝对值很大”而合格。百分位阈值和下界应写成明确的 `score >= max(empirical_quantile, minimum_bound)`，并在同代候选中计算，不能跨训练、验证、测试混算。

论文 Appendix B.4 另说明 S&P500 使用较低的 MI 下界：普通候选 `MI >= 0.012`，elite 候选 `MI >= 0.012`；IC、RankIC、ICIR、RankICIR 的下界仍分别为 0.005、0.005、0.05、0.05，elite 的前四项分别为 0.01、0.01、0.1、0.1。前文表述的 `MI >= 0.02` 仅适用于 CSI300/CSI500 等默认设置，不应无条件套用到 S&P500。

论文的五项指标使用方式是：IC/RankIC/ICIR/RankICIR 以及 MI **全部**通过同一候选的门槛，才进入 qualified/elite；不存在“只要 MI 高就保留”的规则。论文没有说明对每代极小样本如何处理 MI 估计不稳定，青序实现应报告样本数、空值和估计器配置。

## 14. Qlib 数据可用性：官方包与公开社区归档要分开

### 14.1 官方 Qlib 当前状态

核对 Microsoft Qlib 官方仓库当前 README（[Data Preparation](https://github.com/microsoft/qlib#data-preparation)，检索日期 2026-10-19）可得：官方数据集因更严格的数据安全政策**暂时禁用**；README 建议尝试社区数据源，并写明未来会恢复。官方代码仍保留 `scripts/get_data.py` / `qlib.cli.data` 的 CN 1d 下载入口，但这不等于当前官方二进制 URL 可成功下载。

官方文档还确认 Qlib 的 CN 数据格式包含 `features`、`calendars` 和 `instruments`，`csi300.txt` 等 instrument 文件带有股票开始/结束日期区间；官方示例可以查询 `D.instruments(market='csi300')`。但官方文档没有给出当前 CN bundle 的固定 release hash、下载字节数或“2011--2024 一定存在”的保证。Qlib 的 adjusted OHLCV 也有自己的调整约定，官方文档提醒不同数据源的复权价格可能不同。

### 14.2 当前公开社区归档的可行性和大小

Qlib README 推荐的社区项目是 [`chenditc/investment_data`](https://github.com/chenditc/investment_data)，不是 Microsoft 官方发布物。其 README 说明数据表可来自 Tushare、AkShare、Yahoo、Baostock，并将合并后的结果转为 Qlib bin；仓库 license 是 Apache-2.0。该项目当前 latest release 为 `2026-10-18`，公开 manifest 记录：

| 字段 | 值 |
|---|---|
| archive | `qlib_bin.tar.gz` |
| compressed size | `568,280,701` bytes，约 542 MiB（十进制约 568 MB） |
| target trade date | `2026-10-16` |
| future range in manifest | `2026-10-19`--`2026-12-31` |
| archive SHA-256 | `a11eaba954f2d5658c017fa800eea85bb6277f8ffc3d7b0449f791a44e43e7dd` |
| release | [2026-10-18](https://github.com/chenditc/investment_data/releases/tag/2026-10-18) |
| manifest | [qlib_bin.manifest.json](https://github.com/chenditc/investment_data/releases/download/2026-10-18/qlib_bin.manifest.json) |

该归档明显可以作为当前工程的公开 Qlib-format 数据来源，并且项目的索引脚本会生成动态的 `csi300.txt` / `csi500.txt` 区间。但 manifest 只记录 release/target/future 元数据，没有在不下载归档的情况下证明 `calendars/day.txt` 的最早日期或每一条 `instruments/csi300.txt` 的完整历史。因此现阶段严谨结论是：**存在一个约 542 MiB、更新到 2026-10-16 的公开 CN Qlib-format 社区归档；是否完整覆盖 2011--2024 及其成分历史，必须下载后读取 `calendars/day.txt` 和 `instruments/csi300.txt` 验证，不能只根据 release 日期推断。**

Qlib 官方仓库曾在 issue/文档中使用过约 196.5 MB 的 `v2/qlib_data_cn_1d_latest.zip` 示例大小，但当前官方数据已禁用；该历史字节数不能作为当前下载保证。正式复现优先使用已有 Tushare 快照并保存自己的 hash，或明确采用上述社区归档并报告其来源、commit、release 和复权规则。原论文的“从 Qlib 下载”不自动等于“与当前社区归档完全相同”。

### 14.3 成分历史的风险

`csi300.txt` 的区间记录可以表达动态成分，但 Qlib 的 instrument 文件不等于指数权重、自由流通市值或 point-in-time 研究数据库。官方仓库的公开 issue 也记录过 CSI300 instrument 数量和历史区间疑问。复现时应检查：

1. 每个 `(instrument, start, end)` 区间是否有重叠或断裂；
2. 成分切换日是否与指数公告/数据源一致；
3. 训练、验证、测试是否只使用当时可知的成分；
4. benchmark 是 Qlib 的指数序列、等权成分组合还是市值加权指数。

如果不能完成这四项检查，论文结果只能标为 data-adapted reproduction；使用 Tushare 2020--2026 快照时更应明确这是跨样本的机制复现。

## 15. CogAlpha prompt 的许可和归属

作者 prompt 仓库的 [LICENSE](https://raw.githubusercontent.com/uwFengyuan/CogAlpha_Prompt/e6f422ddd8250f03b6c887bcf3a82ce8507c46e4/LICENSE) 是 **MIT License，Copyright (c) 2026 Fengyuan Liu**。MIT 条款允许使用、复制、修改、合并、发布和再许可，但在源码或实质性部分中必须保留版权声明和许可声明，并保留免责声明。

因此青序可以借鉴 prompt 的层级和字段结构；如果把作者 prompt 的实际句子或较长片段复制进仓库、报告或 PPT，应：

- 在相应文件中保留 MIT copyright/license notice；
- 在报告参考文献或代码注释中注明 `uwFengyuan/CogAlpha_Prompt`、commit `e6f422ddd8250f03b6c887bcf3a82ce8507c46e4` 和访问日期；
- 说明平台采用了安全 DSL 的改写，未经作者仓库提供的 runtime、数据和实验输出不能称为原实现；
- 不把作者仓库的 prompt 误写成 Project 2 课程要求，也不把第三方实现的代码当作官方代码。

若只使用“七层组织、五项指标、候选进化”这些方法概念而不复制文本，则仍应引用论文；MIT 的代码/文本归属要求主要针对复制到项目中的 prompt 内容。

## 16. 可直接实施的最小严谨复现设计（第二条轨道）

这是在不调用付费模型、不借用 API key 的情况下，最小而可审计的实施方案。它的目标是复现机制和验证工程管线，不能承诺复现论文主结论。

### 16.1 数据和时间边界

- 数据：使用已有 Tushare 日频快照，建议固定一个只读、带 SHA-256 的 parquet/CSV 快照；当前建议区间为 2020--2026。若实际快照不足七年，报告以真实起止日期为准。
- 股票池：以快照可证明的 A 股股票池为主；如果没有 point-in-time CSI300 成分历史，命名为 `Tushare A-share mechanism reproduction`，不要伪称 CSI300。
- 标签：固定 `forward_10d_open`，因子使用 `t` 日收盘前可知 OHLCV，执行从 `t+1` 开盘开始，十日标签的端点定义写入配置和报告。
- 切分：按时间固定 train/validation/test。已有 Tushare 2020--2026 只用于跨样本机制复现；不能与论文 2011--2019/2020/2021--2024 的数字直接比较。

### 16.2 原 literal、无进化和同预算进化

固定一个小而公开的预算，例如 3 个种子（Listing 1、Listing 2、Listing 3）+ 7 层各 2 个 DSL seed，共 17 个初始候选；所有实验都使用同一候选预算、同一 label、同一切分和同一执行器。

| 组别 | 候选产生 | 预算和选择 |
|---|---|---|
| L0 literal | 原 Listing 1/2/3 的安全 DSL 翻译 | 不进化，保留全部通过 checker 的候选 |
| N0 seeds-only | 七层 deterministic seed cards | 与 L0 相同的检查、候选数和模型 |
| E0 seeds + evolution | N0 候选做有限 mutation/crossover/crossover-then-mutation | 同一 parent/children 上限，固定随机种子，最多 2 轮；每轮保留 elite 和少量 qualified |
| A0 optional LLM | 运行前明确 provider/model；只返回 DSL 候选和解释 | 与 E0 使用同一候选预算和 checker；没有模型时保持 offline 空位，不影响主结果 |

L0 是最重要的 sanity check：先确认论文 literal 公式在青序标签和执行时点下可以计算，避免把后续进化收益归因给论文。E0 每次只改变“是否进行有限进化”，不同时改变 costs、label、标准化或训练器。A0 只作为辅助实验，不得替代 deterministic 主结果。

### 16.3 五项指标和训练器

对每个候选按日期横截面计算 IC、RankIC、ICIR、RankICIR；MI 作为可选第五项，但必须固定估计器、样本数、随机种子并在报告中写清楚。候选门槛按论文方向实现：每一项都要求 `>=` 同代 percentile 与数据集 minimum bound 的较大值；若暂时无法可靠估计 MI，则报告“4-metric provisional selection”，不要冒充论文 5-metric reproduction。

在训练集上构造候选和选择阈值，在 validation 上固定组合权重/模型超参数，test 只运行一次。对同一候选池分别使用：

- `Ridge(alpha=10)`，对应论文 Appendix B.4 的默认正则强度；
- `LightGBM`，若环境可用则采用论文记录的 learning rate `0.0001`、32 leaves、max depth 12、1000 trees、`reg_alpha=reg_lambda=1.0`，并明确本地 LightGBM 版本。

主结果至少报告 L0/N0/E0 三组和两种训练器；A0 的结果单独列为“可选模型辅助”。不允许在 test 上根据收益、Sharpe 或回撤挑选组别。

### 16.4 回测和审计输出

同时保存：配置 JSON、数据 hash、候选表达式、checker 结果、预热行数、每日期标签范围、模型版本、训练/验证/test 指标、交易成本、组合净值、最大回撤和日志摘要。论文 top-50/drop-5 与青序当前策略执行器不同，若实现该组合规则，要在页面显示“paper-adapted execution”；如果沿用青序策略，必须单独标注“platform execution”。

最终可作的严谨结论是：

> 在已有 Tushare 2020--2026 样本上，我们复现了 CogAlpha 的 literal 因子、七层组织、质量检查、五指标筛选和有限进化机制，并比较了无进化与同预算进化在 Ridge/LightGBM 下的 OOS 表现。

不能作的结论是：

> 我们在当前样本上完整复现并验证了 CogAlpha 在论文 CSI300 2011--2024 设定下的主结论。

后一说法要求相同数据、成分历史、标签、Qlib 版本、模型和随机状态；当前证据不足。
