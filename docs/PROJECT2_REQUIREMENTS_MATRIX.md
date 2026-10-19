# Project 2：课程要求与两条独立复现路线

小组：12412408 胡博文、12410819 谢明飞、12410436 王天希。用户确认的截止日期为 2026-11-01，演示按约 15 分钟准备；这些信息不来自课件。

**研究状态：正式实验和材料完成，2026-10-19。** RSRS正式矩阵已有198组结果、账本核对和四件演示材料。CogAlpha三次提案重复、90组投资实验及113项公开证据文件已完成，90项账本检查全部通过；16页报告、16页Beamer、8页讲稿PDF及16页可编辑PPTX全部完成编译/结构核验和全页视觉检查。实际浏览器验收和正式审计六部分均已通过，下表分别说明研究证据与发布状态，不代表已提交Blackboard或教师评分。

课程依据为用户提供的 8 页 `CF2026_Project2.pdf`，SHA-256：

```text
346219417ef143471a8461102dfeef06e1560a1eecd08698f320ad2582fc1ad2
```

课程要求复现一篇方法明确的研究，不要求同时复现两篇，也不要求取得正收益。我们把 RSRS 作为完整研报复现，将 CogAlpha 作为另外一份受限论文机制复现，分别保留协议、实验、报告和演示材料。两项结果不合并成一条策略，不以 CogAlpha 的结果替换 RSRS 的结论，原 V3 年度 LightGBM 主线继续保留。

## 逐页映射

以下是课程要求的概括；原研报、论文和第三方仓库中的建议不自动成为课程要求。

| 课件页 | 正式要求 | RSRS 对应实现与证据 | CogAlpha 对应实现与证据 |
| --- | --- | --- | --- |
| 1：课程主线 | 在 P1 平台上把研究核心结论变成可核验实验。 | 平台“研报复现 · P2”；`src/cfquant/rsrs.py` 接入既有 `engine.py` / `analytics.py`，`rsrs_study.py` 编排。 | 独立“论文复现 · CogAlpha”；`cogalpha.py` / `cogalpha_models.py` / `cogalpha_mining.py` / `cogalpha_study.py` 复用 DSL、数据与回测模块。 |
| 2：目标 | 说明原方法、实现、可比范围与有证据的差异；未复现原结论也有价值。 | [来源审计](PROJECT2_SOURCE_AUDIT.md) 明确公式与未交代细节；`reproduction_table.csv` 同列原值、接近原样本的补充计算与共同窗口结果。 | [论文来源审计](PROJECT2_PAPER2_SOURCE_AUDIT.md) 区分原文事实和本地替代设定；页面突出“受限机制复现 / 与原表非直接可比 / 已观察历史”。原表数值不作为本地成功门槛。 |
| 3：选题 | 推荐多类选题及研报、arXiv 等来源；优先方法明确、数据可得、范围适中。 | 光大证券 2017-05-01 RSRS 研报：市场择时，选定图 25、表 5、图 26；完整来源及 PDF 哈希见来源审计。 | arXiv `2511.18850v4`：因子挖掘；原例公式、质量检查、五指标筛选和有限进化。作者prompt固定提交写入配置；冻结配置中65字符论文哈希笔误由独立 `project2_cogalpha_source_erratum.json` 纠正并在 `source_reference.json` 引用，保留运行身份。推荐资源不是必须接入的系统。 |
| 4：平台、配置、小样本核验、运行证据 | 复用 P1；样本/窗口/费用可追溯；独立核验公式、中间结果与指标；保存快照、版本、种子。 | `configs/project2_rsrs.json`；`formula_checks.json`、`formula_ols_points.csv`、`validation.json`；198 组账本检查，162 组与独立旧实现比较；`manifest.json` 保存输入、代码及输出哈希。 | `configs/project2_cogalpha.json`；DSL白名单、前缀/未来扰动检查、标签退出边界、MI与冻结模型测试。`manifest.json`封存113项公开文件，`validation.json`记录90项账本检查全部通过；不把API连通记录当作实验核验。 |
| 5：原/新结果对照 | 同列条件、目标图表、原值/新值/可比差值及中间证据；说明单位、费用、年化。 | `reference/` 原图；`reproduction_table.csv`、`performance.csv`、`daily/`；“原文对照”页与独立报告并排展示，标明原表不能与共同窗口机械相减。 | 原文表格/Listing与本地实验分别标明数据、模型、指标口径；`factor_metrics.csv`、`pool_summary.csv`、`performance.csv`、`daily/`均已生成。不能把新样本数值放入“复现原表成功”一栏。 |
| 6：差异归因 | 先排查错误，再分别改变样本、预处理、交易或补充假设；列出未解释项，不机械相加。 | `diagnostics.csv`、`diagnostic_checks.json`、`source_compatible.csv`：标准化是否含当日、标准差自由度、MA 索引、执行时点、费用等单项对照；确切 Wind 样本仍不可得。 | 固定16公式、21初代、49槽位新提案、49槽位进化；同候选池的rank/Ridge/LightGBM、指定成本/零成本；`comparison.csv`有18项预设比较，质量与选择记录完整保留。数据、原标签、MI估计器和服务模型差异不能由这些消融完全识别。 |
| 7：自主改进 | 核心复现后改进一项因子/策略，保留原方案，在相同样本/成本/评价口径比较；保留失败。 | 原 `right_ma`，改进 `right_ma_vol15`：过去20日、15%波动目标；加入 `right_ma_cash75` 固定仓位对照。`paired_comparison.csv`、`robustness.csv`、`bootstrap.json` 同时报告收益、风险和换手。 | `llm_evolved_diverse` 在同一进化候选池增加训练期日 Rank IC 相关去冗余，阈值 `abs(corr)≤0.8`；与 `llm_evolved49` 保持其余设定。此项是独立扩展，不能混同论文原方法或事后赢家替换。 |
| 8：提交与演示 | 原文与协议；P1 入口、依赖、配置、数据获取/快照说明；原/新对照、归因、改进、局限、下一步。 | `reports/project2_rsrs/` 四件材料及 LaTeX 源码；[复现说明](PROJECT2_REPRODUCIBILITY.md)、协议、公开聚合证据；完整材料已经生成，最终包与验收以发布清单为准。 | `reports/project2_cogalpha/`独立四件材料、源码、协议和聚合证据已生成，四件材料均已核验；14页正文加2页备份配约15分钟讲稿。浏览器交互及材料下载验收通过，正式发布审计六部分通过，两份材料按各自研究组织。 |

## 两份交付的边界

| 项目 | RSRS | CogAlpha |
| --- | --- | --- |
| 平台入口 | 研报复现 · P2 | 论文复现 · CogAlpha |
| 主实验问题 | 原择时结论能否复现；波动控制是否改善风险收益权衡 | 同候选槽位预算下，训练反馈进化是否优于持续产生新公式 |
| 主比较 | 沪深300，发表后，双边各10bp；原策略/15%波动目标/固定75%仓位 | 三次提案重复；LightGBM；历史留出期；evolved49 对 fresh49 的 Rank IC |
| 数据 | 3个价格指数至2026-10-15；指数是研究代理 | 1000股静态Tushare快照至2026-09-18；不是历史CSI300成分 |
| 公开证据 | `evidence/project2_rsrs/` | `evidence/project2_cogalpha/` |
| 本地完整运行 | `runs/project2_rsrs_formal/` | `runs/project2_cogalpha_formal/` |
| 报告与演示 | `reports/project2_rsrs/CF2026_Project2_*` | `reports/project2_cogalpha/CF2026_Project2_CogAlpha_*` |
| 历史性质 | 已观察的回溯研究；不是独立前瞻盲测 | 已观察的历史留出；另有大模型预训练知识污染可能 |

CogAlpha采用的沪深300比较线是价格指数，股票组合采用复权研究单位且扣费，二者不是完全一致的全收益口径；不能将相对此价格代理的结果写成已经跑赢同口径全收益指数。`deepseek-flash`是API别名，当日官方版本映射不能固定服务商权重；保存公式可离线重新评价，重新抽取API提案不保证同结果。

原样本/原字段无法重建的部分必须报告为不可直接比较，不能用“策略收益为正”“模型真的调用过”或“账本通过”替代论文结论的检验。账本正确仅说明所声明的执行假设内部一致。

本轮CogAlpha主比较未支持进化更优：指定成本、LightGBM、2025-01-02至2026-09-18下，evolved49相对fresh49三次Rank IC差值全部为负，平均−0.006788；三次年化收益差值平均−0.78个百分点。去冗余扩展在2/3次重复提高Rank IC，但年化收益仅1/3次提高，平均差值−1.07个百分点。平均值是三次差值的算术平均，不是一条混合策略收益；三次重复和已观察历史不能证明普遍优劣。完整失败和未改善结果保留，原V3主线不变。

## 验收记录与交付边界

- RSRS：正式证据 `manifest.json` 记录198组主运行；`validation.json` 记录所有主账本通过，独立旧实现核对162组。共享执行器扩展后198组重放一致，浏览器已验收；各自记录为 `release_engine_equivalence.json`、`ui_acceptance.json`。报告、Beamer PDF、可编辑 PPTX、讲稿 PDF 已有文件。
- CogAlpha：三轮共231槽位，197通过质量检查、34拒绝；236提案尝试加1次连通检查，共237次成功调用，保守估费1.312712元、未知预留0。90组组合评价与18项比较完成，账本全部通过，113项清单哈希核对通过；质量通过数不等于五指标合格数或有效策略数。公开审计器 `scripts/audit_project2_cogalpha.py --final` 已独立核对来源、候选、预算、保存账本及比较，六个检查部分全部通过且没有新增API；正式回执为 `evidence/project2_release/cogalpha_release_audit.json`，实际浏览器回执为同目录的 `cogalpha_ui_acceptance.json`。
- 页面默认不加载1000股完整原始表；缺私有数据仍可浏览已发布聚合证据。CogAlpha 页没有模型密钥输入，也不会自动发起付费请求。
- 四件材料中的成员、方法、数值和图注必须与各自证据一致。正式压缩包、文件清单、最终测试/视觉验收及 Blackboard 真实提交状态单独记录，不从目录存在推断已完成。
- 当前自动测试跨记录环境去重后462项全部通过、0项未解决跳过；原Qlib跳过项已在独立Qlib环境实际通过，见 `evidence/project2_release/test_acceptance.json`。这不是一次全套0跳过的运行。审计器另有10项合成自检，不代替正式研究审计。公开模型元数据不含拟合系数/截距或完整模型权重；私有数据、逐股预测及API缓存不进入公开提交。
- 已有PPTX可直接编辑，但源码重建需要artifact工具运行环境；三份TeX可在具备CTeX/Beamer与字体的XeLaTeX环境中独立编译，不依赖模型API或artifact运行时。图表与原相对路径一并交付。
- PPTX部分曲线有约140点的显示抽样，完整日频回测、指标和每日CSV没有抽样。离线重算示例使用新 `runs/reproduction_cogalpha/` 目录，保留已发布证据；打包预检要求正式审计通过、审计脚本身份一致与代码已提交。

历史 `projects/project2_rsrs/` 保留独立旧实现与预研资料，供核验；本轮正式入口是仓库根目录下的 `scripts/run_project2_rsrs.py`。旧版 V2/V3/V4 和 v2.3.0 报告是历史平台档案，不替代这两份 P2 交付。
