# Project 2：查看、核验与复跑

**研究状态：2026-10-19，正式实验和材料完成。** RSRS正式证据和四件材料已生成；CogAlpha三次提案重复、90个投资组合实验和113项公开证据文件完成，90组账本全部通过、清单哈希核对通过。CogAlpha的16页报告、16页Beamer、8页讲稿PDF及16页可编辑PPTX全部完成编译/结构核验和全页视觉检查，实际浏览器验收和正式发布审计六部分均通过。代码远程发布与正式提交包生成由最后的发布步骤另行确认；具体身份与状态见各自 `manifest.json`、`validation.json` 和验收回执。

小组：12412408 胡博文、12410819 谢明飞、12410436 王天希。

## 1. 环境与页面

所有命令从青序仓库根目录执行。本机使用入口为 `D:/Desktop/cf2026-quant-platform`；完整CogAlpha私有运行和唯一付费账本已迁入该根目录下的 `runs/`，逐文件SHA-256核对通过。本轮计算/构建曾在 `D:/Downloads/MM-LaTeX/mm_project1-main/work/qingxu-project2-formal` 完成，该路径仅保留为构建来源历史，不再含可继续付费的账本。正式代码同步与GitHub推送由发布步骤另行确认，不能仅凭私有数据迁移推断源码已发布。`--data-root`可以指向另一份只读私有快照根目录，不等于当前代码目录；以下相对路径始终以实际代码根目录为准。

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-research-lock.txt
.venv/Scripts/python.exe -m pip install -e ".[dev,research]"
.venv/Scripts/python.exe scripts/run_project2_rsrs.py --help
.venv/Scripts/python.exe scripts/run_project2_cogalpha.py --help
.venv/Scripts/python.exe -m streamlit run app.py
```

`pyproject.toml` 允许 Python ≥3.11；锁定依赖包括 NumPy 2.3.5、pandas 2.3.3、SciPy 1.16.3、Streamlit 1.51.0、scikit-learn 1.7.2、LightGBM 4.7.0、PyArrow 21.0.0。Python3.12/3.13是当前说明建议环境；依赖文件是复跑环境依据，不是“任何系统均已测试”的承诺。LaTeX 编译与可编辑 PPTX 构建另需对应工具，不属于查看平台的必要依赖。

平台默认仍为“研究总览”，原年度 LightGBM 主线保留。两个独立入口是：

| 页面 | 默认比较与可用操作 |
| --- | --- |
| 研报复现 · P2 | 沪深300、发表后、单边10bp；原文对照、策略切换、差异诊断、CSV/材料下载。本地有指数快照时可单独参数回放。 |
| 论文复现 · CogAlpha | 历史留出期、LightGBM、指定成本、evolved49，并列同预算fresh49；可切重复、模型、时期、成本及候选，查看失败公式、负收益、预算摘要。 |

两个页面读取公开聚合证据，不自动加载整份股票面板，也不自动调用外部模型。空证据显示未完成状态。RSRS 缺私有行情时参数回放禁用；CogAlpha 页是研究结果查看器，训练/提案由明确的命令行阶段完成。

## 2. 数据、来源与文件身份

公开仓库不包含原始授权行情、Tushare Token、模型密钥、逐股预测、完整持仓账本或付费 API 私有缓存。`data/private/`、`runs/` 和凭据文件被 Git 忽略。公开 `evidence/` 保存研究聚合序列、配置、方法检查和身份哈希；它不能替代原始数据重新训练。

公开 `model_metadata.json` 只保留选中因子、训练边界/行数、固定超参数、训练身份和特征重要性等说明，删除拟合后的 `coefficients`、`intercept`，不公开模型权重。完整训练元数据（含 Ridge 系数）和逐股预测仅在私有运行目录；当前没有保存 LightGBM 完整树结构或序列化模型。离线评价依照固定配置重新拟合，不能把特征重要性当作一套可直接加载的模型权重。

### 论文文件身份勘误

冻结 `configs/project2_cogalpha.json` 的 `paper_sha256` 曾将论文身份误录成65字符；真正的SHA-256应为64字符。为保留已运行配置的字节身份，不直接修改冻结配置，而用 [`project2_cogalpha_source_erratum.json`](../configs/project2_cogalpha_source_erratum.json) 单独记录错误值、正确PDF哈希、冻结配置哈希和勘误范围。`evidence/project2_cogalpha/source_reference.json` 引用正确身份，最终清单也记录勘误文件身份。这只修正来源元数据，不改变公式、提示词、样本、阈值、模型、预算或实验身份，不需要新增付费调用。

### RSRS 指数快照

默认本地目录 `data/private/project2_rsrs/`，三个CSV须含 `date` 和程序需要的 OHLC 字段；使用固定原快照，重新下载的厂商修订数据应另立实验。

| 文件 | 配置记录的 SHA-256 |
| --- | --- |
| `sh000300.csv` | `3d432bc0baab98857fb52053418e880d34d623a64341bbb467402c8dac34e5b9` |
| `sh000016.csv` | `7ccb853cd5c9464683a92bbb0975d8002c00e5f41f18faafd80667768537c451` |
| `sh000905.csv` | `545bae9f221cfc2d828e3e71ae724a01da61649ebdf74921647c199bc57d77a6` |

正式运行在计算前检查这些哈希。协议固定3指数×3时期×2费用×11方案，共198组主实验；162组二元持仓基线另与旧实现比较。指数最新日为2026-10-15；指数价格本身不等于可直接交易证券，未含指数股息、杠杆或真实交易连接。来源和原文补充假设详见 [来源审计](PROJECT2_SOURCE_AUDIT.md)。

### CogAlpha Tushare 快照

本机 `--data-root` 为 `D:/Desktop/cf2026-quant-platform`，以下路径相对该根目录：

| 核心输入 | 当前数据质量记录中的 SHA-256 |
| --- | --- |
| `data/private/mainboard1000_20260918/data/processed/market.csv` | `7feecc242fe6450c3aeb0a4024621c6b6fa3f230540eacb5dc9b4c006af7adf4` |
| `data/private/research_v2/features.parquet` | `d33617783704255209e0fc6dd667cba5f53bcf68c3b950bc68bcabdfd4604514` |
| `data/private/research_v2/benchmark.csv` | `3feb9f40aa8dd5601ac07196e52d93e0a2479538fec4ee4801af5174de47fd55` |

快照有1000只股票、1,656,599行实际行情、1,690个交易日，2019-10-08至2026-09-18。股票池静态，存在幸存者/成员选择偏差，不能称为历史沪深300成分股实验。`features.parquet` 提供已存在的合格股票掩码；本次因子从 OHLCV 重新计算，历史未来收益列不作为因子输入。

当前复用 `benchmark_research.load_inputs`，所以仅提供上表三个文件仍不够。还需要：

- 同行情目录下的 `calendar.csv`。
- `data/private/research_v2/features_cache.json`、`daily_basic.parquet`、`financials.parquet`、`industries.parquet`、`cpi.csv`。
- `features_cache.json` 所列原数据与计算源码身份，包括数据根目录下的 `src/cfquant/features.py`、`data.py`、`__init__.py`。除经过明确识别的历史版本号变化外，缓存输入按字节哈希核对。

本轮复跑直接使用已有快照，不需要 Tushare 凭据或重新拉取行情。其他机器应在有相应数据使用权限的前提下自行准备同身份的私有快照；不同数据另立配置并记录新哈希，不能冒充本次精确重放。

训练2020–2022、验证2023–2024、历史留出2025至2026-09-18。标签为 `open[t+11]/open[t+1]-1`，标签退出必须严格早于下一分段；模型在2023-01-01之前训练一次，后续不重估。因子/标签先按当日合格股票遮罩再做截面排名。ICIR不年化；投资Sharpe按252日年化、无风险利率0计算。这里的留出是流程隔离，不是未观察的前瞻盲测；大模型预训练知识污染另列为局限。

CogAlpha 图中的沪深300基准是**无手续费的价格指数**，以评价期首日开盘为基数；股票组合用连续复权研究单位计收益并扣指定交易费用。价格指数不含股息，与股票复权收益不是完全一致的全收益口径。`aer`、信息比率和相对累计收益因此是相对此价格代理的描述，不能表述成已经战胜同口径沪深300全收益指数。RSRS本身的指数择时实验则在各方案声明的相同指数、时点和费用假设下比较。

## 3. RSRS 正式矩阵与独立探索

为保留发布结果，复跑使用新输出目录：

```powershell
.venv/Scripts/python.exe scripts/run_project2_rsrs.py --data data/private/project2_rsrs --config configs/project2_rsrs.json --output runs/reproduction_rsrs/evidence --private-output runs/reproduction_rsrs/private --workers 3
```

默认不传输出参数时会使用 `evidence/project2_rsrs` 与 `runs/project2_rsrs_formal`。续跑只接受同协议、同计算代码身份和已核验输出；身份变化会拒绝覆盖原证据，须新建运行目录。

主策略 `right_ma` 与改进 `right_ma_vol15` 的信号相同；改进采用20日历史波动、15%目标，并用 `right_ma_cash75` 区分风险缩放与动态波动控制。原表较早样本采用不同预热补充假设，见 `source_compatible.csv`，不能和主矩阵视作完全同口径。

页面参数回放通过统一 `run_rsrs` 接口执行，结果放入唯一 `runs/project2_rsrs_explorations/<时间_标识>/`。改动参数后旧结果不继续冒充当前参数结果。这些探索不改写发布协议、主比较或报告数字。

## 4. CogAlpha：离线评价与模型提案分开

### 离线复算已完成提案

`--stage evaluate` 不构造 `ResearchAPI`，不读取模型密钥，也不调用 `mine`。它要求固定协议、完整的提案运行记录、原预算摘要和有权访问的私有行情；可从保存的已接受公式重新构建缺失因子缓存，再按同包代码/依赖和固定超参数重新拟合模型。它逐项检查提案协议身份、完成状态、公式质量和原训练指标，不会重新向服务商“生成同一批因子”。

正式研究已经完成。离线复跑时，先准备一个**此前不存在**的复跑目录。以下仅复制研究身份、完整挖掘记录（每轮的 `manifest.json`、`prompt_contract.json`、`mining_result.json` 和槽位记录）以及公开预算摘要；不复制API账本、因子面板或模型预测：

```powershell
if (-not (Test-Path -LiteralPath 'evidence/project2_cogalpha/manifest.json')) { throw '正式实验尚未完成，请等待最终清单。' }
$formalRun = (Resolve-Path -LiteralPath 'runs/project2_cogalpha_formal').Path
$replayBase = Join-Path (Get-Location).Path 'runs/reproduction_cogalpha'
if (Test-Path -LiteralPath $replayBase) { throw '复跑目录已存在，请换一个新目录名。' }
$replayEvidence = Join-Path $replayBase 'evidence'
$replayPrivate = Join-Path $replayBase 'private'
New-Item -ItemType Directory -Path $replayEvidence, $replayPrivate | Out-Null
Copy-Item -LiteralPath (Join-Path $formalRun 'identity.json') -Destination $replayPrivate
Copy-Item -LiteralPath (Join-Path $formalRun 'mining') -Destination $replayPrivate -Recurse
Copy-Item -LiteralPath 'evidence/project2_cogalpha/budget.json' -Destination $replayEvidence
.venv/Scripts/python.exe scripts/run_project2_cogalpha.py --data-root 'D:/Desktop/cf2026-quant-platform' --stage evaluate --output runs/reproduction_cogalpha/evidence --private-output runs/reproduction_cogalpha/private --config configs/project2_cogalpha.json
```

此命令只向新的 `runs/reproduction_cogalpha/{evidence,private}` 写入，不覆盖发布的 `evidence/project2_cogalpha` 或原正式运行。`evaluate` 计算所必需的最小提案内容是每轮保存的完整 `mining_result.json`；上例同时保留挖掘身份、提示词约定和槽位收据，便于追溯。`factors/` 的值、日IC和哈希收据可额外复制到新私有目录以节省时间；没有缓存时按保存公式重建，重新拟合所有模型也会耗时。新目录必须对应同一数据与协议。只有页面导出的 `proposals.json` 而没有完整提案运行记录，不构成完整离线输入。

若需要再次独立复跑，应同时改上例复制目标和 `--output`、`--private-output` 为另一组新路径；不要传 `--key-file`。复制的 `budget.json` 仅记载原提案费用，不赋予新付费额度。不完整或已改变的研究记录应先排查，不能把缺失候选当作“模型没有预测能力”。

### 明确启动有预算的提案

以下是有外部模型费用的研究阶段，不是打开页面或离线评价时自动执行的步骤：

```powershell
.venv/Scripts/python.exe scripts/run_project2_cogalpha.py --data-root 'D:/Desktop/cf2026-quant-platform' --stage mine --key-file '<本机私有密钥文件路径>'
```

`--stage mine` 生成并检查候选；`--stage all` 在同一有预算流程后继续组合模型和回测；默认阶段是 `all`，所以只想离线重算时应明确写 `--stage evaluate`。密钥文件路径是占位符，不应把密钥本身放在命令行、聊天、仓库或报告中。

本轮所有提案重复、有限修复、诊断和续跑共用唯一账本 **`D:/Desktop/cf2026-quant-platform/runs/project2_cogalpha_api/`**，通过锁和哈希链收据累计同一预算。上限是50元人民币、400个请求；**50元是停止上限，不是需要花完的金额**。同请求缓存命中不重新付费；没有自动重试。开始请求前预留保守最坏成本，失败/中断的未知费用保留占额，不能当作零。不同目录会形成新的预算账本，因此不能通过换根目录、重建缓存或清空记录绕过这次全局50元约束。

完整私有运行已迁入 `D:/Desktop/cf2026-quant-platform/runs/project2_cogalpha_formal/`，唯一账本已迁入上述桌面路径；源计算目录下的两个旧目录已不存在，不保留第二份可继续付费的账本。迁移先核对源身份、移动后再逐文件核对SHA-256：正式运行2376个文件、4,400,950,389字节；API目录239个文件、3,172,350字节。本机迁移回执为桌面仓库 `runs/project2_private_migration.json`，不收入公开源码包。离线目录可以复制完成的提案、因子和实验结果，但不能把复制的API账本当作第二个付费额度；不得在复制目录重新启动 `mine/all` 来延续额度。离线 `evaluate` 不需要复制或加载API账本。

`budget.json` 是按返回token与固定保守单价估算的研究摘要，不是服务商官方账单。页面区分成功请求估计、未知预留和可用余额；最终实际支出需以服务商账单为准。缓存与收据不公开；公开仅包含经筛选的预算摘要与模型/版本描述。

已完成的挖掘记录：3轮各77个独立槽位，共231槽位；197通过质量检查、34拒绝，未删除失败槽位。236次提案尝试加1次连通检查，共237次成功API调用；保守估费1.312712元，未知费用预留0元。90组组合评价使用保存的提案离线完成，没有新增API调用。槽位通过质量检查不代表五指标全部合格，也不代表投资策略有效；提案计数与预测/投资评价分别报告。

提案使用3个独立组织种子731/732/733；每次21初代槽位，2代各14子代，与持续产生新公式的49槽位对照共享初代。随机种子控制本地抽样，不保证服务商模型生成逐字确定；后续精确评价依靠保存的响应和公式。训练反馈可以进入进化，验证/留出结果不得反馈给模型。最终选择规则和 Ridge/LightGBM 参数固定，不因留出收益翻转因子方向或更换赢家。

请求中的 `deepseek-flash` 是服务商API别名；`observed_version` 是查询当日官方别名映射记录，不是API独立证明的不可变权重版本。服务商权重和采样随机状态不能固定，因此今后重新调用API即使使用相同提示词，也不能保证产生同一批公式。可复跑的对象是本次保存的提案和公式，在同包代码/依赖、同数据、同固定训练参数下的离线评价；新API抽样必须另行标注。映射来源与局限见公开 `source_reference.json`。

### 本次固定主比较的结果

`comparison.csv` 保存三次重复、三种组合模型、两组预设比较，共18行。主比较固定为LightGBM、2025-01-02至2026-09-18历史留出期、指定成本；下表均为前者减后者。Rank IC差值以相关系数原单位表示，年化收益差值以百分点表示。

| 重复种子 | evolved49 − fresh49：Rank IC | 年化收益差值（百分点） | 去冗余 − evolved49：Rank IC | 年化收益差值（百分点） |
| --- | ---: | ---: | ---: | ---: |
| 731 | −0.002539 | +1.99 | −0.003068 | +2.47 |
| 732 | −0.010873 | −9.61 | +0.002988 | −0.46 |
| 733 | −0.006951 | +5.29 | +0.002965 | −5.23 |
| 三次差值的算术平均 | −0.006788 | −0.78 | +0.000962 | −1.07 |

主比较三次Rank IC全部下降，不能支持“训练反馈进化稳定优于继续产生新公式”。去冗余在2/3次重复提高Rank IC，但年化收益仅1/3次提高，投资改善并不一致。上述平均不是一条混合策略的回测收益，也不是统计显著性结论；只有三次重复，且全区间属于已经观察的历史。所有候选、零成本对照、原例公式和负结果保留在90组完整结果中，未根据留出收益替换主比较或原V3策略。

## 5. 核验文件与测试

RSRS `manifest.json` 的 `files` 列出所有公开证据哈希；另含 `source_sha256`、`config_sha256`、`data_snapshot_sha256`。CogAlpha 最终清单使用 `files`、`calculation_sha256`、`protocol_sha256`、`input_sha256`。可以先核对公开文件完整性；这无需行情或 API：

```powershell
@'
import hashlib, json
from pathlib import Path
for name in ("project2_rsrs", "project2_cogalpha"):
    folder = Path("evidence") / name
    manifest = folder / "manifest.json"
    if not manifest.is_file():
        print(name, "INCOMPLETE: no final manifest")
        continue
    data = json.loads(manifest.read_text(encoding="utf-8"))
    bad = [rel for rel, expected in data["files"].items()
           if not (folder / rel).is_file()
           or hashlib.sha256((folder / rel).read_bytes()).hexdigest() != expected]
    print(name, "PASS" if not bad else "FAIL", len(data["files"]), bad)
'@ | .venv/Scripts/python.exe -
```

此命令验证输出字节身份，不替代计算核验。代码继续修改后需另查清单的源文件哈希与当前文件是否相同；清单中的运行时提交不自动等于最后发布提交。

```powershell
.venv/Scripts/python.exe -m pytest tests/test_rsrs_integration.py tests/test_rsrs_diagnostics.py tests/test_rsrs_study.py tests/test_rsrs_ui.py -q
.venv/Scripts/python.exe -m pytest tests/test_cogalpha.py tests/test_cogalpha_models.py tests/test_cogalpha_mining.py tests/test_cogalpha_study.py tests/test_research_api.py tests/test_cogalpha_ui.py -q
```

自动测试使用小样本和替代传输，不消耗真实 API 预算，也不代替完整真实研究。最终通过数、跳过原因、浏览器检查和材料视觉核验应引用发布验收记录，不沿用旧版README的测试总数。

当前发布测试记录 `evidence/project2_release/test_acceptance.json` 汇总去重后**462项全部通过、0项未解决跳过**。原来跳过的Qlib表达式引擎测试已在已有独立Qlib环境实际通过；统计跨主研究环境、独立Qlib环境和成功重跑记录，不是一次不中断的全套测试0跳过。18项研究编排测试包含离线CLI不构造API/不调用挖掘、重建公式评价和公开元数据去除系数检查。RSRS的198组在共享执行器扩展后逐组重放一致，见 `release_engine_equivalence.json`；该次重放没有重跑差异诊断、稳健性扫描或bootstrap。RSRS浏览器验收见 `ui_acceptance.json`。CogAlpha正式清单记录90组组合、113项公开文件，`validation.json`中的90项账本检查均通过；这些结果与软件测试、正式审计、材料/浏览器验收分别记录。

### 公开审计器与打包入口

`scripts/audit_project2_cogalpha.py` 默认仅检查就绪状态，未执行正式审计时 `passed=null`。下面两个模式分别是合成自检与完成后的只读正式审计：

```powershell
.venv/Scripts/python.exe scripts/audit_project2_cogalpha.py --self-test
.venv/Scripts/python.exe scripts/audit_project2_cogalpha.py --final --data-root 'D:/Desktop/cf2026-quant-platform' --private-output runs/project2_cogalpha_formal --api-dir runs/project2_cogalpha_api --output tmp/cogalpha_release_audit.json
```

10项合成自检检查正常/停牌减记账本、拒绝篡改现金/收盘标记价格，无Git源码包的文件哈希验证，以及带额外system和修复消息时按role提取首个user提案/拒绝无效JSON；这些是审计器自检，另计于462项测试，不是正式研究验收。`--final` 要求完整90组合、原私有行情、完整运行和唯一API收据，核对身份、候选、选择、预算、保存账本与18项比较；不读密钥、不联网、不重新训练或回测，缺材料时拒绝认证。审计结果默认只写 `tmp/`，避免改动被核验的来源；正式公开回执为 `evidence/project2_release/cogalpha_release_audit.json`，六个检查部分全部通过，审计脚本SHA与本版一致。核对范围包含113项公开文件、3项核心输入、237次调用及保存实验来源链；没有新增API、重新拟合或重新回测。

源码ZIP不含 `.git` 时，审计器仍核对计算文件的SHA-256，并明确注明没有执行冻结Git历史比较。保存的预测指标会与日指标记录核对，不会重新拟合模型或重新从行情计算所有预测相关性；结论限于所保存的证据与声明的来源链。

所有材料和审计通过、代码提交完成后，可先只做打包预检：

```powershell
.venv/Scripts/python.exe scripts/package_project2_submission.py --destination 'D:/Desktop/青序_Project2_提交包_v2.4.0' --check-only
```

通过后去掉 `--check-only` 才生成文件夹、两份独立研究、共享源码ZIP、文件校验清单和外层总ZIP。打包器不运行API，不复制私有行情/模型缓存/付费账本；拒绝覆盖已有非空目录，拒绝tracked未提交修改，要求正式审计通过且审计脚本哈希匹配。还要求实际浏览器回执 `evidence/project2_release/cogalpha_ui_acceptance.json` 通过交互、四件材料下载入口、报告实际下载和截图视觉检查，且对应本版界面源码及最终证据清单。实际回执已生成：各筛选与风险切换、失败公式保留、4个材料入口和报告实际下载通过，8张截图已视觉检查；下载报告SHA与最终报告一致。这不是用合成测试替代真实浏览器验收。制作提交包不代表已代交Blackboard。

## 6. 独立材料与解释边界

| 交付 | RSRS 文件 | CogAlpha 文件 |
| --- | --- | --- |
| 报告 | `reports/project2_rsrs/CF2026_Project2_Report.pdf` | `reports/project2_cogalpha/CF2026_Project2_CogAlpha_Report.pdf` |
| Beamer | `reports/project2_rsrs/CF2026_Project2_Beamer.pdf` | `reports/project2_cogalpha/CF2026_Project2_CogAlpha_Beamer.pdf` |
| 可编辑PPT | `reports/project2_rsrs/CF2026_Project2_Presentation.pptx` | `reports/project2_cogalpha/CF2026_Project2_CogAlpha_Presentation.pptx` |
| 讲稿 | `reports/project2_rsrs/CF2026_Project2_Speaker_Script.pdf` | `reports/project2_cogalpha/CF2026_Project2_CogAlpha_Speaker_Script.pdf` |

四件CogAlpha材料已完成：报告16页、Beamer16页、讲稿8页、可编辑PPTX16页；14页演示正文加2页备份，对应约15分钟讲稿。三份PDF均至少编译两次并逐页视觉核验，PPTX完成结构与全页视觉检查，材料字节哈希与 `authoring_checks.json`、`visual_qa.json`、`pptx_validation.json` 一致。平台仅在文件存在时提供下载。两项材料分别对应其论文/研报，不混用收益、样本、图号或验收。RSRS历史准备目录 `projects/project2_rsrs/` 与v2.3.0平台材料继续作为档案；本说明及固定配置是本轮正式复跑依据。

已交付PPTX可直接在PowerPoint等兼容软件中编辑文字、图表和表格，不需要重新运行构建程序。**从源码重建PPTX**则依赖 `scripts/build_project2_presentation.mjs` 使用的 `@oai/artifact-tool`、配套 `artifact_tool_utils.mjs` 与字体运行环境，这些不是Python依赖文件能安装的部分；源码保留环境路径/可配置入口，不承诺一条普通Node命令即可在新电脑重建。

PPTX中的部分原生时序曲线会为编辑和展示抽样到约140个点并保留末点，RSRS长序列的显示密度可能接近月频。这仅影响图表显示；绩效指标、回测执行与公开每日CSV仍使用完整日频序列，Beamer/PDF科研图一般使用完整序列。不能把显示抽样误认为回测改成了月频。

LaTeX报告、Beamer和讲稿不依赖该artifact运行环境；保留 `reports/project2_*/`、`figures/` 与相应 `evidence/` 相对布局后，可用安装了CTeX/Beamer及相应字体的XeLaTeX独立编译。例如进入对应报告目录后，对 `final_report.tex`、`beamer.tex`、`speaker_script.tex` 分别运行两次 `xelatex -interaction=nonstopmode -halt-on-error 文件名.tex`。编译所需图表已随材料提供，无需重跑付费模型；新机器仍需检查字体与最终排版。

两条路线都是研究实现，允许现金、不连接交易账户，不承诺未来超额收益或固定回撤上限。RSRS 没有原始 Wind 数据；CogAlpha 更换了数据、时间区间、服务模型、部分选择/估计器与模型规模，因此属于受限机制复现，不是原表数值的精确复现。报告应保留未改善和负结果，清楚区分公式检查通过、账本通过、预测增量与投资结果。
