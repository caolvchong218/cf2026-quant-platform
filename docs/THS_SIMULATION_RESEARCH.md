# 同花顺模拟交易接入核查

核查日期：2026-10-20。目标是把青序策略接入外部**模拟**撮合，保留真实的委托、成交、持仓与收益证据。本次研究没有登录用户账户，没有开户，没有下单。

## 结论与建议

**同花顺确有官方可编程模拟交易方案，不能笼统写成“同花顺没有 API”。** 当前问财 SkillHub 把“模拟炒股”列为官方技能；从其官方安装器下载的技能包包含模拟开户、买卖、持仓、资金、收益、成交查询的 HTTP 文档和 Python 示例。另有 SuperMind 官方研究环境的 `TradeAPI`。两个方案均需与普通手机模拟炒股的现有账户区分。[官方 SkillHub](https://search.10jqka.com.cn/skillhub)、[SuperMind 研究环境文档](https://quant.10jqka.com.cn/view/help/14)

适合青序的优先路径是：**以官方模拟技能的专用账户做自动下单适配；手机号账户继续作为独立的手机观察/手工练习入口。** 在取得真实连通与账户回报之前，平台只能标记“适配器已准备”，不能标记“已连接”；在实际成交前，委托成功也不能记成买入成功。

此处没有证据证明技能新账户会出现在用户手机 App 中。若要求沿用手机已有持仓，必须额外核对账户身份、营业部/比赛编号和官方绑定方式，不能把手机登录名直接塞入资金账号字段。

## 三类入口与能力边界

| 路线 | 已证实的入口和用途 | 仍需核实 |
|---|---|---|
| 普通模拟炒股 | [网页首页](https://moni.10jqka.com.cn/)、[手机网页登录页](https://moni.10jqka.com.cn/mcgiwt/index/nindex)、[官方模拟炒股 App 下载页](https://download.10jqka.com.cn/phone/Monitrade/)；官方提供账户表现、持仓和交易记录入口 | 用户所用手机版本、具体比赛/练习场、与网页/电脑的账户对应关系；普通用户个人成交导出的实际入口 |
| 问财官方模拟技能 | [SkillHub](https://search.10jqka.com.cn/skillhub)；可下载官方模拟交易文档与示例，流程为新建专用模拟账户 | 当前服务连通性、账户初始模拟金、撮合回报、手机展示或绑定方式、频率上限 |
| SuperMind | [量化平台](https://quant.10jqka.com.cn/)、[模拟仿真说明](https://quant.10jqka.com.cn/view/help/10)、[官方文档引用的客户端入口](https://download.10jqka.com.cn/index/download/id/709)；云端策略模拟或客户端仿真柜台 | 当前用户权限、安装后的账户种类、与普通手机模拟盘是否互通 |

普通模拟网站明确提到“大赛管理”的用户数据导出；这不能推导为所有手机个人账户都支持 CSV 导出或批量订单导入。此次没有找到官方证据证明普通手机端支持青序订单文件直接导入。[模拟炒股官网](https://moni.10jqka.com.cn/index.shtml)

官方软件下载中心同时列有 Windows 模拟炒股和手机模拟 App，但“都有客户端”不等于“同一模拟资金账户一定互通”。具体账户仍要实测核对。[官方软件下载中心](https://download.10jqka.com.cn/)

## 官方技能的来源链与可复核材料

这次只阅读公开发行文件，没有执行其中的安装器或账户脚本。下载路径来自官方网站自身公开前端代码及安装器，不是猜测私有接口。

1. SkillHub 页面公开路由脚本：
   [index-78fb9937.js](https://s.thsi.cn/cd/iwc-web-ai-unifiedwap-public/static/7.3.0/index-78fb9937.js)。
2. 脚本指向当前 [0.0.4 安装说明脚本](https://www.iwencai.com/skillhub/static/0.0.4/download_and_install.sh)，其中指定 [官方 CLI ZIP](https://www.iwencai.com/skillhub/static/0.0.4/iwencai-skillhub-cli.zip)。
3. CLI 的 `cli/metadata.json` 指明技能下载模板为 `ms.10jqka.com.cn/gateway/market/api/v1/skills/square/download?name={slug}`。
4. 使用 HTTPS 成功取得 [官方“模拟炒股”技能包](https://ms.10jqka.com.cn/gateway/market/api/v1/skills/square/download?name=%E6%A8%A1%E6%8B%9F%E7%82%92%E8%82%A1)。

2026-10-20 下载包大小为 **19,693 字节**，SHA-256：

```text
851d24ebe6e29476d328432d2cebacfada40ac1e378f8df5078d263158a9fcc7
```

本机只读参考文件放在仓库忽略目录：

```text
tmp/ths_official/official_simulation_skill.zip
tmp/ths_official/download_receipt.json
tmp/ths_official/模拟炒股/SKILL.md
tmp/ths_official/模拟炒股/references/api-spec.md
tmp/ths_official/模拟炒股/references/account-data-format.md
tmp/ths_official/模拟炒股/scripts/open_account.py
tmp/ths_official/模拟炒股/scripts/stock_query.py
tmp/ths_official/模拟炒股/scripts/stock_trading.py
```

该目录用于核验来源，不应随平台公开发布账户资料。技能包未见许可证文件；青序宜依据接口协议独立实现小型适配器，不直接复制整套示例进公共仓库。

### 鉴权与账户的特殊性

SkillHub 的一般安装界面会显示 `IWENCAI_API_KEY` 和 `https://openapi.iwencai.com`。但**此次取得的模拟炒股技能脚本没有读取此密钥，也没有调用这个 OpenAPI 网关**。该技能直接使用 `http://trade.10jqka.com.cn:8088`，默认营业部编号为 `997376`，以 `skill_` 加毫秒时间戳创建专用用户名，再保存返回的模拟资金账号与沪深股东账号。这里的“正式环境”指模拟服务的生产环境，不能理解为真实证券账户。[官方技能包](https://ms.10jqka.com.cn/gateway/market/api/v1/skills/square/download?name=%E6%A8%A1%E6%8B%9F%E7%82%92%E8%82%A1)

因此，不应为了此接口要求用户提供手机登录密码、真实证券密码或额外申请问财 Key。此结论只对应上述已核验包，后续版本可能变化。HTTP 传输与模拟账号资料应作为具体实现限制记录；不要向这个接口发送真实账户凭据。

### 需要处理的官方示例差异

官方包本身存在文档与示例不一致，不能简单复制后宣布可用：

| 项目 | `api-spec.md` | Python 示例 |
|---|---|---|
| 持仓路径 | `/pt_web_qy_stock` | `/pt_web_qry_stock` |
| 资金查询参数 | `usid` | `usrid` |
| 当日/历史成交账号参数 | `usrid` | `usrname`，值仍为资金账号 |
| 持仓返回列表 | `data` | `result` 或 `list` |
| 资金返回列表 | `result.list` | `list` |
| 当日成交返回 | `ret.code`、`ret.item` | 顶层 `errorcode`、`result/list` |

技能说明声称可撤单，但该包的 API 规范与交易脚本没有提供撤单接口。不能据此实现一个猜测的撤单地址。股票名称查询示例还有静态小股票表及名称映射错误，不适合作为青序证券主数据源。

工程验收应使用新建的专用模拟账户进行只读查询和受控模拟委托；仅支持实际确认的返回形态，遇到未知字段必须停止同步并显示错误。下单接口虽使用 GET，依然属于写操作；超时后先核查成交，不可自动重发造成重复下单。研究记录没有进行上述有状态验收。

## SuperMind 作为另一条官方路线

SuperMind 官方要求在客户端登录需要使用的资金账户，再进入研究环境使用 `tick_trade_api`。文档提供资金 `portfolio`、持仓 `positions`、委托 `order`、撤单 `cancel_order`、委托查询与推送；同一接口既可绑定仿真也可绑定真实资金账户。因此若后续接入，必须在账户配置上强制锁定仿真身份，不能仅凭接口名推断安全。多个策略共享资金账户还会相互影响持仓。[研究环境/实盘文档](https://quant.10jqka.com.cn/view/help/14)

其普通策略模拟支持从当前日启动、查看持仓与收益、导出指定时间范围交易明细；仿真柜台则处理更接近实际交易的异步委托与回报。官方说明明确区分 `trade_api=None` 的内部模拟与绑定账户后的柜台交易，`signal_mode` 也会改变查询和下单语义。[模拟仿真文档](https://quant.10jqka.com.cn/view/help/10)

本地接口、SDK 或计算资源可能收费，应先查用户实际权限，不能把网页免费回测等同于免费本地柜台 API。[SuperMind 常见问题](https://quant.10jqka.com.cn/view/help/16)

iFinD/同花顺数据接口提供金融数据服务。拥有它的数据权限不会自动带来模拟下单权限；数据获取与交易连接应保持两个模块。[同花顺数据接口基础概念](https://quantapi.10jqka.com.cn/gwstatic/static/ds_web/quantapi-web/help-center/faq.html)、[SuperMind 数据平台](https://quant.10jqka.com.cn/view/dataplatform)

## 已有程序与复用选择

检索到了 [easytrader](https://github.com/shidenggui/easytrader)。其项目说明包括同花顺客户端界面自动化、券商 miniQMT、雪球组合和跟随其他平台模拟策略。“模拟操作客户端”描述的是操作方式，不能据此认定账户是同花顺免费模拟账户。它是独立开源项目，不是同花顺官方模拟 API；本次优先复用官方接口协议，不引入依赖真实券商窗口的默认执行路径。

[iwencai-skillhub-tools](https://github.com/eoncyrus-lab/iwencai-skillhub-tools) 是第三方批量安装助手，帮助定位了官方发行入口；它不是交易服务的证据来源，旧安装地址也已不适用于当前版本。

## 青序验收要求

以下是本项目的实现建议，属于工程判断：

- 将“历史回测”“青序本地模拟”“同花顺外部模拟”分别展示，避免把三类收益混算。
- 每个候选算法固定版本、参数、资金和启动日期；分别保存持仓账本。一个外部账户上的混合仓位只能作为组合策略结果，不能冒充多个独立策略实测。
- 收盘后用已完成日线计算信号，下一交易日才允许执行；股票使用未复权报价下单，并核验最新数据日期、手数、可用现金和可卖数量。
- 永久保存外部订单号、请求摘要、回报、实际成交与费用。超时订单进入“状态待核对”，禁止自动重复发单。
- 用外部资产快照计算收益、最大回撤和 Sharpe；同步资金转入转出，避免把充值当盈利。
- 明确显示运行状态、最近成功同步时间、未成交委托、失败原因和停止按钮。电脑休眠或网络中断时不能继续声称策略正常运行。
- 用户手机端能否看到新专用账户、撤单能力、实际服务规则和多账户额度均标记待验证；不使用文档未声明的功能作为验收通过条件。
