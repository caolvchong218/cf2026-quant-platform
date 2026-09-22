# V3 基准与验证口径

核对日期：2026-09-22。本文为实施建议；未下载数据、读取凭据或验证接口权限。

## 基准

沪深300价格指数代码为000300（沪市）/399300（深市），除息时自然回落；全收益指数H00300计入红利再投资，净收益指数N00300另作税后处理。策略采用分红再投资复权单位时，主要指数比较应采用H00300，价格指数单列，不能将二者混称。[中证编制方案§4.6、5.2、12.1](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/000300_Index_Methodology_cn.pdf)、[中证指数单张](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/000300factsheet.pdf)

Tushare `index_daily` 接受指数代码，但官方文档未保证H00300覆盖。先在 `index_basic(market='CSI')` 等目录核对名称、代码和发布商，再用返回的 `ts_code` 请求、检查完整日期及异常值；不要猜后缀。无数据时明确写“全收益指数缺失”，不能以000300冒充。[指数目录](https://tushare.pro/document/2?doc_id=94)、[指数日线](https://tushare.pro/document/2?doc_id=95)

另列预先固定的一只沪深300ETF买入持有代理，例如510330.SH。合并 `fund_daily` 与 `fund_adj`，建议计算 `P*=P×factor/factor_ref`；每日收益用相邻复权价比，并核对分红日。复权单位是再投资近似，执行价格、整手和费用须用原价账本；ETF代理不能改名为H00300。`pro_bar`文档的 `adj` 仅针对股票，不能假设基金自动复权。[ETF行情及示例](https://tushare.pro/document/2?doc_id=127)、[基金复权因子](https://tushare.pro/document/2?doc_id=199)、[pro_bar限制](https://tushare.pro/document/2?doc_id=146)

## 公平比较的实施建议

1. 固定共同估值起点、交易日和终点；策略与ETF在相同可执行时点建仓。H00300为收盘序列时，使用共同前收盘归一化，并披露策略首日现金/开盘建仓效应，不能伪造全收益指数开盘价。
2. 策略计全部成交费用与滑点，ETF计建仓、再投资及同一终止规则下的退出费用；价格内已体现的基金费用勿重复扣。现有买10bp/卖15bp须标为研究假设，并报告成本加倍情形。
3. 同时报净收益、相对财富 `(1+R策略)/(1+R基准)-1`、最大回撤、跟踪误差、换手和现金比例；明确收益差的单位为百分点。ETF与全收益指数分别给结论。

## 策略与证据边界

Qlib增强指数策略用于控制相对基准风险；官方示例的权重是手工构造，不能直接当历史真实权重。可复用实现思想，但必须提供当时可得的成分、权重和风险数据。[Qlib策略](https://github.com/microsoft/qlib/blob/main/docs/component/strategy.rst)、[示例及限制](https://github.com/microsoft/qlib/blob/main/examples/portfolio/README.md)

按时间滚动训练/验证，拟合处理器仅见训练数据，并按标签实际退出日期清除跨边界样本；候选、参数和选择规则先冻结。[时间序列验证](https://sklearn.org/stable/modules/cross_validation.html)

项目已查看2025-01-02至2026-09-18的V2结果；据此修改V3后，该区间属于已观察研究回测，即使重新切分或滚动也不能恢复盲测资格。保留全部尝试与失败结果；冻结代码和协议后，以未来未见数据另作前向验证。历史胜出仅支持该区间扣费表现，不能证明未来超额收益。[回测过拟合研究原文](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
