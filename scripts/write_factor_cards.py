from pathlib import Path
from cfquant.features import FEATURE_CARDS
ROOT=Path(__file__).resolve().parents[1]
details={
 'momentum_60_20':('close','61日完整价格','趋势突变、样本风格集中'),
 'reversal_5':('close','6日完整价格','基本面恶化导致持续下跌'),
 'low_volatility_20':('close','20个完整日收益','快速牛市落后，与行业相关'),
 'range_20':('high, low, close','20日完整振幅','与低波动高度重叠'),
 'trend_20':('close','20日完整价格','震荡反复，方向不稳定'),
 'volume_trend':('volume，单位股','5日和20日完整量','事件放量不一定预示上涨'),
 'liquidity_20':('raw_close, volume, close','20日完整收益/成交额','市值与流动性偏差、单位敏感'),
 'turnover':('daily_basic.turnover_rate，百分数','形成日','低换手也可能意味着关注不足'),
 'earnings_yield':('daily_basic.pe_ttm','形成日且PE>0','亏损公司缺失，价值陷阱'),
 'book_yield':('daily_basic.pb','形成日且PB>0','账面资产价值并不稳定'),
 'dividend_yield':('daily_basic.dv_ttm，百分数','形成日','历史分红不代表未来分红'),
 'quality_roe':('fina_indicator.roe, ann_date, end_date','已公告年度，报告期距形成日≤550日','公告滞后、厂商财报修订'),
}
lines=['# 扩展因子卡（V2）','','实现：`src/cfquant/features.py`。原有三个注册式基准因子仍保留。以下方向在结果产生前固定，高分优先；不因最终IC为负而翻转符号。','',
 '## 公共规则','','先对齐完整交易日历，不将停牌压缩。原始NaN保留，不用未来价格或财报填充。每个形成日独立1%/99%截尾与样本z标准化，再可选历史行业/对数市值OLS残差标准化。组合要求至少10/12因子有效；学习模型把剩余标准化特征NaN填为0，未来标签NaN直接排除。','',
 '财报只在公告日后可用；同公告日同报告期的冲突副本剔除。行业按[in_date,out_date)重构，未知/冲突为UNKNOWN。厂商历史修订与分类回溯风险仍然存在。','']
for key,card in FEATURE_CARDS.items():
    fields,window,failure=details[key]
    lines += ['## '+card[0]+' / '+key,'','- 假设：'+card[3]+'。','- 公式：`'+card[1]+'`。','- 字段：'+fields+'。','- 窗口/可得条件：'+window+'。','- 方向：公式结果越大，事前预期收益越高。','- 缺失：窗口或字段不足记NaN，执行公共规则。','- 主要失败情形：'+failure+'。','']
lines += ['## 标准输出与使用','','`features.parquet`是日期×资产长表，原始列、z_列和n_列对应原始、标准化与中性化版本。调用方可将任一列转为date/asset/factor/value四列，或pivot为日期×资产面板供组合模块使用。',
          '完整每日IC、RankIC、样本量、分布、形成/有效组人数与20日组收益在`evidence/research_v2/factors`。这些诊断是重叠标签统计，不是可执行多空净值。','',
          '## 来源','','[Qlib官方量价表达](https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py)启发滚动窗口和比例特征；本项目不是完整Alpha158复制。估值/财报字段分别依据[Tushare daily_basic](https://tushare.pro/document/2?doc_id=32)和[fina_indicator](https://tushare.pro/document/2?doc_id=79)。']
(ROOT/'docs/FACTOR_CARDS_V2.md').write_text('\n'.join(lines),encoding='utf-8')
