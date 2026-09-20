# 使用与验收

## 本机入口

双击根目录launch.cmd，打开 http://127.0.0.1:8501 。只监听本机。五页：研究总览、回测实验、因子诊断、数据与复现、扩展指南。

表单填写后点击“运行并核对账本”。10基点=0.1%。旧结果保留自己的参数标识；修改表单但未提交不会改变结果。停止服务可关闭运行终端；8501已有本平台时直接打开浏览器，不重复启动。

## 组员电脑安装

推荐Python3.12/3.13。根目录执行：

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m cfquant.cli run --config configs/demo.yaml
.\.venv\Scripts\python.exe -m streamlit run app.py
~~~

requirements-lock.txt固定直接依赖；传递依赖由包管理器解析，不声称是完整哈希锁。GitHub Actions用干净Linux/Python3.12检查。Windows也可双击setup.cmd。

仓库附带固定种子合成样本，无Token也可演示；界面会明确标记。真实报告对应本机Tushare快照。

## 获取数据

环境变量TUSHARE_TOKEN或本地文件读取。不要将凭据写入代码、YAML、Git、截图或报告。

~~~powershell
.\.venv\Scripts\python.exe -m cfquant.cli download --token-file D:\Desktop\tushare_token.txt
~~~

默认60只、2022年末规则，首次约182个请求；限速、重试和缓存内置。更改股票数是不同样本，不能称原报告复现。行情在data/raw和data/processed，默认不入Git。本地快照仅在团队获准使用范围内复制。

## 运行

~~~powershell
.\.venv\Scripts\python.exe -m cfquant.cli run --config configs/baseline.yaml
.\.venv\Scripts\python.exe -m cfquant.cli study
~~~

study运行三因子和仅将动量5日调仓改20日的主对照，同时输出因子诊断、因果检查和重复检查，不搜索最优参数。

每个目录有config.yaml、daily.csv、trades.csv、orders.csv、positions.csv、factors.csv、metrics.json、checks.json、provenance.json。分别保存配置、账本、成交、订单、持仓、长表、绩效、复核与版本哈希。

同样数据配置刷新同一参数签名目录；不同参数生成新目录。长期保留实验可复制目录或用新配置名称。主要数值应一致，时间戳和耗时可变。

## 八分钟演示顺序

1. 总览：完整链路，主动展示亏损结果。
2. 数据：快照校验、缺失和复权口径。
3. 因子：三个假设，诊断不等于策略收益。
4. 回测：修改调仓间隔，看费用、净值、未成交。
5. 工程：pytest与因子插件例子。
6. 报告：解释对照、局限和模型接口。

## 常见问题

- API权限不足：不要以原价伪装复权价，检查daily/adj_factor/stk_limit/trade_cal权限，可先合成演示。
- 找不到数据：根目录运行或指定--root。
- 克隆后是合成样本：真实数据不入Git，需自行下载或导入获准快照。
- 订单未成交：查看missing_open、missing_limit、upper_limit、lower_limit、cash_scaled。
- 亏损不是代码错误：核对账本后分析信号和成本，不为了曲线好看改变区间。

