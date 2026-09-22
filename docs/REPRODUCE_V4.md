# V4 平台与材料复现

版本 v2.2.0 增强了工作台，保留 V3 策略及历史研究文件。先阅读 README 中的环境安装步骤。

1. 本机执行 `launch.cmd`，或在已安装环境运行 `streamlit run app.py`。
2. 无原始数据时，研究总览、策略对比和风险透镜读取 `evidence/research_v3/daily` 与 V2 日账本，可离线查看真实聚合结果。
3. 从策略对比选择版本、日期和基准，下载 CSV 与 methodology.json；筛选日期是已有路径切片。真实重新开仓应在回测实验中提交参数。
4. 风险透镜提供月历、回撤恢复、滚动风险及静态现金情景。查看计算假设，缺失值不等于零。
5. 独立12因子页面读取公开 evidence CSV，可离线查看定义、IC与日期诊断。真实行情及逐股成交明细需要对应本机私有数据/运行记录；缺少时按页面说明补足合法快照，不用合成数据冒充实证。
6. 运行 `python -m pytest -q -p no:cacheprovider`；合成示例入口为 `python -m cfquant.cli run --config configs/demo.yaml`。
7. Qlib在隔离环境安装，按接入文档转换本地数据并运行原生表达式。先确认状态证据，不将未验证的账户接口说成已接通。
8. 材料作者入口为 `python scripts/build_platform_materials.py --compile`；最终作者应传入真实Qlib状态、同花顺连接说明和已通过测试数量。无状态证据时材料只记录待验证。

材料位于 `reports/platform_v4`。`deck_content.json`同时提供Beamer、原生可编辑PowerPoint和逐页讲稿内容；PPTX由 `scripts/build_platform_presentation.mjs` 单独构建。使用报告和讲稿时保持同一V4版本。

原始私有数据、凭据和训练缓存不在公开仓库；真实训练需要本机合法快照。V3既有研究目录禁止覆盖，改变假设必须新建运行身份。平台不连接证券交易账户。
