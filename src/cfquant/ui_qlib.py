"""Optional Qlib laboratory: saved evidence works without Qlib in the UI env."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import subprocess

import pandas as pd
import plotly.express as px
import streamlit as st

from .qlib_bridge import bridge_command, MARKET_PATH
from .ui_design import heading, chart


def render(root: Path):
    root = Path(root)
    folder = root/'evidence/qlib_bridge'
    receipt = folder/'result.json'
    heading('Qlib 实验室', '真实 Microsoft Qlib 表达式引擎 · 独立 Python 环境 · Tushare 本地数据桥接')
    result = json.loads(receipt.read_text(encoding='utf-8')) if receipt.exists() else None
    if result:
        if result.get('passed'):
            st.success(f"真实接入已核验 · pyqlib {result['packages']['pyqlib']} · {result['factor_count']} 个表达式逐项对照通过")
        else:
            st.error('最近一次 Qlib 接入尚未通过核对。')
        columns = st.columns(4)
        for col, label, value in zip(columns, ['真实资产', '日线记录', '独立核对因子', '官方 Alpha158'],
                                    [result['assets'], f"{result['source_rows']:,}", result['factor_count'],
                                     f"{result['alpha158']['with_finite_values']} / {result['alpha158']['configured']}"]):
            col.metric(label, value)
        st.caption(f"诊断区间 {result['start']} 至 {result['end']} · {result['diagnostic_sessions']} 个交易日；provider 从 {result['provider_start']} 起保留预热数据。")
        if result['alpha158']['vwap_coverage'] < 1:
            st.warning(f"原始成交额支持的 VWAP 覆盖率为 {result['alpha158']['vwap_coverage']:.1%}；相关 Alpha158 特征保留缺失，未用收盘价替代。")
        st.info('这是可复用的数据与因子接入实验。小样本用于验证接口和计算口径，尚未训练新的投资策略，V3 默认策略保持其原始研究结果。')
        tabs = st.tabs(['因子与诊断', 'Alpha158 因子库', '一致性与来源', '重新运行'])
        with tabs[0]:
            summary = pd.read_csv(folder/'factor_summary.csv')
            labels = dict(zip(summary.factor, summary.label))
            factor = st.selectbox('Qlib 表达式', list(labels), format_func=lambda x: labels[x], key='qlib_factor')
            row = summary[summary.factor == factor].iloc[0]
            st.code(row.expression, language=None)
            cols = st.columns(3)
            cols[0].metric('平均 Rank IC', f'{row.rank_ic_mean:.4f}')
            cols[1].metric('平均因子覆盖', f'{row.mean_factor_coverage:.1%}')
            cols[2].metric('有效 IC 日数', f'{int(row.ic_valid_days):,}')
            daily = pd.read_csv(folder/'daily.csv', parse_dates=['date'])
            daily = daily[daily.factor == factor].copy()
            daily['20日平均 Rank IC'] = daily.rank_ic.rolling(20, min_periods=20).mean()
            groups = pd.read_csv(folder/'groups.csv')
            groups = groups[groups.factor == factor].groupby('group', as_index=False).mean_forward_return.mean()
            a, b = st.columns([3, 2])
            with a:
                fig = px.line(daily, x='date', y='20日平均 Rank IC', labels={'date': '形成日'}, color_discrete_sequence=['#168579'])
                fig.update_layout(height=300, margin=dict(t=15, b=10))
                chart(fig, height=300, key='qlib_ic')
            with b:
                fig = px.bar(groups, x='group', y='mean_forward_return', labels={'group': '从低分到高分', 'mean_forward_return': '平均未来20日收益'}, color_discrete_sequence=['#4265a6'])
                fig.update_layout(height=300, yaxis_tickformat='.1%', margin=dict(t=15, b=10))
                chart(fig, height=300, key='qlib_groups')
            st.caption('标签为下一开盘至20个交易日后开盘收益。按形成时因子分组后匹配标签；最后20余日没有完整标签。因子IC和分组均值是描述性诊断，重叠标签不可连乘成净值。')
            st.dataframe(summary[['label', 'expression', 'ic_mean', 'rank_ic_mean', 'mean_factor_coverage']], hide_index=True, width='stretch')
            st.download_button('下载 Qlib 诊断汇总', (folder/'factor_summary.csv').read_bytes(), 'qlib_factor_summary.csv', 'text/csv')
        with tabs[1]:
            alpha = pd.read_csv(folder/'alpha158_summary.csv')
            st.write('直接调用安装包的 Alpha158DL.get_feature_config，交由 Qlib D.features 执行全部官方表达式。')
            query = st.text_input('按因子名或表达式搜索', placeholder='例如 ROC、CORR、VWAP', key='qlib_alpha_search')
            if query:
                alpha = alpha[alpha.name.str.contains(query, case=False, regex=False) | alpha.expression.str.contains(query, case=False, regex=False)]
            st.dataframe(alpha, hide_index=True, width='stretch', column_config={'coverage': st.column_config.NumberColumn('有效覆盖率', format='%.4f')})
            st.caption('158项已生成并实际运行；目前独立逐项核对的是前页8个基础表达式，未声称验证了全部158项经济有效性。逐股特征留在本地，公开这里的公式与聚合覆盖。')
            st.download_button('下载 Alpha158 公式与覆盖', (folder/'alpha158_summary.csv').read_bytes(), 'qlib_alpha158_summary.csv', 'text/csv')
        with tabs[2]:
            checks = pd.DataFrame([{'factor': key, **value} for key, value in result['checks'].items()])
            st.dataframe(checks, hide_index=True, width='stretch')
            st.write('Qlib 的滚动算子默认允许不完整窗口，标准差使用 ddof=1；独立 pandas 参考遵循同一规则。展示诊断另加完整窗口与有效报价过滤。bin 保存 float32，因此采用明确绝对/相对容差。')
            st.write('OHLC 沿用平台固定参考复权价；Qlib 成交量为原始股数除以复权比例。VWAP 来自 Tushare 原始成交额（千元）与股数，再用同一比例复权，没有用收盘价替代。')
            st.json({k: result[k] for k in ['run_id', 'packages', 'source_market_sha256', 'source_calendar_sha256', 'module_sha256', 'alpha158']}, expanded=False)
            if 'unit_checks' in result:
                st.json(result['unit_checks'], expanded=False)
            st.markdown('[Qlib 官方数据说明](https://qlib.readthedocs.io/en/latest/component/data.html) · [官方 Alpha158 源码](https://github.com/microsoft/qlib/blob/v0.9.7/qlib/contrib/data/loader.py)')
            st.caption('固定历史池的对半交易所、代码排序小样本，不代表全市场；这些历史日期已被研究过，不是独立盲测。')
        with tabs[3]:
            _run_controls(root)
    else:
        st.info('尚无保存的 Qlib 验证记录。安装独立环境并运行下方固定任务后，这里会展示真实结果。')
        _run_controls(root)


def _run_controls(root):
    command = bridge_command(root)
    ready = Path(command[0]).exists() and (root/MARKET_PATH).exists()
    st.write('重新从本地真实快照建立小样本 provider，运行8个独立核对表达式和官方Alpha158。每次生成独立目录，核对失败不会替换已有通过记录。')
    with st.form('qlib_run_form'):
        count = st.selectbox('接入资产数', [20, 10, 40])
        a, b = st.columns(2)
        start = a.date_input('诊断起始日期', pd.Timestamp('2020-01-02').date(), key='qlib_start')
        end = b.date_input('诊断结束日期', pd.Timestamp('2026-09-18').date(), key='qlib_end')
        submitted = st.form_submit_button('运行真实 Qlib 核验', disabled=not ready, type='primary')
    if submitted:
        existing = st.session_state.get('qlib_job')
        if existing and existing['process'].poll() is None:
            st.warning('当前会话已有 Qlib 核验正在运行。')
        else:
            try:
                command = bridge_command(root, count, str(start), str(end))
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                log = root/'data/private/qlib_bridge/jobs'/f'{stamp}.log'
                log.parent.mkdir(parents=True, exist_ok=True)
                env = {**os.environ, 'PYTHONUTF8': '1'}
                flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                with log.open('w', encoding='utf-8') as stream:
                    proc = subprocess.Popen(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT, creationflags=flags)
                st.session_state['qlib_job'] = {'process': proc, 'log': log, 'finished': False}
            except (ValueError, OSError) as exc:
                st.error(str(exc))
    if st.session_state.get('qlib_job'):
        _job_status()
    if not ready:
        st.caption('本机未同时具备独立环境和真实快照；已保存公开结果仍可浏览。安装说明见 docs/QLIB_INTEGRATION.md。')
    with st.expander('独立环境安装与命令行入口'):
        st.code('python -m venv .venv-qlib\n.venv-qlib\\Scripts\\python -m pip install --cache-dir tmp/qlib-pip-cache pyqlib==0.9.7 pandas==2.3.3 numpy==2.3.5 scipy==1.16.3 pyarrow==21.0.0\n.venv-qlib\\Scripts\\python scripts/run_qlib_bridge.py', language='powershell')


@st.fragment(run_every='2s')
def _job_status():
    job = st.session_state['qlib_job']
    code = job['process'].poll()
    if code is None:
        st.info('Qlib 在独立进程中运行，结果核对完成后更新。此页每2秒检查任务状态。')
    elif code == 0:
        if not job['finished']:
            job['finished'] = True
            st.rerun(scope='app')
        st.success('真实 Qlib 核验完成，本页已加载本次结果。')
    else:
        st.error('本次核验未完成，已有通过记录保留；详情见下方本地日志。')
    if job['log'].exists():
        st.code(job['log'].read_text(encoding='utf-8', errors='replace')[-2500:], language=None)
