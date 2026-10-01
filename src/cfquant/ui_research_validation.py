"""Execution stress, descriptive attribution and append-only forward plans."""
from pathlib import Path
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
import re
import pandas as pd
import plotly.express as px
import streamlit as st
from .ui_design import heading,chart,note
from .ui_factor_lab import experiment_folder,_read


def _sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _artifact(path):
    path = Path(path)
    if not path.is_file():
        return None
    content = (_read(path) if path.suffix == '.json' else path.read_text(encoding='utf-8'))
    return {'path': str(path.resolve()), 'sha256': _sha256(path), 'content': content}


def build_freeze_inputs(root, strategy_id):
    """Read concrete strategy state; absent identity/configuration stays explicit."""
    from .research_lab import source_root
    root = Path(root)
    allowed = {'v3/rolling_lightgbm__managed', 'v4/fixed_whitebox', 'v4/mined_rank', 'v4/mined_ridge', 'v4/blend'}
    if strategy_id not in allowed:
        raise ValueError('不支持的冻结策略身份')
    version, candidate = strategy_id.split('/')
    evidence = root / 'evidence' / ('research_' + version)
    data_root = source_root(root)
    local = experiment_folder(root) if version == 'v4' else data_root / 'runs/research_v3'
    local = Path(local) if local else root / 'runs/research_v4'
    required = ['folds', 'decision', 'manifest']
    if version == 'v4':
        required += ['factor_cards', 'protocol', 'selection']
    else:
        required += ['provenance']
    artifacts, missing = {}, []
    for name in required:
        choices = [local / (name + '.json'), evidence / (name + '.json')]
        artifact = next((_artifact(p) for p in choices if p.is_file()), None)
        artifacts[name] = artifact
        if artifact is None:
            missing.append(name + '.json')
    if version == 'v3':
        choices = [evidence / 'protocol.json', local / 'protocol.json',
                   root / 'docs/RESEARCH_PROTOCOL_V3.md', data_root / 'docs/RESEARCH_PROTOCOL_V3.md']
        artifacts['protocol'] = next((_artifact(p) for p in choices if p.is_file()), None)
        if artifacts['protocol'] is None:
            missing.append('V3 research protocol')
    configuration_path = local / (('historical_' if version == 'v4' else 'test_') + candidate) / 'config.json'
    configuration = _artifact(configuration_path)
    artifacts['candidate_config'] = configuration
    if configuration is None:
        missing.append('本地候选 config.json')
    manifest = artifacts['manifest']['content'] if artifacts.get('manifest') else {}
    provenance = artifacts.get('provenance')
    historical_source = manifest.get('source_hashes', {})
    if provenance:
        historical_source = provenance['content'].get('source_hashes', historical_source)
    package = root / 'src/cfquant'
    current_source = {p.name: _sha256(p) for p in sorted(package.glob('*.py'))}
    if not current_source:
        package = Path(__file__).resolve().parent
        current_source = {p.name: _sha256(p) for p in sorted(package.glob('*.py'))}
    config = configuration['content'] if configuration else {}
    cutoff = manifest.get('data_end') or config.get('backtest', {}).get('end')
    if not cutoff:
        missing.append('数据快照截止日')
    published_inputs = manifest.get('input_hashes', {})
    if provenance:
        published_inputs = {**published_inputs, **{name: value for name, value in provenance['content'].items()
                            if name in {'features_sha256', 'scores_sha256', 'protocol_sha256'}}}
    score_path = local / 'scores.parquet'
    actual_outputs = {'scores.parquet': _sha256(score_path)} if score_path.is_file() else {}
    declared_score = manifest.get('output_hashes', {}).get('scores')
    if version=='v4':
        if not score_path.is_file():missing.append('本地 scores.parquet')
        if not isinstance(declared_score,str) or re.fullmatch('[0-9a-fA-F]{64}',declared_score) is None:
            missing.append('manifest 缺少有效 scores SHA256')
        for name,item in artifacts.items():
            published=evidence/(name+'.json')
            if item and published.is_file() and item['sha256']!=_sha256(published):
                missing.append(name+' 与发布研究快照不一致')
    if declared_score and actual_outputs.get('scores.parquet') and declared_score != actual_outputs['scores.parquet']:
        missing.append('本地分数与 manifest 输出哈希不一致')
    if not published_inputs:
        missing.append('研究输入哈希')
    definition = {'strategy_id': strategy_id, 'artifacts': artifacts,
                  'default_mainline_retained': True,
                  'research_scope': 'historical development; requires future observations',
                  'protocol_json_available': (evidence / 'protocol.json').is_file() or (local / 'protocol.json').is_file()}
    return {'ready': not missing, 'missing': missing, 'data_cutoff': cutoff,
            'strategy_definition': definition, 'config': config,
            'code_identity': {'kind': 'current_and_historical_source_sha256',
                              'current_source_hashes': current_source, 'research_source_hashes': historical_source},
            'data_identity': {'kind': 'explicit_research_artifact_sha256', 'input_hashes': published_inputs,
                              'declared_output_hashes': manifest.get('output_hashes', manifest.get('files', {})),
                              'actual_output_hashes': actual_outputs,
                              'artifact_hashes': {name: item['sha256'] for name, item in artifacts.items() if item}}}


def parse_index_weights_csv(payload, *, weights_in_percent=False):
    """Bounded data-only CSV import; preserve leading zeros in asset codes."""
    from .index_enhancement import validate_index_weights
    if not isinstance(payload, bytes) or not payload or len(payload) > 8_000_000:
        raise ValueError('指数权重 CSV 必须是非空文件，且不超过 8 MB')
    try:
        frame = pd.read_csv(BytesIO(payload), encoding='utf-8-sig', dtype={'asset': str}, nrows=100001)
    except (ValueError, UnicodeError, pd.errors.ParserError):
        raise ValueError('无法读取指数权重 CSV，请使用 UTF-8 编码与标准列名') from None
    if len(frame) > 100000:
        raise ValueError('指数权重 CSV 不得超过 100000 行')
    return validate_index_weights(frame, weights_in_percent=weights_in_percent)


def index_data_readiness(root):
    from .research_lab import source_root
    root = Path(root)
    data = source_root(root)
    folder = experiment_folder(root)
    scores_path = folder / 'scores.parquet' if folder else root / 'runs/research_v4/scores.parquet'
    manifest = _read(folder / 'manifest.json', {}) if folder else {}
    required = [data / 'data/private/research_v2' / name for name in
                ['features.parquet', 'features_cache.json', 'benchmark.csv', 'cpi.csv']]
    required += [data / 'data/private/mainboard1000_20260918/data/processed' / name
                 for name in ['market.csv', 'calendar.csv']]
    missing = [p.name for p in required if not p.is_file()]
    if not scores_path.is_file():
        missing.append('本地 V4 scores.parquet')
    if manifest.get('source') != 'real_tushare_snapshot' or manifest.get('status') != 'complete':
        missing.append('完成的真实快照研究 manifest')
    if not manifest.get('output_hashes', {}).get('scores'):
        missing.append('研究分数输出哈希')
    return {'ready': not missing, 'missing': missing, 'data_root': data,
            'scores_path': scores_path, 'research_manifest': manifest}


def run_index_experiment(root, weights, candidate, start, end, *, max_active_stock=.01,
                         max_active_industry=.03, max_exposure=.95, lookback=60,
                         rebalance_every=20, initial_cash=1e6, buy_cost=.001, sell_cost=.0015):
    """Use actual local snapshots/scores, then save a separately reconciled ledger."""
    from .analytics import performance, reconcile
    from .benchmark_research import load_inputs, benchmark_metrics
    from .config import Config
    from .data import write_json
    from .engine import run_backtest
    from .experiment import json_safe
    from .factors import panel
    from .index_enhancement import index_enhanced_targets, validate_index_weights
    from .return_attribution import attribute_result
    root = Path(root)
    if candidate not in {'fixed_whitebox', 'mined_rank', 'mined_ridge', 'mainline'}:
        raise ValueError('请选择已保存的 V4 分数列')
    if pd.Timestamp(start) < pd.Timestamp('2023-01-03'):
        raise ValueError('V4 模型分数只支持 2023-01-03 之后的执行日期')
    ready = index_data_readiness(root)
    if not ready['ready']:
        raise ValueError('缺少真实研究输入：' + '、'.join(ready['missing']))
    if _sha256(ready['scores_path']) != ready['research_manifest']['output_hashes']['scores']:
        raise ValueError('本地 V4 分数与研究 manifest 哈希不一致')
    weights = validate_index_weights(weights)
    cfg = Config(name='index_enhancement_' + candidate, start=str(start), end=str(end),
                 rebalance_every=int(rebalance_every), holdings=1000, initial_cash=initial_cash,
                 buy_cost=buy_cost, sell_cost=sell_cost)
    frame, market, calendar, benchmark, _ = load_inputs(ready['data_root'])
    if pd.Timestamp(end) > calendar[-1]:
        raise ValueError('回测截止日超过本地快照最后交易日：'+str(calendar[-1].date()))
    dates = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    if not len(dates) or not len(calendar[calendar < dates[0]]):
        raise ValueError('请选择含上一交易日的有效本地回测区间')
    first_position = int(calendar.get_loc(dates[0]))
    last_position = int(calendar.get_loc(dates[-1]))
    # Preserve enough daily prehistory for both alpha risk and execution liquidity.
    calendar = calendar[max(0, first_position - max(lookback + 2, 22)):last_position + 1]
    assets = pd.Index(sorted(market.asset.unique()), name='asset')
    raw = pd.read_parquet(ready['scores_path'], columns=['date', 'asset', candidate])
    raw['date'] = pd.to_datetime(raw.date)
    if raw.duplicated(['date', 'asset']).any():
        raise ValueError('本地分数存在重复日期/资产')
    scores = raw.pivot(index='date', columns='asset', values=candidate).reindex(index=calendar, columns=assets)
    close = panel(market, calendar, 'close').reindex(columns=assets)
    industries = frame.pivot(index='date', columns='asset', values='industry').reindex(index=calendar, columns=assets)
    formations = pd.DatetimeIndex([calendar[calendar.get_loc(d) - 1] for d in dates[::cfg.rebalance_every]])
    formation_scores = scores.where(pd.Series(scores.index.isin(formations), index=scores.index), axis=0)
    allocations, diagnostics = index_enhanced_targets(formation_scores, close, industries, weights,
        max_active_stock=max_active_stock, max_active_industry=max_active_industry,
        max_exposure=max_exposure, lookback=lookback)
    diagnostics['used_formation'] = diagnostics.date.isin(formations)
    required = diagnostics[diagnostics.used_formation]
    blocked = required[~required.constraints_passed]
    if len(blocked):
        row = blocked.iloc[0]
        raise ValueError(f"形成日 {pd.Timestamp(row.date).date()} 无法配置：{row.block_reason}；未生成回测或修补缺失权重")
    targets = pd.DataFrame(0., index=calendar, columns=assets)
    current = pd.Series(0., index=assets)
    for date in calendar:
        if date in formations:
            current = allocations.loc[date]
        targets.loc[date] = current
    # The current engine executes only the preceding formation row at rebalance.
    result = run_backtest(market, calendar, scores, cfg, target_weights=targets, participation_limit=.01)
    checks = reconcile(result, cfg.initial_cash, cfg.buy_cost, cfg.sell_cost)
    if not checks['passed']:
        raise ValueError('指数增强账本核对失败，未发布成功结果')
    metrics = performance(result.daily, cfg.initial_cash, trades=result.trades)
    metrics.update(benchmark_metrics(result.daily, benchmark.close, cfg.initial_cash))
    attribution = attribute_result(result, cfg.initial_cash, industries.loc[pd.Timestamp(result.daily.date.max())])
    if not attribution.reconciliation['passed']:
        raise ValueError('指数增强资金归因核对失败')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = root / 'runs' / ('index_enhancement_' + stamp)
    output.mkdir(parents=True, exist_ok=False)
    for name in ['daily', 'trades', 'positions', 'orders']:
        getattr(result, name).to_csv(output / (name + '.csv'), index=False)
    weights.to_csv(output / 'index_weights.csv', index=False)
    diagnostics.to_csv(output / 'allocation_diagnostics.csv', index=False)
    targets.to_parquet(output / 'targets.parquet')
    attribution.stocks.to_csv(output / 'attribution_stocks.csv', index=False)
    attribution.industries.to_csv(output / 'attribution_industries.csv', index=False)
    scope = {'industry_mapping_date': str(result.daily.date.max()),
             'start': str(result.daily.date.min()), 'end': str(result.daily.date.max()),
             'grouping': 'end-date historical labels; descriptive cashflow attribution'}
    configuration = {'backtest': cfg.to_dict(), 'candidate': candidate, 'rebalance_every': cfg.rebalance_every,
                     'max_active_stock': max_active_stock, 'max_active_industry': max_active_industry,
                     'max_exposure': max_exposure, 'lookback': lookback, 'participation_limit': .01,
                     'active_reference': 'exposure_scaled_benchmark', 'historical_development': True,
                     'target_constraints_are_not_actual_position_constraints': True,
                     'benchmark_metrics_scope': 'local CSI300 price series; uploaded membership identity is user-supplied'}
    write_json(output / 'config.json', configuration)
    write_json(output / 'checks.json', checks)
    write_json(output / 'metrics.json', json_safe(metrics))
    write_json(output / 'attribution_check.json', json_safe(attribution.reconciliation))
    write_json(output / 'attribution_scope.json', scope)
    inputs = {name: _sha256(ready['data_root'] / relative) for name, relative in {
        'features': 'data/private/research_v2/features.parquet',
        'feature_identity': 'data/private/research_v2/features_cache.json',
        'market': 'data/private/mainboard1000_20260918/data/processed/market.csv',
        'calendar': 'data/private/mainboard1000_20260918/data/processed/calendar.csv',
        'benchmark': 'data/private/research_v2/benchmark.csv'}.items()}
    inputs.update({'scores': _sha256(ready['scores_path']), 'index_weights': _sha256(output / 'index_weights.csv')})
    write_json(output / 'manifest.json', {'status': 'complete', 'source': 'local_snapshot_and_user_index_weights',
        'input_hashes': inputs, 'research_manifest': ready['research_manifest'],
        'source_hashes': {p.name: _sha256(p) for p in Path(__file__).resolve().parent.glob('*.py')},
        'output_hashes': {p.name: _sha256(p) for p in output.iterdir() if p.is_file()},
        'allocation_scope': 'only used formation dates; nonsignal targets carry previous allocation',
        'actual_execution_reconciled': True, 'independent_blind_test': False})
    return output, json_safe(metrics), diagnostics


def render(root):
    root=Path(root)
    heading('策略验证与观察','把历史开发、实际成交约束与冻结后的记录放在同一研究流程。','VALIDATION / V4')
    execution,attribution,forward,blend,index_tab=st.tabs(['成交约束与压力','收益归因','前向观察','多策略组合','指数增强'])
    evidence=root/'evidence/research_v4'
    with execution:
        st.write('比较整手买入、最低佣金、配置化印花税、滑点现金成本和滞后成交额上限对实际成交的影响。')
        note('研究仍以连续复权单位记账，整手约束换算为原始等价股数；未完整模拟分红现金、企业行动和清算。税费为明确配置的研究情景。')
        if (evidence/'stress.csv').exists():
            rows=pd.read_csv(evidence/'stress.csv')
            st.dataframe(rows[['candidate','scenario','total_return','sharpe','max_drawdown','total_cost','blocked_orders']],hide_index=True,width='stretch')
            st.download_button('下载成交压力对照',rows.to_csv(index=False).encode('utf-8-sig'),'execution_stress.csv')
            st.caption('严格情景费率与原基线不同，不应将收益差异全部解释为整手效应；双倍基线费用单独列出。')
        else:st.info('成交压力结果尚未发布。')
        with st.expander('查看本地约束政策与核对结果'):
            folder=experiment_folder(root)
            paths=sorted(folder.glob('strict_*')) if folder else []
            if paths:
                path=Path(st.selectbox('严格成交实验',[str(p) for p in paths],format_func=lambda x:Path(x).name))
                st.json(_read(path/'policy.json',{}));st.json(_read(path/'checks.json',{}))
                if (path/'orders.csv').exists():
                    orders=pd.read_csv(path/'orders.csv')
                    failed=orders[orders.status!='filled']
                    st.dataframe(failed.head(200),hide_index=True,width='stretch')
            else:st.caption('逐股订单和政策细节需要本地实验文件。')
    with attribution:
        st.write('逐股损益 = 卖出现金流 − 买入现金流 − 实际成本 + 期末持仓价值。贡献相加核对组合净收益。')
        st.caption('这是资金账本归因。行业分组采用明确提供的标签，不能据此证明某个因子的因果贡献。')
        folder=experiment_folder(root)
        paths=sorted(folder.glob('historical_*/attribution_stocks.csv')) if folder else []
        paths+=sorted((root/'runs').glob('index_enhancement_*/attribution_stocks.csv'))
        if paths:
            p=Path(st.selectbox('归因策略',[str(p) for p in paths],format_func=lambda x:Path(x).parent.name,key='attribution_strategy'))
            frame=pd.read_csv(p,dtype={'asset':str});st.json(_read(p.parent/'attribution_check.json',{}))
            scope=_read(p.parent/'attribution_scope.json',{})
            if scope:
                st.caption('行业标签对应日期：'+str(scope.get('industry_mapping_date','未记录')))
                st.json(scope)
                st.download_button('下载归因日期与分组范围',json.dumps(scope,ensure_ascii=False,indent=2),
                                   'attribution_scope.json',key='attribution_scope_download')
            else:st.caption('此实验未记录行业标签日期；行业归因范围待核对。')
            col='net_pnl' if 'net_pnl' in frame else 'pnl'
            if col in frame and not frame.empty:
                frame[col]=pd.to_numeric(frame[col],errors='coerce')
                ranked=pd.concat([frame.nsmallest(10,col),frame.nlargest(10,col)]).drop_duplicates('asset')
                chart(px.bar(ranked.sort_values(col),x=col,y='asset',orientation='h'),460)
            elif frame.empty:st.caption('此账本没有逐股成交或持仓，收益归因保留为空。')
            st.dataframe(frame,hide_index=True,width='stretch')
            st.download_button('下载逐股资金归因',frame.to_csv(index=False).encode('utf-8-sig'),'stock_attribution.csv')
            industry_path=p.parent/'attribution_industries.csv'
            if industry_path.exists():
                industries=pd.read_csv(industry_path)
                st.markdown('**行业资金归因**')
                st.dataframe(industries,hide_index=True,width='stretch')
                st.download_button('下载行业资金归因',industries.to_csv(index=False).encode('utf-8-sig'),
                                   'industry_attribution.csv',key='attribution_industry_download')
        else:st.info('公开版本只展示聚合研究，逐股收益归因需本地完整实验。运行真实数据实验即可生成。')
    with forward:
        from .forward_observer import freeze_plan,load_plan,append_observation,observation_summary
        observation_root=root/'runs/forward_observations'
        st.write('先冻结策略和数据身份，再添加截止日之后的观察。已观察历史不能通过修改文件名变成盲测。')
        specs=['v3/rolling_lightgbm__managed']+['v4/'+x for x in ['fixed_whitebox','mined_rank','mined_ridge','blend']]
        selected=st.selectbox('待观察策略',specs,key='forward_strategy')
        try:
            frozen=build_freeze_inputs(root,selected)
        except (ValueError,OSError,KeyError) as exc:
            frozen={'ready':False,'missing':['状态文件读取失败：'+str(exc)],'data_cutoff':None}
        cutoff=frozen.get('data_cutoff')
        if cutoff:st.caption('所选策略快照截止 '+str(cutoff)+'。冻结时点前已经发生的新增历史，与真正后续观察分别记录。')
        if not frozen['ready']:st.info('冻结所需状态尚未齐备：'+'、'.join(frozen['missing']))
        with st.expander('查看将冻结的策略状态与身份'):
            if frozen.get('strategy_definition'):
                st.json(frozen['strategy_definition'])
                st.json(frozen['data_identity'])
            st.caption('冻结内容包含公式/训练折/选择记录、具体候选配置、当前与研究源码哈希，以及输入输出身份。')
        if st.button('冻结新的观察计划',key='forward_freeze',disabled=not frozen['ready']):
            try:
                plan=freeze_plan(observation_root,frozen['strategy_definition'],cutoff,config=frozen['config'],
                                 code_identity=frozen['code_identity'],data_identity=frozen['data_identity'])
                st.success('计划已冻结：'+plan['plan_id'])
            except (ValueError,OSError) as exc:st.error(str(exc))
        plans=sorted(observation_root.glob('*/plan.json'))
        if plans:
            ident=st.selectbox('观察计划',[p.parent.name for p in plans],key='forward_plan')
            plan=load_plan(observation_root,ident)
            st.json(observation_summary(observation_root,ident))
            st.download_button('下载冻结计划',json.dumps(plan,ensure_ascii=False,indent=2),'forward_plan.json')
            with st.expander('登记新观察记录'):
                st.caption('手工登记仅是本地观察记录，不证明成交真实性；需保留信号形成时间和原始回执。')
                with st.form('forward_append_form'):
                    date=st.date_input('观察交易日',pd.Timestamp.now().date(),key='forward_date')
                    signal=st.date_input('信号形成日',pd.Timestamp.now().date()-pd.Timedelta(days=1),key='forward_signal')
                    nav=st.number_input('观察净值（元）',1.,1e10,1e6,key='forward_nav')
                    submitted=st.form_submit_button('追加观察记录')
                if submitted:
                    try:
                        append_observation(observation_root,ident,str(date),str(signal),metrics={'nav':nav,'source':'manual_paper_observation'})
                        st.success('观察记录已追加。')
                    except ValueError as exc:st.error(str(exc))
        else:st.info('尚未建立冻结计划。当前前向表现待观察，历史收益不会填入此处。')
    with blend:
        st.write('主线与白箱的目标权重各占50%，合并后重新经过同一个成交引擎，计入重叠持仓、现金与费用。')
        st.caption('组合权重固定，不按已观察最终区间反复优化。因子中性化也不等于实际持仓完全中性。')
        p=evidence/'candidates.csv'
        if p.exists():
            rows=pd.read_csv(p);rows=rows[(rows.phase=='historical')&rows.candidate.isin(['mainline','mined_rank','blend'])]
            st.dataframe(rows[['label','total_return','sharpe','max_drawdown','mean_turnover','total_cost']],hide_index=True,width='stretch')
            curves=[]
            for name in rows.candidate:
                frame=pd.read_csv(evidence/'daily'/f'{name}.csv',parse_dates=['date'])
                frame['return']=frame.nav.pct_change();frame['candidate']=name;curves.append(frame)
            if curves:
                wide=pd.concat(curves).pivot(index='date',columns='candidate',values='return')
                st.markdown('**本历史区间日收益相关性**');st.dataframe(wide.corr())
        else:st.info('组合回测尚未发布。')
    with index_tab:
        st.write('导入形成日已经公布的完整历史指数成分和权重，再将本地 V4 分数用于基准相对配置。')
        st.caption('必需列：date、asset、weight、known_at。asset 必须与本地代码完全一致（如 000001.SZ），保留前导零。每个快照权重合计为1；known_at 是真实可获得时间。完整快照、价格或行业覆盖缺失会阻止相关形成日的回测。')
        st.caption('个股和行业主动偏离相对同一权益暴露下的基准权重计算。目标配置限制不等于实际成交后的持仓限制；历史数据回测属于开发比较。')
        sample=b'date,asset,weight,known_at\n2025-01-02,600000.SH,0.5,2025-01-02\n2025-01-02,000001.SZ,0.5,2025-01-02\n'
        st.download_button('下载格式样例（虚构权重）',sample,'index_weights_format_example.csv',key='index_sample_download')
        uploaded=st.file_uploader('上传历史指数权重 CSV',type=['csv'],key='index_weights_upload')
        percent=st.checkbox('上传文件 weight 使用百分数（显式除以100）',key='index_weights_percent')
        weights=None
        if uploaded is not None:
            try:
                weights=parse_index_weights_csv(uploaded.getvalue(),weights_in_percent=percent)
                st.success(f'结构验证通过：{weights.date.nunique()} 个完整权重快照，{len(weights)} 行。数据来源仍需用户保留原始证据。')
                st.dataframe(weights.head(60),hide_index=True,width='stretch')
            except ValueError as exc:st.error(str(exc))
        else:st.info('当前本地缓存未提供历史指数权重。上传有效快照后才可运行；此处尚无实测指数增强结论。')
        try:
            readiness=index_data_readiness(root)
        except (ValueError,OSError,KeyError) as exc:
            readiness={'ready':False,'missing':[str(exc)],'research_manifest':{}}
        if not readiness['ready']:st.info('本地真实回测输入未齐备：'+'、'.join(readiness['missing']))
        candidate=st.selectbox('指数增强使用的 V4 分数',['fixed_whitebox','mined_rank','mined_ridge','mainline'],key='index_score')
        c1,c2,c3=st.columns(3)
        stock=c1.number_input('个股主动偏离上限',0.,.2,.01,.005,key='index_active_stock')
        sector=c2.number_input('行业主动偏离上限',0.,.5,.03,.005,key='index_active_industry')
        exposure=c3.number_input('权益目标暴露',.01,1.,.95,.01,key='index_exposure')
        c1,c2,c3=st.columns(3)
        start=c1.date_input('指数回测起始日',pd.Timestamp('2025-01-02').date(),key='index_start')
        data_end=readiness.get('research_manifest',{}).get('data_end','2026-09-18')
        end=c2.date_input('指数回测截止日',pd.Timestamp(data_end).date(),key='index_end')
        rebalance=c3.number_input('指数调仓间隔（交易日）',1,60,20,1,key='index_rebalance')
        st.caption('收益对照使用现有本地沪深300价格序列；上传快照的指数身份由用户提供，软件不认证其来源。成交采用既有复权单位账本、下一期开盘、费用与滞后成交额上限。')
        if st.button('运行并保存指数增强回测',disabled=weights is None or not readiness['ready'],key='index_run'):
            try:
                with st.spinner('验证形成日覆盖、计算目标并核对实际成交账本…'):
                    output,metrics,diagnostics=run_index_experiment(root,weights,candidate,str(start),str(end),
                        max_active_stock=stock,max_active_industry=sector,max_exposure=exposure,
                        rebalance_every=int(rebalance))
                st.session_state['validation_index_output']=str(output)
                st.success('独立实验已保存：'+output.name+'；账本核对通过。')
            except (ValueError,OSError,KeyError,AssertionError) as exc:st.error(str(exc))
        saved=st.session_state.get('validation_index_output')
        if saved and Path(saved).is_dir():
            output=Path(saved)
            st.caption('以下结果来自已保存的独立实验：'+output.name)
            st.json(_read(output/'metrics.json',{}))
            diagnostics=pd.read_csv(output/'allocation_diagnostics.csv')
            formations=diagnostics[diagnostics.used_formation]
            baseline_only=int((formations.status=='ready_baseline_only').sum())
            if baseline_only:
                st.info(f'{baseline_only}/{len(formations)} 个形成日仅保持基准配置，没有形成主动因子倾斜。')
            st.dataframe(formations,hide_index=True,width='stretch')
            st.download_button('下载指数配置诊断',diagnostics.to_csv(index=False).encode('utf-8-sig'),
                               'index_allocation_diagnostics.csv',key='index_diagnostics_download')
