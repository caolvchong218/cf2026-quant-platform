"""Incremental white-box research; every candidate uses the current cash ledger.

Selection uses realized training labels only. Published 2025-2026 observations
are historical development comparisons, never relabeled as an unseen test.
"""
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from .analytics import performance, reconcile
from .benchmark_research import load_inputs, annual_windows, benchmark_metrics
from .config import Config
from .data import digest, write_json
from .engine import run_backtest
from .experiment import json_safe
from .factors import panel
from .features import FEATURES, FEATURE_CARDS, standardize, neutralize
from .risk import RiskPolicy, build_targets

LABEL_HORIZON = 20
POLICY = RiskPolicy(holdings=50, rebalance_every=20, target_volatility=.20,
                    max_exposure=.95, defensive_scale=.75)
LABELS = {'fixed_whitebox': '固定经济因子 · 白箱',
          'mined_rank': '训练期筛选 · 白箱排名组合',
          'mined_ridge': '训练期筛选 · 年度 Ridge',
          'blend': '主线与白箱 · 各50%目标组合',
          'mainline': '原主线 · 年度 LightGBM'}


def source_root(root):
    """Local-only configuration: no private path is required in a distribution."""
    p = Path(root)/'runs/local_data_source.json'
    return Path(json.loads(p.read_text(encoding='utf-8'))['data_root']) if p.exists() else Path(root)


def training_dates(calendar, label_exit, cutoff, years=3):
    cutoff = pd.Timestamp(cutoff)
    return ((calendar >= cutoff-pd.DateOffset(years=years)) & (calendar < cutoff)
            & (pd.DatetimeIndex(label_exit) < cutoff))


def rank_ic_daily(values, labels):
    return values.rank(axis=1, pct=True).corrwith(labels.rank(axis=1, pct=True), axis=1)


def neutralize_panel(values, cap, industry):
    """Apply the existing single-cross-section convention independently per day."""
    result=pd.DataFrame(np.nan,index=values.index,columns=values.columns)
    for date,row in values.iterrows():
        result.loc[date]=neutralize(row.rename('factor').to_frame(),cap.loc[date],industry.loc[date])['factor']
    return result


def research_cards(custom_candidates=None):
    """Keep economic and DSL identities distinct; reject ambiguous user IDs."""
    from .factor_lab import builtin_candidates, validate_expression
    economic=[{'id':n,'label':FEATURE_CARDS[n][0],'expression':FEATURE_CARDS[n][1],
               'hypothesis':FEATURE_CARDS[n][3],'source':'existing_economic_factor'} for n in FEATURES]
    formulas=[{**c,'id':'dsl_'+c['id'],'source':'offline_formula_catalogue'} for c in builtin_candidates()]
    formulas+=list(custom_candidates or [])
    seen=set(FEATURES)
    for card in formulas:
        if card['id'] in seen:raise ValueError('Duplicate factor identity: '+card['id'])
        validate_expression(card['expression'])
        seen.add(card['id'])
    return economic+formulas


def select_factors(factors, labels, dates, limit=6, correlation_limit=.75):
    """Fixed budget, training-only direction/strength and redundancy screening."""
    ranked, selected = [], []
    for name, values in factors.items():
        x, y = values.loc[dates], labels.loc[dates]
        ic = rank_ic_daily(x, y).dropna()
        coverage = float((x.notna() & y.notna()).sum().sum()/max(1,y.notna().sum().sum()))
        mean = float(ic.mean()) if len(ic) else 0.
        ranked.append({'id':name, 'rank_ic':mean, 'direction':1 if mean >= 0 else -1,
                       'coverage':coverage, 'valid_days':len(ic), 'strength':abs(mean)})
    ranked.sort(key=lambda r:(-r['strength'],r['id']))
    for row in ranked:
        if row['coverage'] < .80 or row['valid_days'] < 60:
            row['reason']='coverage_or_days'; continue
        x = factors[row['id']].loc[dates].to_numpy().ravel()
        duplicate=False
        for old in selected:
            y=factors[old['id']].loc[dates].to_numpy().ravel()
            mask=np.isfinite(x)&np.isfinite(y)
            corr=float(np.corrcoef(x[mask],y[mask])[0,1]) if mask.sum()>100 else 0.
            if abs(corr)>correlation_limit:
                duplicate=True;row['reason']='redundant_with_'+old['id'];break
        if not duplicate and len(selected)<limit:
            row['reason']='selected';selected.append(row)
        elif not duplicate:row['reason']='fixed_budget'
    if not selected:raise ValueError('No factor met coverage and training-day requirements')
    return [x['id'] for x in selected], ranked


def blend_target_weights(first, second, first_weight=.5):
    if not 0<=first_weight<=1:raise ValueError('Blend weight must lie in [0,1]')
    if not first.index.equals(second.index) or not first.columns.equals(second.columns):
        raise ValueError('Blend requires identical complete date/asset axes')
    for x in (first,second):
        a=x.to_numpy()
        if not np.isfinite(a).all() or (a<0).any() or (a.sum(axis=1)>1+1e-10).any():
            raise ValueError('Invalid component target weights')
    return first*first_weight+second*(1-first_weight)


def _models(factors, selected, labels, exits, calendar, eligible):
    """Freeze factor identities before 2023; re-estimate signs/coefficients annually."""
    from sklearn.linear_model import Ridge
    scores={n:pd.DataFrame(np.nan,index=calendar,columns=labels.columns)
            for n in ('mined_rank','mined_ridge')}
    folds=[]
    for year,cutoff,end in annual_windows(calendar, range(2023,2027)):
        dates=calendar[training_dates(calendar, exits, cutoff)]
        pred_dates=calendar[(calendar>=cutoff)&(calendar<end)]
        signs={n:(1 if rank_ic_daily(factors[n].loc[dates],labels.loc[dates]).mean()>=0 else -1)
               for n in selected}
        panels=[factors[n].rank(axis=1,pct=True)-.5 for n in selected]
        all_valid=np.logical_and.reduce([p.notna().to_numpy() for p in panels]) & eligible.to_numpy()
        ranking=sum(p*signs[n] for n,p in zip(selected,panels))/len(selected)
        scores['mined_rank'].loc[pred_dates]=ranking.loc[pred_dates].where(all_valid[calendar.get_indexer(pred_dates)])
        x=np.stack([p.loc[dates].to_numpy() for p in panels],axis=-1).reshape(-1,len(selected))
        y=(labels.loc[dates].rank(axis=1,pct=True)-.5).to_numpy().ravel()
        mask=np.isfinite(y)&np.isfinite(x).all(axis=1)&eligible.loc[dates].to_numpy().ravel()
        if mask.sum()<100:raise ValueError('Insufficient realized training observations')
        model=Ridge(alpha=100.).fit(x[mask],y[mask])
        xp=np.stack([p.loc[pred_dates].to_numpy() for p in panels],axis=-1).reshape(-1,len(selected))
        valid=np.isfinite(xp).all(axis=1)&eligible.loc[pred_dates].to_numpy().ravel()
        out=np.full(len(xp),np.nan);out[valid]=model.predict(xp[valid])
        scores['mined_ridge'].loc[pred_dates]=out.reshape(len(pred_dates),len(labels.columns))
        folds.append({'year':year,'cutoff':str(cutoff.date()),'selected':selected,'directions':signs,
                      'weights':{n:signs[n]/len(selected) for n in selected},
                      'ridge_coefficients':dict(zip(selected,model.coef_.tolist())),
                      'ridge_intercept':float(model.intercept_), 'train_rows':int(mask.sum()),
                      'max_label_exit':str(exits.loc[dates].max().date())})
    return scores,folds


def run_research(root, data_root=None, output=None, *, demo=False, custom_candidates=None):
    """Run bounded reproducible research. Output is immutable once created."""
    from .factor_lab import builtin_candidates, evaluate_expression
    from .return_attribution import attribute_result
    root=Path(root); data_root=Path(data_root or source_root(root))
    output=Path(output or root/'runs'/('factor_lab_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
    if output.exists():raise FileExistsError('Use a new experiment directory')
    output.mkdir(parents=True)
    started=time.monotonic()
    if demo:
        return _run_demo(root,output)
    frame,market,calendar,bench,_=load_inputs(data_root)
    assets=pd.Index(sorted(market.asset.unique()),name='asset')
    def pivot(name):return frame.pivot(index='date',columns='asset',values=name).reindex(index=calendar,columns=assets)
    eligible=pivot('eligible').fillna(False).astype(bool)
    labels=pivot('label').where(eligible)
    exits=frame.groupby('date').label_exit.first().reindex(calendar)
    factors={n:pivot('n_'+n).where(eligible) for n in FEATURES}
    cards=research_cards(custom_candidates)
    fields={n:panel(market,calendar,n).reindex(columns=assets) for n in ('open','high','low','close','volume')}
    cap,industry=pivot('log_cap'),pivot('industry')
    initial_cutoff=calendar[calendar<pd.Timestamp('2023-01-01')][-1]
    train_dates=calendar[training_dates(calendar,exits,initial_cutoff)]
    protocol={'schema':'qingxu.research.v4','data_end':str(calendar[-1].date()),'factor_budget':6,
              'candidate_budget':len(cards),
              'label_horizon_sessions':LABEL_HORIZON,'initial_cutoff':str(initial_cutoff.date()),
              'training':'preceding 3 years; label exits strictly before formation',
              'rank_direction':'training-only, annual update','ridge_alpha':100,
              'comparison_start':'2025-01-02','comparison_end':str(calendar[-1].date()),
              'historical_development':True,'independent_blind_test':False,
              'llm_called':False,'selection_correlation_limit':.75,'policy':asdict(POLICY)}
    write_json(output/'protocol.json',protocol)
    for card in cards[len(FEATURES):]:
        print('Evaluate formula: '+card['id'],flush=True)
        raw=evaluate_expression(card['expression'],fields)
        norm=standardize(raw.T).T
        factors[card['id']]=neutralize_panel(norm,cap,industry).where(eligible)
    selected,screening=select_factors(factors,labels,train_dates)
    pd.DataFrame(screening).to_csv(output/'screening.csv',index=False)
    write_json(output/'factor_cards.json',cards)
    factor_rows=[];ic_rows=[]
    for name,values in factors.items():
        for phase,dates in [('training',train_dates),('validation',calendar[(calendar>='2023-01-03')&(calendar<='2024-12-31')]),
                            ('historical',calendar[calendar>='2025-01-02'])]:
            # Only realized endpoint labels in each declared interval.
            dates=dates[pd.DatetimeIndex(exits.reindex(dates))<dates[-1]] if len(dates) else dates
            ic=rank_ic_daily(values.loc[dates],labels.loc[dates]).dropna()
            factor_rows.append({'id':name,'phase':phase,'rank_ic':float(ic.mean()),
                 'rank_ic_ir':float(ic.mean()/ic.std(ddof=1)) if ic.std(ddof=1)>1e-12 else None,
                 'days':len(ic),'selected':name in selected})
            ic_rows.extend({'date':d,'id':name,'phase':phase,'rank_ic':v} for d,v in ic.items())
    pd.DataFrame(factor_rows).to_csv(output/'factor_metrics.csv',index=False)
    pd.DataFrame(ic_rows).to_csv(output/'factor_ic_daily.csv',index=False)
    scores,folds=_models(factors,selected,labels,exits,calendar,eligible)
    fixed=['earnings_yield','book_yield','dividend_yield','quality_roe','low_volatility_20','range_20']
    scores['fixed_whitebox']=sum(factors[n].rank(axis=1,pct=True)-.5 for n in fixed)/len(fixed)
    scores['fixed_whitebox']=scores['fixed_whitebox'].where(eligible)
    existing=pd.read_parquet(data_root/'runs/research_v3/scores.parquet',columns=['date','asset','rolling_lightgbm'])
    scores['mainline']=existing.pivot(index='date',columns='asset',values='rolling_lightgbm').reindex(index=calendar,columns=assets)
    write_json(output/'folds.json',folds)
    long=pd.concat({n:s.stack(dropna=False) for n,s in scores.items()},axis=1).reset_index()
    long.to_parquet(output/'scores.parquet',index=False)
    rows=[];targets_by_phase={};runs={};preselected=None
    for phase,start,end in [('validation','2023-01-03','2024-12-31'),('historical','2025-01-02',str(calendar[-1].date()))]:
        targets={}
        for name,score in scores.items():
            targets[name],_=build_targets(score,frame,fields['close'],bench.close,start,POLICY)
        targets['blend']=blend_target_weights(targets['mainline'],targets['mined_rank'])
        targets_by_phase[phase]=targets
        for name in LABELS:
            print('Backtest '+phase+': '+name,flush=True)
            cfg=Config(name=name,start=start,end=end,holdings=50,rebalance_every=20,buy_cost=.001,sell_cost=.0015)
            score=scores.get(name,scores['mainline'])
            result=run_backtest(market,calendar,score,cfg,target_weights=targets[name],participation_limit=.01)
            checks=reconcile(result,cfg.initial_cash,cfg.buy_cost,cfg.sell_cost)
            if not checks['passed']:raise AssertionError(checks)
            metrics=performance(result.daily,cfg.initial_cash,trades=result.trades)
            metrics.update(benchmark_metrics(result.daily,bench.close))
            tr=pd.read_csv(data_root/'data/private/benchmark_v3/total_return.csv')
            tr['date']=pd.to_datetime(tr.trade_date.astype(str));tr=tr.set_index('date').close.sort_index()
            tr_metrics=benchmark_metrics(result.daily,tr)
            metrics.update({'benchmark_total_return':tr_metrics['benchmark_return'],
                            'excess_vs_total_return':metrics['total_return']-tr_metrics['benchmark_return']})
            folder=output/(phase+'_'+name);folder.mkdir()
            for table in ('daily','trades','positions','orders'):getattr(result,table).to_csv(folder/(table+'.csv'),index=False)
            targets[name].to_parquet(folder/'targets.parquet')
            write_json(folder/'metrics.json',json_safe(metrics));write_json(folder/'checks.json',checks)
            write_json(folder/'config.json',{'backtest':{**cfg.to_dict(),'strategy_id':'v4/'+name},'policy':asdict(POLICY),
                       'candidate':name,'historical_development':True})
            attribution=attribute_result(result,cfg.initial_cash,industry.loc[pd.Timestamp(result.daily.date.max())])
            if not attribution.reconciliation['passed']:raise AssertionError(attribution.reconciliation)
            attribution.stocks.to_csv(folder/'attribution_stocks.csv',index=False)
            attribution.industries.to_csv(folder/'attribution_industries.csv',index=False)
            write_json(folder/'attribution_scope.json',{'industry_mapping_date':str(result.daily.date.max()),
                       'grouping':'end-date historical industry labels; descriptive cashflow attribution'})
            write_json(folder/'attribution_check.json',json_safe(attribution.reconciliation))
            rows.append({'candidate':name,'label':LABELS[name],'phase':phase,**metrics})
            runs[(phase,name)]=(result,cfg)
        if phase=='validation':
            validation=pd.DataFrame(rows)
            validation=validation[validation.candidate.isin(['fixed_whitebox','mined_rank','mined_ridge'])]
            preselected=validation.sort_values(['sharpe','candidate'],ascending=[False,True]).iloc[0].candidate
            write_json(output/'selection.json',{'whitebox_preselected':preselected,
                       'selection_rule':'validation Sharpe descending, candidate ID ties',
                       'written_before_historical_comparison':True,
                       'previous_historical_results_already_observed':True,
                       'independent_blind_test':False})
    summary=pd.DataFrame(rows);summary.to_csv(output/'candidates.csv',index=False)
    assert preselected is not None
    chosen_result,_=runs[('historical',preselected)]
    last_signal=pd.Timestamp(chosen_result.trades.signal_date.max())
    fold=next(f for f in folds if f['year']==last_signal.year)
    held=chosen_result.positions[chosen_result.positions.date==chosen_result.daily.date.max()]
    explanations=[]
    for row in held.itertuples():
        names=fixed if preselected=='fixed_whitebox' else selected
        contributions=[]
        for name in names:
            normalized=float(factors[name].loc[last_signal].rank(pct=True).get(row.asset,np.nan)-.5)
            weight=(1/len(names) if preselected=='fixed_whitebox' else
                    fold['ridge_coefficients'][name] if preselected=='mined_ridge' else fold['weights'][name])
            card=next(c for c in cards if c['id']==name)
            contributions.append({'factor':name,'expression':card['expression'],
                                  'normalized_value':normalized,'weight':weight,
                                  'contribution':normalized*weight})
        explanations.append({'asset':row.asset,'date':str(last_signal.date()),'candidate':preselected,
                             'actual_weight':float(row.weight),'contributions':contributions,
                             'ridge_intercept':fold['ridge_intercept'] if preselected=='mined_ridge' else 0.,
                             'note':'形成日因子贡献；末日实际持仓权重随价格漂移。'})
    write_json(output/'holding_explanations.json',json_safe(explanations))
    from .execution_constraints import ExecutionPolicy,run_constrained_backtest,audit_constrained_result
    stress=[]
    for name in ['mainline',preselected,'blend']:
        _,cfg=runs[('historical',name)];score=scores.get(name,scores['mainline'])
        ep=ExecutionPolicy()
        strict=run_constrained_backtest(market,calendar,score,cfg,target_weights=targets_by_phase['historical'][name],policy=ep)
        audit=audit_constrained_result(strict,cfg.initial_cash,ep)
        if not audit.get('passed',False):raise AssertionError(audit)
        folder=output/('strict_'+name);folder.mkdir()
        for table in ('daily','trades','positions','orders'):getattr(strict,table).to_csv(folder/(table+'.csv'),index=False)
        sm=performance(strict.daily,cfg.initial_cash,trades=strict.trades);sm.update(benchmark_metrics(strict.daily,bench.close))
        stress.append({'candidate':name,'scenario':'lot_min_fee_tax_slippage',**sm})
        write_json(folder/'metrics.json',json_safe(sm))
        write_json(folder/'config.json',{'backtest':{**cfg.to_dict(),'strategy_id':'v4/'+name},
                   'candidate':name,'execution_policy':asdict(ep),'historical_development':True})
        write_json(folder/'checks.json',json_safe(audit));write_json(folder/'policy.json',asdict(ep))
    # Equal comparison stress uses common original fee rates for both routes.
    for name in ['mainline',preselected]:
        _,cfg=runs[('historical',name)];score=scores[name]
        double=replace(cfg,buy_cost=.002,sell_cost=.003)
        result=run_backtest(market,calendar,score,double,target_weights=targets_by_phase['historical'][name],participation_limit=.01)
        cm=performance(result.daily,double.initial_cash,trades=result.trades);cm.update(benchmark_metrics(result.daily,bench.close))
        stress.append({'candidate':name,'scenario':'double_baseline_fees',**cm})
    pd.DataFrame(stress).to_csv(output/'stress.csv',index=False)
    write_json(output/'decision.json',{'mainline_retained':'v3/rolling_lightgbm__managed','whitebox_preselected':preselected,
               'selection_rule':'validation Sharpe descending, candidate ID ties; no historical reranking',
               'historical_development':True,'independent_blind_test':False,'forward_validation_required':True,
               'llm_called':False,'selected_factors':selected})
    manifest={'status':'complete','source':'real_tushare_snapshot','data_end':str(calendar[-1].date()),
              'market_rows':len(market),'assets':len(assets),'protocol':protocol,'seconds':time.monotonic()-started,
              'input_hashes':{name:digest(data_root/path) for name,path in {
                  'features':'data/private/research_v2/features.parquet',
                  'feature_identity':'data/private/research_v2/features_cache.json',
                  'market':'data/private/mainboard1000_20260918/data/processed/market.csv',
                  'calendar':'data/private/mainboard1000_20260918/data/processed/calendar.csv',
                  'price_benchmark':'data/private/research_v2/benchmark.csv',
                  'total_return_benchmark':'data/private/benchmark_v3/total_return.csv',
                  'mainline_scores':'runs/research_v3/scores.parquet'}.items()},
              'output_hashes':{'scores':digest(output/'scores.parquet'),'folds':digest(output/'folds.json'),
                               'factor_cards':digest(output/'factor_cards.json')},
              'source_hashes':{p.name:digest(p) for p in (root/'src/cfquant').glob('*.py')}}
    write_json(output/'manifest.json',json_safe(manifest))
    return output


def _run_demo(root,output):
    from .data import load_market,load_calendar
    from .factor_lab import builtin_candidates,evaluate_expression
    cfg=Config.load(root/'configs/demo.yaml')
    market=load_market(root/cfg.data_path);calendar=load_calendar(root/cfg.calendar_path)
    fields={n:panel(market,calendar,n) for n in ('open','high','low','close','volume')}
    card=builtin_candidates()[0];score=evaluate_expression(card['expression'],fields)
    result=run_backtest(market,calendar,score,cfg)
    folder=output/'historical_demo';folder.mkdir()
    for table in ('daily','trades','positions','orders'):getattr(result,table).to_csv(folder/(table+'.csv'),index=False)
    write_json(output/'factor_cards.json',[card]);write_json(output/'manifest.json',{'status':'complete','source':'synthetic_demo','investment_evidence':False})
    pd.DataFrame([{'candidate':'demo','label':'合成流程验证','phase':'historical',**performance(result.daily,cfg.initial_cash,trades=result.trades)}]).to_csv(output/'candidates.csv',index=False)
    return output


def publish_evidence(root,output):
    """Publish aggregate metrics/ledgers only; never scores or stock-level trades."""
    import shutil
    root,output=Path(root),Path(output);dest=root/'evidence/research_v4'
    if dest.exists():raise FileExistsError('Preserve existing evidence before a new release')
    dest.mkdir(parents=True);(dest/'daily').mkdir()
    for name in ['manifest.json','protocol.json','selection.json','decision.json','candidates.csv','factor_cards.json',
                 'factor_metrics.csv','factor_ic_daily.csv','screening.csv','folds.json','stress.csv']:
        shutil.copy2(output/name,dest/name)
    rows=pd.read_csv(output/'candidates.csv')
    rows[rows.phase=='historical'].to_csv(dest/'test.csv',index=False)
    rows[rows.phase=='validation'].to_csv(dest/'validation.csv',index=False)
    for name in LABELS:
        shutil.copy2(output/('historical_'+name)/'daily.csv',dest/'daily'/f'{name}.csv')
    for folder in output.glob('strict_*'):
        shutil.copy2(folder/'daily.csv',dest/'daily'/f'{folder.name}.csv')
    # Private experiment location stays local; public evidence is portable.
    write_json(root/'runs/research_v4_location.json',{'output':str(output.resolve())})
    hashes={str(p.relative_to(dest)).replace('\\','/'):digest(p) for p in dest.rglob('*') if p.is_file()}
    write_json(dest/'SHA256.json',hashes)
    return dest
