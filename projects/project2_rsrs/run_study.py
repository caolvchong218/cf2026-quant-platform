"""One-command fixed-parameter replication; writes all results, including losses."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import itertools
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rsrs import SPECS, strategy, decisions, backtest, metrics, audit_ledger

NAMES={"sh000300":"CSI 300", "sh000016":"SSE 50", "sh000905":"CSI 500"}
COLORS={"buy_hold":"#8492a6", "ma20":"#d49b39", "slope":"#b597bb",
        "standard":"#4878b8", "corrected":"#909d3d", "right":"#c06658",
        "standard_ma":"#8270ab", "corrected_ma":"#326a88", "right_ma":"#148779"}


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path,parse_dates=["date"]).set_index("date").sort_index()


def bootstrap_excess(strategy_returns, benchmark_returns, repetitions=2000, block=20):
    x=np.asarray(strategy_returns)-np.asarray(benchmark_returns)
    rng=np.random.default_rng(20261016); n=len(x); estimates=[]
    for _ in range(repetitions):
        starts=rng.integers(0,n,size=int(np.ceil(n/block)))
        ix=((starts[:,None]+np.arange(block)[None,:])%n).ravel()[:n]
        estimates.append(x[ix].mean()*252)
    return {"annualized_mean_daily_excess":float(x.mean()*252),
            "ci95_low":float(np.quantile(estimates,.025)),"ci95_high":float(np.quantile(estimates,.975)),
            "block_sessions":block,"replicates":repetitions,"seed":20261016,
            "interpretation":"Exploratory circular block bootstrap of paired daily arithmetic excess; not a CAGR interval or selection-adjusted proof."}


def main(data_dir: Path, output: Path, config_path: Path, supplemental: Path | None):
    cfg=json.loads(config_path.read_text(encoding="utf-8")); output.mkdir(parents=True,exist_ok=True)
    for sub in ["ledgers","figures"]:(output/sub).mkdir(exist_ok=True)
    manifest=json.loads((data_dir/"manifest.json").read_text(encoding="utf-8"))
    if hashlib.sha256(config_path.read_bytes()).hexdigest()!=manifest['config_sha256']:
        raise ValueError('Configuration differs from frozen data manifest; use the delivered configuration')
    for name,meta in manifest["files"].items():
        assert hashlib.sha256((data_dir/name).read_bytes()).hexdigest()==meta["sha256"],name
    all_metrics=[];all_audits=[]; all_ledgers={}; frames={};factor_frames={};sigmap={}
    for symbol in cfg["symbols"]:
        d=load_data(data_dir/f"{symbol}.csv");frames[symbol]=d
        sigs={"buy_hold":pd.Series(1,index=d.index),"ma20":(d.close>d.close.rolling(20).mean()).astype(int)}
        for spec in SPECS:
            f,s=strategy(d,spec);sigs[spec.name]=s;factor_frames[(symbol,spec.name)]=f
        sigmap[symbol]=sigs
        for period,(start,end) in cfg["periods"].items():
            for name,s in sigs.items():
                for cost in [0,cfg["cost_bps_each_side"]]:
                    ledger=backtest(d,s,start,end,cost)
                    key=f"{symbol}_{period}_{name}_{cost}bps"
                    ledger.to_csv(output/"ledgers"/f"{key}.csv",float_format="%.12g",lineterminator="\n")
                    all_metrics.append({"symbol":symbol,"period":period,"strategy":name,"cost_bps":cost,**metrics(ledger)})
                    all_audits.append({"run":key,**audit_ledger(ledger,cost)})
                    all_ledgers[(symbol,period,name,cost)]=ledger
        print('evaluated',symbol,flush=True)
    table=pd.DataFrame(all_metrics)
    benchmark=table[table.strategy=='buy_hold'][['symbol','period','cost_bps','total_return','cagr']]
    table=table.merge(benchmark,on=['symbol','period','cost_bps'],suffixes=('','_benchmark'))
    table['total_return_difference_pp']=100*(table.total_return-table.total_return_benchmark)
    table['cagr_difference_pp']=100*(table.cagr-table.cagr_benchmark)
    table.to_csv(output/'performance.csv',index=False,float_format='%.12g')
    pd.DataFrame(all_audits).to_csv(output/'ledger_audit.csv',index=False)

    # No optimization or deployment choice: show the entire preset neighbourhood.
    sensitivity=[]; d=frames['sh000300']; start,end=cfg['periods']['post_publication']
    base=next(s for s in SPECS if s.name=='right_ma')
    for n,m,s in itertools.product([14,16,18],[250,300,600],[.5,.7,.9]):
        spec=replace(base,n=n,m=m,upper=s,lower=-s)
        _,signal=strategy(d,spec); ledger=backtest(d,signal,start,end,10)
        sensitivity.append({'N':n,'M':m,'S':s,**metrics(ledger)})
    pd.DataFrame(sensitivity).to_csv(output/'sensitivity.csv',index=False)

    # Report wording admits more than one interpretation: display, never hide.
    diagnostics=[]
    for variant,lag,ddof,current in [('main_priorM_lag2',2,0,False),('trend_lag3',3,0,False),
                                   ('current_in_M_ddof0',2,0,True),('current_in_M_ddof1',2,1,True)]:
        spec=replace(base,trend_lag=lag);f,signal=strategy(d,spec)
        if current:
            roll=f.beta.rolling(spec.m,min_periods=spec.m)
            z=(f.beta-roll.mean())/roll.std(ddof=ddof)
            signal=decisions(z*f.r2*f.beta,.7,-.7,f.ma20>f.ma20.shift(lag))
        for period,(lo,hi) in cfg['periods'].items():
            diagnostics.append({'variant':variant,'period':period,**metrics(backtest(d,signal,lo,hi,10))})
    pd.DataFrame(diagnostics).to_csv(output/'implementation_sensitivity.csv',index=False)

    # Labels are confined to diagnostics; never used in decision or execution.
    correlations=[];bins_out=[]
    for period,(lo,hi) in cfg['periods'].items():
        q=factor_frames[('sh000300','right')]['right'].rename('score').to_frame()
        q['forward10']=d.close.shift(-10)/d.close-1
        q['label_end']=pd.Series(d.index,index=d.index).shift(-10)
        q=q.loc[lo:hi];q=q[q.label_end<=pd.Timestamp(hi)].dropna()
        q['bin']=pd.cut(q.score,bins=np.arange(-3,4.01,.2))
        grouped=q.groupby('bin',observed=True).agg(score=('score','mean'),mean_return=('forward10','mean'),count=('forward10','size'))
        grouped=grouped[grouped['count']>=10]
        correlations.append({'period':period,'daily_pearson':q.score.corr(q.forward10),
                             'daily_spearman':q.score.corr(q.forward10,method='spearman'),
                             'bin_mean_correlation_min10':grouped.score.corr(grouped.mean_return),
                             'daily_n':len(q),'bins_n':len(grouped)})
        grouped['period']=period;bins_out.append(grouped.reset_index())
    pd.DataFrame(correlations).to_csv(output/'predictive_diagnostics.csv',index=False)
    pd.concat(bins_out).to_csv(output/'score_bins.csv',index=False)

    robustness={}
    for symbol in cfg['symbols']:
        s=all_ledgers[(symbol,'post_publication','right_ma',10)]
        b=all_ledgers[(symbol,'post_publication','buy_hold',10)]
        robustness[symbol]=bootstrap_excess(s['return'],b['return'])
    (output/'bootstrap.json').write_text(json.dumps(robustness,indent=2),encoding='utf-8')
    years=[]
    for name in ['buy_hold','standard','right','right_ma','ma20']:
        ledger=all_ledgers[('sh000300','post_publication',name,10)]
        for year,part in ledger.groupby(ledger.index.year):
            years.append({'year':year,'strategy':name,'return':(1+part['return']).prod()-1,'sessions':len(part)})
    pd.DataFrame(years).to_csv(output/'yearly.csv',index=False)

    supplementary_audit={}
    if supplemental:
        old=pd.read_csv(supplemental/'benchmark.csv');old['date']=pd.to_datetime(old.trade_date.astype(str));old=old.set_index('date')
        overlap=d[['open','close']].join(old[['open','close']],how='inner',rsuffix='_tushare')
        supplementary_audit['tencent_tushare_overlap']={'rows':len(overlap),'start':str(overlap.index.min().date()),'end':str(overlap.index.max().date()),
          'max_abs_open_difference':float((overlap.open-overlap.open_tushare).abs().max()),
          'max_abs_close_difference':float((overlap.close-overlap.close_tushare).abs().max())}
        etf=pd.read_csv(supplemental/'etf.csv');etf['date']=pd.to_datetime(etf.trade_date.astype(str));etf=etf.set_index('date').sort_index()
        for col in ['open','close']:etf[col]=etf[col]*etf.adj_factor/etf.adj_factor.iloc[-1]
        for name in ['buy_hold','right_ma']:
            signal=sigmap['sh000300'][name].reindex(etf.index)
            assert signal.notna().all()
            ledger=backtest(etf,signal,'2020-01-02','2026-09-18',10)
            ledger.to_csv(output/'ledgers'/f'ETF510330_{name}.csv',float_format='%.12g')
            supplementary_audit[f'etf_{name}']=metrics(ledger)
            supplementary_audit[f'etf_{name}_ledger_audit']=audit_ledger(ledger,10)
    (output/'supplementary.json').write_text(json.dumps(supplementary_audit,indent=2),encoding='utf-8')
    plot_results(table,all_ledgers,pd.DataFrame(sensitivity),output)
    evidence={'config':cfg,'data_manifest':manifest,'environment':{'python':platform.python_version(),'pandas':pd.__version__,
      'numpy':np.__version__,'matplotlib':matplotlib.__version__},'source_sha256':{},'analysis_status':'retrospective replication, not untouched live validation'}
    if supplemental:
        evidence['supplemental_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in supplemental.glob('*.csv')}
    for file in Path(__file__).parent.glob('*.py'):evidence['source_sha256'][file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
    (output/'run_manifest.json').write_text(json.dumps(evidence,indent=2,ensure_ascii=False),encoding='utf-8')
    print(table[(table.symbol=='sh000300')&(table.period=='post_publication')&(table.cost_bps==10)][['strategy','total_return','cagr','sharpe','max_drawdown','total_return_difference_pp']].to_string(index=False))


def plot_results(table,ledgers,sensitivity,output):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.18,'figure.facecolor':'white','savefig.bbox':'tight'})
    for period in ['historical_overlap','post_publication','recent']:
        fig,axes=plt.subplots(2,1,figsize=(10.5,6.4),sharex=True,gridspec_kw={'height_ratios':[2.4,1]})
        for name in ['buy_hold','standard','right','right_ma','ma20']:
            l=ledgers[('sh000300',period,name,10)];nav=l.equity
            axes[0].plot(l.index,nav,label=name,color=COLORS[name],lw=1.5)
            if name in ['buy_hold','right_ma']:
                peak=np.maximum.accumulate(np.r_[1.,nav.to_numpy()])[1:]
                axes[1].plot(l.index,100*(nav.to_numpy()/peak-1),color=COLORS[name],lw=1.3)
        axes[0].set(ylabel='Net asset value',title=f'CSI 300 | {period.replace("_"," ")} | 10 bp each side')
        axes[0].legend(ncol=3,loc='upper left',fontsize=9);axes[1].set(ylabel='Drawdown (%)')
        fig.tight_layout();fig.savefig(output/'figures'/f'nav_{period}.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(11,3.5),sharey=True)
    for ax,s in zip(axs,[.5,.7,.9]):
        p=sensitivity[sensitivity.S==s].pivot(index='N',columns='M',values='cagr')*100
        ax.imshow(p,vmin=sensitivity.cagr.min()*100,vmax=sensitivity.cagr.max()*100,cmap='YlGnBu')
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{p.iloc[i,j]:.1f}%',ha='center',va='center',color='black',fontsize=10)
        ax.set(xticks=range(3),xticklabels=p.columns,yticks=range(3),yticklabels=p.index,xlabel='M',title=f'S = {s}');ax.grid(False)
    axs[0].set_ylabel('N');fig.suptitle('Post-publication CAGR: all 27 preset neighbours (no winner selection)')
    fig.tight_layout();fig.savefig(output/'figures'/'sensitivity.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,3.7))
    t=table[(table.period=='post_publication')&(table.cost_bps==10)&table.strategy.isin(['buy_hold','right_ma'])]
    x=np.arange(3)
    for j,name in enumerate(['buy_hold','right_ma']):
        values=t[t.strategy==name].set_index('symbol').reindex(NAMES.keys())
        ax.bar(x+(j-.5)*.32,values.cagr*100,.32,label=name,color=COLORS[name])
    ax.set(xticks=x,xticklabels=list(NAMES.values()),ylabel='CAGR (%)',title='Cross-index post-publication check');ax.legend()
    fig.tight_layout();fig.savefig(output/'figures'/'cross_index.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'));p.add_argument('--supplemental',type=Path)
    a=p.parse_args();main(a.data,a.output,a.config,a.supplemental)
