"""Predeclared 54-candidate search; final period evaluated once for selected model."""
import ast,json,pathlib
import numpy as np
import pandas as pd
import yfinance as yf
TICKERS=['SPY','QQQ','IWM','DIA','XLK','XLF','XLE','XLV','XLI','XLP','XLY','XLU','XLB','XLRE','SMH','AAPL','MSFT','AMZN','GOOGL','META','NVDA','AMD','TSLA','COST','JPM']
OUT=pathlib.Path('candidate_results');OUT.mkdir(exist_ok=True)
tree=ast.parse(pathlib.Path('rsi2_five_year.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['rma','indicators']],type_ignores=[]),'indicators','exec'))
bulk=yf.download(TICKERS,start='2020-01-01',end='2026-09-12',auto_adjust=True,group_by='ticker',threads=False,progress=False)
frames={}
for t in TICKERS:
    f=bulk[t].dropna(subset=['Open','High','Low','Close']).copy()
    f.index=pd.to_datetime(f.index).tz_localize(None)
    if len(f)<1200:raise RuntimeError('Missing data '+t)
    f=indicators(f)
    f['ibs']=(f.Close-f.Low)/(f.High-f.Low).replace(0,np.nan)
    f['ma20']=f.Close.rolling(20).mean()
    frames[t]=f
    f.to_csv(OUT/(t+'.csv'))
def signals(f,kind):
    # Original candidate: buy weak closes after a pullback in a rising intermediate trend.
    if kind=='recovery':
        return (f.Close<f.ma20)&(f.ma20>f.ma20.shift(5))&(f.ibs<.25)&(f.rsi<30)
    if kind=='trend_dip':
        return (f.Close>f.ma200)&(f.rsi<20)&(f.ibs<.4)
    return (f.Close>f.ma200)&(f.Close<f.Close.shift())&(f.ibs<.15)
def simulate(config,start,end,slip=.0005,save=False):
    trades=[]
    for t,f in frames.items():
        sig=signals(f,config['entry']).to_numpy()
        o,h,l,c,a=(f[k].to_numpy() for k in ['Open','High','Low','Close','atr'])
        dates=f.index;busy=-1
        for i in range(200,len(f)-1):
            if dates[i]<pd.Timestamp(start) or dates[i]>=pd.Timestamp(end) or i<busy or not sig[i]:continue
            j=i+1
            if dates[j]>pd.Timestamp(end):continue
            entry=o[j]*(1+slip);stop=entry-config['stop']*a[i];target=entry+config['target']*a[i]
            last=min(j+config['hold']-1,dates.searchsorted(pd.Timestamp(end),side='right')-1)
            if last<j:continue
            raw=None;reason='time'
            for k in range(j,last+1):
                if o[k]<=stop:raw=o[k]*(1-slip);reason='gap';break
                if o[k]>=target:raw=target;reason='target';break
                if l[k]<=stop:raw=stop*(1-slip);reason='stop';break
                if h[k]>=target:raw=target;reason='target';break
            if raw is None:raw=c[last]*(1-slip);k=last
            net=(raw-entry-.0001*(entry+raw))/entry
            trades.append({'ticker':t,'entry_date':str(dates[j].date()),'exit_date':str(dates[k].date()),'return_pct':100*net,'reason':reason})
            busy=k
    p=np.array([x['return_pct'] for x in trades])
    loss=-p[p<0].sum()
    result={'trades':len(p),'win_rate':float((p>0).mean()*100) if len(p) else 0,
      'profit_factor':float(p[p>0].sum()/loss) if loss else None,
      'mean_net_trade_pct':float(p.mean()) if len(p) else 0}
    if save:
        pd.DataFrame(trades).to_csv(OUT/f"final_trades_{slip}.csv",index=False)
        result['yearly']={}
        for y in ['2025','2026']:
            q=np.array([x['return_pct'] for x in trades if x['exit_date'].startswith(y)])
            result['yearly'][y]={'trades':len(q),'win_rate':float((q>0).mean()*100) if len(q) else None,'mean_net_trade_pct':float(q.mean()) if len(q) else None}
    return result
candidates=[]
for entry in ['recovery','trend_dip','weak_close']:
 for target in [.5,1.,1.5]:
  for stop in [1.,2.,3.]:
   for hold in [5,10]:
    config=dict(entry=entry,target=target,stop=stop,hold=hold)
    train=simulate(config,'2021-09-14','2023-12-29')
    validation=simulate(config,'2024-01-01','2024-12-31')
    candidates.append(dict(config=config,training=train,validation=validation))
    print('CANDIDATE',json.dumps(candidates[-1]),flush=True)
eligible=[x for x in candidates if x['training']['trades']>=200 and x['validation']['trades']>=80 and
          (x['training']['profit_factor'] or 0)>=1.2 and (x['validation']['profit_factor'] or 0)>=1.2 and
          x['training']['win_rate']>=80 and x['validation']['win_rate']>=80]
# If the requested bar is missed, select a diagnostic fallback without claiming success.
pool=eligible or [x for x in candidates if x['training']['trades']>=200 and x['validation']['trades']>=80]
if not pool:pool=candidates
selected=max(pool,key=lambda x:(min(x['training']['profit_factor'] or 0,x['validation']['profit_factor'] or 0),
                               x['validation']['mean_net_trade_pct']))
final=simulate(selected['config'],'2025-01-01','2026-09-11',save=True)
stress=simulate(selected['config'],'2025-01-01','2026-09-11',slip=.0015,save=True)
report={'candidate_count':len(candidates),'eligible_count':len(eligible),'selected':selected,'final_2025_2026':final,'final_higher_costs':stress,
 'meets_final_target':bool(eligible and final['win_rate']>=80 and (final['profit_factor'] or 0)>=1.2),
 'limitations':['Final period was excluded from candidate scoring, but previously viewed in other experiments; not truly untouched',
 'Fixed present-day 25-instrument universe, selection bias possible','Per-trade equal-notional statistics, NOT account returns; simultaneous correlated positions possible',
 'Fees 1bp each side, adverse market slippage 5bp baseline/15bp stress; targets modeled as limit fills',
 'No parameter changes after final evaluation; strategy is an original combination, not a claim of novelty',
 'Five-year window split into training, validation and final; incomplete 2021/2026','No same-stock overlaps; no portfolio capital cap']}
(OUT/'candidates.json').write_text(json.dumps(candidates,indent=2))
(OUT/'report.json').write_text(json.dumps(report,indent=2))
print('SEARCH_REPORT_START\n'+json.dumps(report,indent=2)+'\nSEARCH_REPORT_END',flush=True)
