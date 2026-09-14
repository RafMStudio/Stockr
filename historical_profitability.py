"""Reproducible historical evaluation; no strategy parameter fitting."""
import ast, json, pathlib, sys
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from backtest_engine import simulate_deal

tree = ast.parse(pathlib.Path('v12.py').read_text())
nodes = [n for n in tree.body if
         isinstance(n, ast.FunctionDef) and n.name == 'calculate_expert_strategy' or
         isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'TICKERS' for t in n.targets)]
exec(compile(ast.Module(body=nodes, type_ignores=[]), 'strategy', 'exec'))
out = pathlib.Path('historical_results'); out.mkdir(exist_ok=True)
frames, failed = {}, []
# Use the original universe, report exclusions, no replacement stocks.
for k in range(0,len(TICKERS),10):
    tickers=TICKERS[k:k+10]
    bulk=yf.download(tickers, period='700d', interval='1h', group_by='ticker',
                     auto_adjust=True, progress=False, threads=False)
    for t in tickers:
        try:
            df=bulk[t].dropna(subset=['Open','High','Low','Close']).copy()
            df=df[df.index + pd.Timedelta(hours=1) <= pd.Timestamp.now(tz='UTC')]
            if len(df)<400: raise ValueError('insufficient bars')
            df=calculate_expert_strategy(df)
            df['ATR']=ta.atr(df.High,df.Low,df.Close,length=14)
            frames[t]=df
            df.to_csv(out/(t+'.csv'))
        except Exception as e:
            failed.append({'ticker':t,'reason':str(e)})
if len(frames)<10:
    raise RuntimeError('Too few symbols with usable Yahoo data: '+json.dumps(failed))
times=sorted(set().union(*(set(d.index[200:]) for d in frames.values())))
split=times[int(len(times)*.7)]
signals=[]
for t,df in frames.items():
    for ts,row in df.iloc[200:].iterrows():
        side=1 if row.Buy_Signal else -1 if row.Sell_Signal else 0
        if side: signals.append((ts,t,side))
signals.sort()
# Freeze first-two-per-day House Bot selection. No retrospective scanner ranking.
selected=[]; daily={}; busy={}
for ts,t,side in signals:
    if t in busy and (busy[t] is None or ts<busy[t]): continue
    day=ts.date()
    if daily.get(day,0)>=2: continue
    rec=simulate_deal(frames[t],ts,side,.75,4)
    if rec is None: continue
    busy[t]=None if rec['OUTCOME'] in ('OPEN','PENDING') else rec['EXIT TIME']
    daily[day]=daily.get(day,0)+1
    selected.append((ts,t,side))
def evaluate(start,end,slip):
    trades=[]
    for ts,t,side in selected:
        if ts<start or ts>end: continue
        df=frames[t].loc[:end]
        rec=simulate_deal(df,ts,side,.75,4,slippage_bps=slip,fee_bps=1)
        if rec and rec['OUTCOME']!='PENDING':
            rec.update(TICKER=t,side=side); trades.append(rec)
    trades.sort(key=lambda x:(x['ENTRY TIME'],x['TICKER']))
    # Hypothetical $1,000 account: <=10% notional and <=0.5% planned stop risk
    # per position, <=100% gross exposure. Fractional shares; shorts fully collateralized.
    balance=1000.; active=[]; curve=[1000.]; accepted=[]; rejected=0
    schedule={}
    for tr in trades: schedule.setdefault(tr['ENTRY TIME'],[]).append(tr)
    period_times=[x for x in times if start<=x<=end]
    marks={}
    for t,df in frames.items():
        marks[t]=df.Close.reindex(pd.DatetimeIndex(period_times),method='ffill')
    for ts in period_times:
        remaining=[]
        for p in active:
            tr=p['trade']
            if pd.notna(tr['EXIT TIME']) and tr['EXIT TIME']<ts:
                balance+=p['notional']*tr['PNL %']/100
            else: remaining.append(p)
        active=remaining
        equity=balance+sum(p['qty']*p['trade']['side']*(float(marks[p['trade']['TICKER']].loc[ts])-p['trade']['ENTRY'])-p['notional']*.0001 for p in active)
        for tr in schedule.get(ts,[]):
            if any(p['trade']['TICKER']==tr['TICKER'] for p in active):
                rejected+=1;continue
            atr=float(frames[tr['TICKER']].loc[tr['SIGNAL TIME'],'ATR'])
            notional=max(0,min(equity*.1,equity*.005/(4*atr/tr['ENTRY'])))
            if sum(p['notional'] for p in active)+notional>equity or notional<=0:
                rejected+=1;continue
            p={'trade':tr,'notional':notional,'qty':notional/tr['ENTRY']}
            active.append(p);accepted.append(p)
        remaining=[]
        for p in active:
            tr=p['trade']
            if pd.notna(tr['EXIT TIME']) and tr['EXIT TIME']==ts:
                balance+=p['notional']*tr['PNL %']/100
            else: remaining.append(p)
        active=remaining
        equity=balance+sum(p['qty']*p['trade']['side']*(float(marks[p['trade']['TICKER']].loc[ts])-p['trade']['ENTRY'])-p['notional']*.0001 for p in active)
        curve.append(equity)
    # Liquidation estimate includes costs on all open positions.
    final=balance+sum(p['notional']*p['trade']['PNL %']/100 for p in active)
    curve.append(final)
    pnl=[p['notional']*p['trade']['PNL %']/100 for p in accepted]
    closed=[p for p in accepted if p['trade']['OUTCOME']!='OPEN']
    gross_loss=-sum(x for x in pnl if x<0)
    arr=np.array(curve); dd=(arr/np.maximum.accumulate(arr)-1)*100
    result={'from':str(start),'to':str(end),'slippage_bps':slip,
            'accepted':len(accepted),'rejected':rejected,'closed':len(closed),'open':len(active),
            'closed_win_rate':100*sum(p['trade']['PNL %']>0 for p in closed)/len(closed) if closed else None,
            'ending_equity':final,'return_pct':(final/1000-1)*100,
            'max_drawdown_pct':float(dd.min()),
            'profit_factor_including_open_marks':sum(x for x in pnl if x>0)/gross_loss if gross_loss else None}
    return result
results=[]
for name,start,end in [('full',times[0],times[-1]),('early_70pct',times[0],times[int(len(times)*.7)-1]),('later_30pct',split,times[-1])]:
    for slip in [5,15]:
        r=evaluate(start,end,slip);r['period']=name;results.append(r)
report={'universe_count':len(TICKERS),'loaded_count':len(frames),'failed':failed,
        'bars':sum(len(d) for d in frames.values()),'signals':len(signals),
        'house_selected':len(selected),'results':results,
        'limitations':['Current fixed universe, survivorship bias possible','Historical split is not genuinely unseen: prior strategy development may have used these dates',
        'Yahoo adjusted hourly bars; bar ordering unknown, conservative stop-first rule',
        'Fractional shares, no borrow fees/locate constraints, taxes or dividends separately',
        'Portfolio sizing is an explicit research assumption; original app has no funded account',
        'No scanner ranking backtest: current full-history ranking would leak future outcomes']}
(out/'report.json').write_text(json.dumps(report,indent=2))
print('HISTORICAL_REPORT_START\n'+json.dumps(report,indent=2)+'\nHISTORICAL_REPORT_END',flush=True)
