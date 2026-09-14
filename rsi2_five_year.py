"""Fixed RSI2 experiment. No parameter search. All dates and rules set before run."""
import json, pathlib, math
import numpy as np
import pandas as pd
import yfinance as yf
TICKERS=['SPY','QQQ','IWM','DIA']
START='2021-09-14'; END='2026-09-11'
OUT=pathlib.Path('rsi2_results'); OUT.mkdir(exist_ok=True)

def rma(x,n):
    # Wilder smoothing seeded with the mean of the first n non-missing values.
    vals=x.to_numpy(dtype=float); result=np.full(len(vals),np.nan)
    good=np.flatnonzero(np.isfinite(vals))
    if len(good)>=n:
        j=good[n-1]; result[j]=np.mean(vals[good[:n]])
        for k in range(j+1,len(vals)):
            result[k]=(result[k-1]*(n-1)+vals[k])/n
    return pd.Series(result,index=x.index)

def indicators(df):
    d=df.copy(); delta=d.Close.diff()
    gain=rma(delta.clip(lower=0),2); loss=rma(-delta.clip(upper=0),2)
    d['rsi']=100-100/(1+gain/loss)
    d.loc[(loss==0)&(gain>0),'rsi']=100
    d.loc[(loss==0)&(gain==0),'rsi']=50
    tr=pd.concat([d.High-d.Low,(d.High-d.Close.shift()).abs(),(d.Low-d.Close.shift()).abs()],axis=1).max(axis=1)
    d['atr']=rma(tr,14);d['ma200']=d.Close.rolling(200).mean();d['ma5']=d.Close.rolling(5).mean()
    d['signal']=(d.Close>d.ma200)&(d.rsi<5)&(d.Close<d.Close.shift())&(d.Close.shift()<d.Close.shift(2))
    return d

def stop_fill(op,low,stop):
    return op if op<=stop else stop if low<=stop else None

def stats(trades):
    p=np.array([x['pnl'] for x in trades],dtype=float)
    pct=np.array([x['return_pct'] for x in trades],dtype=float)
    wins=p>0; losses=p<0
    return {'trades':len(p),'win_rate':float(wins.mean()*100) if len(p) else None,
        'profit_factor':float(p[wins].sum()/-p[losses].sum()) if losses.any() else None,
        'average_win_pct':float(pct[wins].mean()) if wins.any() else None,
        'average_loss_pct':float(pct[losses].mean()) if losses.any() else None,
        'net_pnl':float(p.sum())}

def run(frames,slippage_bps):
    slip=slippage_bps/10000; fee=.0001
    dates=frames['SPY'].loc[START:END].index
    assert all(dates.equals(f.loc[START:END].index) for f in frames.values()),'ETF calendars differ'
    cash=1000.; positions={}; trades=[]; curve=[]; signals=0
    def close(t,date,raw,reason):
        nonlocal cash
        p=positions.pop(t);price=raw*(1-slip)
        proceeds=p['qty']*price*(1-fee);cash+=proceeds
        pnl=proceeds-p['cost']
        trades.append({'ticker':t,'entry_date':str(p['date'].date()),'exit_date':str(date.date()),
            'entry':p['entry'],'exit':price,'qty':p['qty'],'pnl':pnl,
            'return_pct':100*pnl/p['cost'],'reason':reason,'sessions':p['held']})
    for date in dates:
        rows={t:f.loc[date] for t,f in frames.items()}
        prev={t:f.iloc[f.index.get_loc(date)-1] for t,f in frames.items()}
        # Previously known exits execute at the open, before new orders.
        for t in list(positions):
            p=positions[t]; op=float(rows[t].Open)
            if op<=p['stop']:close(t,date,op,'gap_stop')
            elif p['held']>=10:close(t,date,op,'time')
            elif prev[t].Close>prev[t].ma5:close(t,date,op,'recovery')
        equity_open=cash+sum(p['qty']*rows[t].Open for t,p in positions.items())
        for t in sorted(TICKERS):
            if t in positions or not bool(prev[t].signal):continue
            signals+=1
            atr=float(prev[t].atr);entry=float(rows[t].Open)*(1+slip)
            if not math.isfinite(atr) or atr<=0:continue
            qty=min(equity_open*.005/(2*atr),equity_open*.20/entry,cash/(entry*(1+fee)))
            if qty<=0:continue
            cost=qty*entry*(1+fee);cash-=cost
            positions[t]={'qty':qty,'entry':entry,'cost':cost,'stop':entry-2*atr,'held':0,'date':date}
        for t in list(positions):
            positions[t]['held']+=1
            raw=stop_fill(float(rows[t].Open),float(rows[t].Low),positions[t]['stop'])
            if raw is not None:close(t,date,raw,'stop')
        if date==dates[-1]:
            for t in list(positions):close(t,date,float(rows[t].Close),'end_of_test')
        equity=cash+sum(p['qty']*rows[t].Close for t,p in positions.items())
        assert cash>=-1e-7 and equity>0
        curve.append({'date':date,'equity':float(equity)})
    eq=pd.DataFrame(curve).set_index('date').equity
    peak=np.maximum.accumulate(np.r_[1000.,eq.values])
    dd=(np.r_[1000.,eq.values]/peak-1)*100
    result=stats(trades)
    result.update(slippage_bps=slippage_bps,ending_equity=float(eq.iloc[-1]),
        return_pct=float((eq.iloc[-1]/1000-1)*100),max_drawdown_pct=float(dd.min()),entry_opportunities=signals)
    yearly=[];base=1000.
    for year,g in eq.groupby(eq.index.year):
        s=stats([x for x in trades if x['exit_date'].startswith(str(year))])
        vals=np.r_[base,g.values];yd=(vals/np.maximum.accumulate(vals)-1)*100
        s.update(year=int(year),return_pct=float((g.iloc[-1]/base-1)*100),max_drawdown_pct=float(yd.min()))
        yearly.append(s);base=float(g.iloc[-1])
    result['yearly']=yearly
    result['per_etf']={t:stats([x for x in trades if x['ticker']==t]) for t in TICKERS}
    pd.DataFrame(trades).to_csv(OUT/f'trades_{slippage_bps}bps.csv',index=False)
    eq.to_csv(OUT/f'equity_{slippage_bps}bps.csv')
    # Equal-weight buy-and-hold, fully invested, same entry/exit costs.
    bh=pd.Series(0.,index=dates)
    for t,f in frames.items():
        q=250/(float(f.loc[dates[0],'Open'])*(1+slip)*(1+fee))
        bh+=q*f.loc[dates,'Close']
    bh.iloc[-1]*=(1-slip)*(1-fee)
    vals=np.r_[1000.,bh.values]
    result['buy_hold_equal_weight']={'return_pct':float((bh.iloc[-1]/1000-1)*100),
        'ending_equity':float(bh.iloc[-1]),'max_drawdown_pct':float(((vals/np.maximum.accumulate(vals)-1)*100).min())}
    return result

def checks():
    assert stop_fill(90,89,96)==90
    assert stop_fill(100,95,96)==96
    assert stop_fill(100,97,96) is None
    x=pd.Series([np.nan,1.,3.,5.]);assert rma(x,2).iloc[2]==2 and rma(x,2).iloc[3]==3.5
    test=stats([{'pnl':1,'return_pct':1},{'pnl':-2,'return_pct':-2}])
    assert test['win_rate']==50 and test['profit_factor']==.5
    print('Deterministic calculation checks passed',flush=True)

if __name__=='__main__':
    checks()
    bulk=yf.download(TICKERS,start='2020-01-01',end='2026-09-12',interval='1d',
        auto_adjust=True,group_by='ticker',progress=False,threads=False)
    frames={}
    for t in TICKERS:
        f=bulk[t].dropna(subset=['Open','High','Low','Close']).copy()
        f.index=pd.to_datetime(f.index).tz_localize(None)
        assert f.index.is_unique and f.index.is_monotonic_increasing
        assert str(f.loc[START:END].index[-1].date())==END,'Incomplete coverage '+t
        assert len(f.loc[:START])>200,'Insufficient warmup'
        frames[t]=indicators(f)
        frames[t].to_csv(OUT/(t+'_inputs.csv'))
    report={'start':START,'end':END,'daily_bars':{t:len(f.loc[START:END]) for t,f in frames.items()},
        'rules':'RSI2<5, above SMA200, two down closes; next-open long; 2 ATR stop; SMA5 recovery next open; ten sessions then next open',
        'assumptions':'1000 USD, fractional shares, 0.5% stop risk, max20% per ETF, no leverage, 1bp fee per side, idle cash earns zero. Adjusted OHLC dividend/split proxy.',
        'results':[run(frames,5),run(frames,15)],
        'limitations':['Fixed four-ETF selection; no optimization','2021 and 2026 calendar-year results are partial',
        'End-of-test positions liquidated at final close with costs','Daily closes understate possible intraday drawdown',
        'Historical evaluation is not a guarantee or genuinely prospective validation','Buy-and-hold is fully invested; strategy often holds substantial cash']}
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('RSI_REPORT_START\n'+json.dumps(report,indent=2,allow_nan=False)+'\nRSI_REPORT_END',flush=True)
