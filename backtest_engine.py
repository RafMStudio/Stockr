"""Shared bar-based research simulator. No brokerage or portfolio return claims."""
import math
import pandas as pd

def simulate_deal(df, ts, side, target_mult, stop_mult, max_hold=35,
                  slippage_bps=5.0, fee_bps=1.0):
    i = df.index.get_loc(ts)
    atr = float(df['ATR'].iloc[i])
    if not math.isfinite(atr) or atr <= 0:
        return None
    if side not in (-1, 1) or min(target_mult, stop_mult, max_hold) <= 0:
        raise ValueError("Invalid side or exit parameters")
    if slippage_bps < 0 or fee_bps < 0:
        raise ValueError("Costs must be non-negative")
    # A completed signal is visible immediately, but has no invented fill.
    if i + 1 >= len(df):
        return {'SIGNAL TIME': ts, 'ENTRY TIME': ts, 'SIDE': 'LONG' if side == 1 else 'SHORT',
                'ENTRY': float('nan'), 'EXIT TIME': pd.NaT, 'EXIT': float('nan'),
                'OUTCOME': 'PENDING', 'PNL %': float('nan'), 'R': float('nan')}
    start = i + 1
    slip = slippage_bps / 10000
    entry = float(df['Open'].iloc[start]) * (1 + side * slip)
    if not math.isfinite(entry) or entry <= 0:
        return None
    target = entry + side * target_mult * atr
    stop = entry - side * stop_mult * atr
    end = min(len(df), start + max_hold)
    outcome = None
    for j in range(start, end):
        op, hi, lo = (float(df[c].iloc[j]) for c in ('Open', 'High', 'Low'))
        # Gaps execute at the available open, never at a skipped stop.
        if side * (op - stop) <= 0:
            raw, outcome = op, 'LOSS'
        elif side * (op - target) >= 0:
            raw, outcome = target, 'WIN'  # conservative limit fill
        elif (lo <= stop if side == 1 else hi >= stop):
            raw, outcome = stop, 'LOSS'
        elif (hi >= target if side == 1 else lo <= target):
            raw, outcome = target, 'WIN'
        if outcome is not None:
            break
    if outcome is None:
        j = end - 1
        raw = float(df['Close'].iloc[j])
        outcome = 'OPEN' if end - start < max_hold else 'TIME'
    # Target is a limit; market exits incur adverse slippage.
    price = raw if outcome == 'WIN' else raw * (1 - side * slip)
    fees = fee_bps / 10000 * (entry + price)
    net = side * (price - entry) - fees
    if outcome == 'TIME':
        outcome = 'TIME-WIN' if net > 0 else 'TIME-LOSS'
    return {'SIGNAL TIME': ts, 'SIDE': 'LONG' if side == 1 else 'SHORT',
            'ENTRY TIME': df.index[start], 'ENTRY': entry,
            'EXIT TIME': pd.NaT if outcome == 'OPEN' else df.index[j], 'EXIT': price,
            'OUTCOME': outcome, 'PNL %': net / entry * 100,
            'R': net / (stop_mult * atr)}

def signal_history(df, target_mult, stop_mult, max_hold=35):
    """One active position per ticker; pending signals have no P&L."""
    records = []
    blocked_until = None
    for ts, row in df.iterrows():
        if blocked_until is not None and ts < blocked_until:
            continue
        side = 1 if row['Buy_Signal'] else (-1 if row['Sell_Signal'] else 0)
        if not side:
            continue
        rec = simulate_deal(df, ts, side, target_mult, stop_mult, max_hold)
        if rec is None:
            continue
        records.append(rec)
        if rec['OUTCOME'] in ('OPEN', 'PENDING'):
            break
        blocked_until = rec['EXIT TIME']
    return records
