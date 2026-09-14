---
name: markov-2-hedge-fund-method
description: Markov 2.0 — Hedge Fund Method (corrected). Build regime transition matrices from stride-sampled (non-overlapping) return windows, self-verify state labels, choose FILTER or STANDALONE mode, and produce walk-forward regime signals for any ticker or strategy. Use when the user asks for regime analysis, transition matrices, Markov signals, or to gate a strategy by market regime.
---

# Markov 2.0 — Hedge Fund Method (corrected)

Same core as the original Markov hedge fund method (states → transition matrix
→ stickiness → signal), with three documented flaws fixed. A reference
implementation lives in this repo: the `🧮 Markov Regime` agent in `v12.py`
(engine functions `_markov_states`, `_verify_state_labels`,
`_transition_matrix`, `_markov_agent_signals`, `markov_lab`).

## The method

1. **States.** Default: 20-day cumulative return ≥ +5% = BULL, ≤ −5% = BEAR,
   else SIDEWAYS. Label the asset's full history. (On hourly bars, 20 trading
   days ≈ 140 bars.)
2. **Transition matrix.** Count state→state transitions, convert rows to
   probabilities (rows sum to 1). Report stickiness (the diagonal).
3. **Signal.** P(bull tomorrow) − P(bear tomorrow). Sign = direction,
   magnitude = conviction. Trade only beyond a conviction threshold (±0.15
   default).
4. **Multi-day forecasts** by matrix powers; note convergence to the
   stationary distribution — long-horizon forecasts carry no signal.
5. **Hidden Markov mode (optional):** fit an HMM (e.g. `hmmlearn`, not in
   requirements.txt by default) with no hand-made labels and report where it
   agrees/disagrees with the threshold labels — agreement is the green light.

## The three fixes (non-negotiable — this is what makes it 2.0)

**FIX 1 — Stride sampling (the autocorrelation flaw).** NEVER build the matrix
from overlapping rolling windows: consecutive 20-day windows share 19 days,
which fakes persistence on the diagonal. Count transitions between
NON-overlapping windows (stride = window length). Always compute BOTH matrices
— overlapping (legacy) and stride-sampled (true) — and show them side by side
with a one-line warning that only the stride-sampled one is statistically
honest.

**FIX 2 — Label verification.** After building any table, chart or matrix
display, SELF-CHECK the state labels programmatically: the window with the
highest return must be labeled BULL, the lowest BEAR, and every |return| < 5%
window SIDEWAYS. If the rendered display disagrees with the data, fix it
before showing the user — and refuse to trade/display on failed verification.
The original version shipped with bull/bear swapped in a display; never repeat
that class of bug.

**FIX 3 — Two explicit modes.** Ask the user which they want; never leave it
ambiguous:
- **FILTER mode** (default): the regime gates an existing strategy — longs
  only when signal > threshold, shorts only when below, flat in chop. The
  user's strategy stays theirs; Markov 2.0 decides WHEN it is allowed to act.
- **STANDALONE mode**: trade the differential directly, position size scaled
  to |signal| with a user-set cap. (The dashboard's `🧮 Markov Regime` agent
  runs standalone.)

## Optional richer states (offer, don't force)

Price-only states are the default. Offer "enhanced states": cluster on 20-day
return + ATR (volatility) + relative volume, so "bear and violent" ≠ "bear and
asleep". If chosen, report how the matrix and signal change vs price-only.

## Proof, not promises

Backtests must be walk-forward — never test on data the matrix has learned
from; recalculate as you walk. Report honestly: win rate, profit factor, max
drawdown, equity curve, and a before-fix vs after-fix comparison (the
overlapping matrix's inflated diagonal vs the stride-sampled truth). Then say
exactly this caveat: "Backtests flatter. The fixed matrix shows uglier, truer
numbers — those are the only ones worth trading." Offer to re-run on any
ticker the user names.
