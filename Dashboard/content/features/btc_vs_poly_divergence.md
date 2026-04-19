## Expression

```
btc_signal  = tanh(500 · btc_return_60s)       # maps ±0.2% return → ±0.76
poly_signal = 2 · yes_price − 1                 # maps [0,1] price → [−1,+1]

feat_btc_vs_poly_divergence = poly_signal − btc_signal
```

Difference between the Polymarket-implied directional signal and the BTC momentum signal,
both mapped to `[−1, +1]`.

---

## Motivation

This is a **cross-market lead-lag signal** that exploits the latency between
BTC price moves on Binance and Polymarket's repricing:

**Positive divergence** (`poly > btc`): Polymarket is more bullish than the recent BTC
price action justifies. This suggests either:
- Smart money on Polymarket is pricing in a move that hasn't happened yet on Binance.
- The Polymarket consensus has not yet fully updated to a recent BTC downtick.

In either case, we expect the two signals to converge — either BTC catches up to poly
(BTC goes up → YES settles), or poly mean-reverts to BTC (YES price falls).

**Negative divergence** (`btc > poly`): BTC has moved up but poly hasn't repriced yet.
This is the classic **arbitrage signal**: if BTC is up 0.3% in the last minute but
YES is only at 0.55, the market may be under-reacting. Expected convergence: YES rises.

**Why tanh for BTC?** The tanh function softly saturates the BTC return at `±1`, so
very large BTC moves don't dominate the divergence calculation. The scale factor 500
ensures that a ±0.2% move (typical 60-second range) maps to roughly ±0.76.
