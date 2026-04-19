## Expression

```
returns_60s = [btc_mid[t−i+1] / btc_mid[t−i] − 1  for i in 1..60]

feat_btc_realized_vol_60s = std(returns_60s)
```

Standard deviation of 1-second BTC returns over the last 60 seconds — a short-horizon
realised volatility estimate.

---

## Motivation

Volatility affects prediction difficulty in two ways:

1. **Uncertainty about settlement**: High vol means BTC could move in either direction
   quickly before expiry. The logistic model's P(YES) should be closer to 0.5 in
   high-volatility regimes, and the entry threshold should be harder to cross.

2. **Calibration of momentum signals**: A 0.1% move during low-vol periods is
   significant; the same move during high-vol is noise. By including realised vol
   as an explicit feature, the model can learn to *discount* momentum signals when
   volatility is elevated.

**Connection to Kelly sizing**: The strategy uses half-Kelly position sizing. In
volatile markets, the true edge is less certain, and the model's P(YES) estimates are
less reliable — so both the feature and the Kelly fraction naturally reduce exposure.

**Why 60 seconds?** Shorter windows (e.g., 10s) are too noisy; longer windows
(e.g., 300s) include past volatility regimes that are no longer relevant. 60 seconds
balances responsiveness with statistical stability.
