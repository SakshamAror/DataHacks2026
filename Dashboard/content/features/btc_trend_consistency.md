## Expression

```
signs = [sign(btc_mid[t-i+1] - btc_mid[t-i])  for i in 1..60]
# +1 if up-tick, -1 if down-tick, 0 if unchanged

feat_btc_trend_consistency = mean(signs)
```

Mean sign of 1-second BTC price changes over the last 60 seconds.
Range: `[−1, 1]`.

---

## Motivation

The short-horizon return features (`btc_return_30s`, `btc_return_60s`) capture the
*magnitude* of price movement. But two scenarios with the same 60-second return can
look very different:

1. **Consistent trend**: BTC ticked up 45 out of 60 seconds → `consistency ≈ +0.50`
2. **Volatile mean-reversion**: BTC alternated up/down every second but ended higher →
   `consistency ≈ 0.03`

The trend consistency metric distinguishes these cases. A high consistency value
indicates **persistent directional order flow** — many sequential buyers (or sellers)
rather than a noisy equilibrium. Persistent one-directional flow is more likely to
continue, making it a stronger signal for YES/NO settlement.

**Complementary to realized volatility:**
- High volatility + low consistency = chaotic, hard to predict.
- High volatility + high consistency = strong trending move, easier to call direction.
- Low volatility + high consistency = quiet grind in one direction.

This feature was added as an enhancement to the original feature set to improve
discrimination in low-volatility trending regimes.
