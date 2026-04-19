## Expression

```
feat_btc_return_300s = (btc_mid[t] − btc_mid[t−300]) / btc_mid[t−300]
```

Fractional price change of BTC/USD over the last 300 seconds (5 minutes).

---

## Motivation

The prediction horizon for a 5-minute Polymarket contract is **exactly 5 minutes**.
The 300-second return therefore answers: *has BTC already moved the full distance it
needs to move, or is there still room?*

This feature encodes two distinct signals:

1. **Continuation**: If BTC has risen 0.3% in the past 5 minutes, the YES side is already
   in the money. A large positive return makes it more likely the market has already
   repriced and the contract will settle YES.

2. **Mean reversion**: Conversely, if the move is very large (±1%+), momentum may
   exhaust itself before settlement. The logistic model learns the appropriate non-linear
   response through its standardised coefficient.

**Design note:** Unlike the 30s and 60s features which are more about *recent impulse*,
the 300s feature measures the *full-window cumulative move* — directly matching the
economic quantity that determines settlement.
