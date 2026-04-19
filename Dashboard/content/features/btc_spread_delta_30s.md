## Expression

```
feat_btc_spread_delta_30s = (spread[t] − spread[t−30]) / btc_mid[t]
```

Change in the Binance BTC bid-ask spread over the past 30 seconds, normalised by
the current mid-price.

---

## Motivation

While `btc_spread_norm` captures the *level* of market uncertainty, this feature captures
the *change* — whether conditions are **improving** (tightening) or **deteriorating** (widening).

**Positive value** → spread is widening → uncertainty is increasing. This often occurs:
- When an economic announcement is imminent.
- When a large market order just hit one side of the book.
- When BTC is about to make a directional move (front-running behaviour).

**Negative value** → spread is tightening → liquidity is returning after a disruption.

**Why 30 seconds?** Spread changes on spot crypto exchanges are highly autocorrelated
at short horizons. A 30-second delta is long enough to capture a meaningful regime shift
but short enough to be actionable within a 5-minute contract window.

**Complementary role:** This feature works in combination with `btc_spread_norm`.
A wide spread that is continuing to widen (`btc_spread_norm` high, `btc_spread_delta_30s`
positive) is a stronger warning sign than either feature alone.
