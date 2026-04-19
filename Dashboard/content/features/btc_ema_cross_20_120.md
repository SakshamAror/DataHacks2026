## Expression

```
ema20  = EMA(btc_mids, span=20)    # ~20-second exponential moving average
ema120 = EMA(btc_mids, span=120)   # ~2-minute exponential moving average

feat_btc_ema_cross_20_120 = (ema20 − ema120) / ema120
```

EMA crossover ratio: normalised distance between the short-term and long-term
exponential moving averages of BTC/USD mid-price.

**Positive** → short-term trend is above long-term (bullish).  
**Negative** → short-term trend is below long-term (bearish).

---

## Motivation

Simple point-to-point returns (30s, 60s) are noisy because a single anomalous tick
can dominate the value. EMAs smooth the price series by weighting recent prices more
heavily, giving a cleaner picture of the trend.

The **crossover** between a fast EMA (20s) and a slow EMA (120s) is one of the
most widely used technical signals in algorithmic trading:
- When the fast line crosses above the slow line → recent acceleration upward.
- The ratio normalises by the long-term level, making the feature scale-invariant
  across different BTC price levels.

**Why these spans?**
- 20 seconds captures intra-minute momentum — fast enough to react to news-driven moves.
- 120 seconds captures the 2-minute trend — slow enough to filter out short-term reversals.
- Together they detect *early trend formation* within the 5-minute market window.
