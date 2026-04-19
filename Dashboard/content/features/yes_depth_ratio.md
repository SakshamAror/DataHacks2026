## Expression

```
total_bid_size = Σ size  for all bid levels
total_ask_size = Σ size  for all ask levels

feat_yes_depth_ratio = log(total_bid_size / total_ask_size)
```

Natural log of the ratio of total YES bid depth to total YES ask depth across
all visible order book levels.

---

## Motivation

While level-1 OBI captures *immediate* pressure at the best bid/ask, this feature
captures the **aggregate inventory imbalance** across the full order book.

A market where bids dominate the entire book depth — not just the top level — suggests
broad, persistent buying interest rather than a single large order creating a temporary
imbalance. This is harder for a single participant to fake and therefore a more reliable
signal of genuine directional bias.

**Why log?** Without the log transformation, the distribution of this ratio is highly
right-skewed (it can range from near-zero to hundreds). The log transformation:
1. Makes the feature roughly Gaussian-distributed, which logistic regression handles better.
2. Makes it symmetric: `log(2x / x) = log(2)`, `log(x / 2x) = −log(2)` — equal-magnitude
   signals for equal-but-opposite imbalances.
3. Penalises extreme values less harshly, preventing single outlier ticks from distorting
   the model.

**Expected range:** Typically `[−2, 2]`, centred near 0 in balanced markets.
