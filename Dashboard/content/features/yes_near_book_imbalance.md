## Expression

```
near_bids = Σ bid_size   for levels where |bid_price − mid| ≤ 0.05
near_asks = Σ ask_size   for levels where |ask_price − mid| ≤ 0.05

feat_yes_near_book_imbalance = (near_bids − near_asks) / (near_bids + near_asks)
```

OBI restricted to order book levels within ±0.05 probability units of the YES mid-price.

---

## Motivation

Far-from-money orders in a prediction market are economically different from near-money
orders. An order to buy YES at $0.20 when the market is at $0.65 is unlikely to fill
within the remaining market window — it is essentially noise for short-horizon prediction.

By restricting to levels within 0.05 of mid, this feature focuses on the **actionable
near-money depth**: orders that will influence the YES price movement in the next few ticks.

**Complementary to `feat_yes_obi_multilevel`:**
- The multi-level OBI uses *soft* distance weighting (exponential decay).
- The near-book imbalance uses a *hard* cutoff (threshold filter).
- These two different aggregation approaches can be complementarily informative —
  the model can weight one over the other depending on what the data supports.

**Why 0.05?** In a [0,1]-bounded prediction market, a 0.05 window covers the
most competitive part of the book — roughly 5 cents on either side of mid.
This is wide enough to capture multiple price levels but tight enough to exclude
speculative far-from-money resting orders.
