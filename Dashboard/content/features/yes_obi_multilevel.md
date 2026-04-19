## Expression

```
λ = 10   # decay rate in probability units

bid_weight = Σ  exp(−λ · |bid_price − mid|) · bid_size     for all bid levels
ask_weight = Σ  exp(−λ · |ask_price − mid|) · ask_size     for all ask levels

feat_yes_obi_multilevel = (bid_weight − ask_weight) / (bid_weight + ask_weight)
```

Multi-level OBI: each level's volume is weighted by an exponential decay in price
distance from the mid, so near-money orders matter more than far-from-money orders.

---

## Motivation

The level-1 OBI only looks at the single best bid and ask. But in Polymarket's YES book,
there are typically multiple price levels, and the *distribution* of volume across them
carries additional information:

- If a large resting bid sits 3 cents below the mid, it acts as a **support level** —
  even if it does not affect the immediate next trade, it limits downward price movement.
- Conversely, a large ask wall above the mid suppresses further YES price rises.

The exponential decay weighting (`exp(−λ · distance)`) is a principled way to
down-weight far orders:
- A level 0.1 away from mid has weight `exp(−10 × 0.1) ≈ 0.37` of a level at mid.
- A level 0.3 away has weight `≈ 0.05` — essentially negligible.

**Why λ = 10?** At this decay rate, orders within ~0.15 probability units of mid dominate.
In a [0, 1]-bounded probability market, this covers the most economically relevant depth.
