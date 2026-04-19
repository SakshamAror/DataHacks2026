## Expression

```
microprice = (best_ask · best_bid_size + best_bid · best_ask_size) / (best_bid_size + best_ask_size)

mid = (best_bid + best_ask) / 2

feat_yes_microprice_dev = (microprice − mid) / mid
```

Signed normalised deviation of the YES microprice from the quoted mid-price.

---

## Motivation

The **microprice** is the volume-weighted average of the best bid and best ask, using
the *opposing* side's size as the weight. This makes intuitive sense:

- If the bid has 800 shares and the ask has 200 shares, the next trade is much more
  likely to be a sell order (hitting the bid) than a buy order (lifting the ask).
  The microprice reflects this by weighting the bid price more heavily: `microprice < mid`.

Wait — that seems backwards. Let's re-examine:
- Heavy bid size (lots of YES buyers resting) → the ask side is being "used up" faster →
  the microprice is pulled toward the ask price → `microprice > mid` → **bullish signal**.

**Positive deviation** → bid has more volume → price is likely to be pushed upward.  
**Negative deviation** → ask has more volume → price is likely to be pushed downward.

**Why not just use OBI?** The microprice expresses the same information as OBI level-1
but as a *price level* rather than a volume ratio. This creates a feature that is
decorrelated from `feat_yes_obi_level1` in the standardised feature space, providing
the model with an additional independent view of the same underlying signal.
