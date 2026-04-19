## Expression

```
feat_btc_spread_norm = (btc_ask − btc_bid) / btc_mid
```

Relative bid-ask spread of BTC/USDT on Binance — the spread normalised by the mid-price.

---

## Motivation

The bid-ask spread on a spot exchange is a real-time signal of **market maker uncertainty**.
When informed traders arrive with directional flow, market makers widen their quotes to
protect against adverse selection. A widening spread therefore often *precedes* a
directional price move.

**Why normalise by mid?** Raw spreads in dollar terms grow with the price level.
At BTC = $50,000 a $10 spread is the same *percentage* cost as a $5 spread at $25,000.
Normalising makes the feature comparable across different price regimes.

**Expected model behaviour:**
- Wide spread → higher uncertainty → the model should be more conservative.
- Narrow spread → tight market → market makers confident, prices well-anchored.

This feature primarily acts as a **regime filter**: in thin-spread, liquid conditions,
momentum and OBI signals are more reliable; in wide-spread conditions, they are noisier.
