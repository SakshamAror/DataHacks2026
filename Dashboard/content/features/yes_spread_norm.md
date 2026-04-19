## Expression

```
feat_yes_spread_norm = (yes_ask − yes_bid) / yes_mid
```

YES bid-ask spread on Polymarket divided by the YES mid-price — the relative spread.

---

## Motivation

The relative spread on the Polymarket YES side is a measure of how **uncertain or
illiquid** the prediction market is at this tick. Wide spreads have two implications:

1. **Transaction cost warning**: A wide spread means we would buy YES at `yes_ask`
   (above fair value) and could only sell at `yes_bid` (below fair value). High spreads
   reduce the expected profit of any position.

2. **Information signal**: Market makers in prediction markets widen their quotes when
   they are uncertain about the true probability — often because new information has
   arrived or is expected. A wide spread can be a leading indicator that the market
   is about to reprice.

**Contrast with BTC spread (`feat_btc_spread_norm`):**
- The Binance spread reflects uncertainty on the underlying asset (BTC price).
- The YES spread reflects uncertainty on the *prediction market contract* itself —
  which can diverge from the underlying due to liquidity fragmentation, smart-money
  positioning, or imminent settlement.

In practice, a very wide YES spread (e.g., 10+ cents) means the market is thin and
any model prediction should be treated with extra skepticism.
