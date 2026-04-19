## Expression

```
feat_yes_price_mom_30s = yes_mid[t] − yes_mid[t−30]
```

Change in the Polymarket YES mid-price over the last 30 seconds (in probability units).

---

## Motivation

While `feat_yes_price` captures the *level* of market consensus, this feature captures
**how quickly that consensus is changing** — the momentum of smart money.

When the YES price has been rising over the past 30 seconds, it means market participants
have been net-buying YES tokens. This could indicate:
- New information arriving that makes a YES outcome more likely.
- Informed traders front-running a BTC move that has not yet fully materialised.
- Order flow imbalance that will persist for a few more ticks.

**Why this matters for prediction:** Polymarket participants include professional traders
who monitor BTC order flow closely. A sustained YES price increase over 30 seconds
suggests these participants are accumulating ahead of an expected upward move —
providing a leading signal that the exchange price does not yet fully reflect.

**Scale:** The feature is in probability units (e.g., +0.02 means the YES price rose
2 cents over 30 seconds). After StandardScaler normalisation, the model interprets
this in terms of standard deviations from the historical mean.
