## Expression

```
feat_yes_price = yes_mid  =  (yes_bid + yes_ask) / 2
```

The raw Polymarket YES token mid-price — the crowd's consensus estimate of P(BTC UP)
at this tick.

---

## Motivation

This is the **single most important baseline feature** in the model. It has the highest
absolute coefficient weight (`≈ 2.14`) by a wide margin.

The YES price directly encodes the aggregated belief of all Polymarket participants.
Prediction markets are known to be well-calibrated — when the YES price is 0.70,
the outcome is YES roughly 70% of the time. Including this as a feature therefore
gives the model a strong prior that it then *adjusts* using the other 20 signals.

**Why include it if it is already calibrated?**  
The market consensus is not perfect:
- It reacts to information with a lag.
- It is affected by noise traders and liquidity constraints.
- It does not fully account for the current BTC momentum or order book state.

The logistic model's job is to find the gap between the market's implied probability
and the true probability — and `feat_yes_price` anchors that estimate.

**Neutralization note:** Because this feature IS the market consensus, all other features
should be evaluated for their *incremental* contribution relative to `feat_yes_price`.
See the [Factor Library](/factor-library) neutralization analysis for details.
