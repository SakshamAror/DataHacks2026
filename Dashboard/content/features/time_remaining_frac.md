## Expression

```
feat_time_remaining_frac = seconds_remaining / total_duration_seconds
```

Fraction of the market's total duration that remains.
- `1.0` = market just opened.
- `0.5` = halfway through.
- `0.0` = settlement imminent.

For 5-minute markets: `total_duration_seconds = 300`.

---

## Motivation

The time remaining until settlement fundamentally changes the dynamics of a binary
prediction market:

**Early (frac ≈ 0.9):** The YES price is highly uncertain — it reflects the market's
prior belief before much information has been revealed. Predictions made here have
a long runway for BTC to move in either direction.

**Middle (frac ≈ 0.5):** The most informative window. Enough time has passed for
BTC to demonstrate direction, but enough time remains that the outcome isn't locked in.
The strategy only enters when `0.15 < frac < 0.85`.

**Late (frac ≈ 0.1):** The YES price is rapidly converging to 0 or 1. Small BTC moves
have outsized effects on the probability. Trading here is risky due to the convexity
of the payout function near settlement.

**Why include this as a feature?** The model needs to know *where in the lifecycle*
it is operating. The information content of all other features changes depending on
time remaining — a 0.3% BTC return with 30 seconds left is very different from
the same return with 200 seconds left.
