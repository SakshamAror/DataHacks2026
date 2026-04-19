## Expression

```
curr_spread = yes_ask[t] − yes_bid[t]
prev_spread = yes_ask[t−30] − yes_bid[t−30]

feat_yes_spread_delta_30s = (curr_spread − prev_spread) / yes_mid[t]
```

Change in the Polymarket YES spread over 30 seconds, normalised by the current mid.

---

## Motivation

While `feat_yes_spread_norm` measures the current level of market uncertainty,
this feature captures whether that uncertainty is **increasing or decreasing**.

**Positive value** (spread widening) → market makers are pulling back quotes, becoming
more cautious. This often precedes a price move as informed order flow arrives.
Potential action: delay entry until the spread stabilises.

**Negative value** (spread tightening) → market makers are gaining confidence, adding
depth. This often follows a period of uncertainty resolution. The price is likely to
stabilise or consolidate.

**Interaction with `feat_yes_spread_norm`:**
- High level + widening → danger zone: thin market getting thinner.
- Low level + tightening → ideal entry conditions: liquid and improving.
- High level + tightening → recovering from a spike in uncertainty.
- Low level + widening → early warning of an upcoming disruption.

The 30-second window was chosen to match the other 30-second momentum and spread
features, keeping the feature set temporally consistent.
