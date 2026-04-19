## Expression

```
feat_yes_price_mom_60s = yes_mid[t] − yes_mid[t−60]
```

Change in the Polymarket YES mid-price over the last 60 seconds (in probability units).

---

## Motivation

This feature serves the same purpose as `feat_yes_price_mom_30s` but at a longer horizon,
capturing **sustained directional consensus** rather than the most recent impulse.

A YES price that has been rising for a full minute represents a stronger, more persistent
signal than one that has only risen in the last 30 seconds. Short-term momentum can be
caused by a single large order; 60-second momentum is more likely to represent genuine
information diffusion across multiple market participants.

**Interaction with the 30-second version:**
- **Both positive**: strong, consistent YES buying pressure across timescales.
- **30s positive, 60s negative**: recent reversal after a prior downtrend — weaker signal.
- **30s negative, 60s positive**: short-term pullback within a broader uptrend — moderate signal.

The model learns these interactions implicitly through the joint coefficient structure,
allowing it to distinguish between these scenarios during inference.
