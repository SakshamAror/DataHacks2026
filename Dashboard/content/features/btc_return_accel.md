## Expression

```
feat_btc_return_accel = feat_btc_return_30s - feat_btc_return_60s
```

Momentum acceleration: the difference between the short-horizon (30s) return and the
medium-horizon (60s) return.

---

## Motivation

**Momentum acceleration** captures whether the pace of price movement is speeding up
or slowing down:

**Positive value** (`return_30s > return_60s`): The most recent 30 seconds saw
a *higher* return rate than the overall 60-second period. This means the move is
*accelerating* — price has been rising faster recently than it did in the earlier part
of the 60-second window. Accelerating momentum tends to continue.

**Negative value** (`return_30s < return_60s`): The most recent 30 seconds saw
a *lower* return rate than the full 60-second period. The move is *decelerating* —
suggesting the trend is losing steam and may reverse.

**Near zero**: The return rate has been roughly constant over the 60-second window —
a stable, persistent trend.

**Economic intuition:** This is the discrete-time analogue of the *second derivative*
of price — measuring the rate of change of the rate of change. In options, this concept
maps to gamma; in trend-following, it maps to filter curvature. For 5-minute binary
contracts, detecting whether momentum is building or exhausting is directly actionable.
