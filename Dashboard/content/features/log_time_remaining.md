## Expression

```
feat_log_time_remaining = ln(max(seconds_remaining, 1))
```

Natural logarithm of the seconds remaining until settlement, floored at 1 to avoid `ln(0)`.

---

## Motivation

Time-to-expiry effects in options and prediction markets are inherently **non-linear**.
The linear feature `feat_time_remaining_frac` captures the proportional position in
the market lifecycle, but the *rate of change* of uncertainty is not constant:

- At 200 seconds remaining, losing 10 more seconds barely changes the odds.
- At 10 seconds remaining, losing the same 10 seconds is catastrophic — the outcome
  is almost certainly locked in.

The log transformation captures this **accelerating convergence** near expiry:
- `ln(200) ≈ 5.3` → `ln(190) ≈ 5.25`: small change (0.05 drop).
- `ln(10) ≈ 2.3` → `ln(1) = 0`: large change (full drop).

**Complementary to `feat_time_remaining_frac`:**  
Together, the two timing features allow the model to fit both the linear and
logarithmic components of time-decay:
- Linear: how far along we are proportionally.
- Logarithmic: the non-linear acceleration of convergence near expiry.

This is analogous to including both an option's theta and its gamma as features
in an options-pricing model.
