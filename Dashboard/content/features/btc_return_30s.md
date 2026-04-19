## Expression

```
feat_btc_return_30s = (btc_mid[t] − btc_mid[t−30]) / btc_mid[t−30]
```

Fractional price change of the BTC/USD mid-price on Binance over the last 30 seconds.
Falls back gracefully to a shorter window if fewer than 30 ticks of history are available.

---

## Motivation

A 5-minute Polymarket contract asks whether BTC will be **higher or lower** at settlement
than at market open. If BTC has risen sharply in the last 30 seconds, momentum effects
suggest the move may continue — at least in the short term — making a YES outcome more likely.

A 30-second window is short enough to capture *recent* directional impulse without
over-averaging past periods where the price was moving in the opposite direction.

**Why 30 seconds specifically?**  
- Long enough to smooth out single-tick noise in the Binance order book.
- Short enough to react quickly when momentum reverses.
- Complements the 60-second and 300-second return features, which capture medium and full-horizon momentum.

The feature is standardised before model fitting, so the coefficient captures the
**direction** and **relative strength** of this signal compared to the other 20 features.
