## Expression

```
feat_btc_return_60s = (btc_mid[t] − btc_mid[t−60]) / btc_mid[t−60]
```

Fractional price change of the BTC/USD mid-price on Binance over the last 60 seconds.

---

## Motivation

The 60-second return bridges the very short (30s) and the full-horizon (300s) momentum
windows. It captures *medium-term* directional pressure — moves that have been sustaining
for a full minute are more likely to represent genuine order flow than fleeting noise.

In empirical microstructure research, returns at the 1-minute horizon have the highest
**autocorrelation** among short-window returns, making this a particularly informative
feature for short-horizon binary prediction.

**Interaction with other features:** When `feat_btc_return_30s` and `feat_btc_return_60s`
point in the same direction, the signal is consistent across timescales — the model should
be more confident. When they diverge (e.g., 30s down but 60s still up), the market may
be in a short-term retracement within a longer uptrend.
