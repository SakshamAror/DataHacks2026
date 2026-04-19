## Expression

```
feat_chainlink_vs_binance = (chainlink_btc[t] − binance_btc_mid[t]) / binance_btc_mid[t]
```

Fractional difference between the Chainlink oracle price and the Binance spot mid-price
for BTC/USD.

---

## Motivation

Chainlink is a decentralised oracle network that aggregates BTC price data from multiple
sources, but it updates on a slower heartbeat than the Binance spot exchange (which
updates in real time). This creates a predictable **lag structure**:

**Positive value** (oracle above spot): The last Chainlink update was above the current
Binance price. This means BTC has *fallen* since the oracle last updated. The next
oracle update will be a downward revision — and Polymarket traders who track the oracle
will likely push the YES price down when it updates.

**Negative value** (oracle below spot): BTC has *risen* since the oracle last updated.
When the oracle catches up, it will push YES prices higher on Polymarket.

**Why does Polymarket care about Chainlink?** Many Polymarket binary contracts settle
based on whether BTC (as measured by an oracle) is above or below a threshold. Traders
who know the oracle lags spot can position ahead of the oracle update to capture
the expected YES/NO repricing.

**The alpha:** This is essentially a **cross-venue arbitrage signal** — exploiting
the information asymmetry between a real-time exchange and a slower oracle feed.
